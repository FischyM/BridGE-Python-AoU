import argparse, psutil, signal, sys, threading, time
import multiprocessing as mp
from os import path

import datatools
from corefuns import matrix_operations_par as ci
from corefuns import genstats_perm as gs
from corefuns import fdrsampleperm as fdr
from corefuns import collectresults as cl



MODULE_CHOICES = ('DataProcess', 'ComputeInteraction', 'ComputeStats', 'ComputeFDR', 'Summarize')
VALID_MODELS = ('RR', 'RD', 'DD', 'combined')
SIM_MEASURES = ('jaccard', 'overlap', 'either')


def parse_args():
    p = argparse.ArgumentParser(description='BridGE pipeline', allow_abbrev=False)
    
    # required arguments
    p.add_argument('--projectDir', dest='project_dir', required=True)
    p.add_argument('--module', choices=MODULE_CHOICES, required=True)
    
    # common arguments
    p.add_argument('--model', choices=VALID_MODELS, default='combined')
    p.add_argument('--nWorker', dest='n_workers', type=int, default=None)  # None will use all available cores
    p.add_argument('--nJobs', dest='n_jobs', type=int, default=10)
    p.add_argument('--i', type=int, default=-1)
    p.add_argument('--R', dest='r', type=int, default=-1)
    p.add_argument('--densityCutoff', dest='density_cutoff', type=float, default=None)
    p.add_argument('--seed', dest='seed', type=int, default=42)
    p.add_argument('--ssmFile', dest='ssm_file', default=None)
    p.add_argument('--noMem', dest='no_mem', action='store_true')
    
    # data processing arguments
    p.add_argument('--plinkFile', dest='plinkfile', default='')
    p.add_argument('--geneSets', dest='gene_sets', default='')
    p.add_argument('--geneAnnotation', dest='gene_annotation', default='')
    p.add_argument('--mappingDistance',  dest='mapping_distance', type=int, default=50000)
    p.add_argument('--minPath', dest='min_path_size', type=int, default=10)
    p.add_argument('--maxPath', dest='max_path_size', type=int, default=300)
    p.add_argument('--simMeasure', dest='sim_measure', choices=SIM_MEASURES, default='jaccard')
    p.add_argument('--jaccardCutoff', dest='jaccard_cutoff', type=float, default=0.5)
    p.add_argument('--overlapCutoff', dest='overlap_cutoff', type=float, default=0.5)
    
    # compute interaction arguments
    p.add_argument('--alpha1', type=float, default=0.05)
    p.add_argument('--alpha2', type=float, default=0.05)
    
    # compute stats arguments
    p.add_argument('--binaryNetwork', dest='binary_network', action='store_true')
    p.add_argument('--snpPerms', dest='snp_perms', type=int, default=10000)
    
    # compute fdr arguments
    p.add_argument('--pvalueCutoff', dest='pval_cutoff', type=float, default=0.05)
    
    # summarize arguments
    p.add_argument('--fdrCutoff', dest='fdr_cutoff', type=float, default=0.25)
    
    return p.parse_args()


def require_exists(*filepaths):
    for filepath in filepaths:
        if not path.exists(filepath):
            sys.exit(f'{filepath} not found')


def run_data_process(args):
    print('Data Processing')

    # load in plink files and convert to pkl format. Additionally converts missing genotypes to 0
    print('converting plink files to pkl format...')
    pgen_file = f"{args.project_dir}/raw/{args.plinkfile}.pgen"
    pvar_file = f"{args.project_dir}/raw/{args.plinkfile}.pvar"
    psam_file = f"{args.project_dir}/raw/{args.plinkfile}.psam"
    require_exists(pgen_file, pvar_file, psam_file)
    snp_data_pkl = f"{args.project_dir}/intermediate/snp_data.pkl"
    datatools.plink2pkl(pgen_file, pvar_file, psam_file, snp_data_pkl)
    # TODO: Issues with changing MAF after all the preprocessing?

    # create a gene to pathway mapping from MSigDB gene set file.
    print('filtering and creating gene to pathway (gene set) mapping...')
    symbols_file = f"{args.project_dir}/raw/{args.gene_sets}.symbols.gmt"
    entrez_file = f"{args.project_dir}/raw/{args.gene_sets}.entrez.gmt"
    require_exists(symbols_file, entrez_file)
    gene_pathway_pkl = f"{args.project_dir}/intermediate/gene_pathway_mapping.pkl"
    datatools.msigdb2pkl(symbols_file, entrez_file, args.sim_measure, args.jaccard_cutoff,
                         args.overlap_cutoff, args.min_path_size, args.max_path_size, gene_pathway_pkl)

    # create a mapping of SNPs to genes with a mapping_distance extension to the start and end of each gene
    print('creating SNP to gene mapping...')
    # using a gene annotation file downloaded from Plink.
    gene_annotation_file = f"{args.project_dir}/raw/{args.gene_annotation}"
    require_exists(gene_annotation_file)
    snp_gene_pkl = f"{args.project_dir}/intermediate/snp_gene_mapping.pkl"
    datatools.mapsnp2gene(pvar_file, gene_annotation_file, args.mapping_distance, snp_gene_pkl)

    # create a mapping of SNPs to pathways using the snp_date.pkl, snp_gene_mapping.pkl and gene_pathway_mapping.pkl files.
    print('creating SNP to pathway mapping...')
    snp_pathway_pkl = f"{args.project_dir}/intermediate/snp_pathway_mapping.pkl"
    datatools.snppathway(args.project_dir, args.min_path_size, args.max_path_size, snp_pathway_pkl)
    
    # create a mapping of SNP indices for BPM/WPM sets using the snp_pathway_mapping.pkl file.
    print('creating SNP indices for BPM/WPM sets...')
    pathway_inds_pkl = f"{args.project_dir}/intermediate/pathway_indices.pkl"
    datatools.bpmind(args.project_dir, args.min_path_size, pathway_inds_pkl)


def run_compute_interaction(args):
    print("Computing SNP-SNP Interactions")
    
    # TODO: fix plink1 cluster file use for phenotype permutation
    
    # setup memory tracking
    def get_used_mem():
        return psutil.virtual_memory().total - psutil.virtual_memory().available

    initial_virtual_mem = get_used_mem()
    peak = 0
    stop_event = threading.Event()
    
    def monitor(interval=0.1):
        nonlocal peak
        while not stop_event.is_set():
            peak = max(peak, get_used_mem())
            time.sleep(interval)

    monitor_thread = threading.Thread(target=monitor)
    monitor_thread.start()

    # provide a more elegant way to cancel the worker pool on Ctrl+C
    def init_worker():
        # Ignore SIGINT in worker processes; only the main process should handle it
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    
    pool = mp.Pool(processes=args.n_workers, initializer=init_worker)
    try:
        indices = range(args.r + 1) if args.r >= 0 else [args.i]
        for i in indices:
            if args.model == 'combined':
                ci.combine(args.project_dir, args.alpha1, args.alpha2, args.n_jobs, args.n_workers, pool, i, args.seed)
            else:
                ci.run(args.project_dir, args.model, args.alpha1, args.alpha2, args.n_jobs, args.n_workers, pool, i, args.seed)
        pool.close()
        pool.join()
        
    except KeyboardInterrupt:
        print("\nCtrl+C received — terminating worker pool...")
        pool.terminate()
        pool.join()
        
    stop_event.set()
    monitor_thread.join()
    if not args.no_mem:
        print(f"peak memory of whole system: {(peak) / 1024**3:.2f} GB")
        print(f"peak memory attributed to BridGE: {(peak - initial_virtual_mem) / 1024**3:.2f} GB")
        
        
def run_compute_stats(args):
    
    # setup memory tracking
    def get_used_mem():
        return psutil.virtual_memory().total - psutil.virtual_memory().available

    initial_virtual_mem = get_used_mem()
    peak = 0
    stop_event = threading.Event()
    
    def monitor(interval=0.1):
        nonlocal peak
        while not stop_event.is_set():
            peak = max(peak, get_used_mem())
            time.sleep(interval)

    monitor_thread = threading.Thread(target=monitor)
    monitor_thread.start()
    
    if args.n_jobs < 2:
        args.n_jobs = 2
        print("n_jobs should never be less than 2 for computing stats. n_jobs will be changed to 2")
        
    if args.ssm_file is not None:
        ssm_file = f"{args.project_dir}/intermediate/{args.ssm_file}"
        print(f'Computing statistics on {args.ssm_file}')
        gs.genstats(args.project_dir, ssm_file, args.binary_network, args.density_cutoff, 
                    args.snp_perms, args.n_jobs, args.n_workers, args.seed)

    else:
        indices = range(args.r + 1) if args.r >= 0 else [args.i]
        for i in indices:
            ssm_file = f"{args.project_dir}/intermediate/ssM_mhygessi_{args.model}_R{i}.pkl"
            print(f'Computing statistics on {args.model}_R{i}')
            gs.genstats(args.project_dir, ssm_file, args.binary_network, args.density_cutoff,
                        args.snp_perms, args.n_jobs, args.n_workers, args.seed)
            
    stop_event.set()
    monitor_thread.join()
    if not args.no_mem:
        print(f"peak memory of whole system: {(peak) / 1024**3:.2f} GB")
        print(f"peak memory attributed to BridGE: {(peak - initial_virtual_mem) / 1024**3:.2f} GB")
        
        
def run_compute_fdr(args):
    if args.ssm_file is None:
        ssm_file = f"{args.project_dir}/intermediate/ssM_mhygessi_{args.model}_R0.pkl"
    else:
        ssm_file = f"{args.project_dir}/intermediate/{args.ssm_file}"

    print(f'Computing FDR')
    fdr.fdrsampleperm(args.project_dir, ssm_file, args.pval_cutoff, args.R)


def run_summarize(args):
    if args.ssm_file is None:
        imported = False
        ssm_file = f"{args.project_dir}/intermediate/ssM_mhygessi_{args.model}_R0.pkl"
    else:
        imported = True
        ssm_file = f"{args.project_dir}/intermediate/{args.ssm_file}"
        
    cl.collectresults(args.project_dir, ssm_file, args.model, args.fdr_cutoff, imported, args.density_cutoff)


MODULES_RUN = {
    'DataProcess': run_data_process,
    'ComputeInteraction': run_compute_interaction,
    'ComputeStats': run_compute_stats,
    'ComputeFDR': run_compute_fdr,
    'Summarize': run_summarize,
}

def main():
    args = parse_args()
    module_fn = MODULES_RUN.get(args.module)
    if module_fn is None:
        sys.exit(f'unknown module: {args.module!r}')
    module_fn(args)

if __name__ == '__main__':
    main()
    
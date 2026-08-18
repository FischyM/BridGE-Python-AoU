import argparse, sys
from os import path
import multiprocessing as mp

from corefuns import collectresults as cl
from corefuns import fdrsampleperm as fdr
from corefuns import genstats_perm as gs
from corefuns import matrix_operations_par as ci
import datatools


MODULE_CHOICES = ('DataProcess', 'ComputeInteraction', 'ComputeStats', 'ComputeFDR', 'Summarize')
VALID_MODELS = ('RR', 'RD', 'DD', 'combined')


def parse_args():
    p = argparse.ArgumentParser(description='BridGE pipeline', allow_abbrev=False, suggest_on_error=True)
    
    # required arguments
    p.add_argument('--projectDir', dest='project_dir', required=True)
    p.add_argument('--module', dest='module', choices=MODULE_CHOICES, required=True)
    
    # common arguments
    p.add_argument('--model', choices=VALID_MODELS, default='combined')
    p.add_argument('--nJobs', dest='n_jobs', type=int, default=10)
    p.add_argument('--nWorker', dest='n_workers', type=int, default=None)  # None means use all available cores
    p.add_argument('--minPath', type=int, default=10)  # TODO: is this needed for compute stats? How can this permeate through?
    p.add_argument('--i', type=int, default=-1)
    p.add_argument('--R', dest='r', type=int, default=-1)
    p.add_argument('--densityCutoff', dest='densitycutoff', type=float, default=None)
    p.add_argument('--ssmfile', default=None)
    
    # data processing arguments
    p.add_argument('--plinkFile', dest='plinkfile', default='')
    p.add_argument('--genesets', default='c2.cp.v7.1')
    p.add_argument('--geneAnnotation', dest='gene_annotation', default='glist-hg38')
    p.add_argument('--mappingDistance', type=int, default=50000)
    p.add_argument('--maxPath', type=int, default=300)
    
    # compute interaction arguments
    p.add_argument('--alpha1', type=float, default=0.05)
    p.add_argument('--alpha2', type=float, default=0.05)
    
    # compute stats arguments
    p.add_argument('--binaryNetwork', action='store_true')
    p.add_argument('--snpPerms', dest='snp_perms', type=int, default=10000)
    
    # compute fdr arguments
    p.add_argument('--samplePerms', dest='sample_perms', type=int, default=10)  # this is just R, so maybe it isn't needed to be a separate argument? but it is for now
    p.add_argument('--pvalueCutoff', dest='pval_cutoff', type=float, default=0.05)
    
    # summarize arguments
    p.add_argument('--fdrcut', type=float, default=0.25)
    p.add_argument('--snpPathFile', dest='snppathwayfile', default='snp_pathway_min10_max300.pkl') # TODO: may not need this. save as an unambigous file with min and max saved to the dataclass
    
    return p.parse_args()

def require_exists(*filepaths):
    for filepath in filepaths:
        if not path.exists(filepath):
            sys.exit(f'{filepath} not found')

def run_data_process(args):
    print('data processing...')

    pgen_file = f"{args.project_dir}/raw/{args.plinkfile}.pgen"
    pvar_file = f"{args.project_dir}/raw/{args.plinkfile}.pvar"
    psam_file = f"{args.project_dir}/raw/{args.plinkfile}.psam"
    require_exists(pgen_file, pvar_file, psam_file)
    snp_data_pkl = f"{args.project_dir}/intermediate/snp_data.pkl"
    datatools.plink2pkl(pgen_file, pvar_file, psam_file, snp_data_pkl)

    symbols_file = f"{args.project_dir}/raw/{args.genesets}.symbols.gmt"
    entrez_file = f"{args.project_dir}/raw/{args.genesets}.entrez.gmt"
    require_exists(symbols_file, entrez_file)
    gene_pathway_pkl = f"{args.project_dir}/intermediate/gene_pathway_mapping.pkl"
    datatools.msigdb2pkl(symbols_file, entrez_file, gene_pathway_pkl)
    # TODO: reduce gene set based on Jaccard similarity?

    gene_annotation_file = f"{args.project_dir}/raw/{args.gene_annotation}"
    require_exists(gene_annotation_file)
    snp_gene_pkl = f"{args.project_dir}/intermediate/snp_gene_mapping.pkl"
    datatools.mapsnp2gene(pvar_file, gene_annotation_file, args.mappingDistance, snp_gene_pkl)

    snp_pathway_pkl = f"{args.project_dir}/intermediate/snp_pathway_mapping.pkl"
    datatools.snppathway(args.project_dir, args.minPath, args.maxPath, snp_pathway_pkl)
    
    pathway_inds_pkl = f"{args.project_dir}/intermediate/pathway_indices.pkl"
    datatools.bpmind(args.project_dir, pathway_inds_pkl)
    # TODO: add in min_path here to remove pathways that are too small (after removing SNPs in both BPM pathways)
    # that are thrown out anyways in Compute Stats

def run_compute_interaction(args):

    # TODO: add in memory tracking? See test_code-small_snps.ipynb for example
    
    pool = mp.Pool(processes=args.n_workers)
    
    indices = range(args.r + 1) if args.r >= 0 else [args.i]
    for i in indices:
        if args.model == 'combined':
            ci.combine(args.project_dir, args.alpha1, args.alpha2, args.n_jobs, args.n_workers, pool, i)
            print("\n")
        else:
            ci.run(args.project_dir, args.model, args.alpha1, args.alpha2, args.n_jobs, args.n_workers, pool, i)

    pool.close()
    pool.join()

def run_compute_stats(args):

    bpmfile = f"{args.project_dir}/intermediate/pathway_indices.pkl"
    require_exists(bpmfile)

    if args.n_jobs < 2:
        args.n_jobs = 2
        print("n_jobs should never be less than 2 for computing stats. n_jobs will be changed to 2")
        
    if args.ssmfile is not None:
        ssmfile = f"{args.project_dir}/intermediate/{args.ssmfile}"
        print(f'Computing statistics on {args.ssmfile}')
        gs.genstats(ssmfile, bpmfile, args.binaryNetwork, args.snp_perms,
                    args.minPath, args.n_jobs, args.n_workers, args.densitycutoff)
    else:
        indices = range(args.r + 1) if args.r >= 0 else [args.i]
        for i in indices:
            ssmfile = f"{args.project_dir}/intermediate/ssM_mhygessi_{args.model}_R{i}.pkl"
            print(f'Computing statistics on {args.model}_R{i}')
            gs.genstats(ssmfile, bpmfile, args.binaryNetwork, args.snp_perms,
                        args.minPath, args.n_jobs, args.n_workers, args.densitycutoff)

def run_compute_fdr(args):
    if args.ssmfile is None:
        ssmfile = f"{args.project_dir}/intermediate/ssM_mhygessi_{args.model}_R0.pkl"
    else:
        ssmfile = f"{args.project_dir}/intermediate/{args.ssmfile}"
    require_exists(ssmfile)

    print(f'Computing FDR')
    fdr.fdrsampleperm(ssmfile, args.pval_cutoff, args.sample_perms)

def run_summarize(args):
    bpmfile = f"{args.project_dir}/intermediate/BPM_WPM_indices.pkl"
    require_exists(bpmfile)

    snppathwayfile = f"{args.project_dir}/intermediate/{args.snppathwayfile}"
    require_exists(snppathwayfile)

    # TODO: this should be made into a dataclass that is unambiguous to load
    snpgenemappingfile = f"{args.project_dir}/intermediate/snpgenemapping_{args.mappingDistance // 1000}kb.pkl"
    require_exists(snpgenemappingfile)

    if args.ssmfile is None:
        imported = False
        ssmfile = f"{args.project_dir}/intermediate/ssM_mhygessi_{args.model}_R0.pkl"
        resultsfile = f"{args.project_dir}/intermediate/results_ssM_mhygessi_{args.model}_R0.pkl"
    else:
        imported = True
        ssmfile = f"{args.project_dir}/intermediate/{args.ssmfile}"
        resultsfile = f"{args.project_dir}/intermediate/results_{args.ssmfile}"
    require_exists(ssmfile, resultsfile)

    cl.collectresults(resultsfile, args.fdrcut, ssmfile, bpmfile,
                       snppathwayfile, snpgenemappingfile, imported, args.densitycutoff)

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
    
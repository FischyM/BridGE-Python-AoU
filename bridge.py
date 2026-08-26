import argparse, sys
import multiprocessing as mp
from os import path

import datatools
from corefuns import matrix_operations_par as ci
from corefuns import genstats_perm as gs
from corefuns import fdrsampleperm as fdr
from corefuns import collectresults as cl


MODULE_CHOICES = ('DataProcess', 'ComputeInteraction', 'ComputeStats', 'ComputeFDR', 'Summarize')
VALID_MODELS = ('RR', 'RD', 'DD', 'combined')


def parse_args():
    p = argparse.ArgumentParser(description='BridGE pipeline', allow_abbrev=False)
    
    # required arguments
    p.add_argument('--projectDir', dest='project_dir', required=True)
    p.add_argument('--module', dest='module', choices=MODULE_CHOICES, required=True)
    
    # common arguments
    p.add_argument('--model', choices=VALID_MODELS, default='combined')
    p.add_argument('--nJobs', dest='n_jobs', type=int, default=10)
    p.add_argument('--nWorker', dest='n_workers', type=int, default=None)  # None will use all available cores
    p.add_argument('--i', type=int, default=-1)
    p.add_argument('--R', dest='r', type=int, default=-1)
    p.add_argument('--densityCutoff', dest='densitycutoff', type=float, default=None)
    p.add_argument('--seed', dest='seed', type=int, default=42)
    p.add_argument('--ssmFile', dest='ssmfile', default=None)
    
    # data processing arguments
    p.add_argument('--plinkFile', dest='plinkfile', default='')
    p.add_argument('--genesets', default='c2.cp.v7.1')
    p.add_argument('--geneAnnotation', dest='gene_annotation', default='glist-hg38')
    p.add_argument('--mappingDistance', type=int, default=50000)
    p.add_argument('--minPath', type=int, default=10)
    p.add_argument('--maxPath', type=int, default=300)
    
    # compute interaction arguments
    p.add_argument('--alpha1', type=float, default=0.05)
    p.add_argument('--alpha2', type=float, default=0.05)
    
    # compute stats arguments
    p.add_argument('--binaryNetwork', action='store_true')
    p.add_argument('--snpPerms', dest='snp_perms', type=int, default=10000)
    
    # compute fdr arguments
    p.add_argument('--pvalueCutoff', dest='pval_cutoff', type=float, default=0.05)
    
    # summarize arguments
    p.add_argument('--fdrcut', type=float, default=0.25)
    
    return p.parse_args()

def require_exists(*filepaths):
    for filepath in filepaths:
        if not path.exists(filepath):
            sys.exit(f'{filepath} not found')

def run_data_process(args):
    print('data processing...')

    # load in plink files and convert to pkl format. Additionally converts missing genotypes to 0
    pgen_file = f"{args.project_dir}/raw/{args.plinkfile}.pgen"
    pvar_file = f"{args.project_dir}/raw/{args.plinkfile}.pvar"
    psam_file = f"{args.project_dir}/raw/{args.plinkfile}.psam"
    require_exists(pgen_file, pvar_file, psam_file)
    snp_data_pkl = f"{args.project_dir}/intermediate/snp_data.pkl"
    datatools.plink2pkl(pgen_file, pvar_file, psam_file, snp_data_pkl)

    # create a gene to pathway mapping from MSigDB gene set file.
    symbols_file = f"{args.project_dir}/raw/{args.genesets}.symbols.gmt"
    entrez_file = f"{args.project_dir}/raw/{args.genesets}.entrez.gmt"
    require_exists(symbols_file, entrez_file)
    gene_pathway_pkl = f"{args.project_dir}/intermediate/gene_pathway_mapping.pkl"
    datatools.msigdb2pkl(symbols_file, entrez_file, gene_pathway_pkl)
    # TODO: reduce gene set based on Jaccard similarity?


    # create a mapping of SNPs to genes with a mappingDistance extension to the start and end of each gene
    # using a gene annotation file downloaded from Plink.
    gene_annotation_file = f"{args.project_dir}/raw/{args.gene_annotation}"
    require_exists(gene_annotation_file)
    snp_gene_pkl = f"{args.project_dir}/intermediate/snp_gene_mapping.pkl"
    datatools.mapsnp2gene(pvar_file, gene_annotation_file, args.mappingDistance, snp_gene_pkl)

    # create a mapping of SNPs to pathways using the snp_date.pkl, snp_gene_mapping.pkl and gene_pathway_mapping.pkl files.
    snp_pathway_pkl = f"{args.project_dir}/intermediate/snp_pathway_mapping.pkl"
    datatools.snppathway(args.project_dir, args.minPath, args.maxPath, snp_pathway_pkl)
    
    # create a mapping of SNP indices for BPM/WPM sets using the snp_pathway_mapping.pkl file.
    pathway_inds_pkl = f"{args.project_dir}/intermediate/pathway_indices.pkl"
    datatools.bpmind(args.project_dir, args.minPath, pathway_inds_pkl)

def run_compute_interaction(args):

    # TODO: add in memory tracking?
    # TODO: redo/remove per job split print statements?
    
    pool = mp.Pool(processes=args.n_workers)
    
    indices = range(args.r + 1) if args.r >= 0 else [args.i]
    for i in indices:
        if args.model == 'combined':
            ci.combine(args.project_dir, args.alpha1, args.alpha2, args.n_jobs, args.n_workers, pool, i, args.seed)
        else:
            ci.run(args.project_dir, args.model, args.alpha1, args.alpha2, args.n_jobs, args.n_workers, pool, i, args.seed)
        print(flush=True)
        
    pool.close()
    pool.join()

def run_compute_stats(args):
    
    # TODO: add in memory tracking?

    if args.n_jobs < 2:
        args.n_jobs = 2
        print("n_jobs should never be less than 2 for computing stats. n_jobs will be changed to 2")
        
    if args.ssmfile is not None:
        ssmfile = f"{args.project_dir}/intermediate/{args.ssmfile}"
        print(f'Computing statistics on {args.ssmfile}')
        gs.genstats(args.project_dir, ssmfile, args.binaryNetwork, args.densitycutoff, 
                    args.snp_perms, args.n_jobs, args.n_workers, args.seed)

    else:
        indices = range(args.r + 1) if args.r >= 0 else [args.i]
        for i in indices:
            ssmfile = f"{args.project_dir}/intermediate/ssM_mhygessi_{args.model}_R{i}.pkl"
            print(f'Computing statistics on {args.model}_R{i}')
            gs.genstats(args.project_dir, ssmfile, args.binaryNetwork, args.densitycutoff,
                        args.snp_perms, args.n_jobs, args.n_workers, args.seed)

def run_compute_fdr(args):
    if args.ssmfile is None:
        ssmfile = f"{args.project_dir}/intermediate/ssM_mhygessi_{args.model}_R0.pkl"
    else:
        ssmfile = f"{args.project_dir}/intermediate/{args.ssmfile}"

    print(f'Computing FDR')
    fdr.fdrsampleperm(args.project_dir, ssmfile, args.pval_cutoff, args.R)

def run_summarize(args):
    if args.ssmfile is None:
        imported = False
        ssmfile = f"{args.project_dir}/intermediate/ssM_mhygessi_{args.model}_R0.pkl"
    else:
        imported = True
        ssmfile = f"{args.project_dir}/intermediate/{args.ssmfile}"
        
    cl.collectresults(args.project_dir, ssmfile, args.model, args.fdrcut, imported, args.densitycutoff)

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
    
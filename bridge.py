import argparse, sys
from os import path
import multiprocessing as mp

from corefuns import collectresults as cl
from corefuns import fdrsampleperm as fdr
from corefuns import genstats_perm as gs
from corefuns import matrix_operations_par as ci
from datatools import bindataa as ba
from datatools import bpmind as bpm
from datatools import mapsnp2gene as snp2gene
from datatools import msigdb2pkl as msig2p
from datatools import plink2pkl as p2p
from datatools import snppathway as snpp

VALID_MODELS = {'RR', 'RD', 'DD', 'combined'}


def parse_args():
    p = argparse.ArgumentParser(description='BridGE pipeline', allow_abbrev=False)
    p.add_argument('--job', default='')
    p.add_argument('--plinkFile', dest='plinkfile', default='')
    p.add_argument('--genesets', default='c2.cp.v7.1')
    p.add_argument('--geneAnnotation', dest='gene_annotation', default='glist-hg38')
    p.add_argument('--mappingDistance', type=int, default=50000)
    p.add_argument('--minPath', type=int, default=10)
    p.add_argument('--maxPath', type=int, default=300)
    p.add_argument('--model', default=None)
    p.add_argument('--nJobs', dest='n_jobs', type=int, default=10)
    p.add_argument('--nWorker', dest='n_workers', default=None)  # None means use all available cores
    p.add_argument('--samplePerms', dest='sample_perms', type=int, default=10)
    p.add_argument('--binaryNetwork', type=int, default=0)
    p.add_argument('--snpPerms', type=int, default=10000)
    p.add_argument('--pvalueCutoff', dest='pval_cutoff', type=float, default=0.05)
    p.add_argument('--i', type=int, default=-1)
    p.add_argument('--fdrcut', type=float, default=0.25)
    p.add_argument('--snpPathFile', dest='snppathwayfile', default='snp_pathway_min10_max300.pkl')
    p.add_argument('--projectDir', dest='project_dir', default='data')
    p.add_argument('--densityCutoff', dest='densitycutoff', type=float, default=None)
    p.add_argument('--ssmfile', default=None)
    p.add_argument('--R', dest='r', type=int, default=-1)

    args = p.parse_args()
    # alpha1/alpha2 were hardcoded constants in the original; no CLI flag
    # ever set them, so keep them fixed here.
    args.alpha1 = 0.05
    args.alpha2 = 0.05
    args.binaryNetwork = bool(args.binaryNetwork)
    return args

def require_exists(*filepaths):
    for filepath in filepaths:
        if not path.exists(filepath):
            sys.exit(f'{filepath} not found')

def _require_model(args, allow_ssmfile=False):
    if args.model in VALID_MODELS:
        return
    if allow_ssmfile and args.ssmfile is not None:
        return
    sys.exit('wrong model')

def _snp_data_files(args):
    snp_data_ad = f"{args.project_dir}/intermediate/SNPdataAD.pkl"
    snp_data_ar = f"{args.project_dir}/intermediate/SNPdataAR.pkl"
    require_exists(snp_data_ad, snp_data_ar)
    return snp_data_ad, snp_data_ar

def _ssm_filename(project_dir, model, r_index):
    return f"{project_dir}/intermediate/ssM_mhygessi_{model}_R{r_index}.pkl"


# Module Functions

def run_data_process(args):
    print('data processing...')
    sys.stdout.flush()

    if not args.plinkfile:
        sys.exit('plinkFile not provided')

    # TODO: change these to use pgen files
    rawfile = f"{args.project_dir}/intermediate/{args.plinkfile}.raw"
    bimfile = f"{args.project_dir}/intermediate/{args.plinkfile}.bim"
    famfile = f"{args.project_dir}/intermediate/{args.plinkfile}.fam"
    require_exists(rawfile, bimfile, famfile)

    finalfile = f"{args.project_dir}/intermediate/{args.plinkfile}.pkl"
    p2p.plink2pkl(rawfile, bimfile, famfile, finalfile)
    # TODO: read in pgen file directly instead of converting to raw first

    ba.bindataa(args.project_dir, finalfile, 'r')
    ba.bindataa(args.project_dir, finalfile, 'd')
    # TODO: remove these and simply load the original data and change to r or d as needed.

    symbolsfile = f"{args.project_dir}/raw/{args.genesets}.symbols.gmt"
    entrezfile = f"{args.project_dir}/raw/{args.genesets}.entrez.gmt"
    require_exists(symbolsfile, entrezfile)
    msig2p.msigdb2pkl(symbolsfile, entrezfile)
    # TODO: reduce gene set based on Jaccard similarity?

    gene_annotation_file = f"{args.project_dir}/raw/{args.gene_annotation}"
    require_exists(gene_annotation_file)
    sgmfile = f"{args.project_dir}/intermediate/snpgenemapping_{args.mappingDistance // 1000}kb.pkl"
    snp2gene.mapsnp2gene(bimfile, gene_annotation_file, args.mappingDistance, 'matrix', sgmfile)
    # TODO: shouldn't this output also be a dataclass?

    geneset_pkl = f"{args.project_dir}/intermediate/{args.genesets}.pkl"
    outfile = snpp.snppathway(finalfile, sgmfile, geneset_pkl, args.minPath, args.maxPath)
    bpm.bpmind(outfile)

def run_compute_interaction(args):
    _require_model(args)
    _snp_data_files(args)

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
    _require_model(args, allow_ssmfile=True)

    bpmfile = f"{args.project_dir}/intermediate/BPMind.pkl"
    require_exists(bpmfile)
    _snp_data_files(args)

    if args.n_jobs < 2:
        args.n_jobs = 2
        print("n_jobs should never be less than 2 for computing stats. n_jobs will be changed to 2")
        
    if args.ssmfile is not None:
        ssmfile = f"{args.project_dir}/intermediate/{args.ssmfile}"
        print(f'Computing statistics on {args.ssmfile}')
        gs.genstats(ssmfile, bpmfile, args.binaryNetwork, args.snpPerms,
                    args.minPath, args.n_jobs, args.n_workers, args.densitycutoff)
    else:
        indices = range(args.r + 1) if args.r >= 0 else [args.i]
        for i in indices:
            ssmfile = _ssm_filename(args.project_dir, args.model, i)
            print(f'Computing statistics on {args.model}_R{i}')
            gs.genstats(ssmfile, bpmfile, args.binaryNetwork, args.snpPerms,
                        args.minPath, args.n_jobs, args.n_workers, args.densitycutoff)

def run_compute_fdr(args):
    if args.ssmfile is None:
        ssmfile = _ssm_filename(args.project_dir, args.model, 0)
    else:
        ssmfile = f"{args.project_dir}/intermediate/{args.ssmfile}"
    require_exists(ssmfile)

    print(f'Computing FDR')
    fdr.fdrsampleperm(ssmfile, args.pval_cutoff, args.sample_perms)

def run_summarize(args):
    bpmfile = f"{args.project_dir}/intermediate/BPMind.pkl"
    require_exists(bpmfile)

    snppathwayfile = f"{args.project_dir}/intermediate/{args.snppathwayfile}"
    require_exists(snppathwayfile)

    snpgenemappingfile = f"{args.project_dir}/intermediate/snpgenemapping_{args.mappingDistance // 1000}kb.pkl"
    require_exists(snpgenemappingfile)

    if args.ssmfile is None:
        imported = False
        resultsfile = f"{args.project_dir}/intermediate/results_ssM_mhygessi_{args.model}_R0.pkl"
        ssmfile = _ssm_filename(args.project_dir, args.model, 0)
    else:
        imported = True
        resultsfile = f"{args.project_dir}/intermediate/results_{args.ssmfile}"
        ssmfile = f"{args.project_dir}/intermediate/{args.ssmfile}"

    require_exists(ssmfile, resultsfile)

    cl.collectresults(resultsfile, args.fdrcut, ssmfile, bpmfile,
                       snppathwayfile, snpgenemappingfile, imported, args.densitycutoff)

JOBS = {
    'DataProcess': run_data_process,
    'ComputeInteraction': run_compute_interaction,
    'ComputeStats': run_compute_stats,
    'ComputeFDR': run_compute_fdr,
    'Summarize': run_summarize,
}

def main():
    args = parse_args()
    job_fn = JOBS.get(args.job)
    if job_fn is None:
        sys.exit(f'unknown job: {args.job!r}')
    job_fn(args)

if __name__ == '__main__':
    main()
    
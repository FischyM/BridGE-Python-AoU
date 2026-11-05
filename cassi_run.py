"""Compute SNP-SNP interactions with CASSI and convert them to BridGE's sparse
interaction network format (.npz).

Options:
    --plinkFile:   input plink file without the extension
    --GI:          genetic interaction type, similar to CASSI. Check CASSI for more information.
                   lr: logistic regression, lin: linear regression, je: joint effects,
                   awu: adjusted Wu, afe: adjusted fast epistasis, wz: Wellek Ziegler
    --pvalue:      p-value cutoff for filtering interactions in CASSI.
    --R:           number of sample permutations to be used for generating random interaction networks.
    --i:           random network number (0 means real network) to be computed. This can be used
                   instead of --R for creating only one network.
    --seed:        random seed for reproducibility (default 42).
    --nWorker:     number of networks (CASSI runs and conversions) to compute at once, one CPU
                   core each (default 1; -1 uses all available cores). With more than one, CASSI's
                   screen output is suppressed; each run still writes <model>-R<i>.log.
    --convertOnly: skip CASSI and only convert existing CASSI output files to .npz
                   (--pvalue is then not needed).

Random network i (i >= 1) permutes the .fam phenotypes exactly as bridge.py's ComputeInteraction
does for R<i>: the same seed (seed * i, or i when --seed=0), and, if <project>/raw/bridge.cluster.tsv
exists, shuffling only within its genetic similarity clusters. The permuted .fam is written to
<project>/intermediate next to links to the .bed/.bim, and removed once CASSI has run.

https://www.staff.ncl.ac.uk/richard.howey/cassi/

Plink files live in <project>/raw; CASSI output (<model>-R<i>.txt) and networks go to
<project>/intermediate, the networks named like the ones bridge.py computes
(snp_interaction_networks-<model>-R<i>.npz) with model cassi_<GI>. Use them by giving --model=cassi_<GI> (e.g. --model=cassi_LR) to bridge.py's
ComputeStats, ComputeFDR, and Summarize.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy.sparse import coo_array

from src.compute_interactions import run_pheno_perm, save_sparse_array
from src.polars_io import read_whitespace

# CASSI threshold options and BridGE model name for each interaction type
GI_TYPES = {
    "lr": (["-lr", "-lr-th"], "cassi_LR"),
    "lin": (["-lin", "-lin-th"], "cassi_LIN"),
    "je": (["-je", "-je-thcc"], "cassi_JE"),
    "awu": (["-awu", "-awu-thcc"], "cassi_AWU"),
    "afe": (["-afe", "-afe-thcc"], "cassi_AFE"),
    "wz": (["-wz", "-wz-thcc"], "cassi_WZ"),
}


def add_suffix(prefix, suffix):
    """Append to a plink file prefix (Path.with_suffix would replace any dot in the name)."""
    return prefix.with_name(prefix.name + suffix)


def check_snp_order(bim_file, snp_data_file):
    """Exit unless the .bim lists the same SNPs, in the same order, as snp_data.npz.

    CASSI numbers SNPs by their .bim row, and the networks are indexed that way, while
    ComputeStats and Summarize index them by snp_data.npz row (pathway_indices, varid).
    """
    if not snp_data_file.exists():
        print(f"warning: {snp_data_file} not found; cannot check that {bim_file.name} has the same "
              f"SNPs in the same order (run DataProcess first)", flush=True)
        return
    bim_ids = np.array([line.split()[1] for line in bim_file.read_text().splitlines() if line.strip()])
    with np.load(snp_data_file) as f:
        varid = f['varid']
    if bim_ids.size != varid.size:
        sys.exit(f"{bim_file} has {bim_ids.size} SNPs but {snp_data_file} has {varid.size}; "
                 f"the CASSI network would not line up with the BridGE SNP indices")
    mismatch = np.flatnonzero(bim_ids != varid)
    if mismatch.size:
        k = mismatch[0]
        sys.exit(f"{bim_file} and {snp_data_file} list SNPs in a different order ({mismatch.size} rows differ, "
                 f"first at row {k + 1}: {bim_ids[k]} vs {varid[k]}); the CASSI network would not line up "
                 f"with the BridGE SNP indices")


def permute_fam(fam_file, out_fam, seed, i, cluster_file, snp_data_file):
    """Write a copy of a plink .fam with its phenotypes permuted the way
    compute_interactions.run_single permutes them for random network R<i>.

    Args:
        fam_file (Path): plink .fam of the real data (phenotypes coded 1=control, 2=case)
        out_fam (Path): permuted .fam to write
        seed (int): bridge.py's --seed
        i (int): random network number (>= 1)
        cluster_file (Path): bridge.cluster.tsv; permute within clusters if it exists
        snp_data_file (Path): snp_data.npz; only used to warn if its sample order differs,
            since the permutation is only identical to BridGE's for the same sample order
    """
    rows = [line.split() for line in fam_file.read_text().splitlines() if line.strip()]
    iid = np.array([r[1] for r in rows])
    pheno = np.array([int(r[5]) for r in rows]) - 1
    if not np.isin(pheno, (0, 1)).all():
        sys.exit(f"{fam_file}: phenotypes must be 1 (control) or 2 (case)")

    if snp_data_file.exists():
        with np.load(snp_data_file) as f:
            if not np.array_equal(f['iid'], iid):
                print(f"warning: sample order of {fam_file.name} differs from {snp_data_file.name}; "
                      f"R{i} permutation will not match BridGE's", flush=True)

    # same generator as compute_interactions.run_single
    rng = np.random.default_rng(seed * i) if seed != 0 else np.random.default_rng(i)
    if cluster_file.exists():
        print(f"permuting phenotypes for R{i} within clusters of {cluster_file.name}", flush=True)
        pheno = run_pheno_perm(rng, cluster_file, pheno, iid)
    else:
        print(f"permuting phenotypes for R{i}", flush=True)
        pheno = pheno[rng.permutation(pheno.size)]

    for r, ph in zip(rows, pheno):
        r[5] = str(ph + 1)
    out_fam.write_text("".join("\t".join(r) + "\n" for r in rows))


def to_csr_network(rows, cols, values, p):
    """Symmetric p x p csr_array from (row, col, value) triples.

    A pair listed more than once keeps its last value, and each pair is mirrored
    with the elementwise max, as the earlier dense version did.
    """
    keys = rows * p + cols
    _, last_reversed = np.unique(keys[::-1], return_index=True)
    keep = keys.size - 1 - last_reversed
    network = coo_array((values[keep], (rows[keep], cols[keep])), shape=(p, p)).tocsr()
    network = network.maximum(network.T).tocsr()
    network.eliminate_zeros()
    return network


def cassi_to_network(bim_file, cassi_file, gi, output_file):
    """Convert CASSI output to BridGE's sparse interaction network format, the same
    .npz layout compute_interactions.py writes (read back with load_sparse_array).

    Args:
        bim_file (Path): plink .bim file of the data CASSI ran on (sets the number of SNPs)
        cassi_file (Path): CASSI output file
        gi (str): genetic interaction type (a key of GI_TYPES)
        output_file (Path): output file name (.npz)
    """
    print(f"converting {cassi_file} -> {output_file}", flush=True)
    ssm_cassi = read_whitespace(cassi_file)
    p = read_whitespace(bim_file, has_header=False).shape[0]

    def col(name):
        return ssm_cassi[name].to_numpy()

    x = col('SNP1') - 1
    y = col('SNP2') - 1

    # -log10 p-value column, and which interactions are protective / risk
    if gi == 'lr':
        z = -np.log10(col('LR_P'))
        protective = col('LR_LOG_OR') < 0
        risk = col('LR_LOG_OR') > 0
    elif gi == 'lin':
        z = -np.log10(col('LIN_P'))
        protective = col('LIN_BETA') < 0
        risk = col('LIN_BETA') >= 0
    elif gi in ('je', 'awu', 'afe'):
        prefix = gi.upper()
        z = -np.log10(col(f'{prefix}_CC_P'))
        case_log_or = col(f'{prefix}_CASE_LOG_OR')
        ctrl_log_or = col(f'{prefix}_CTRL_LOG_OR')
        protective = ctrl_log_or > case_log_or
        risk = ctrl_log_or <= case_log_or
    elif gi == 'wz':
        z = -np.log10(col('WZ_CC_P'))
        protective = col('WZ_CTRL_R') > col('WZ_CASE_R')
        risk = col('WZ_CTRL_R') <= col('WZ_CASE_R')
    else:
        raise ValueError(f"unknown GI type {gi!r}; expected one of {', '.join(GI_TYPES)}")

    save_sparse_array(output_file, {
        'risk': to_csr_network(x[risk], y[risk], z[risk], p),
        'protective': to_csr_network(x[protective], y[protective], z[protective], p),
    })


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plinkFile", dest="infile", type=Path)
    parser.add_argument("--GI", dest="gi")
    parser.add_argument("--pvalue", dest="pval")
    parser.add_argument("--R", dest="n", type=int)
    parser.add_argument("--i", dest="net_id", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nWorker", dest="n_workers", type=int, default=1)  # -1 uses all available cores
    parser.add_argument("--convertOnly", action="store_true")
    args = parser.parse_args()

    if not args.infile:
        sys.exit("plink input file not provided")
    if not args.gi:
        sys.exit("CASSI interaction type not provided")
    if not args.pval and not args.convertOnly:
        sys.exit("p-value cutoff not provided")
    if args.n is None and args.net_id is None:
        sys.exit("number of sample permutations and network id not provided")
    if args.gi not in GI_TYPES:
        sys.exit(f"unknown CASSI interaction type: {args.gi}")
    return args


def main(args):
    infile, gi, pval, seed = args.infile, args.gi, args.pval, args.seed
    th_command, model = GI_TYPES[gi]

    project_dir = infile.parent.parent
    intermediate_dir = project_dir / "intermediate"
    cluster_file = project_dir / "raw" / "bridge.cluster.tsv"
    snp_data_file = intermediate_dir / "snp_data.npz"
    print(intermediate_dir)
    check_snp_order(add_suffix(infile, ".bim"), snp_data_file)

    ids = list(range(args.n + 1)) if args.net_id is None else [args.net_id]
    parallel = args.n_workers != 1 and len(ids) > 1

    def compute_network(i):
        cassi_out = intermediate_dir / f"{model}-R{i}.txt"
        if not args.convertOnly:
            if i == 0:
                prefix = infile
                temp_files = []
            else:
                # CASSI reads <prefix>.bed/.bim/.fam: link the real .bed/.bim and write a permuted .fam
                prefix = intermediate_dir / f"{infile.name}_R{i}"
                temp_files = [add_suffix(prefix, ext) for ext in (".bed", ".bim", ".fam")]
                for f in temp_files:
                    f.unlink(missing_ok=True)
                for ext in (".bed", ".bim"):
                    add_suffix(prefix, ext).symlink_to(add_suffix(infile, ext).resolve())
                permute_fam(add_suffix(infile, ".fam"), add_suffix(prefix, ".fam"),
                            seed, i, cluster_file, snp_data_file)

            # Step 1, compute SNP-SNP interaction using CASSI
            bed = add_suffix(prefix, ".bed")
            cassi_cmd = f"cassi -i {bed} {' '.join(th_command)} {pval} -max 0 -mem2 -o {cassi_out}"
            if parallel:
                # concurrent runs would interleave their screen output; it is also in the .log
                cassi_cmd += " -so"
            print(f"R{i}: {cassi_cmd}", flush=True)
            try:
                subprocess.run(cassi_cmd, shell=True, check=True)
            finally:
                for f in temp_files:
                    f.unlink(missing_ok=True)

        # Step 2, convert SNP-SNP interaction to BridGE's sparse network format
        save_file = intermediate_dir / f"snp_interaction_networks-{model}-R{i}.npz"
        cassi_to_network(add_suffix(infile, ".bim"), cassi_out, gi, save_file)

    if parallel:
        # each worker runs one network at a time, so at most n_workers CASSI processes run at once
        Parallel(n_jobs=args.n_workers)(delayed(compute_network)(i) for i in ids)
    else:
        for i in ids:
            compute_network(i)


if __name__ == "__main__":
    args = parse_args()
    main(args)

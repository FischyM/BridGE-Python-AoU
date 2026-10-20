import argparse
import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

from plink_ids import read_header, read_table, read_psam, sample_key


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pcaPrefix", required=True, help="PLINK2 prefix for .eigenvec/.eigenval files")
    p.add_argument("--psam", required=True, help="PLINK2 .psam file (phenotypes)")
    p.add_argument("--out", required=True, help="output cluster TSV, written to the same directory as what is given but called 'plink.cluster.tsv'")
    p.add_argument("--npcs", type=int, default=10, help="number of leading PCs to cluster on (default: 10)")
    p.add_argument("--kmax", type=int, default=255, help="largest k on the 3, 5, 9, 17, ... grid (default: 255)")
    p.add_argument("--n-init", type=int, default=10, help="k-means restarts per k (default: 10)")
    p.add_argument("--workers", type=int, default=1, help="parallel processes (default: 1)")
    p.add_argument("--seed", type=int, default=42, help="random seed (default: 42)")
    p.add_argument("--pheno-col", default=None, help="phenotype column in .psam (default: PHENO1)")
    return p.parse_args()

def read_eigenvec(path, npcs):
    """Return (keys, pcs, has_fid) for a PLINK2 .eigenvec file."""
    fields, has_fid = read_header(path)
    pc_cols = sorted([f for f in fields if f.startswith("PC")], key=lambda f: int(f[2:]))
    if len(pc_cols) < npcs:
        sys.exit(f"{path}: only {len(pc_cols)} PCs present, --npcs {npcs} requested")
    pc_cols = pc_cols[:npcs]

    keys, rows = [], []
    for rec, _ in read_table(path):
        keys.append(sample_key(rec, has_fid))
        rows.append([float(rec[c]) for c in pc_cols])
    return keys, np.asarray(rows, dtype=float), has_fid

def scale_pcs(pcs, eigval):
    """z-score each PC, then weight it by sqrt(eigenvalue / first eigenvalue)."""
    sd = pcs.std(axis=0, ddof=1)
    sd[sd == 0] = 1.0
    scaled = (pcs - pcs.mean(axis=0)) / sd
    return scaled * np.sqrt(eigval / eigval[0])

def valid_clusters(labels, pheno):
    """Boolean per cluster label: True if it has >= 1 case and >= 1 control."""
    n_clusters = labels.max() + 1
    n_case = np.bincount(labels, weights=pheno, minlength=n_clusters)
    n_total = np.bincount(labels, minlength=n_clusters)
    return (n_case > 0) & (n_case < n_total)

def kmeans_once(X, k, seed, restart):
    """One single-threaded k-means restart. Returns (k, restart, inertia, labels)."""
    rs = int(np.random.SeedSequence(seed, spawn_key=(k, restart)).generate_state(1)[0])
    with threadpool_limits(limits=1):
        km = KMeans(n_clusters=k, n_init=1, random_state=rs).fit(X)
    return k, restart, km.inertia_, km.labels_.astype(np.int64)

def cluster_samples(X, pheno, kmax, n_init, workers, seed):
    """Returns (labels, k, merged)."""
    n = X.shape[0]
    grid = []
    i = 1
    while 2**i + 1 <= min(kmax, n):
        grid.append(2**i + 1)
        i += 1
    if kmax not in grid:
        grid.append(kmax)
    if len(grid) == 0:
        sys.exit(f"--kmax {kmax} is below the smallest grid value (3) or the sample size ({n})")

    # largest k first so the slowest tasks start early (better load balancing)
    tasks = [(k, r) for k in reversed(grid) for r in range(n_init)]
    results = Parallel(n_jobs=workers)(delayed(kmeans_once)(X, k, seed, r) for k, r in tasks)

    # per k keep the lowest-inertia restart (ties -> lowest restart index)
    best = {}
    for k, r, inertia, labels in results:
        if k not in best or (inertia, r) < (best[k][0], best[k][1]):
            best[k] = (inertia, r, labels)

    chosen_k = None
    for k in grid:
        ok = valid_clusters(best[k][2], pheno)
        print(f"    k={k}: {ok.sum()}/{k} clusters valid", flush=True)
        if ok.all():
            chosen_k = k
    if chosen_k is not None:
        return best[chosen_k][2], chosen_k, False

    # fallback: no k on the grid is valid -> use k = 3 and merge invalid clusters
    k = grid[0]
    labels = best[k][2].copy()
    ok = valid_clusters(labels, pheno)
    if not ok.any():
        print("WARNING: no valid cluster at any k; using a single cluster "
              "(equivalent to a global phenotype permutation)", flush=True)
        return np.zeros(n, dtype=np.int64), 1, True

    print(f"WARNING: no k on the grid gave all-valid clusters; using k={k} and "
          f"reassigning members of {(~ok).sum()} invalid cluster(s) to the nearest "
          f"valid cluster centroid", flush=True)
    valid_ids = np.flatnonzero(ok)
    centroids = np.vstack([X[labels == c].mean(axis=0) for c in valid_ids])
    bad = ~ok[labels]
    dists = ((X[bad, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    labels[bad] = valid_ids[dists.argmin(axis=1)]
    labels = np.unique(labels, return_inverse=True)[1]  # relabel to 0..m-1
    return labels, k, True

def main():
    """Cluster samples into genetic similarity groups for within-cluster phenotype
    permutation (used by BridGE ComputeInteraction for R > 0).

    Reads a PLINK2 `--pca` .eigenvec/.eigenval pair and the matching .psam, then:

    1. z-scores each PC and weights it by sqrt(eigenvalue_i / eigenvalue_1), the
        same scaling as match_case_control_helper.py --weight-eigenvalues.
    2. Runs k-means for every k on the odd geometric grid 3, 5, 9, 17, 33, ...
        (2^i + 1) up to --kmax, and keeps the LARGEST k whose clusters are all
        valid. A cluster is valid if it contains at least one case and at least one
        control (this also rules out single-sample clusters).
    3. Fallback, only if no k on the grid is valid: take k = 3 and reassign each
        member of an invalid cluster to the nearest valid cluster centroid. If no
        cluster is valid at all, every sample is put in one cluster (equivalent to a
        global permutation) and a warning is printed.

    Parallelism: each k-means restart (n_init of them per k) is an independent
    single-threaded task, so the whole grid (len(grid) * n_init tasks) is spread over
    --workers processes. For each k the restart with the lowest inertia is kept, as
    sklearn does internally for n_init > 1. Restart seeds come from
    SeedSequence(seed, spawn_key=(k, restart)), so the result does not depend on
    --workers.

    Output: a TSV with columns IID, PHENO (0 = control, 1 = case, matching
    snp_data.npz), CLUSTER, preceded by a '#' comment line recording the settings.
    """
    args = parse_args()
    eigenvec_path = f"{args.pcaPrefix}.eigenvec"
    eigenval_path = f"{args.pcaPrefix}.eigenval"

    _, pheno_map, psam_has_fid = read_psam(args.psam, args.pheno_col)
    keys, pcs, vec_has_fid = read_eigenvec(eigenvec_path, args.npcs)
    
    if vec_has_fid != psam_has_fid:
        sys.exit("FID presence differs between .eigenvec and .psam")
        
    missing = [k for k in keys if k not in pheno_map]
    if missing:
        sys.exit(f"{len(missing)} sample(s) in {eigenvec_path} absent from "
                 f"{args.psam}, e.g. {missing[0]}")
        
    no_pheno = [k for k in keys if pheno_map[k] == "missing"]
    if no_pheno:
        sys.exit(f"{len(no_pheno)} sample(s) have a missing phenotype, e.g. {no_pheno[0]}")
    pheno = np.array([1 if pheno_map[k] == "case" else 0 for k in keys], dtype=np.int64)

    eigval = np.loadtxt(eigenval_path, dtype=float, ndmin=1)[:args.npcs]
    if len(eigval) < args.npcs:
        sys.exit(f"{eigenval_path}: fewer than {args.npcs} eigenvalues")
    X = scale_pcs(pcs, eigval)

    print(f"clustering {len(keys)} samples on {args.npcs} PCs "
          f"(kmax={args.kmax}, n_init={args.n_init}, workers={args.workers}, seed={args.seed})",
          flush=True)
    labels, k, merged = cluster_samples(X, pheno, args.kmax, args.n_init, args.workers, args.seed)

    sizes = np.bincount(labels)
    case_frac = np.bincount(labels, weights=pheno) / sizes
    print(f"chosen k={k}{' (merged)' if merged else ''}: {sizes.size} clusters, "
          f"size min/median/max = {sizes.min()}/{int(np.median(sizes))}/{sizes.max()}, "
          f"case fraction min/max = {case_frac.min():.2f}/{case_frac.max():.2f}", flush=True)

    with open(Path(args.out).with_name(f'plink.cluster.tsv'), "w") as out:
        # out.write(f"# k={k} merged={merged} kmax={args.kmax} n_init={args.n_init} "
        #           f"seed={args.seed} npcs={args.npcs}\n")
        out.write("IID\tPHENO\tCLUSTER\n")
        for key, ph, cl in zip(keys, pheno, labels):
            out.write(f"{key[-1]}\t{ph}\t{cl}\n")

if __name__ == "__main__":
    main()

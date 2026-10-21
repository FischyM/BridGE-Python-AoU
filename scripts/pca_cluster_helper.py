"""Cluster samples into genetic similarity groups for within-cluster phenotype
permutation (used by BridGE ComputeInteraction for R > 0).

Reads a PLINK2 `--pca` .eigenvec/.eigenval pair and the matching .psam, then:

1. z-scores each PC and weights it by sqrt(eigenvalue_i / eigenvalue_1), the
    same scaling as match_case_control_helper.py --weight-eigenvalues.
2. Runs k-means for every k on the odd geometric grid 3, 5, 9, 17, 33, ...
    (2^i + 1) below a limit, plus a final k equal to the limit (minus 1 if it
    is even). With n <= 10,000 samples the limit is n. With n > 10,000 it is
    --kmax, or 10,000 if --kmax is not given (never more than n).
3. At every k, merges each invalid cluster as a whole into the valid cluster
    with the nearest centroid. A cluster is valid if it contains at least one
    case and at least one control (so single-sample clusters are invalid). If no
    cluster at that k is valid, all samples form one cluster.
4. Keeps the k with the MOST clusters after merging (ties go to the smaller k).
    The per-k log shows how many clusters were merged and the cluster count
    after merging. k is scanned in increasing order and the scan stops once
    --patience consecutive k values have all failed to exceed the best cluster
    count so far (a tie counts as not exceeding), so the most expensive k values
    (close to n) can be skipped when the peak comes early enough.

Parallelism: at each k the n_init k-means restarts are independent
single-threaded tasks spread over --workers processes, so at most n_init
workers are busy at once. For each k the restart with the lowest inertia is
kept, as sklearn does internally for n_init > 1. Restart seeds come from
SeedSequence(seed, spawn_key=(k, restart)), so the result does not depend on
--workers.

Output: a TSV with columns IID, PHENO (0 = control, 1 = case, matching
snp_data.npz), CLUSTER, preceded by a '#' comment line recording the settings.
"""
    
import argparse
import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy.spatial import KDTree
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

from plink_ids import read_header, read_table, read_psam, sample_key


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pcaPrefix", required=True, help="PLINK2 prefix for .eigenvec/.eigenval files")
    p.add_argument("--psam", required=True, help="PLINK2 .psam file (phenotypes)")
    p.add_argument("--out", required=True, help="output cluster TSV, written to the same directory as what is given but called 'plink.cluster.tsv'")
    p.add_argument("--npcs", type=int, default=10, help="number of leading PCs to cluster on (default: 10)")
    p.add_argument("--n-init", type=int, default=10, help="k-means restarts per k (default: 10)")
    p.add_argument("--kmax", type=int, default=None, help="largest k to test when there are more than "
                   "10,000 samples (default: 10,000; ignored at 10,000 samples or fewer, where the grid goes up to n)")
    p.add_argument("--patience", type=int, default=2, help="stop after this many consecutive k fail to exceed the best cluster count so far (default: 2)")
    p.add_argument("--workers", type=int, default=None, help="parallel processes (default: None, which uses all available cores)")
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

def merge_invalid(X, labels, pheno):
    """Merge every invalid cluster (single sample, or only one phenotype) as a whole
    into the valid cluster with the nearest centroid. Centroids are those of the
    valid clusters before merging, so the order of merges does not matter.

    Returns (labels relabelled to 0..m-1, n_invalid_clusters, n_samples_moved).
    If no cluster is valid, every sample is put in one cluster.
    """
    ok = valid_clusters(labels, pheno)
    n_bad = int((~ok).sum())
    if n_bad == 0:
        return labels, 0, 0
    if not ok.any():
        return np.zeros_like(labels), n_bad, labels.size

    valid_ids = np.flatnonzero(ok)
    bad_ids = np.flatnonzero(~ok)
    n_clusters = labels.max() + 1
    counts = np.bincount(labels, minlength=n_clusters)
    sums = np.zeros((n_clusters, X.shape[1]))
    np.add.at(sums, labels, X)
    centroids = sums / np.maximum(counts, 1)[:, None]

    # nearest valid centroid for each invalid cluster's centroid (KD-tree: memory is
    # linear in the number of clusters, unlike a full invalid x valid distance matrix)
    _, nearest = KDTree(centroids[valid_ids]).query(centroids[bad_ids], k=1)
    target = np.arange(n_clusters)
    target[bad_ids] = valid_ids[nearest]

    merged = target[labels]
    merged = np.unique(merged, return_inverse=True)[1]  # relabel to 0..m-1
    return merged, n_bad, int(counts[bad_ids].sum())

LARGE_N = 10_000  # above this many samples the grid stops at kmax instead of n

def get_kmax(n, kmax=None):
    """Largest (odd) k to test.

    n <= 10,000: k goes up to n, and any user kmax is ignored.
    n >  10,000: k goes up to the user's kmax, or 10,000 if none was given (never above n).
    The limit is reduced by one if it is even, so every k on the grid is odd.
    """
    if n <= LARGE_N:
        if kmax is not None:
            print(f"    note: --kmax ignored, {n} samples <= {LARGE_N}; testing k up to n", flush=True)
        limit = n
    else:
        limit = min(LARGE_N if kmax is None else kmax, n)
    limit = limit if limit % 2 == 1 else limit - 1
    if limit < 3:
        sys.exit(f"the largest k to test must be at least 3 (got {limit} from n={n}, --kmax={kmax})")
    return limit

def cluster_samples(X, pheno, n_init, workers, seed, patience=2, kmax=None):
    """Returns (labels, k, n_bad, n_moved) for the chosen k."""
    n = X.shape[0]
    kmax = get_kmax(n, kmax)
    grid = []
    i = 1
    while 2**i + 1 <= kmax:
        grid.append(2**i + 1)
        i += 1
    if kmax not in grid:
        grid.append(kmax)
    print(f"    k grid (up to {kmax}): {', '.join(map(str, grid))}", flush=True)

    # Scan k in increasing order. At each k the n_init restarts run in parallel, the
    # lowest-inertia restart is kept (ties -> lowest restart index), invalid clusters
    # are merged, and the k with the most clusters after merging is kept (ties ->
    # smaller k). The scan stops early once `patience` consecutive k values have all
    # failed to exceed the best count so far (ties count as not exceeding), which
    # skips the costly k ~ n tail.
    chosen = None
    n_worse = 0
    with Parallel(n_jobs=workers) as parallel:  # reuse the same worker processes for every k
        for k in grid:
            results = parallel(delayed(kmeans_once)(X, k, seed, r) for r in range(n_init))
            _, _, _, best_labels = min(results, key=lambda res: (res[2], res[1]))

            labels, n_bad, n_moved = merge_invalid(X, best_labels, pheno)
            n_after = labels.max() + 1
            print(f"    k={k}: {n_bad} invalid cluster(s) merged ({n_moved} samples moved) "
                  f"-> {n_after} clusters", flush=True)

            if chosen is None or n_after > chosen[0]:
                chosen = (n_after, k, labels, n_bad, n_moved)
                n_worse = 0
            else:
                n_worse += 1
                if n_worse >= patience and k != grid[-1]:
                    print(f"    stopping: {patience} consecutive k did not exceed the "
                          f"best cluster count so far (k={chosen[1]})", flush=True)
                    break

    n_after, k, labels, n_bad, n_moved = chosen
    if n_after == 1:
        print("WARNING: no k produced more than one valid cluster; using a single cluster "
              "(equivalent to a global phenotype permutation)", flush=True)
    return labels, k, n_bad, n_moved

def main():
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
          f"(n_init={args.n_init}, patience={args.patience}, workers={args.workers}, seed={args.seed})",
          flush=True)
    labels, k, n_bad, n_moved = cluster_samples(X, pheno, args.n_init, args.workers, args.seed, args.patience, args.kmax)

    sizes = np.bincount(labels)
    case_frac = np.bincount(labels, weights=pheno) / sizes
    print(f"chosen k={k} ({n_bad} invalid cluster(s) merged, {n_moved} samples moved): "
          f"{sizes.size} clusters, "
          f"size min/median/max = {sizes.min()}/{int(np.median(sizes))}/{sizes.max()}, "
          f"case fraction min/max = {case_frac.min():.2f}/{case_frac.max():.2f}", flush=True)

    with open(Path(args.out).with_name(f'plink.cluster.tsv'), "w") as out:
        out.write("IID\tPHENO\tCLUSTER\n")
        for key, ph, cl in zip(keys, pheno, labels):
            out.write(f"{key[-1]}\t{ph}\t{cl}\n")

if __name__ == "__main__":
    main()

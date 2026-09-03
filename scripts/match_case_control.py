"""Match controls to cases in principal-component space.

Reads a PLINK2 `--pca` .eigenvec file and the corresponding .psam, then does
greedy nearest-neighbour matching without replacement: each case claims its
closest still-unclaimed control(s) in PC space.

Distance is Euclidean on PCs that have been standardised to unit variance. With
--weight-eigenvalues each PC is then rescaled by sqrt(eigenvalue / eigenvalue_1),
so PC1 dominates the distance in proportion to the variance it explains. Which
you want depends on intent: unweighted treats every retained PC as an equally
important axis of ancestry, weighted keeps the emphasis on the dominant axes.

A caliper (--caliper, in standardised PC units) is strongly recommended. Greedy
matching always returns a match if any control is left, however far away it is;
the caliper is what stops a case from being paired with a genetically unrelated
control. Cases with no control inside the caliper are dropped and reported.
"""

import argparse
import sys

import numpy as np
from scipy.spatial import KDTree

from plink_ids import read_header, read_table, read_psam, sample_key, write_id_file


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eigenvec", required=True, help="PLINK2 .eigenvec file")
    p.add_argument("--psam", required=True, help="PLINK2 .psam file")
    p.add_argument("--out", required=True,
                   help="output --keep ID file of matched cases and controls")
    p.add_argument("--pairs", default=None,
                   help="optional TSV report of matched pairs and distances")
    p.add_argument("--eigenval", default=None,
                   help="PLINK2 .eigenval file (required with --weight-eigenvalues)")
    p.add_argument("--weight-eigenvalues", action="store_true",
                   help="scale each PC by sqrt(eigenvalue / first eigenvalue)")
    p.add_argument("--npcs", type=int, default=5,
                   help="number of leading PCs to match on (default: 5)")
    p.add_argument("--ratio", type=int, default=1,
                   help="controls per case (default: 1)")
    p.add_argument("--caliper", type=float, default=None,
                   help="maximum allowed distance in standardised PC units")
    p.add_argument("--drop-partial", action="store_true",
                   help="discard cases that could not be matched to the full "
                        "--ratio, releasing their controls for other cases "
                        "(use for a strict fixed-ratio design)")
    p.add_argument("--pheno-col", default=None,
                   help="phenotype column in .psam (default: PHENO1)")
    p.add_argument("--seed", type=int, default=0,
                   help="seed for the order in which cases are processed")
    return p.parse_args()

def read_eigenvec(path, npcs):
    fields, has_fid = read_header(path)
    pc_cols = [f for f in fields if f.startswith("PC")]
    pc_cols.sort(key=lambda f: int(f[2:]))
    if len(pc_cols) < npcs:
        raise ValueError(f"{path}: only {len(pc_cols)} PCs present, "
                         f"--npcs {npcs} requested")
    pc_cols = pc_cols[:npcs]

    keys, rows = [], []
    for rec, _ in read_table(path):
        keys.append(sample_key(rec, has_fid))
        rows.append([float(rec[c]) for c in pc_cols])
    return keys, np.asarray(rows, dtype=float), has_fid

def scale_pcs(pcs, eigenval_path, weight, npcs):
    sd = pcs.std(axis=0, ddof=1)
    sd[sd == 0] = 1.0
    scaled = (pcs - pcs.mean(axis=0)) / sd

    if weight:
        if eigenval_path is None:
            sys.exit("--weight-eigenvalues requires --eigenval")
        vals = np.loadtxt(eigenval_path, dtype=float, ndmin=1)[:npcs]
        if len(vals) < npcs:
            sys.exit(f"{eigenval_path}: fewer than {npcs} eigenvalues")
        scaled = scaled * np.sqrt(vals / vals[0])
    return scaled

def match(case_pcs, control_pcs, ratio, caliper, drop_partial, rng):
    """Greedy nearest-neighbour matching without replacement.

    Returns (pairs, short) where pairs is a list of
    (case_index, control_index, distance) and short is a list of
    (case_index, n_controls_found) for cases that fell below --ratio.
    """
    n_controls = len(control_pcs)
    tree = KDTree(control_pcs)
    taken = np.zeros(n_controls, dtype=bool)
    pairs, short = [], []

    for ci in rng.permutation(len(case_pcs)):
        mine = []
        k, seen, exhausted = max(ratio * 4, 8), 0, False

        while len(mine) < ratio and not exhausted:
            k = min(k, n_controls)
            q_dists, q_idxs = tree.query(case_pcs[ci], k=k)
            dists, idxs = np.atleast_1d(q_dists), np.atleast_1d(q_idxs)
            for d, j in zip(dists[seen:], idxs[seen:]):
                if not np.isfinite(d):
                    continue
                if caliper is not None and d > caliper:
                    exhausted = True  # neighbours only get farther from here
                    break
                if taken[j]:
                    continue
                taken[j] = True
                mine.append((int(ci), int(j), float(d)))
                if len(mine) >= ratio:
                    break
            seen = k
            if k >= n_controls:
                exhausted = True
            k = min(k * 4, n_controls)

        if len(mine) < ratio:
            short.append((int(ci), len(mine)))
            if drop_partial:
                for _, j, _ in mine:
                    taken[j] = False  # release for a case that can use them
                continue
        pairs.extend(mine)

    return pairs, short

def main():
    args = parse_args()
    if args.ratio < 1:
        sys.exit("--ratio must be >= 1")

    _, pheno, psam_has_fid = read_psam(args.psam, args.pheno_col)
    keys, pcs, vec_has_fid = read_eigenvec(args.eigenvec, args.npcs)
    if vec_has_fid != psam_has_fid:
        sys.exit("FID presence differs between .eigenvec and .psam")

    missing = [k for k in keys if k not in pheno]
    if missing:
        sys.exit(f"{len(missing)} sample(s) in {args.eigenvec} absent from "
                 f"{args.psam}, e.g. {missing[0]}")

    scaled = scale_pcs(pcs, args.eigenval, args.weight_eigenvalues, args.npcs)

    case_idx = [i for i, k in enumerate(keys) if pheno[k] == "case"]
    control_idx = [i for i, k in enumerate(keys) if pheno[k] == "control"]
    dropped_missing = len(keys) - len(case_idx) - len(control_idx)

    if not case_idx:
        sys.exit("no cases found; check --pheno-col and phenotype coding")
    if len(control_idx) < len(case_idx) * args.ratio:
        print(f"warning: {len(control_idx)} controls available but "
              f"{len(case_idx) * args.ratio} needed for {args.ratio}:1 matching",
              file=sys.stderr)

    rng = np.random.default_rng(args.seed)
    pairs, short = match(scaled[case_idx], scaled[control_idx],
                         args.ratio, args.caliper, args.drop_partial, rng)

    matched_cases = sorted({p[0] for p in pairs})
    matched_controls = sorted({p[1] for p in pairs})
    keep = ([keys[case_idx[i]] for i in matched_cases] +
            [keys[control_idx[j]] for j in matched_controls])
    keep_set = set(keep)
    write_id_file(args.out, [k for k in keys if k in keep_set], psam_has_fid)

    if args.pairs:
        with open(args.pairs, "w") as out:
            out.write("CASE\tCONTROL\tDIST\n")
            for ci, cj, d in sorted(pairs):
                out.write(f"{'_'.join(keys[case_idx[ci]])}\t"
                          f"{'_'.join(keys[control_idx[cj]])}\t{d:.6f}\n")

    dists = np.array([p[2] for p in pairs]) if pairs else np.array([0.0])
    print(f"PCs used                 : {args.npcs}"
          f"{' (eigenvalue-weighted)' if args.weight_eigenvalues else ''}")
    print(f"cases / controls in PCA  : {len(case_idx)} / {len(control_idx)}"
          + (f"  ({dropped_missing} missing phenotype ignored)"
             if dropped_missing else ""))
    print(f"pairs formed             : {len(pairs)}")
    print(f"cases fully matched      : {len(case_idx) - len(short)}")
    print(f"cases below --ratio      : {len(short)}"
          f"{' (dropped)' if args.drop_partial else ' (kept, partially matched)'}")
    print(f"match distance mean/max  : {dists.mean():.4f} / {dists.max():.4f}")
    print(f"retained -> {args.out}  : {len(keep)}")

if __name__ == "__main__":
    main()
    
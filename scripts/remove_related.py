"""Select a near-maximal unrelated sample set, preferring to retain cases.

Input is a PLINK2 `--make-king-table --king-table-filter <cutoff>` .kin0 file,
which lists only the pairs at or above the kinship cutoff. Those pairs form a
graph; the goal is a large independent set. Finding the true maximum is
NP-hard, so this uses the standard greedy heuristic (repeatedly delete the
highest-degree vertex) with one modification: controls are deleted before
cases.

Consequence worth understanding: prioritising cases can cost total sample size.
If a control sits between two cases in the relatedness graph, dropping that one
control saves two cases, which is a good trade. But when a high-degree case is
connected only to controls, this rule deletes all of those controls to keep the
single case. The --max-controls-per-case guard caps that behaviour.
"""

import argparse
from collections import defaultdict

from plink_ids import read_header, read_table, read_psam, write_id_file


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kin0", required=True,
                   help="PLINK2 .kin0 file from --make-king-table")
    p.add_argument("--psam", required=True, help="PLINK2 .psam file")
    p.add_argument("--out", required=True,
                   help="output --keep ID file of unrelated samples")
    p.add_argument("--removed", default=None,
                   help="optional output ID file of removed samples")
    p.add_argument("--pheno-col", default=None,
                   help="phenotype column in .psam (default: PHENO1)")
    p.add_argument("--kinship-cutoff", type=float, default=None,
                   help="ignore pairs below this KINSHIP value; use if the "
                        "kin0 file was written with a looser --king-table-filter")
    p.add_argument("--max-controls-per-case", type=int, default=None,
                   help="if a case has more than this many related controls, "
                        "drop the case instead of all the controls")
    return p.parse_args()

def read_pairs(path, cutoff):
    fields, has_fid = read_header(path)
    if "KINSHIP" not in fields:
        raise ValueError(f"{path}: no KINSHIP column; is this a .kin0 file?")
    pairs = []
    for rec, _ in read_table(path):
        kin = float(rec["KINSHIP"])
        if cutoff is not None and kin < cutoff:
            continue
        if has_fid:
            a = (rec["FID1"], rec["IID1"])
            b = (rec["FID2"], rec["IID2"])
        else:
            a = (rec["IID1"],)
            b = (rec["IID2"],)
        pairs.append((a, b, kin))
    return pairs, has_fid

def build_graph(pairs):
    # defaultdict avoids manually checking if a key exists, then add an empty set
    # such as this:
    # if a not in adj:
    #     adj[a] = set()
    # adj[a].add(b)
    # Note: any use of "if adj[x]:" on a defaultdict will create the key with an empty set if it doesn't exist
    # as these are created on read access, not just write access.
    # This doesn't happen anywhere in this script, but useful to know when working with defaultdicts
    adj = defaultdict(set)
    for a, b, _ in pairs:
        if a == b:
            continue
        adj[a].add(b)
        adj[b].add(a)
    return adj

def priority(node, adj, pheno, max_controls_per_case):
    """Sort key; the node with the smallest key is deleted first."""
    status = pheno.get(node, "missing")
    degree = len(adj[node])
    if status == "case" and max_controls_per_case is not None:
        related_controls = sum(1 for n in adj[node] if pheno.get(n) == "control")
        if related_controls > max_controls_per_case:
            status = "control"  # demote: not worth the controls it would cost

    # missing phenotypes are useless downstream, so shed them first
    rank = {"missing": 0, "control": 1, "case": 2}[status]
    return (rank, -degree, node)

def main():
    args = parse_args()

    samples, pheno, psam_has_fid = read_psam(args.psam, args.pheno_col)
    pairs, kin_has_fid = read_pairs(args.kin0, args.kinship_cutoff)

    if pairs and kin_has_fid != psam_has_fid:
        raise ValueError("FID presence differs between .kin0 and .psam; "
                         "regenerate both from the same dataset")

    known = set(samples)
    adj = build_graph(pairs)
    unknown = [n for n in adj if n not in known]
    if unknown:
        raise ValueError(f"{len(unknown)} sample(s) in {args.kin0} are absent "
                         f"from {args.psam}, e.g. {unknown[0]}")

    for node in samples:
        adj.setdefault(node, set())

    removed = set()
    active = {n for n in adj if adj[n]}

    while active:
        sample = min(active, key=lambda n: priority(n, adj, pheno, args.max_controls_per_case))
        removed.add(sample)
        for nbr in adj[sample]:
            adj[nbr].discard(sample)
            if not adj[nbr]:
                active.discard(nbr)
        adj[sample].clear()
        active.discard(sample)

    kept = [s for s in samples if s not in removed]
    write_id_file(args.out, kept, psam_has_fid)
    if args.removed:
        write_id_file(args.removed, [s for s in samples if s in removed], psam_has_fid)

    def tally(keys):
        # this supplies 0 automatically, so incrementing can simply procede without a missing key check
        counts = defaultdict(int)
        for k in keys:
            counts[pheno.get(k, "missing")] += 1
        return counts

    kept_n, removed_n = tally(kept), tally(removed)
    print(f"related pairs read      : {len(pairs)}")
    print(f"samples in .psam        : {len(samples)}")
    print(f"removed                 : {len(removed)} "
          f"({removed_n['case']} case, {removed_n['control']} control, "
          f"{removed_n['missing']} missing)")
    print(f"retained -> {args.out} : {len(kept)} "
          f"({kept_n['case']} case, {kept_n['control']} control, "
          f"{kept_n['missing']} missing)")

if __name__ == "__main__":
    main()
"""Helpers for the population-check step (see example-check_population.sh).

The step compares a study cohort against 1000 Genomes reference populations on
the top principal components. Everything PLINK2 cannot do itself lives here:

  variant-ids     list the unique variant IDs in a .pvar/.bim
  shared-variants list variants the two filesets agree on, i.e. same ID and the
                  same (possibly REF/ALT-swapped) allele pair
  ref-samples     list the reference samples belonging to the requested
                  populations
  combine         join the reference and study projections into one coordinate
                  table plus a one-hot population membership file

The old shell implementation discovered allele conflicts by attempting a merge,
reading PLINK 1.9's .missnp file and re-merging in a loop. `shared-variants`
determines the same set up front from the two .pvar files, so no merge is
needed at all.
"""

import argparse
import sys

from plink_ids import read_header, write_id_file

# PLINK 1.9 --biallelic-only strict, expressed as a predicate on a .pvar row
def _biallelic(ref, alt):
    return "," not in ref and "," not in alt


def read_pvar(path):
    """Yield (id, chrom, pos, ref, alt) from a .pvar or .bim file."""
    is_bim = path.endswith(".bim")
    min_fields = 6 if is_bim else 5
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < min_fields:  # whitespace- rather than tab-delimited
                f = line.split()
            if len(f) < min_fields:
                continue
            if is_bim:  # CHROM ID CM POS A1 A2
                yield f[1], f[0], f[3], f[5], f[4]
            else:       # CHROM POS ID REF ALT
                yield f[2], f[0], f[1], f[3], f[4]


def unique_ids(path):
    """Return (ids_in_file_order, n_dropped).

    IDs that are missing or appear more than once are dropped: PLINK2 refuses
    --extract lists that are ambiguous, and a duplicated ID cannot be matched
    unambiguously across two filesets anyway.
    """
    seen, dup = {}, set()
    for vid, _, _, _, _ in read_pvar(path):
        if vid in (".", ""):
            dup.add(vid)
            continue
        if vid in seen:
            dup.add(vid)
        else:
            seen[vid] = None
    for vid in dup:
        seen.pop(vid, None)
    return list(seen), len(dup)


def cmd_variant_ids(args):
    ids, dropped = unique_ids(args.pvar)
    with open(args.out, "w") as out:
        out.write("\n".join(ids) + "\n")
    print(f"{args.pvar}: {len(ids)} unique variant IDs "
          f"({dropped} dropped as missing/duplicated) -> {args.out}")


def cmd_shared_variants(args):
    """Variants the two panels agree on: same ID and the same allele pair."""
    study = {}
    dup = set()
    for vid, chrom, pos, ref, alt in read_pvar(args.study_pvar):
        if vid in (".", "") or vid in study:
            dup.add(vid)
            continue
        study[vid] = (chrom, pos, ref, alt)
    for vid in dup:
        study.pop(vid, None)

    shared, seen = [], set()
    n_pos, n_allele, n_multi = 0, 0, 0
    for vid, chrom, pos, ref, alt in read_pvar(args.ref_pvar):
        rec = study.get(vid)
        if rec is None or vid in seen:
            continue
        s_chrom, s_pos, s_ref, s_alt = rec
        if not (_biallelic(ref, alt) and _biallelic(s_ref, s_alt)):
            n_multi += 1
            continue
        # REF/ALT may be swapped between panels, and PLINK2 --score matches on
        # the allele code, so a swap is fine. A genuinely different allele pair
        # is the "3+ alleles" case that used to break the merge.
        if {ref, alt} != {s_ref, s_alt}:
            n_allele += 1
            continue
        # Coordinates are not used downstream (--score matches on ID), but a
        # large disagreement means the two panels are on different genome
        # builds, which is worth knowing about.
        if pos != s_pos or chrom != s_chrom:
            n_pos += 1
        seen.add(vid)
        shared.append(vid)

    with open(args.out, "w") as out:
        out.write("\n".join(shared) + "\n")
    print(f"shared variants          : {len(shared)} -> {args.out}")
    print(f"dropped, allele mismatch : {n_allele}")
    print(f"dropped, multiallelic    : {n_multi}")
    if not shared:
        sys.exit("No variants are shared between the two filesets; check that "
                 "they use the same variant naming.")
    if n_pos > len(shared) // 10:
        print(f"WARNING: {n_pos} of {len(shared)} shared variants sit at "
              f"different coordinates in the two filesets. They are almost "
              f"certainly on different genome builds. Population axes are "
              f"still usable (variants are matched by ID), but LD pruning "
              f"uses the reference coordinates only.", file=sys.stderr)


def read_pop_map(path):
    """Read the reference population file: FID IID POP per line."""
    pop_of = {}
    with open(path) as fh:
        for line in fh:
            f = line.split()
            if len(f) < 3 or f[0].startswith("#"):
                continue
            pop_of[(f[0], f[1])] = f[2]
    if not pop_of:
        sys.exit(f"{path}: no 'FID IID POP' rows found")
    return pop_of


def read_psam_ids(path):
    """Return (list of ID tuples in file order, has_fid)."""
    fields, has_fid = read_header(path)
    iid = fields.index("IID")
    keys = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.split()
            if f:
                keys.append((f[0], f[iid]) if has_fid else (f[iid],))
    return keys, has_fid


def lookup_pop(pop_of, key):
    """Population of a sample, tolerating a missing FID on either side."""
    if len(key) == 2:
        return pop_of.get(key)
    iid = key[0]
    return pop_of.get((iid, iid))


def cmd_ref_samples(args):
    pops = args.pops.split(",")
    pop_of = read_pop_map(args.pop_id)
    keys, has_fid = read_psam_ids(args.ref_psam)

    wanted = set(pops)
    kept = [k for k in keys if lookup_pop(pop_of, k) in wanted]
    if not kept:
        sys.exit(f"None of the populations {args.pops} are present in both "
                 f"{args.pop_id} and {args.ref_psam}")
    write_id_file(args.out, kept, has_fid)

    counts = {}
    for k in kept:
        pop = lookup_pop(pop_of, k)
        counts[pop] = counts.get(pop, 0) + 1
    summary = ", ".join(f"{p}={counts.get(p, 0)}" for p in pops)
    print(f"reference samples       : {len(kept)} ({summary}) -> {args.out}")


def read_sscore(path, npcs):
    """Read a PLINK2 --score .sscore file, returning (id_tuple, [pc, ...]) rows.

    The projected PCs are the PC<i>_AVG columns; both projections are produced
    by the same --score call, so reference and study coordinates are directly
    comparable without any rescaling.
    """
    fields, has_fid = read_header(path)
    cols = [f"PC{i}_AVG" for i in range(1, npcs + 1)]
    missing = [c for c in cols if c not in fields]
    if missing:
        sys.exit(f"{path}: missing score columns {', '.join(missing)}; "
                 f"available: {', '.join(fields)}")
    idx = [fields.index(c) for c in cols]
    iid = fields.index("IID")

    rows = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.split()
            if not f:
                continue
            key = (f[0], f[iid]) if has_fid else (f[iid], f[iid])
            rows.append((key, [f[i] for i in idx]))
    return rows


def cmd_combine(args):
    """Write the coordinate table and the matching one-hot population file.

    One row per individual. A sample that appears in both cohorts -- the
    bundled example draws its study data from the reference panel, so all 479
    reference samples do -- is written once, with the study projection for its
    coordinates and its reference population for its label. That is what the
    old plink 1.9 --bmerge did, since it merged same-ID samples into one.

    Both files are written from the same row list so they cannot disagree.
    """
    pops = args.pops.split(",")
    columns = pops + ["StudyPop"]
    pop_of = read_pop_map(args.pop_id)
    index = {p: i for i, p in enumerate(pops)}

    rows = []  # (fid, iid, pcs, column index)
    study_keys = set()
    for key, pcs in read_sscore(args.study_sscore, args.npcs):
        study_keys.add(key)
        # a study sample of a known reference population is drawn as that
        # population, exactly as the merged fileset used to be labelled
        col = index.get(lookup_pop(pop_of, key), len(pops))
        rows.append((key[0], key[1], pcs, col))
    n_study = len(rows)

    for key, pcs in read_sscore(args.ref_sscore, args.npcs):
        if key in study_keys:
            continue
        pop = lookup_pop(pop_of, key)
        if pop not in index:
            continue  # not one of the requested reference populations
        rows.append((key[0], key[1], pcs, index[pop]))

    pc_names = [f"PC{i}" for i in range(1, args.npcs + 1)]
    with open(args.out_coords, "w") as out:
        out.write("\t".join(["#FID", "IID"] + pc_names) + "\n")
        for fid, iid, pcs, _ in rows:
            out.write("\t".join([fid, iid] + pcs) + "\n")

    counts = {c: 0 for c in columns}
    with open(args.out_labels, "w") as out:
        out.write(" ".join(columns) + "\n")
        for fid, iid, _, col in rows:
            counts[columns[col]] += 1
            onehot = ["0"] * len(columns)
            onehot[col] = "1"
            out.write(" ".join([fid, iid] + onehot) + "\n")

    summary = ", ".join(f"{c}={counts[c]}" for c in columns)
    print(f"population axes          : {len(rows)} samples "
          f"({n_study} study, {len(rows) - n_study} reference-only) "
          f"-> {args.out_coords}")
    print(f"population labels        : {summary} -> {args.out_labels}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("variant-ids", help="unique variant IDs in a .pvar/.bim")
    s.add_argument("--pvar", required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_variant_ids)

    s = sub.add_parser("shared-variants", help="variants the two filesets agree on")
    s.add_argument("--study-pvar", required=True)
    s.add_argument("--ref-pvar", required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_shared_variants)

    s = sub.add_parser("ref-samples", help="reference samples in the requested populations")
    s.add_argument("--ref-psam", required=True)
    s.add_argument("--pop-id", required=True, help="reference population file, 'FID IID POP' per line")
    s.add_argument("--pops", required=True, help="comma-separated population codes")
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_ref_samples)

    s = sub.add_parser("combine", help="join the two projections into one coordinate table")
    s.add_argument("--ref-sscore", required=True, help=".sscore of the reference samples")
    s.add_argument("--study-sscore", required=True, help=".sscore of the study samples")
    s.add_argument("--pop-id", required=True)
    s.add_argument("--pops", required=True)
    s.add_argument("--npcs", type=int, required=True)
    s.add_argument("--out-coords", required=True, help="output .eigenvec-style coordinate table")
    s.add_argument("--out-labels", required=True, help="output one-hot population membership file")
    s.set_defaults(func=cmd_combine)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.func(args)

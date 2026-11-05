#!/usr/bin/env python3
"""
Collapse duplicate-probe variants (same ID) into one consensus variant.

Rules, applied per duplicate ID (all thresholds are 0.01):
  1. Probes with missingness > MISS_THRESH are "bad" and removed.
  2. 3+ good probes  -> per sample, most common called genotype
                        (tie -> missing). Drop the site if the tie rate
                        (among samples with >=1 call) > DISC_THRESH.
  3. 2 good probes   -> best probe (lowest missingness), missing calls
                        filled from the other probe. Drop the site if
                        their discordance (both called) > DISC_THRESH.
  4. 0-1 good probes -> base = the surviving probe, or the lowest-missingness
                        probe if none survive. Fill its missing calls with
                        the most common called genotype of all other probes
                        (tie -> missing). Drop the site if the filled
                        missingness > MISS_THRESH, or if base disagrees with
                        the others' consensus in > DISC_THRESH of samples
                        where both are called.
A dropped site loses all of its copies. A kept site is written once, at the
position of its first copy. All other variants are copied unchanged.

Only hard calls are carried over (fine for array data); the script stops if
the input has phased hard calls, multiallelic records, or a mix of
provisional (INFO "PR") and non-provisional REF alleles.

Usage:
  python collapse_dups.py <in_prefix> <dup_ids.txt> <out_prefix>
    in_prefix   : .pgen/.pvar/.psam prefix (e.g. IPDGC/preprocess/tmp2)
    dup_ids.txt : one ID per line (the .rmdup.mismatch file)
    out_prefix  : writes <out>.pgen/.pvar/.psam and <out>.collapse_decisions.tsv
"""
import shutil
import sys
from collections import defaultdict

import numpy as np
import pgenlib

MISS_THRESH = 0.01
DISC_THRESH = 0.01
CHUNK = 10000

in_prefix, ids_path, out_prefix = sys.argv[1:4]

# ---------------------------------------------------------------- .pvar
header_lines, records = [], []
with open(f"{in_prefix}.pvar") as f:
    for line in f:
        if line.startswith("##"):
            header_lines.append(line)
        elif line.startswith("#"):
            header_lines.append(line)
            cols = line.lstrip("#").rstrip("\n").split("\t")
        else:
            records.append(line)
id_col, alt_col = cols.index("ID"), cols.index("ALT")
info_col = cols.index("INFO") if "INFO" in cols else None
var_ids = [r.rstrip("\n").split("\t")[id_col] for r in records]

if any("," in r.rstrip("\n").split("\t")[alt_col] for r in records):
    sys.exit("Multiallelic records found; this script handles biallelic only.")

def is_pr(rec):
    if info_col is None:
        return False
    return "PR" in rec.rstrip("\n").split("\t")[info_col].split(";")

pr = [is_pr(r) for r in records]
if any(pr) and not all(pr):
    sys.exit("Mixed provisional/non-provisional REF alleles; cannot preserve "
             "per-variant REF flags with pgenlib. Stopping.")
nonref_all = all(pr) and len(pr) > 0

# ---------------------------------------------------------------- .pgen
reader = pgenlib.PgenReader(f"{in_prefix}.pgen".encode())
n_samp = reader.get_raw_sample_ct()
n_var = reader.get_variant_ct()
if n_var != len(records):
    sys.exit(f".pgen has {n_var} variants but .pvar has {len(records)}.")
if reader.hardcall_phase_present():
    sys.exit("Phased hard calls present; this script writes unphased calls.")

dup_ids = {l.strip() for l in open(ids_path) if l.strip()}
groups = defaultdict(list)                       # ID -> variant indices
for i, v in enumerate(var_ids):
    if v in dup_ids:
        groups[v].append(i)
missing_ids = dup_ids - set(groups)
if missing_ids:
    print(f"Warning: {len(missing_ids)} listed IDs not found in .pvar")

def mode_call(G):
    """Most common called genotype per sample (rows = probes). Tie/none -> -9."""
    counts = np.stack([(G == g).sum(0) for g in (0, 1, 2)])   # 3 x samples
    top = counts.max(0)
    n_top = (counts == top).sum(0)
    out = counts.argmax(0).astype(np.int8)
    out[(top == 0) | (n_top > 1)] = -9
    return out, (top > 0) & (n_top > 1)            # consensus, tie mask

buf = np.empty(n_samp, dtype=np.int8)
consensus, decisions = {}, []
for vid, idx in groups.items():
    G = np.empty((len(idx), n_samp), dtype=np.int8)
    for k, vi in enumerate(idx):
        reader.read(vi, buf)
        G[k] = buf
    miss = (G == -9).mean(1)
    good = [k for k in range(len(idx)) if miss[k] <= MISS_THRESH]
    d = {"ID": vid, "n_copies": len(idx),
         "copy_missingness": ",".join(f"{m:.4g}" for m in miss),
         "bad_copies_removed": ",".join(str(k + 1) for k in range(len(idx))
                                         if k not in good) or "none"}

    if len(good) >= 3:
        cons, tie = mode_call(G[good])
        any_call = (G[good] != -9).any(0)
        metric = tie.sum() / max(any_call.sum(), 1)
        rule, base = "majority", "-"
    elif len(good) == 2:
        b, o = sorted(good, key=lambda k: (miss[k], k))
        cons = G[b].copy()
        fill = cons == -9
        cons[fill] = G[o][fill]
        both = (G[b] != -9) & (G[o] != -9)
        metric = (G[b][both] != G[o][both]).sum() / max(both.sum(), 1)
        rule, base = "best_plus_fill", b + 1
    else:
        b = good[0] if good else int(np.argmin(miss))
        others = [k for k in range(len(idx)) if k != b]
        oc, _ = mode_call(G[others])
        cons = G[b].copy()
        fill = cons == -9
        cons[fill] = oc[fill]
        both = (G[b] != -9) & (oc != -9)
        metric = (G[b][both] != oc[both]).sum() / max(both.sum(), 1)
        rule, base = "fallback", b + 1

    final_miss = (cons == -9).mean()
    drop = metric > DISC_THRESH
    if rule == "fallback":
        drop = drop or final_miss > MISS_THRESH
    d.update(rule=rule, base_copy=base, discordance_metric=metric,
             final_missingness=final_miss,
             action="drop_site" if drop else "keep_consensus")
    decisions.append(d)
    if not drop:
        consensus[idx[0]] = cons

# Variant indices to write; consensus replaces the first copy of kept IDs
skip = {vi for idx in groups.values() for vi in idx} - set(consensus)
keep_idx = [i for i in range(n_var) if i not in skip]

# ---------------------------------------------------------------- write
writer = pgenlib.PgenWriter(f"{out_prefix}.pgen".encode(), n_samp,
                            len(keep_idx), nonref_flags=nonref_all)
block = np.empty((CHUNK, n_samp), dtype=np.int8)
for start in range(0, n_var, CHUNK):
    end = min(start + CHUNK, n_var)
    reader.read_range(start, end, block[: end - start])
    out = []
    for i in range(start, end):
        if i in skip:
            continue
        out.append(consensus[i] if i in consensus else block[i - start])
    if out:
        writer.append_biallelic_batch(np.ascontiguousarray(np.stack(out)))
writer.close()
reader.close()

with open(f"{out_prefix}.pvar", "w") as f:
    f.writelines(header_lines)
    f.writelines(records[i] for i in keep_idx)
shutil.copyfile(f"{in_prefix}.psam", f"{out_prefix}.psam")

# ---------------------------------------------------------------- report
cols_out = ["ID", "n_copies", "copy_missingness", "bad_copies_removed", "rule",
            "base_copy", "discordance_metric", "final_missingness", "action"]
decisions.sort(key=lambda d: -d["discordance_metric"])
with open(f"{out_prefix}.collapse_decisions.tsv", "w") as f:
    f.write("\t".join(cols_out) + "\n")
    for d in decisions:
        f.write("\t".join(f"{d[c]:.5g}" if isinstance(d[c], float) else
                          str(d[c]) for c in cols_out) + "\n")

n_drop = sum(d["action"] == "drop_site" for d in decisions)
print(f"{len(decisions)} duplicate IDs: {len(decisions) - n_drop} collapsed, "
      f"{n_drop} dropped")
by_rule = defaultdict(int)
for d in decisions:
    by_rule[d["rule"]] += 1
print("Rule used:", dict(by_rule))
print(f"Variants: {n_var} in -> {len(keep_idx)} out")
print(f"REF provisional flag written as: {nonref_all}")
for d in decisions:
    if d["action"] == "drop_site":
        print(f"  dropped {d['ID']} ({d['rule']}, discordance "
              f"{d['discordance_metric']:.4g}, missingness "
              f"{d['final_missingness']:.4g})")

"""Shared helpers for reading PLINK2 text output (.psam, .kin0, .eigenvec).

PLINK2 omits the FID column entirely when the input had no meaningful FIDs, so
every reader here discovers the ID columns from the header rather than assuming
a fixed layout.
"""

import csv


def read_header(path):
    """Return (header_fields, has_fid) for a PLINK2 text file."""
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                fields = line.lstrip("#").rstrip("\n").split()
                # .psam/.eigenvec use FID; .kin0 uses FID1
                return fields, fields[0] in ("FID", "FID1")
    raise ValueError(f"{path}: no header line starting with '#' found")

def read_table(path):
    """Yield dicts keyed by header field name, skipping the '#' prefix."""
    fields, has_fid = read_header(path)
    with open(path) as fh:
        reader = csv.reader((l for l in fh if not l.startswith("#")),
                            delimiter="\t")
        for row in reader:
            if len(row) == 1:  # whitespace- rather than tab-delimited
                row = row[0].split()
            if not row:
                continue
            yield dict(zip(fields, row)), has_fid

def sample_key(rec, has_fid):
    """Canonical tuple identifying a sample."""
    return (rec["FID"], rec["IID"]) if has_fid else (rec["IID"],)

def read_psam(path, pheno_col=None):
    """Return (samples, pheno, has_fid).

    samples : list of ID tuples in file order
    pheno   : dict ID tuple -> 'case' | 'control' | 'missing'
    """
    fields, has_fid = read_header(path)
    if pheno_col is None:
        pheno_col = "PHENO1" if "PHENO1" in fields else fields[-1]
    if pheno_col not in fields:
        raise ValueError(f"{path}: phenotype column {pheno_col!r} not present "
                         f"(columns: {', '.join(fields)})")

    samples, pheno = [], {}
    for rec, _ in read_table(path):
        key = sample_key(rec, has_fid)
        samples.append(key)
        raw = rec[pheno_col].strip()
        if raw in ("2", "case", "Case", "CASE"):
            pheno[key] = "case"
        elif raw in ("1", "control", "Control", "CONTROL"):
            pheno[key] = "control"
        else:  # 0, -9, NA, blank -> unusable for matching
            pheno[key] = "missing"
    return samples, pheno, has_fid

def write_id_file(path, keys, has_fid):
    """Write a PLINK2 --keep / --remove compatible ID file with a header."""
    with open(path, "w") as out:
        out.write("#FID\tIID\n" if has_fid else "#IID\n")
        for key in keys:
            out.write("\t".join(key) + "\n")

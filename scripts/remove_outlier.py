"""Keep the samples that fall inside a rectangle on the top two population axes.

Reads the coordinates written by example-check_population.sh (a PLINK2
.eigenvec: '#FID IID PC1 PC2 ...') and writes a PLINK2 --keep ID file. A legacy
PLINK 1.9 --mds-plot file ('FID IID SOL C1 C2') is also accepted, in which case
C1/C2 are used.

Bounds are inclusive and each one is optional, so a single axis can be filtered
on its own. Read the cutoffs off the plot produced by the population check.
"""

import argparse
import sys

from plink_ids import write_id_file


def read_coordinates(path, xcol=None, ycol=None):
    """Return (rows, has_fid, xcol, ycol); rows are (id_tuple, x, y)."""
    with open(path) as fh:
        header = fh.readline()
        fields = header.lstrip("#").split()
        rows = [line.split() for line in fh if line.strip()]

    if "IID" not in fields:
        sys.exit(f"{path}: no IID column in header line: {header.strip()}")
    has_fid = fields[0] == "FID"

    if xcol is None or ycol is None:
        # plink2 --pca writes PC1/PC2; plink 1.9 --mds-plot writes C1/C2
        for a, b in (("PC1", "PC2"), ("C1", "C2")):
            if a in fields and b in fields:
                xcol, ycol = xcol or a, ycol or b
                break
        else:
            sys.exit(f"{path}: cannot find PC1/PC2 or C1/C2 among {fields}")
    for col in (xcol, ycol):
        if col not in fields:
            sys.exit(f"{path}: no column {col!r}; available: {', '.join(fields)}")

    iid, xi, yi = fields.index("IID"), fields.index(xcol), fields.index(ycol)
    out = []
    for f in rows:
        if len(f) <= max(iid, xi, yi):
            sys.exit(f"{path}: short line: {' '.join(f)}")
        key = (f[0], f[iid]) if has_fid else (f[iid],)
        out.append((key, float(f[xi]), float(f[yi])))
    return out, has_fid, xcol, ycol


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--coords", required=True,
                   help="PLINK2 .eigenvec (or legacy PLINK 1.9 .mds) file")
    p.add_argument("--out", required=True, help="output --keep ID file")
    p.add_argument("--removed", default=None,
                   help="optional output ID file of the excluded samples")
    p.add_argument("--x1", type=float, default=float("-inf"),
                   help="minimum value on the first axis (inclusive)")
    p.add_argument("--x2", type=float, default=float("inf"),
                   help="maximum value on the first axis (inclusive)")
    p.add_argument("--y1", type=float, default=float("-inf"),
                   help="minimum value on the second axis (inclusive)")
    p.add_argument("--y2", type=float, default=float("inf"),
                   help="maximum value on the second axis (inclusive)")
    p.add_argument("--xcol", default=None, help="override the first axis column")
    p.add_argument("--ycol", default=None, help="override the second axis column")
    return p.parse_args()


def main():
    args = parse_args()
    if args.x1 > args.x2 or args.y1 > args.y2:
        sys.exit("Empty selection: x1 must be <= x2 and y1 must be <= y2")

    rows, has_fid, xcol, ycol = read_coordinates(args.coords, args.xcol, args.ycol)
    kept, dropped = [], []
    for key, x, y in rows:
        inside = args.x1 <= x <= args.x2 and args.y1 <= y <= args.y2
        (kept if inside else dropped).append(key)

    if not kept:
        sys.exit(f"No samples fall inside {xcol} in [{args.x1}, {args.x2}] and "
                 f"{ycol} in [{args.y1}, {args.y2}]; check the cutoffs against "
                 f"{args.coords}")

    write_id_file(args.out, kept, has_fid)
    if args.removed:
        write_id_file(args.removed, dropped, has_fid)

    print(f"samples in {args.coords}: {len(rows)}")
    print(f"selection               : {xcol} in [{args.x1}, {args.x2}], "
          f"{ycol} in [{args.y1}, {args.y2}]")
    print(f"removed as outliers     : {len(dropped)}")
    print(f"retained -> {args.out} : {len(kept)}")


if __name__ == "__main__":
    main()

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import polars as pl

VALID_CONTIG = r"^([1-9]|1[0-9]|2[0-2]|X|Y|MT|M)$"


def _open_pvar(pfile_prefix: str):
    """Return a polars-readable source for <prefix>.pvar or <prefix>.pvar.zst,
    streaming the decompression through a pipe rather than writing a temp file."""
    plain = Path(f"{pfile_prefix}.pvar")
    zst = Path(f"{pfile_prefix}.pvar.zst")
    if zst.exists():
        if shutil.which("zstd") is None:
            sys.exit("ERROR: found .pvar.zst but 'zstd' is not on PATH.")
        proc = subprocess.Popen(["zstd", "-dc", str(zst)], stdout=subprocess.PIPE)
        return proc.stdout, proc
    if plain.exists():
        return str(plain), None
    sys.exit(f"ERROR: could not find {plain} or {zst}")


def cmd_extract_bed(args):
    source, proc = _open_pvar(args.pfile_prefix)

    df = pl.read_csv(
        source,
        separator="\t",
        has_header=False,
        comment_prefix="#",  # drops both "##..." meta lines and the "#CHROM" header row
        new_columns=["chrom", "pos", "id", "ref", "alt"],
        schema_overrides={"chrom": pl.Utf8, "pos": pl.Int64, "id": pl.Utf8},
    )
    if proc is not None:
        proc.wait()
        if proc.returncode != 0:
            sys.exit(f"ERROR: zstd decompression failed (exit {proc.returncode})")

    df = df.with_columns(pl.col("chrom").str.replace(r"^chr", "").alias("c_norm"))
    df = df.with_columns(
        pl.when(pl.col("c_norm") == "MT").then(pl.lit("M")).otherwise(pl.col("c_norm")).alias("c_ucsc"),
        pl.col("c_norm").str.contains(VALID_CONTIG).alias("is_valid"),
    )

    valid = df.filter(pl.col("is_valid"))
    skipped = df.filter(~pl.col("is_valid"))

    bed = valid.select(
        (pl.lit("chr") + pl.col("c_ucsc")).alias("chrom_bed"),
        (pl.col("pos") - 1).alias("start"),  # BED is 0-based half-open
        pl.col("pos").alias("end"),
        pl.col("id"),
    )
    bed.write_csv(args.bed_out, separator="\t", include_header=False)

    skipped.select(
        pl.col("id"),
        (pl.lit("unrecognized_contig:") + pl.col("chrom")).alias("reason"),
    ).write_csv(args.skipped_out, separator="\t", include_header=False)

    print(
        f"[extract-bed] {df.height:,} variants read, {bed.height:,} queued for liftOver, "
        f"{skipped.height:,} skipped (non-standard contig)"
    )


def cmd_parse_results(args):
    mapped = pl.read_csv(
        args.mapped_bed,
        separator="\t",
        has_header=False,
        new_columns=["chrom_bed", "start", "end", "id"],
        schema_overrides={"chrom_bed": pl.Utf8, "start": pl.Int64, "end": pl.Int64, "id": pl.Utf8},
    )
    mapped = mapped.with_columns(
        pl.col("chrom_bed").str.replace(r"^chr", "").alias("c_raw"),
        (pl.col("start") + 1).alias("pos"),  # back to 1-based
    )
    mapped = mapped.with_columns(
        pl.when(pl.col("c_raw") == "M").then(pl.lit("MT")).otherwise(pl.col("c_raw")).alias("chrom")
    )

    # Flag variants whose new (chrom, pos) collides with another variant's.
    # A single-pass is_duplicated() on a combined key is faster here than a groupby+join for this row shape.
    mapped = mapped.with_columns((pl.col("chrom") + pl.lit(":") + pl.col("pos").cast(pl.Utf8)).alias("key"))
    mapped = mapped.with_columns(pl.col("key").is_duplicated().alias("is_dup"))
    unique = mapped.filter(~pl.col("is_dup"))
    dup = mapped.filter(pl.col("is_dup"))

    # unmapped.bed alternates a "#<reason>" comment line with the failed BED
    # record for that variant -- inherently row-order-dependent, so it's read
    # with a plain sequential pass. This file is normally small (liftOver
    # failures are typically a small fraction of the input).
    unmapped_rows = []
    reason = None
    with open(args.unmapped_bed) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith("#"):
                reason = line[1:]
            else:
                vid = line.split("\t")[3]
                unmapped_rows.append((vid, reason or "unknown"))

    unique.select("id").write_csv(args.keep_ids_out, include_header=False)
    unique.select("id", "chrom").write_csv(args.chr_update_out, separator="\t", include_header=False)
    unique.select("id", "pos").write_csv(args.pos_update_out, separator="\t", include_header=False)

    skipped_rows = []
    if args.skipped_contigs and Path(args.skipped_contigs).exists():
        with open(args.skipped_contigs) as fh:
            for line in fh:
                line = line.rstrip("\n")
                if line:
                    vid, reason_txt = line.split("\t", 1)
                    skipped_rows.append((vid, reason_txt))

    with open(args.removed_log_out, "w") as fh:
        fh.write("variant_id\treason\n")
        for vid, r in unmapped_rows:
            fh.write(f"{vid}\tliftover_failed:{r}\n")
        for vid in dup.select("id").to_series().to_list():
            fh.write(f"{vid}\tduplicate_new_position\n")
        for vid, r in skipped_rows:
            fh.write(f"{vid}\t{r}\n")

    print(
        f"[parse-results] mapped={mapped.height:,}  unmapped={len(unmapped_rows):,}  "
        f"duplicate_position_dropped={dup.height:,}  skipped_contig={len(skipped_rows):,}  "
        f"final_kept={unique.height:,}"
    )


def main():
    parser = argparse.ArgumentParser(description="Data-processing steps for the hg19->hg38 pgen liftover pipeline.")
    sub = parser.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("extract-bed", help="pvar -> liftOver BED")
    p1.add_argument("--pfile-prefix", required=True)
    p1.add_argument("--bed-out", required=True)
    p1.add_argument("--skipped-out", required=True)
    p1.set_defaults(func=cmd_extract_bed)

    p2 = sub.add_parser("parse-results", help="liftOver output -> plink2 update files + log")
    p2.add_argument("--mapped-bed", required=True)
    p2.add_argument("--unmapped-bed", required=True)
    p2.add_argument("--skipped-contigs")
    p2.add_argument("--keep-ids-out", required=True)
    p2.add_argument("--chr-update-out", required=True)
    p2.add_argument("--pos-update-out", required=True)
    p2.add_argument("--removed-log-out", required=True)
    p2.set_defaults(func=cmd_parse_results)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
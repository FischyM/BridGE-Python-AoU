import sys
from pathlib import Path

import polars as pl


def main():
    """Modify gene list file from PLINK2 by adjusting the start and end gene boundary by mapping_distance.
    The resulting file will have the suffix '.bed' and will be used in PLINK2 to filter out snps that are 
    not within or near coding genes.
    
    In addition, a second file will be created with the suffix '-chr.bed' that will have the chromosome 
    column modified to include the 'chr' prefix. This is useful for tools that require the chromosome column 
    to be in this format.
    """
    
    if len(sys.argv) < 3:
        sys.exit('not enough parameters')
    glist_file = Path(sys.argv[1])
    mapping_dist = int(sys.argv[2])
    
    df = pl.read_csv(glist_file, separator=" ", has_header=False)
    df.columns = ['chrom', 'start', 'end', 'gene']
    
    # save modified gene list with adjusted start and end boundaries
    df = df.with_columns(
        (pl.col("chrom").cast(pl.String)).alias("chrom"),
        (pl.col("start") - mapping_dist).clip(lower_bound=1).alias("start"),
        (pl.col("end") + mapping_dist).alias("end"),
    )
    out_file = glist_file.with_name(glist_file.stem + f"-{mapping_dist}.bed")
    print(f"Writing modified gene list to {out_file}")
    df.write_csv(out_file, include_header=False, separator=" ")
    
    # save modified gene list with adjusted start and end boundaries and 'chr' prefix in chromosome column
    df = df.with_columns(
        ("chr" + pl.col("chrom").cast(pl.String)).alias("chrom"),
        (pl.col("start") - mapping_dist).clip(lower_bound=1).alias("start"),
        (pl.col("end") + mapping_dist).alias("end"),
    )
    out_file = glist_file.with_name(glist_file.stem + f"-{mapping_dist}-chr.bed")
    print(f"Writing modified gene list to {out_file}")
    df.write_csv(out_file, include_header=False, separator=" ")

if __name__== '__main__':
    main()

import argparse
from pathlib import Path

import numpy as np
import polars as pl
from scipy import sparse as sps


def main():
    parser = argparse.ArgumentParser(description="Stitch LD matrices together")
    parser.add_argument("--prefix", required=True, help="Prefix of the LD matrix files")
    args = parser.parse_args()

    prefix_path = Path(args.prefix)
    
    all_variants_list = []
    sparse_array_list = []
    for i in range(1, 23):
        # read in variant file to find dimension to reshape each binary file to
        chrom_vars_file = f'{prefix_path}.ld_{i}.unphased.vcor2.bin.vars' 
        vars_df = pl.read_csv(chrom_vars_file, separator=" ", has_header=False)
        all_variants_list.append(vars_df)
        n = len(vars_df)
        print(f"Embedding Chromosome {i} ({n} x {n} array)...")

        # read the raw bin4 data for this chromosome, reshape into square array and convert to scipy sparse coo array
        chrom_bin_file = f'{prefix_path}.ld_{i}.unphased.vcor2.bin' 
        sparse_array_list.append(
            sps.coo_array(
                np.fromfile(chrom_bin_file, dtype=np.float32).reshape((n, n)),
                dtype=np.float32,
            )
        )

    # concat and save list of all variants
    total_variants_df = pl.concat(all_variants_list, how='vertical')
    total_variants_file = f'{prefix_path}.r2.unphased.vcor2.bin.vars'
    total_variants_df.write_csv(total_variants_file, separator='\t', include_header=False)

    # stitch the separate sparse blocks together along the diagonal and convert to sparse.COO for fast slicing
    print("\nStiching together all blocked sparse arrays into one...")
    sparse_array = sps.block_diag(sparse_array_list, format='coo', dtype=np.float32)
    del sparse_array_list
    if not sparse_array.has_canonical_format:
        print("Converting sparse array is into canonical format")
        sparse_array.eliminate_zeros()
        sparse_array.sum_duplicates()
    
    # save sparse matrix in npz format
    sparse_array_file = f'{prefix_path}.r2.unphased.vcor2.bin.sparse_csr.npz'
    sps.save_npz(sparse_array_file, sparse_array.tocsr())
    print(f"Sparsity: {sparse_array.nnz / (sparse_array.shape[0] * sparse_array.shape[1]) * 100:.2f}%")


if __name__ == "__main__":
    main()

#!/bin/bash
set -e
set -u
set -o pipefail


plink_file=$1       # base name of the PLINK file (without extension)
output_file=$2      # base name of the output file (without extension)
gene_list=$3        # path to the gene list file (in BED format) to filter SNPs by gene boundaries

npcs=10             # number of PCs to compute for PCA and to use in matching cases to controls


# recalculate PCA on matched dataset to use in permutation testing
plink2 --pfile "${plink_file}" \
    --pca ${npcs} \
    --out "${plink_file}.pca" --silent

python -m pca_cluster_helper \
    --pcaPrefix "${plink_file}.pca" \
    --psam "${plink_file}.psam" \
    --npcs ${npcs} \
    --out "${output_file}.cluster"


# remove SNPs outside gene boundaries
plink2 --pfile "${plink_file}" \
    --extract bed1 "${gene_list}" \
    --make-pgen --out "${output_file}"


# compute LD matrix for use in get_interaction_list
for i in {1..22}; do
    plink2 --pfile "${output_file}" \
    --r2-unphased square bin4 \
    --chr "${i}" \
    --out "${output_file}.ld_${i}" --silent
done

python -m stitch_ld --prefix "${output_file}"

rm "${output_file}".ld_*

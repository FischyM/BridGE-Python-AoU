#!/bin/bash
set -e
set -u
set -o pipefail


plink_file=$1       # base name of the PLINK file (without extension)
output_file=$2      # base name of the output file (without extension)

king_filter=$3      # cutoff for king relatedness


# filtering out related samples
plink2 --pfile "${plink_file}" \
    --make-king-table --king-table-filter "${king_filter}" \
    --out "${output_file}.king" --silent

python -m remove_related_helper \
    --kin0 "${output_file}.king.kin0" \
    --psam "${plink_file}.psam" \
    --out "${output_file}.unrelated.id" \
    --removed "${output_file}.related.id"

plink2 --pfile "${plink_file}" \
    --keep "${output_file}.unrelated.id" \
    --make-pgen --out "${output_file}"

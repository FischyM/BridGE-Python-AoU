#!/bin/bash
set -e
set -u
set -o pipefail



plink_file=$1       # base name of the PLINK file (without extension)
output_file=$2      # base name of the output file (without extension)

npcs=10             # number of PCs to compute for PCA and to use in matching cases to controls
ratio=1             # number of controls to match to each case eg., <ratio>:1


# match cases to controls
plink2 --pfile "${plink_file}" \
    --pca ${npcs} \
    --out "${output_file}.pca" --silent

python -m match_case_control_helper \
    --eigenvec "${output_file}.pca.eigenvec" \
    --eigenval "${output_file}.pca.eigenval" \
    --psam "${plink_file}.psam" \
    --weight-eigenvalues --npcs ${npcs} --ratio ${ratio} \
    --out "${output_file}.matched.id" \
    --pairs "${output_file}.matched.pairs.tsv"
    
plink2 --pfile "${plink_file}" \
    --keep "${output_file}.matched.id" \
    --make-pgen --out "${output_file}"

#!/bin/bash
set -e
set -u
set -o pipefail



plinkFile=$1
n_pcs=$2
match_ratio=$3

# match cases to controls

# PCs are computed on unrelated samples over the LD-pruned variant set. Add
# 'approx' to --pca once the sample count exceeds a few thousand.
plink2 --pfile "${plinkFile}" \
    --keep "${plinkFile}_unrelated.id" \
    --pca "${n_pcs}" \
    --out "${plinkFile}_pca"
 
python3 match_case_control.py \
    --eigenvec "${plinkFile}_pca.eigenvec" \
    --eigenval "${plinkFile}_pca.eigenval" \
    --weight-eigenvalues \
    --psam "${plinkFile}.psam" \
    --npcs "${n_pcs}" \
    --ratio "${match_ratio}" \
    --out "${plinkFile}_matched.id" \
    --pairs "${plinkFile}_matched.pairs.tsv"

# TODO: remove files?
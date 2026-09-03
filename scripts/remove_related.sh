#!/bin/bash
set -e
set -u
set -o pipefail


# remove related individuals
# KING-robust kinship is estimated on the LD-pruned set. --king-cutoff is not
# used here because its greedy selection is phenotype-blind and will discard
# cases; instead the related pairs are listed and resolved in favour of cases.


plinkFile=$1
king_cutoff=0.125   # First-degree relations (parent-child, full siblings) correspond to ~0.25
                    # second-degree relations correspond to ~0.125, etc.
                    # use a cutoff of ~0.354 (the geometric mean of 0.5 and 0.25) to screen for monozygotic twins 
                    # and duplicate samples, ~0.177 to add first-degree relations, etc.

                    
plink2 --pfile "${plinkFile}" --make-king-table --king-table-filter "${king_cutoff}" --out "${plinkFile}.king"
 
python3 remove_related.py \
    --kin0 "${plinkFile}.king.kin0" \
    --psam "${plinkFile}.psam" \
    --out "${plinkFile}.unrelated.id" \
    --removed "${plinkFile}.related.id"

plink2 --pfile "${plinkFile}" --keep "${plinkFile}.unrelated.id" --make-pgen --out "${plinkFile}.unrelated"

# TODO: remove files?
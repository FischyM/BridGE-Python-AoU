#!/bin/bash
set -e
set -u
set -o pipefail

# Drop samples that sit outside a rectangle on the top two population axes.
# Replaces scripts/data_removeoutlier.sh.
#
# The cutoffs are read off the plot produced by example-check_population.sh.
# That script now reports principal components rather than MDS dimensions, so
# cutoffs carried over from the plink 1.9 pipeline will not transfer.
#
# Usage:
#   ./example-remove_outlier.sh <plinkFile> <coordsFile> <outPrefix> <x1> <x2> <y1> <y2>
#
#   plinkFile   study fileset prefix (.pgen/.pvar/.psam)
#   coordsFile  <outPrefix>.eigenvec from example-check_population.sh
#   outPrefix   prefix for the filtered fileset
#   x1 x2       inclusive bounds on the first axis (PC1)
#   y1 y2       inclusive bounds on the second axis (PC2)

plinkFile=$1
coordsFile=$2
out=$3
x1=$4
x2=$5
y1=$6
y2=$7

python3 -m remove_outlier \
    --coords "${coordsFile}" \
    --x1 "${x1}" --x2 "${x2}" --y1 "${y1}" --y2 "${y2}" \
    --out "${out}.keep.id" --removed "${out}.outlier.id"

# --keep is an intersection, so reference-only samples in the coordinates file
# are ignored here and only study samples reach the output fileset.
plink2 --pfile "${plinkFile}" --keep "${out}.keep.id" --make-pgen --out "${out}" --silent

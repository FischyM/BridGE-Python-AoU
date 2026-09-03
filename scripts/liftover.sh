#!/bin/bash
set -e
set -u
set -o pipefail

# Lift a PLINK2 fileset to another genome build.
#
# The bundled example data needs this: example/raw/gwas_subset is GRCh37 while
# the 1000 Genomes panel it is compared against, and the glist-hg38 annotation
# used later in example-run.sh, are GRCh38.
#
# Get the chain file from UCSC, e.g.
#   curl -O https://hgdownload.soe.ucsc.edu/goldenPath/hg19/liftOver/hg19ToHg38.over.chain.gz
#
# Usage:
#   ./example-liftover.sh <plinkFile> <chainFile> <outPrefix>
#
#   plinkFile   input fileset prefix (.pgen/.pvar/.psam)
#   chainFile   UCSC .over.chain[.gz] for the direction you want
#   outPrefix   prefix for the lifted fileset
#
# Variants that do not lift are dropped rather than left behind at their old
# coordinates; <outPrefix>.unmapped.tsv lists them with a reason.

PLINK2=${PLINK2:-./plink2}

plinkFile=$1
chainFile=$2
out=$3

python3 liftover.py \
    --pvar "${plinkFile}.pvar" \
    --chain "${chainFile}" \
    --out-map "${out}.newpos" \
    --out-chr "${out}.newchr" \
    --out-mapped "${out}.mapped.id" \
    --out-unmapped "${out}.unmapped.tsv"

# --update-chr requires --sort-vars, and the new coordinates need re-sorting
# anyway. --extract drops the variants that did not lift.
${PLINK2} --pfile "${plinkFile}" \
    --extract "${out}.mapped.id" \
    --update-chr "${out}.newchr" \
    --update-map "${out}.newpos" \
    --sort-vars --make-pgen --out "${out}"

rm -f "${out}.newpos" "${out}.newchr" "${out}.mapped.id"

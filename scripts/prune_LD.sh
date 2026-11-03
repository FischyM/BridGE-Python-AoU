#!/bin/bash
set -e
set -u
set -o pipefail


plink_file=$1       # base name of the PLINK file (without extension)
output_file=$2      # base name of the output file (without extension)

ld_window=50        # window size for LD pruning in variant count (append 'kb' for kilobase units)
ld_step=5           # step size for LD pruning in variant count (required to be 1 if using kb for window size)
ld_r2=0.1           # maximum squared correlation coefficient for LD pruning


# get a less redundant set of SNPs using LD pruning
plink2 --pfile "${plink_file}" \
    --indep-pairwise ${ld_window} ${ld_step} ${ld_r2} \
    --out "${output_file}.LD" --silent

plink2 --pfile "${plink_file}" \
    --extract "${output_file}.LD.prune.in" \
    --make-pgen --out "${output_file}"

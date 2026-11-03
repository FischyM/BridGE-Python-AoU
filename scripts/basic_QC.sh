#!/bin/bash
set -e
set -u
set -o pipefail


plink_file=$1       # base name of the PLINK file (without extension)
out_file=$2         # base name of the output file (without extension)

mind=$3             # maximum allowed fraction of missing genotypes per sample
geno=$4             # maximum allowed fraction of missing genotypes per variant
hwe_p=$5            # minimum HWE p-value for variants to be included
hwe_k=$6            # minimum HWE p-value adjustment factor for variants to be included
maf=$7              # minimum allele frequency for variants to be included


# basic QC
plink2 --pfile "${plink_file}" \
    --autosome --snps-only just-acgt --exclude-palindromic-snps \
    --mind "${mind}" --geno "${geno}" --hwe "${hwe_p}" "${hwe_k}" midp keep-fewhet --maf "${maf}" \
    --sort-vars --make-pgen --out "${out_file}"

#!/bin/bash
set -e
set -u
set -o pipefail





# basic QC
plink2 --pfile "${plink_file}" \
    --autosome --snps-only just-acgt --exclude-palindromic-snps \
    --mind ${mind} --geno ${geno} --hwe ${hwe_p} ${hwe_k} midp keep-fewhet --maf ${maf} \
    --sort-vars --make-pgen --out "${plink_file}.step1.basicQC" --silent


# get a less redundant set of SNPs using LD pruning
plink2 --pfile "${plink_file}.step1.basicQC" \
    --indep-pairwise ${ld_window} ${ld_step} ${ld_r2} \
    --out "${plink_file}.step2.LD" --silent

plink2 --pfile "${plink_file}.step1.basicQC" \
    --extract "${plink_file}.step2.LD.prune.in" \
    --make-pgen --out "${plink_file}.step2.pruned" --silent

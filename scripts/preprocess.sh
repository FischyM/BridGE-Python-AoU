#!/bin/bash
set -e
set -u
set -o pipefail


### order of operations for commands in this script:

# Note chromosome filter (--chr, --not-chr, --autosome, --autosome-par)
# Exclude variants with multi-character allele codes (--snps-only)
# Exclude palindromic SNPs (--exclude-palindromic-snps)
# Assign chromosome-and-position-based names to variants (--set-all-var-ids, --set-missing-var-ids)
# Read main genotype file's header (--[b]pfile, --bfile, or freshly autoconverted)
# Calculate per-sample genotyping rate, remove samples below threshold (--mind)
# Remove variants below genotyping rate threshold (--geno)
# Hardy-Weinberg equilibrium report and/or exact test (--hardy, --hwe)
# Apply minor allele frequency and count filters (--maf, --max-maf, --mac, --max-mac)
# Write PLINK 1 or 2 binary fileset, first updating chromosome information if necessary (--make-[b]pgen)
# Perform LD-based pruning (--indep-pairwise)
# Write LD-statistic matrix/table to disk (--r[2]-[un]phased)


plink_file=$1       # base name of the PLINK file (without extension)
output_file=$2      # base name of the output file (without extension)
gene_list=$3        # path to the gene list file (in BED format) to filter SNPs by gene boundaries

# basic QC
mind=0.05           # maximum allowed fraction of missing genotypes per sample
geno=0.01           # maximum allowed fraction of missing genotypes per variant
hwe_p=0.000001      # minimum HWE p-value for variants to be included
hwe_k=0.001         # minimum HWE p-value adjustment factor for variants to be included
maf=0.01            # minimum allele frequency for variants to be included
# get a less redundant set of SNPs using LD pruning
ld_window=50        # window size for LD pruning in variant count (append 'kb' for kilobase units)
ld_step=5           # step size for LD pruning in variant count (required to be 1 if using kb for window size)
ld_r2=0.1           # maximum squared correlation coefficient for LD pruning
# filtering out related samples
king_filter=0.125   # cutoff for king relatedness
# match cases to controls
npcs=10             # number of PCs to compute for PCA and to use in matching cases to controls
ratio=1             # number of controls to match to each case eg., <ratio>:1




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


# filtering out related samples
plink2 --pfile "${plink_file}.step2.pruned" \
    --make-king-table --king-table-filter ${king_filter} \
    --out "${plink_file}.step3.king" --silent

python -m remove_related_helper \
    --kin0 "${plink_file}.step3.king.kin0" \
    --psam "${plink_file}.step2.pruned.psam" \
    --out "${plink_file}.step3.unrelated.id" \
    --removed "${plink_file}.step3.related.id"

plink2 --pfile "${plink_file}.step2.pruned" \
    --keep "${plink_file}.step3.unrelated.id" \
    --make-pgen --out "${plink_file}.step3.unrelated" --silent


# match cases to controls
plink2 --pfile "${plink_file}.step3.unrelated" \
    --pca ${npcs} \
    --out "${plink_file}.step4.pca" --silent

python -m match_case_control_helper \
    --eigenvec "${plink_file}.step4.pca.eigenvec" \
    --eigenval "${plink_file}.step4.pca.eigenval" \
    --psam "${plink_file}.step3.unrelated.psam" \
    --weight-eigenvalues --npcs ${npcs} --ratio ${ratio} \
    --out "${plink_file}.step4.matched.id" \
    --pairs "${plink_file}.step4.matched.pairs.tsv"
    
plink2 --pfile "${plink_file}.step3.unrelated" \
    --keep "${plink_file}.step4.matched.id" \
    --make-pgen --out "${plink_file}.step4.matched" --silent



# ---------------------------------------------------------------------------
# required to run BridGE
# ---------------------------------------------------------------------------

# recalculate PCA on matched dataset to use in permutation testing
plink2 --pfile "${plink_file}.step4.matched" \
    --pca ${npcs} \
    --out "${output_file}.pca" --silent

# remove SNPs outside gene boundaries
plink2 --pfile "${plink_file}.step4.matched" \
    --extract bed1 "${gene_list}" \
    --make-pgen --out "${output_file}" --silent

# compute LD matrix for use in get_interaction_list
for i in {1..22}; do
    plink2 --pfile "${output_file}" \
    --r2-unphased square bin4 \
    --chr "${i}" \
    --out "${output_file}.ld_${i}" --silent
done

python -m stitch_ld --prefix "${output_file}"

rm "${output_file}".ld_*

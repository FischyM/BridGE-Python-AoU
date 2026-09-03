#!/bin/bash
set -e
set -u
set -o pipefail

# Compare a study cohort against 1000 Genomes reference populations on the top
# two population axes, so that samples outside the intended ancestry can be
# spotted and removed (see example-remove_outlier.sh).
#
# Replaces scripts/data_checkpopulation.sh. Three changes are worth knowing:
#
#  * The two panels are no longer merged. plink2 (through v2.0.0-a.7) can only
#    concatenate filesets that share their samples; a sample-wise merge exits
#    with "Non-concatenating --pmerge[-list] is under development". Instead the
#    axes are defined by a PCA of the reference panel alone and both cohorts
#    are projected onto them with --score, which is also the better-behaved
#    option: study samples can no longer rotate the reference axes.
#  * Those axes come from --pca rather than plink 1.9 --genome + --cluster
#    --mds-plot. --genome builds an all-pairs IBD matrix, which is O(samples^2)
#    and was the slowest step by far. Coordinates are therefore PCs, not MDS
#    dimensions, and are on a different scale: read new cutoffs off the plot.
#  * Allele conflicts between the panels are resolved up front by comparing the
#    two .pvar files, rather than by attempting a merge, reading plink 1.9's
#    .missnp output and re-merging in a loop.
#
# Usage:
#   ./example-check_population.sh <studyPfile> <refPfile> <popIDFile> <outPrefix> [pops]
#
#   studyPfile  study fileset prefix (.pgen/.pvar/.psam)
#   refPfile    1000 Genomes fileset prefix
#   popIDFile   reference population file, "FID IID POP" per line
#   outPrefix   prefix for all outputs, e.g. example/intermediate/gwas_subset_prj1000
#   pops        comma-separated reference populations to plot against; this is
#               the complete list, not an addition to a built-in default
#               (see internationalgenome.org/category/population)
#               CEU: European
#               CHB: Han Chinese
#               ASW: African-American Southwest
#               YRI: Youruba
#               GIH: Gujarati Indian   
#
# Outputs:
#   <outPrefix>.eigenvec    PC1..PCn, one row per individual. Samples present
#                           in both cohorts appear once, with the study
#                           projection for their coordinates and their 1000
#                           Genomes population for their label.
#   <outPrefix>.eigenval    eigenvalues of the reference PCA
#   <outPrefix>.popid.txt   one-hot population membership, same rows
#   <outPrefix>.pdf         study cohort plotted against the reference panels

PLINK2=${PLINK2:-./plink2}

studyPfile=$1
refPfile=$2
popIDFile=$3
out=$4
pops=${5:-CEU,CHB,ASW,YRI,GIH}

maf=0.01            # PCs are unstable on rare variants, and --score cannot variance-standardize a monomorphic one
ld_window=50        # window size for LD pruning, in variant count
ld_step=5           # step size for LD pruning, in variant count
ld_r2=0.2           # maximum squared correlation between retained variants
npcs=10             # PCs to compute; the first two are plotted. Decrease this to save computation time.
legendPos=best      # southeast, southwest, northeast, northwest or best

mkdir -p "$(dirname "${out}")"


# 1. reference samples in the requested populations, and the study variant IDs
python3 check_population.py ref-samples \
    --ref-psam "${refPfile}.psam" --pop-id "${popIDFile}" --pops "${pops}" \
    --out "${out}.ref.id"

python3 check_population.py variant-ids \
    --pvar "${studyPfile}.pvar" --out "${out}.study.snps"


# 2. cut the reference panel down to those samples and variants. This is the
#    only pass over the full reference genotypes, so everything that can be
#    filtered here is: --keep-founders replaces plink 1.9's --filter-founders
#    and --max-alleles 2 replaces --biallelic-only strict.
${PLINK2} --pfile "${refPfile}" \
    --keep "${out}.ref.id" --keep-founders \
    --extract "${out}.study.snps" --max-alleles 2 --maf ${maf} \
    --make-pgen --out "${out}.ref" --silent


# 3. variants the two panels agree on: same ID, same allele pair
python3 check_population.py shared-variants \
    --study-pvar "${studyPfile}.pvar" --ref-pvar "${out}.ref.pvar" \
    --out "${out}.shared.snps"


# 4. PCA needs variants that are not too correlated with each other
${PLINK2} --pfile "${out}.ref" --extract "${out}.shared.snps" \
    --indep-pairwise ${ld_window} ${ld_step} ${ld_r2} --out "${out}.prune" --silent


# 5. define the population axes from the reference panel. --freq counts gives
#    --score the reference allele frequencies to standardise both cohorts by.
${PLINK2} --pfile "${out}.ref" --extract "${out}.prune.prune.in" \
    --freq counts --pca allele-wts ${npcs} --out "${out}.refpca" --silent


# 6. project both cohorts onto those axes. .eigenvec.allele is
#    "#CHROM ID REF ALT PROVISIONAL_REF? A1 PC1 ... PCn", hence ID in column 2,
#    A1 in column 6 and the weights from column 7 on. Running the same --score
#    over both cohorts is what makes their coordinates comparable.
score_cols="7-$(( 6 + npcs ))"

for cohort in ref study; do
    case ${cohort} in
        ref)   pfile="${out}.ref" ;;
        study) pfile="${studyPfile}" ;;
    esac
    ${PLINK2} --pfile "${pfile}" --extract "${out}.prune.prune.in" \
        --read-freq "${out}.refpca.acount" \
        --score "${out}.refpca.eigenvec.allele" 2 6 header-read \
                no-mean-imputation variance-standardize \
        --score-col-nums ${score_cols} \
        --out "${out}.${cohort}proj"
done


# 7. one coordinate table and one population-label file, written together
python3 check_population.py combine \
    --ref-sscore "${out}.refproj.sscore" \
    --study-sscore "${out}.studyproj.sscore" \
    --pop-id "${popIDFile}" --pops "${pops}" --npcs ${npcs} \
    --out-coords "${out}.eigenvec" --out-labels "${out}.popid.txt"

cp "${out}.refpca.eigenval" "${out}.eigenval"


# 8. plot the study cohort against the reference populations
python3 scripts/plotmds.py "${out}.eigenvec" "${out}.popid.txt" "${legendPos}" "${out}"


rm -f "${out}.ref.id" "${out}.study.snps" "${out}.shared.snps" \
      "${out}.prune."{prune.in,prune.out,log} \
      "${out}.ref."{pgen,pvar,psam,log} \
      "${out}.refpca."{acount,eigenval,eigenvec,eigenvec.allele,log} \
      "${out}."{ref,study}"proj."{sscore,log}

echo "population plot written to ${out}.pdf"

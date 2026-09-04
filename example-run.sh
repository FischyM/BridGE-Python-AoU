#!/bin/bash
set -e
set -u
set -o pipefail


# if you do not have an environment manager, download and install miniforge3 for your system
# and example is below
wget https://github.com/conda-forge/miniforge/releases/download/26.5.3-0/Miniforge3-26.5.3-0-Linux-x86_64.sh
bash Miniforge3-26.5.3-0-Linux-x86_64.sh
conda init

# install BridGE environment
conda env create -f env.yml
conda activate bridge

# make directories
mkdir -p example/raw
mkdir -p example/preprocess
mkdir -p example/intermediate
mkdir -p example/results

# download example data
wget https://zenodo.org/record/8067407/files/gwas_subset.bed -P example/raw
wget https://zenodo.org/record/8067407/files/gwas_subset.bim -P example/raw
wget https://zenodo.org/record/8067407/files/gwas_subset.fam -P example/raw
wget https://zenodo.org/records/8067407/files/ALL.shapeit2_integrated_v1a.GRCh38.20181129.phased.rsid.bed -P example/raw
wget https://zenodo.org/records/8067407/files/ALL.shapeit2_integrated_v1a.GRCh38.20181129.phased.rsid.bim -P example/raw
wget https://zenodo.org/records/8067407/files/ALL.shapeit2_integrated_v1a.GRCh38.20181129.phased.rsid.fam -P example/raw
wget https://zenodo.org/records/8067407/files/allpopid.txt -P example/raw

# example data needs to be converted to GRCh38. 
# This should always be done first to avoid inconsistencies with genome builds.
# download liftover and chain file for lifting data from GRCh37 to GRCh38
wget https://hgdownload.soe.ucsc.edu/admin/exe/linux.x86_64/liftOver -P example/preprocess
wget https://hgdownload.soe.ucsc.edu/goldenPath/hg19/liftOver/hg19ToHg38.over.chain.gz -P example/preprocess
chmod +x example/preprocess/liftOver


conda activate bridge-aou
cd BridGE-Python-AoU/
source setup.sh


# convert data to pgen format
plink2 --bfile example/raw/gwas_subset --make-pgen --out example/preprocess/gwas_subset
plink2 --bfile example/raw/ALL.shapeit2_integrated_v1a.GRCh38.20181129.phased.rsid --make-pgen --out example/preprocess/ALL.shapeit2_integrated_v1a.GRCh38.20181129.phased.rsid

# then, convert .pvar file to .bed file, run liftover, and convert back to .pvar file
python liftover_helper.py extract-bed \
    --pfile-prefix=example/preprocess/gwas_subset \
    --bed-out=example/preprocess/prelift.bed \
    --skipped-out=example/preprocess/prelift.skipped.txt
# liftover the BED file to GRCh38
./example/preprocess/liftOver \
    example/preprocess/prelift.bed \
    example/preprocess/hg19ToHg38.over.chain.gz \
    example/preprocess/lifted.hg38.mapped.bed \
    example/preprocess/lifted.hg38.unmapped.bed
# parse liftover output
python liftover_helper.py parse-results \
    --mapped-bed=example/preprocess/lifted.hg38.mapped.bed \
    --unmapped-bed=example/preprocess/lifted.hg38.unmapped.bed \
    --skipped-contigs=example/preprocess/prelift.skipped.txt \
    --keep-ids-out=example/preprocess/lifted.hg38.keep_ids.txt \
    --chr-update-out=example/preprocess/lifted.hg38.chr_update.txt \
    --pos-update-out=example/preprocess/lifted.hg38.pos_update.txt \
    --removed-log-out=example/preprocess/lifted.hg38.removed.log
# update the .pvar file with the new coordinates
plink2 --pfile example/preprocess/gwas_subset \
    --extract example/preprocess/lifted.hg38.keep_ids.txt \
    --update-chr example/preprocess/lifted.hg38.chr_update.txt \
    --update-map example/preprocess/lifted.hg38.pos_update.txt \
    --sort-vars --make-pgen \
    --out example/preprocess/gwas_subset.hg38
# save in plink bed format to run with BridGE 2.0
plink2 --pfile example/preprocess/gwas_subset.hg38 --autosome --make-bed --out ../BridGE-Python/testdata/raw/gwas_subset.hg38



# check the study population against 1000 Genomes reference populations.
# writes example/intermediate/gwas_subset_prj1000.{eigenvec,eigenval,popid.txt,pdf}
check_population.sh \
    example/preprocess/gwas_subset.hg38 \
    example/preprocess/ALL.shapeit2_integrated_v1a.GRCh38.20181129.phased.rsid \
    example/raw/allpopid.txt \
    example/preprocess/gwas_subset_prj1000

# drop samples that sit outside the intended cluster. The four cutoffs bound
# PC1 then PC2 and are read off example/intermediate/gwas_subset_prj1000.pdf;
# adjust it for your own data.
remove_outlier.sh \
    example/preprocess/gwas_subset.hg38 \
    example/preprocess/gwas_subset_prj1000.eigenvec \
    example/preprocess/gwas_subset.hg38.rmoutlier \
    0.075 0.11 0.075 0.12


# Preprocess the data to remove related samples, match cases to controls, and prune SNPs for LD.
preprocess.sh example/preprocess/gwas_subset.hg38.rmoutlier example/raw/gwas_final

# Run BridGE
python bridge.py --projectDir=example --module=DataProcess \
    --plinkFile=gwas_final \
    --geneAnnotation=glist-hg38 \
    --geneSets=c2.cp.v2026.1.Hs \
    --simMeasure=either \
    --jaccardCutoff=0.33 \
    --overlapCutoff=0.5
# transfer the preprocessed genotype data and filtered gene sets over to BridGE 2.0 to test
plink2 --pfile example/raw/gwas_final --make-bed --out ../BridGE-Python/testdata/intermediate/gwas_final.new
cp example/raw/c2.cp.v2026.1.Hs.* ../BridGE-Python/testdata/raw/

python bridge.py --projectDir=example --module=ComputeInteraction --model=combined \
    --nWorker=30 --nJobs=5 --R=0 --seed=42

python bridge.py --projectDir=example --module=ComputeStats --model=combined --nWorker=10 --snpPerms=100 --minPath=10 --R=5

python bridge.py --projectDir=example --module=ComputeFDR --model=combined --pvalueCutoff=0.05 --minPath=10 --samplePerms=5

python bridge.py --projectDir=example --module=Summarize --model=combined --fdrcut=0.25 --snpPathFile=snp_pathway_min10_max300.pkl

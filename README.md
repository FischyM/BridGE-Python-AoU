# BridGE 3.0

This is the github repository for the Python 3.0 version of Bridging Geneset with Epistasis (BridGE).

## Major changes from 2.0

General updates

- Updated python libraries.
- Uses PLINK2's more efficient pgen file format.
- Implemented the use of sparse arrays.
- More verbose and informative output.
- Tracking memory usage.
- Tunable parameter to split the total work by 'njobs'. This helps to reduce memory usage on systems that may be constrained.
- Permutations of phenotype and SNPs are now deterministic and reproducible regardless of the number of random networks.
- Removed using classes to store data via pickle and instead save as numpy arrays or parquet data frames. This allows for interoperability so that you do not need knowledge of of the classes to load in any of the data generated in BridGE.

Data Processing

- New filtering procedure for pathways using Jaccard similarity, overlap, or both.
- General speed ups to better scale to larger input data.

Computing SNP Interactions

- Genotype data is no longer saved separately for dominant and recessive counts.
- More accurate computing of the hypergeometric p-value using the survival function.
- Reduced the number of matrix multiplications and removes some dense intermediates.
- Phenotype permutation is now deterministic.
- Phenotype can now be permutated with respects to PCA of the genotype matrix. This works to preserve the fine-grained genotypic structure found in the provided samples.

## Installation

We recommend to use miniforge/mamba to create the same python environment. Instructions to download and install Miniforge can be found here: [https://github.com/conda-forge/miniforge](https://github.com/conda-forge/miniforge)

Once Miniforge is installed, clone this repository and install the python environment from file

```bash
git clone https://github.com/FischyM/BridGE-Python-AoU.git
cd BridGE-Python-AoU
mamba env create -f environment.yml
```

## Running the example data

For those familiar with BridGE, here are all the commands to run a BridGE analysis on our example data. See the sections below for detailed information on running each section.

Download and set up example data

```bash
# navigate to the cloned repository
cd BridGE-Python-AoU

# activate BridGE environment that was installed as described above and run the setup script
mamba activate bridge
# the setup.sh script must be run for every new session when running a BridGE analysis.
source setup.sh  

# make directories
mkdir -p example/raw
mkdir -p example/preprocess
mkdir -p example/intermediate
mkdir -p example/results

# download example data
# TODO: convert to pgen, upload to a new Zenodo repo.
wget https://zenodo.org/record/8067407/files/gwas_subset.bed -P example/raw
wget https://zenodo.org/record/8067407/files/gwas_subset.bim -P example/raw
wget https://zenodo.org/record/8067407/files/gwas_subset.fam -P example/raw
wget https://zenodo.org/records/8067407/files/ALL.shapeit2_integrated_v1a.GRCh38.20181129.phased.rsid.bed -P example/raw
wget https://zenodo.org/records/8067407/files/ALL.shapeit2_integrated_v1a.GRCh38.20181129.phased.rsid.bim -P example/raw
wget https://zenodo.org/records/8067407/files/ALL.shapeit2_integrated_v1a.GRCh38.20181129.phased.rsid.fam -P example/raw
wget https://zenodo.org/records/8067407/files/allpopid.txt -P example/raw
```

Preprocess example data

```bash
```

Run BridGE analysis

```bash
python bridge.py --projectDir=example --module=DataProcess --plinkFile=gwas_final --geneAnnotation=glist-hg38-50000-chr.bed --geneSets=c2.cp.v2026.1.Hs --simMeasure=either --jaccardCutoff=0.5 --overlapCutoff=0.5
python bridge.py --projectDir=example --module=ComputeInteraction --model=combined --nWorker=30 --nJobs=1 --seed=42 --R=10
python bridge.py --projectDir=example --module=ComputeStats --model=combined --nWorker=30 --nJobs=1 --snpPerms=10000 --seed=42 --R=10
python bridge.py --projectDir=example --module=ComputeFDR --model=combined --pvalueCutoff=0.05 --R=10
python bridge.py --projectDir=example --module=Summarize --model=combined --fdrCutoff=0.25
# 314 samples, 45610 SNPs, 949 pathways
```

## Preprocessing GWAS data

It is recommended that your GWAS data go through some preprocessing steps for quality control. We have created separate scripts so that you can decide what is right for your data. These scipts can easily be modified, but our recommended settings are hard-coded within each script. Feel free to modify the scripts and test different settings as the upgrade to PLINK2 makes many of these commands quite quick. In addition, some scripts require you choose certain arguements based on a plot, such as in scripts/check_populations.sh.

These are the preprocessing scripts that we have created to be used on GWAS data before running BridGE.

- check_populations.sh: identify and filter out samples based on ancestry of interest.
- QC.sh: basic quality control for GWAS data
- reduce_LD.sh: get a less redundant set of SNPs using LD pruning
- remove_related.sh: filter out samples based on a king relatedness threshold
- match_CC.sh: match cases to controls using PCA
- required.sh: necessary computations for BridGE, such as running PCA to guide phenotype permutations, removing SNPs outside gene boundaries, and computing an LD matrix for discovering SNP interactions when writing out the result files.

## Usage

BridGE is controlled through the bridge.py file and is separated into 5 different modules.

- Data Processing (DataProcess)
- Computing SNP-SNP Interaction Networks (ComputeInteraction)
- Computing Pathway-level Statistics (ComputeStats)
- Computing False Discovery Rate (ComputeFDR)
- Writing Results (Summarize)

ComputeInteraction and ComputeStats are where the heavy computational work is done, can be parallelized with 'nWorker', and memory usage can be limited with 'nJobs'.

There are only two required args for all of the 5 modules, however, the example run above shows args that we suggest you also use, at least for your first analysis.

- --projectDir (directory for the analysis. This stays the same for each module)
- --module (can be either 'DataProcess', 'ComputeInteraction', 'ComputeStats', 'ComputeFDR', or 'Summarize')

### Data Processing (DataProcess)

This module takes in mutliple input files and transforms them into data structures that BridGE can use. All functions in DataProcess are fairly efficient, but the work for generating the pathway indices for WPMs and BPMs can be accelerated with multiprocessing. It does not scale linearly, and doesn't benefit from anymore 'n_workers' than there are physical cores on the machine. Large benefits can be seen using 4-8 'n_workers' and anything more than that shows very marginal gains. This may change based on however many SNPs and pathways are used to create the pathway indicies.

Input files should be in the 'raw' directory and include the following:

- GWAS data, formatted in PLINK2's pgen file. This should already have been preprocessed as described above, as the only processing to the sample SNP data will be to convert any missing genotype data to '0'.
- Two files of gene sets, downloaded from Molecular Signature Database (MSigDB). For example, 'c2.cp.v2026.1.Hs.entrez.gmt' and 'c2.cp.v2026.1.Hs.symbols.gmt'. These are used generate a gene-to-pathway mapping and has been provided for you in the 'refdata' directory. Feel free to download other gene sets that may be of interest.
- A file of gene boundaries in GRCh38 called 'glist-hg38'. This is used to generate a SNP-to-gene mapping and is given to you in the 'refdata' directory. Additionally, the 'refdata/convert_glist.py' script will create two files that extend the gene boundary in both directions by a 'mapping_distance', and saves two files: one where the chromosome is simply the number (ie. 19) and one that saves the chromsome with 'chr' appended (ie. chr19). You should use whichever file matches your data's formatted chromosomes.

Args

- asd
- asd
- asd

Output files will be located in the 'intermediate' directory and include the following:

- snp_data.npz
- gene_pathway_mapping.parquet
- snp_gene_mapping.parquet
- snp_pathway_mapping.parquet
- pathway_indices-wpm.parquet
- pathway_indices-bpm.parquet

### Computing SNP-SNP Interaction Network (ComputeInteraction)

### Computing Pathway-level Statistics (ComputeStats)

### Computing False Discovery Rates (ComputeFDR)

### Writing Results (Summarize)

## Tips and Tricks

- Users should empirically identify proper settings for 'nWorker' and 'nJobs' by running a single network and seeing how long it takes and how much memory is used. This can be done using the '--i' parameter instead of '--R'. Generally, the more cpus/cores/nWorkers, the faster the computations will run. However, this could use too much RAM than what is available by the system. Therefore, BridGE by default will print how much memory BridGE used for either ComputeInteraction or ComputeStats.
- BridGE generates networks of SNP interactions for the same input of SNPs. Therefore, ComputeInteraction only needs to be run once, and ComputeStats can then be run on different sets of pathways. This will require running DataProcess to set up the list of pathways to test, but it does remove one of two large computational steps.
- ComputeInteraction uses a cache when calculating hypergeometric p-values. This means that every input and output is cached, so if an identical input into the hypergeometric function is seen again, the cached output is used automatically and the hypergeometric function will not need to compute the resulting p-value. BridGE takes advantage of this if the user runs all real and random SNP-interaction networks sequentially in one command, such as using '--R=20'. For small sized cohorts (less than 10k) this cache won't grow large enough to be a burden. Memory tracking for ComputeInteraction is available by default, so for much larger sample sizes (100k or more) the user should empirically determine how much memory is being used. If this happens, one could run each network by itself '--i=0', '--i=1', ...etc., or one could change the decoration of the '_hyge_single' function in src/HygeCache.py to '@functools.lru_cache(maxsize=2_000_000)', which instead uses the Least Recently Used Cache decoration to cap the number of cached entries to 2,000,000, or any other number that makes sense for your use case.

## SLURM Scheduler

We have provided files that you can modify in the slurm_scripts directory to run BridGE. We have found that for HPC systems it can be easier to run the heavy computational parts (ComputeInteractions, ComputeStats) within a single sbatch command using arrays, as demonstrated in slurm_scripts/computations.sh. The results also can run quite fast, so combining ComputeFDR and Summarize together is easy to do as well, as shown in slurm_scripts/results.sh. For preprocessing the GWAS data and the DataProcess module, we find it easier to run this locally first until you are satisfied, then run the heavy computational load on an HPC system.

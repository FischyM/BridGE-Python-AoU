# BridGE 3.0

This is the github repository for the 3.0 version of Bridging Geneset with Epistasis (BridGE). Functionality is the same as our previous version (see section How to Cite) but there have been many updates to speed up the whole pipeline and considerably reduce the storage requirements.

## Major changes from 2.0

General updates

- Updated python libraries.
- Uses PLINK2's more efficient pgen file format.
- Implemented the use of sparse arrays.
- More verbose and informative output.
- Tracking memory usage.
- Tunable parameter to split the total work by 'nJobs'. This helps to reduce memory usage on systems that may be constrained. The arg 'nWorkers' now controls the number of CPU cores to paralleize the work.
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

For those familiar with BridGE, here are all the commands to run a BridGE analysis on our example data. See 'Usage' section for more details.

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
wget https://zenodo.org/record/23769209/files/gwas_subset.hg38.pgen -P example/raw/
wget https://zenodo.org/record/23769209/files/gwas_subset.hg38.pvar -P example/raw/
wget https://zenodo.org/record/23769209/files/gwas_subset.hg38.psam -P example/raw/
wget https://zenodo.org/records/23769209/files/ALL.shapeit2_integrated_v1a.GRCh38.20181129.phased.rsid.pgen -P example/raw/
wget https://zenodo.org/records/23769209/files/ALL.shapeit2_integrated_v1a.GRCh38.20181129.phased.rsid.pvar -P example/raw/
wget https://zenodo.org/records/23769209/files/ALL.shapeit2_integrated_v1a.GRCh38.20181129.phased.rsid.psam -P example/raw/
wget https://zenodo.org/records/23769209/files/allpopid.txt -P example/raw/
```

Preprocess example data

```bash
```

Run BridGE analysis

```bash
python bridge.py --projectDir=example --module=DataProcess --plinkFile=gwas_final --geneAnnotation=glist-hg38-50000.bed --geneSets=c2.cp.v2026.1.Hs --simMeasure=either --jaccardCutoff=0.5 --overlapCutoff=0.5 --nWorker=30
python bridge.py --projectDir=example --module=ComputeInteraction --model=combined --nWorker=30 --nJobs=4 --seed=42 --R=10
python bridge.py --projectDir=example --module=ComputeStats --model=combined --nWorker=30 --nJobs=4 --snpPerms=10000 --seed=42 --R=10
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

BridGE is controlled through bridge.py and is separated into 5 different modules.

- Data Processing (DataProcess)
- Computing SNP-SNP Interaction Networks (ComputeInteraction)
- Computing Pathway-level Statistics (ComputeStats)
- Computing False Discovery Rate (ComputeFDR)
- Writing Results (Summarize)

ComputeInteraction and ComputeStats are where the heavy computational work is done, can be parallelized with 'nWorker', and memory usage can be limited with 'nJobs'.

There are only two required args for all of the 5 modules, however, the example run above shows args that we suggest you also use.

- projectDir: directory for the analysis. This stays the same for each module
- module: can be either 'DataProcess', 'ComputeInteraction', 'ComputeStats', 'ComputeFDR', or 'Summarize'

Then, there are common args that are used in more than one module

- model: indicates the SNP interaction model which can be 'RR', 'RD', 'DD', or 'combined'. We suggest to use combined.
- nWorker: the number of cores/CPUs used to paralleize work. More workers will use more RAM.
- nJobs: the number of segments used to split up the work. This can be used to reduce the overall RAM usage without greatly extending computational time.
- i: a single network to run. For the real network use '0', and any random network will be '1' or higher.
- R: Used in ComputeInteractions and ComputeStats, instead of running one network, run up to 'R' many networks. This includes the real network. For example, if '--R=2', then BridGE will run networks 0, 1, and 2. For ComputeFDR, R indicates how many random networks were generated so that those results can be loaded in to compute the false discovery rates.
- densityCutoff: can be between 0 and 1, this is used in ComputeStats and Summarize to binarize the SNP interaction network. The same value should be used for both modules so that the written results follow the same networks.
- seed: value to initialize random sampling for deterministic results.
- ssmFile: an alternative SNP interaction network to BridGE that is addititve and can be generated by CASSI.
- noMem: used to prevent the memory usage of BridGE in ComputeInteraction and ComputeStats.
- compressed: compress the output files. This will make saving files take longer but reduces the storage space of the intermediate files.

### File details

To see what files are created by BridGE, see FILE_METADATA.md

### Data Processing (DataProcess)

This module takes in mutliple input files and transforms them into data structures that BridGE can use. All functions in DataProcess are fairly efficient, but the work for generating the pathway indices for WPMs and BPMs can be accelerated with multiprocessing. It does not scale linearly, and doesn't benefit from anymore 'n_workers' than there are physical cores on the machine. Large benefits can be seen using 4-8 'n_workers' and anything more than that shows very marginal gains. This may change based on however many SNPs and pathways are used to create the pathway indicies.

Input:

Files must be in the 'raw' directory and include the following:

- GWAS data, formatted in PLINK2's pgen file. This should already have been preprocessed as described above, as the only processing to the sample SNP data will be to convert any missing genotype data to '0'.
- Two files of gene sets, downloaded from Molecular Signature Database (MSigDB). For example, 'c2.cp.v2026.1.Hs.entrez.gmt' and 'c2.cp.v2026.1.Hs.symbols.gmt'. These are used generate a gene-to-pathway mapping and has been provided for you in the 'refdata' directory. Feel free to download other gene sets that may be of interest.
- A file of gene boundaries in GRCh38 called 'glist-hg38'. This is used to generate a SNP-to-gene mapping and is given to you in the 'refdata' directory. Additionally, the 'refdata/convert_glist.py' script will create two files that extend the gene boundary in both directions by a 'mapping_distance', and saves two files: one where the chromosome is simply the number (ie. 19) and one that saves the chromsome with 'chr' appended (ie. chr19). You should use whichever file matches your data's formatted chromosomes.

Args:

- plinkFile: name of your plink2 file without a file extension. This needs to be in the 'raw' directory.
- geneSets: Gene set (Pathway) file, typically from MSigDB. This needs to be in the 'raw' directory.
- geneAnnotation: A file that has the chromosome, start, and end position of genes. This is provided for you in the 'refdata' directory. This needs to be in the 'raw' directory.
- minPath: Minimum number of genes required to keep a pathway. We recommend never using less than 10. This also is used as a minimum filter for SNPs that have been mapped to WPMs and BPMs.
- maxPath: Maximum number of genes required to keep a pathway. We suggest 300, but this limit is more flexible than minPath.
- simMeasure: Type of similarity metric to remove pathways with. Can be jaccard, overlap, either, or none. Either uses both the jaccard and overlap metrics. It is highly recommended that there is at least some fitlering to remove highly redundant pathways.
- jaccardCutoff: Cutoff for jaccard similarity. Can be between 0 and 1.
- overlapCutoff: Cuttoff for overlap similarity. Can be between 0 and 1.

Common Args:

- nWorkers
- compressed

Output:

Files will be located in the 'intermediate' directory and include the following

- snp_data.npz
- gene_pathway_mapping.parquet
- snp_gene_mapping.parquet
- snp_pathway_mapping.parquet
- pathway_indices-wpm.parquet
- pathway_indices-bpm.parquet

### Computing SNP-SNP Interaction Network (ComputeInteraction)

This module computes the modified hypergeometric score for SNP-SNP interactions under a risk and protective model assumptions. This module scales nearly linearly with increasing nWorkers and the working memory can be limited by splitting up the amount of work in nJobs number of seqments. For random networks, sample phenotypes are randomly permuted, which can be deterministicly controlled using the '--seed' arg. The same seed for one random network will not result in the permutation for a different random network when using the same seed. The actual value given for deterministic sampling with the random network number multipled by the value of seed. See Tips and Tricks section for other suggestions.

Input:

There are no external inputs for this module

Args:

- alpha1: upper bound p-value cutoff for carriers of both SNPs (p11)
- alpha2: lower bound p-value cutoff for the other three possibilities (p01, p10, p00)

Common args:

- nWorkers
- nJobs
- i/R
- seed
- compressed
- noMem

Output:

Files will be located in the 'intermediate' directory and are specific to the interaction model and network number. Therefore, for a run of 1 real network and 10 random networks for the combined model, there will be 44 file (11 x 4) as the combined model uses the maximum score of the dominant-dominant, recessive-dominant, and recessive-recessive models.

For example, a 'combined' model on random network '10' will be 'snp_interaction_networks-combined-R10.npz'

### Computing Pathway-level Statistics (ComputeStats)

This module computes per-pathway statistics for a protective and risk model assumptions. This model is limited to bandwidth, so using more nWorkers or more nJobs will not scale linearly. See section 'Tips and Tricks' for more details on this. This module uses SNP permutations for real and random networks to generate an empirical distribution for assessing the pathway signficance. These permutations are controlled using the '--seed' arg. As the random SNP interactions networks have already been permuted, the randomness for this model will be deterministic to the SNP permutation number, meaning if this module is ran with 1,000 or 10,000 permutation, the first 1,000 SNP permutations will be the same between both runs.

Input:

Optionally, one can use an additive evaluation of SNP interactions using the CASSI software, which can be supplied to BridGE with the 'ssmFile' arg. See 'CASSI' section for details on how to run this.

Args:

- binaryNetwork: binarize the SNP interaction network with a given densityCutoff.
- snpPerms: number of SNP permutations to perform to create an empirical distribution for pathway statistics.

Common args:

- ssmFile
- densityCutoff
- nWorkers
- nJobs
- i/R
- seed
- compressed
- noMem

Output:

Files will be located in the 'intermediate' directory and are specific to the interaction model and network number. Two files per network are generated: protective risk pathway statistics. For example, using model 'combined' for random network '10' would result in two file: 'pathway_stats-prot-combined-R10.npz' and 'pathway_stats-risk-combined-R10.npz'.

### Computing False Discovery Rates (ComputeFDR)

This module uses the results from the real and random networks to estiamte false discovery rate (FDR) for each pathway tested. If your SNP interaction network was generated by CASSI, you will need to supply the same file name as you did in ComputeStats so this module can find the correct pathway statistic files. The common arg 'R' here is used to indicate how many random networks were generated to use in FDR calculations.

Input:

There are no external inputs for this module.

Args:

- pvalueCutoff: cutoff to determine significant pathways

Common args:

- ssmFile
- R
- compressed

Output:

A single file is created in the 'intermediate' directory that is specific to the model and real network number. For example, for model 'combined' using the SNP interactions generated from BridGE, there will be one file saved named 'results-combined-R0.npz'

### Writing Results (Summarize)

This module takes the FDR results and exports the significant results to an excel file. If significant pathways were found, additional output files are created. See 'Output' below for an overview.

Input:

There are no external inputs for this module.

Args:

- fdrCutoff: cutoff to determine significant pathways

Common args:

- ssmFile
- densityCutoff

Output:

All files are saved to the 'results' directory. The following files can be created.

- result file 'output-combined-R0.xlsx'
- WPM interaction list 'interaction_list_wpm_combined_0.25.xlsx'
- BPM interactions list 'interaction_list_bpm_combined_0.25.xlsx'
- network map 'network-map-results_ssM_mhygessi_combined_R0.pdf'

## Tips and Tricks

### Choosing approrpiate nJobs and nWorkers

Users should empirically identify proper settings for 'nWorker' and 'nJobs' by running a single network and seeing how long it takes and how much memory is used. This can be done using the '--i' parameter instead of '--R'. Generally, the more cpus/cores/nWorkers, the faster the computations will run. However, this could use too much RAM than what is available by the system. Therefore, BridGE by default will print how much memory BridGE used for either ComputeInteraction or ComputeStats. It is highly recommended to do this for ComputeStats, as the benefit of more nWorkers is not linear and appears to be capped by memory bandwidth, meaning larger nJobs and larger nWorkers won't necessarily lead to better run times. A good rule of thumb is to test 3 different nWorkers (ie. 10, 20, 30) and 3 different nJobs (ie. 5, 10, 20) and see which of the 9 combinations are most effective for your data and your system.

### Not rerunning ComputeInteractions

BridGE generates networks of SNP interactions for the same input of SNPs. Therefore, ComputeInteraction only needs to be run once, and ComputeStats can then be run on different sets of pathways. This will require running DataProcess to generate different lists of pathways to test, but it does remove one of the two large computational steps. This is helpful when determining what pathway filtering criteria you'd like to use, such as filter for pathway size and similarity.
  
### Running ComputeInteractions sequentially

The most time consuming step of ComputeInteraction is the hypergeometric test for every SNP interaction. BridGE uses a cache when calculating these tests which means that every input and output is saved, so if an identical input into the hypergeometric function is seen again, the cached output is used automatically. BridGE takes advantage of this if the user runs all real and random SNP-interaction networks sequentially in one command, such as using '--R=20'. For small sized cohorts (less than 10k) this cache won't grow large enough to be a burden. Memory tracking for ComputeInteraction is available by default, so for much larger sample sizes (100k or more) the user should empirically determine how much memory is being used. If this happens, one could run each network by itself '--i=0', '--i=1', ...etc., or one could change the decoration of the '_hyge_single' function in src/HygeCache.py to '@functools.lru_cache(maxsize=2_000_000)', which instead uses the Least Recently Used Cache decoration to cap the number of cached entries to 2,000,000, or any other number that makes sense for your use case.

### Reducing storage space

Due to the nature of BridGE using many random networks to derive empirical p-values, there is a need for ample storage space for all the intermediate files. Currently, all files are saved in efficient formats such as Numpy arrays, however, these files can be further compressed using the '--compressed' arg. This will lead to longer run times, but can be beneficial if many random networks are being generated.

## SLURM Scheduler

We have provided files that you can modify in the slurm_scripts directory to run BridGE. We have found that for HPC systems it can be easier to run the heavy computational parts (ComputeInteractions, ComputeStats) within a single sbatch command using arrays, as demonstrated in slurm_scripts/computations.sh. The results also can run quite fast, so combining ComputeFDR and Summarize together is easy to do as well, as shown in slurm_scripts/results.sh. For preprocessing the GWAS data and the DataProcess module, we find it easier to run this locally first until you are satisfied, then run the heavy computational load on an HPC system.

## How to cite

If using BridGE for academic work, please cite our Nature Protocol paper here: <https://www.nature.com/articles/s41596-024-00954-8>

Hajiaghabozorgi, M., Fischbach, M., Albrecht, M., Wang, W. & Myers, C. L. BridGE: a pathway-based analysis tool for detecting genetic interactions from GWAS. Nat. Protoc. 19, 1400–1435 (2024).
  
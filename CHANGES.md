# List of changes made to BridGE when refactoring to work with biobank scale whole genome sequencing data

## Run times

python bridge.py --projectDir=testing --job=ComputeInteraction --model=combined --nWorker=30 --njobs=10 --R=5

| Module             | time old         | time new           | speed up |
| ------------------ | ---------------- | ------------------ | -------- |
| DataProcess        | 11 min           | 4 min              | 2.74x    |
| ComputeInteraction | 1 hr per network | 6 min per network  | 10x      |
| ComputeStats       | 3 hr per network | 20 min per network | 9x       |
| ComputeFDR         | 4 hr             | 3 sec              | 4800x    |
| Summarize          | 5 min            | 30 sec             | 10x      |

If extended to 1 real network and 20 random networks (using the test data provided [446 samples, 15k SNPs, 2232 pathways] and only 100 snp perms instead of 10000), this would originally take 88 hrs sequentially vs. now it takes 9 hours sequentially and uses less storage for interaction networks.

## Python packages and environment

- all packages were downloaded to their latest version for python 3.12
- TODO: use polars for dataframe operations instead of pandas

This appears to be what is needed for the python version of BridGE

```bash
# use miniforge to replace future conda enviornments 
conda create -name bridge-aou -c conda-forge python=3.12 matplotlib networkx numpy pandas scipy seaborn cython
```

however, I would add the following:

```bash
conda create -name bridge-aou -c conda-forge python=3.12 matplotlib networkx numpy pandas scipy seaborn cython jupyterlab ipython scikit-learn polars openpyxl bioconda::pgenlib
pip install psutil
conda env export > env.yml
# conda env create -f updated-environment.yml
# TODO: remake the environment with just the essential packages
```

## Plink

- updated to plink2 softare and pgen file format
- plink2 AVX2 optimized version was downloaded

## Processing before BridGE / Filtering of genotype data

### The following is what we have previously filtered for in 3-get_genotypes.ipynb

| description                                                               | command line arg               |
| ------------------------------------------------------------------------- | ------------------------------ |
| remove samples identified by AoU                                          | --remove samples_to_remove.txt |
| keep only snps                                                            | --snps-only just-acgt          |
| exclude all biallelic A/T and C/G SNPs                                    | --exclude-palindromic-snps     |
| Exclude all unplaced and non-autosomal variants.                          | --autosome                     |
| filters out variants with missing call rates exceeding the provided value | --geno 0.01                    |
| filters out variants with allele frequency below the provided threshold   | --maf 0.05                     |

### We will preprocess the data before running BridGE in 4-prepare_data.ipynb

In BridGE, GWAS data is preprocessed in the following way (found in preprocessgwas.sh)

Remove samples and variants that are missing for than the given threshold.
``plink2 --pfile {inPlinkFile} --hwe 0.00001 0.001 --make-pgen --out {outPlinkFile}``

These scripts can be skipped as they are taken care in the following ways

- extractchr1-22.sh: can be replaced with --autosome, which was implemented in the previous filtering step
- excludenogenotypesnps.sh: can be replaced by using pgen file format which automatically removes variants without assigned genotypes
- removerelatedindividual.sh: Remove related individuals, which was already calculated by AoU and implemented in the previous filtering step
- matchcasecontrol.sh: matches case to controls by computing IBD, clustering with .genome file, and selects for each case a corresponding control that are most similar to eachother. This could get expensive for AoU, so we will instead skip this step and keep all samples

Get a less redundant SNP set based on R2
``plink2 --pfile {inPlinkFile} --indep-pairwise 50 5 0.1 --out {outPlinkFile}``

Compute all R2 pairwise data to later be used in get_interaction_list. In plink2, you will need to specify if the data is phased or unphased.
``plink2 --pfile {inPlinkFile} --r2-[un]phased square``

Overall, the plink2 commands that have run in part 4-prepare_data are these:

```bash
# filter out for HWE and LD. just write snplist that passes and filter on that. No need to duplicate genotype data
plink2 --pfile {input_file} --hwe 0.000001 0.001 midp keep-fewhet --write-snplist --indep-pairwise 50 5 0.1 --out {output_file}

# save filtered variants and filtered samples into pgen file
plink2 --pfile {input_file} --extract-intersect {snplist1} {snplist2} --keep {keeplist} --make-pgen --out {output_file}

# modify gene list file from Plink to filter out snps that are not within or near coding genes
plink2 --pfile {input_file} --extract bed1 {gene_list} --make-pgen --out {output_file}

# calculate pairwise R2 for each chromosome
plink2 --pfile {input_file} --r2-unphased square bin4 --out {output_file} --chr {i}
# stich chromosome R2 together as diagonal blocks of a whole R2 matrix. Done with python.
```

TODO: redo shell scripts to change everything to plink2 that will run the same way as we do for the AoU data

### Population check and outlier removal

`scripts/data_checkpopulation.sh` and `scripts/data_removeoutlier.sh` were
replaced by `example-check_population.sh` + `check_population.py` and
`example-remove_outlier.sh` + `remove_outlier.py`. Same two outputs as before
(a plot of the study cohort against the 1000 Genomes reference populations, and
a genotype fileset with the off-cluster samples dropped), but:

- Population axes come from `plink2 --pca` instead of plink 1.9
  `--genome` + `--cluster --mds-plot 2`. `--genome` builds an all-pairs IBD
  matrix, which is O(samples^2) and dominated the old runtime. Coordinates are
  now PCs, not MDS dimensions, so the `--x1/--x2/--y1/--y2` cutoffs are on a
  different scale and have to be re-read off the new plot.
- **The two panels are no longer merged.** plink2 through v2.0.0-a.7 cannot do
  a sample-wise merge: `--pmerge` exits with "Non-concatenating
  `--pmerge[-list]` is under development", whether or not the sample sets
  overlap. So the axes are defined by a PCA of the reference panel alone
  (`--pca allele-wts`) and both cohorts are projected onto them with `--score`.
  Running the same `--score` over both cohorts is what makes the coordinates
  comparable; verified by projecting the reference panel onto its own PCs and
  recovering `--pca`'s eigenvectors to within 1e-6 (up to the per-PC sign and
  scale that a shared projection cancels out). This is also the better-behaved
  choice statistically: study samples can no longer rotate the reference axes.
- Allele conflicts between the study panel and the reference panel are found by
  comparing the two `.pvar` files up front (`check_population.py
  shared-variants`). The old script discovered them by attempting a merge,
  reading plink 1.9's `.missnp` output and re-merging in a `while` loop, which
  meant merging the data up to three times.
- The reference panel is subset (populations, founders, biallelic, MAF, study
  variants) in a single `plink2` pass rather than four chained plink 1.9
  `--make-bed` calls, each of which rewrote the whole genotype matrix.
- The one-hot population file was built by one `grep` over the population file
  per sample, plus an `awk` per sample to place the 1. It is now a single pass
  with a dict lookup, written from the same row list as the coordinate table so
  the two cannot disagree.
- The coordinate and label files hold one row per individual. A sample present
  in both cohorts -- the bundled example draws its study data from the
  reference panel, so all 479 reference samples are also study samples -- gets
  the study projection for its coordinates and its 1000 Genomes population for
  its label. That reproduces plink 1.9 `--bmerge`, which merged same-ID samples
  into one; keeping both rows instead buries the reference clouds underneath
  their study twins in the plot.
- Outlier selection reads the `.eigenvec` directly instead of `awk`-ing fixed
  column positions, so it does not break when plink2 omits the FID column, and
  each of the four bounds is optional.
- Both scripts write pgen rather than bed/bim/fam; `example-preprocess.sh` now
  takes `--pfile` to match.

Noticed while testing: **the bundled example data was on two genome builds.**
`gwas_subset` is GRCh37 while the 1000 Genomes panel is GRCh38 (e.g.
rs144434834 is at 1:723918 in the study and 1:788538 in the reference); 836207
of the 843458 shared variants disagreed on position. The old script never
noticed because plink 1.x `--bmerge` matches on variant ID alone. It did not
invalidate the population axes -- `--score` also matches on ID, and genotypes
are build-independent -- but LD pruning uses coordinates, and `example-run.sh`
goes on to annotate with `glist-hg38`.
`check_population.py shared-variants` warns when this happens.

### Liftover

`example-liftover.sh` + `liftover.py` were added, and `example-run.sh` now
lifts `gwas_subset` from GRCh37 to GRCh38 before anything else.

PLINK cannot lift over, and the usual helpers (UCSC `liftOver`, CrossMap) are
extra installs, so the chain lookup is done in `liftover.py`. A UCSC
`.over.chain` is small -- hg19ToHg38 is ~1300 chains and ~56000 blocks -- and
SNPs only need point lookups, so a heavier dependency buys nothing. It follows
liftOver's rules: highest-scoring chain wins where several cover a position,
non-primary contigs (alt/random/unplaced) are dropped, and minus-strand targets
are dropped because their alleles would need complementing and `--update-map`
cannot do that. It emits `--update-map`, `--update-chr` and `--extract` files;
the `--extract` list matters, because `--update-map` leaves unlisted variants
at their old coordinates and would silently mix two builds in one fileset.

On `gwas_subset`: 997729 of 1000767 variants lift (99.70%) in about 10 seconds.
Of the 3038 dropped, 2514 land on the minus strand, 348 on non-primary contigs
and 176 have no aligned block. Validated against the GRCh38 1000 Genomes panel,
which is an independent source of coordinates for the same rsIDs: of the 842322
lifted variants also in the panel, 842303 agree (99.998%). The 19 that disagree
all sit in the 1q21 segmental duplications, where the chain is genuinely
ambiguous and real `liftOver` behaves the same way.

Alleles are not checked against the GRCh38 reference. A small number of sites
differ in which allele is reference between builds; BridGE treats REF/ALT
symmetrically so it does not matter here, but
`plink2 --fa <GRCh38>.fa --ref-from-fa force` will fix it if something
downstream needs a correct REF.

### Imputation

Since AoU has diverse ancestry samples, we fill in any missing variant values (we are not imputing variants that were not genotypes) that were set that way by AoU quality filtering. This involves selecting the statistically phased genomic regions that overlap with out data. This is done in with plink, bcftools, and python.

## Files - Other

- added function definitions where reasonable TODO:
- cyadd.pyx: removed as this is no longer needed
- bridge.py was refactored to use argparse and reusbale functions

## Classes

- merge all separate classes into one python file
- TODO: rename classes to be more informative

## DataProcess using Datatools

- merge all separate files into one python file
- plink2pkl.py
  - This implementation's result matches the older version.
  - Changed to use pgen file format with --export A option in plink2.
  - Change to loading with pgenlib to make it cleaner and so that I don't have to save a large genotype file as a raw text file
  - sparseness of the genotype file is assessed to see whether or not it would be worth saveing as a sparse array
- bindataa.py
  - This implementation's result matches the older version.
  - redundantly saves SNPdata class. Instead, run the code in this file whenever a dominant or recessive data type is needed.
- bpmind.py
  - This implementation's result matches the older version.
  - spmatrix was refactored so that there are no values larger than 1, these checks aren't needed anymore
  - wpmdata size does not divide by two like you would for an n choose k problem where k=2, however, this is accounted for later on.
  - use min_path to remove pathways that are too small that would have been removed anyways in ComputeStats.
- imputesnp.py
  - no longer needed as imputation should be done outside of BridGE, as detailed in my AoU repo. This is to account for the fact that the All of Us data has a diverse ancestry and basic imputation would only work for samples of the same ancestry.
- mapsnp2gene.py
  - This implementation's result matches the older version.
  - adjustments for pgen file and variant ids is renamed since we can't use rsIDs for whole genome SNPs. This change was propagated throughout the code and classes.
  - replaced lambda filtering for numpy boolean arrays to speed up computations.
  - snp-gene matrix is saved using bools instead of int.
- msigdb2pkl.py
  - This implementation's result matches the older version.
  - jagged csv files are read differently. I keep 3 columns, of which, the gene name column holds a list of genes that are in each pathway.
  - a binary (boolean) matrix is created and used that fills entries array-wise based on genes in each pathway.
  - Implemented Jaccard and overlap filtering criteria to the pathways.
- snppathway.py
  - This implementation's result matches the older version.
  - speed improvements with numpy array broadcasting when testing if pathway size is between 10 and 300
  - create a snp to pathway matrix using sparse dot product which speeds up this calculation tremendously
  - checks again now if pathways size is between 10 and 300 for SNPs this time.
  - removing any SNPs not in pathways, and any pathways with no SNPs.

## ComputeInteraction

- This implementation's result matches the older version.
- TODO: add in memory tracking for users to identify best configuration of n_workers and n_jobs
- matrix_operations_par.py
  - TODO: add in a seed arg to control what random permutations get seeded with.
  - kept the splitting of jobs implementation, however, I noticed that numpy uses all available CPUs for mat mul calculations. Therefore, instead of running split jobs simultaneously across workers (which would could also increase RAM usage with a large number of SNPs), I split jobs with n_jobs and n_workers are used within each job. This means that we can adjust how big the total SNP-SNP interaction computation is (n_jobs, reduce RAM usage) while still using many workers to run all the hypergeometric tests.
    - n_jobs won't make this module run any faster, but helps to keep RAM usage down if you have a system with that restriction
    - n_workers will reduce the time it takes to run this module
    - This could be advantageous for the VM options given by AoU, such as using the high-cpu VMs.
  - Using sparse arrays for interaction network to save space.
  - Claude found a way to reduce 12 mat muls to 2, so instead of computing g10/g01/g00/x10/x01/x00 separately for both protective and risk networks, these are derived from row/column sums of g11/x11. This also removes dense intermediate arrays.
  - removed multiprocessing initialization of args and global variables.
  - updated random seed generation of permuted pheno index.
  - parallel pool is creatd in bridge.py so that we don't have to keep creating and closing workers, especially if we can use --R=5 to run all of the random networks sequentially, which is more doable with the efficient computations that have been implemented.
- hygetest.py
  - combined into one HygeCache.py file.
  - Changed lru_cache to cache. No noticible increase in RAM with test data.
  - Switched from computing cumulative distribution function `1 - hypergeom.cdf()` to equivalent survival function `hypergeom.df()` as this will give greater numeric accuracy.
  - double checked and confirmed that even though the arg order is different to `_hyge_single()`, it is equivalent to the previous version, but this ordering made more sense to me.
- HygeCache.py
  - added in functionality that splits the hypergeom tests into chunks per number of workers in a multiprocessing pool. More workers means this part will run faster and it won't increase RAM usage dramatically.

## ComputeStats

- TODO: add in memory tracking for users to identify best configuration of n_workers and n_jobs
- bpmind.py saves wpmsize as (n^2 - n)
  - In WPM chi2 calculations, it does appear that wpmgi is calculated as the full matrix, so then the size would be doubled and this would then be accounted for
- binarizing the network when binary_flag is false TODO:
  - uses a cutoff of 0.2, or ~0.63 pvalue
  - should this be lowered to a threshold of 1.0 which corresponds to a pvalue threshold of 0.1, the same threshold used for chi2 filtering? TODO:
- wpm ranksum calculation
  - original code had "density_wpm" misspelled as "denisty_wpm", so the density_wpm variable had the old calculated densities from WPM chi2 calculations.
  - for WPMs that pass ranksum and are kept have their densities updated from a new wpmsum, but the older densities that passed chi2 but not ranksums are kept in this returned array
  - This isn't an issue and has no consequences as density_wpm is save but not used in calculating fdr or summarizing results
- Noticebly, the number of pathways that pass chi2 and then ranksums aren't too different, only a small reduction most of the time
- changed dense arrays to sparse arrays
- n_workers is used to speed up chi2 and ranksum for BPMs and for SNP perms
- Similar to ComputeInteractions, use n_jobs arg to control how much RAM is used by splitting the problem into smaller parts.
  - Recommend a test run with n_jobs=10, monitor RAM, and lower n_jobs to speed up computation time for the rest of the jobs you want to run.
- Random seed generation is now reproducible. Before, your permutations would differ if you used a different number of workers. Now each worker sets the same seed, but depending on what section of the total SNP perms that a worker will compute, it burns the previous number of SNP perms to get to the same permutation for the number of SNP perms for a given seed. TODO: add in a seed arg to control what random permutations get seeded with. Allows for reproducibility but also flexibility if someone wants to do different seeds for whatever reason.

## ComputeFDR

- exact same output
- Runs in near instant time, down from 4 hours.
- used same p-value threshold for BPM, WPM, and Path (WPM and Path were hard-coded to 0.05)
- Use numpy for filtering instead of pandas
- minPath filter already is used in ComputeStats, so no need to check for this again since they were masked out to be O for bpm_local or 1 for bpm_local_pv. This also means that reading in BPMindFile is not needed anymore.

## Summarize

- no changes to output file
- runs in 30s instead of 5 min

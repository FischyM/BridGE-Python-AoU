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
- use polars for dataframe operations instead of pandas TODO: use pip install Pgenlib

This appears to be what is needed for the python version of BridGE

```bash
# use miniforge to replace future conda enviornments 
conda create -name bridge-aou -c conda-forge python=3.12 matplotlib networkx numpy pandas scipy seaborn cython
```

however, I would add the following:

```bash
conda create -name bridge-aou -c conda-forge python=3.12 matplotlib networkx numpy pandas scipy seaborn cython jupyterlab ipython scikit-learn polars bioconda::pgenlib openpyxl
pip install polars-bio  # might not be needed TODO:
conda env export > updated-environment.yml
# conda env create -f updated-environment.yml
```

TODO: Then, find a consensus between what the updated BridGE environment needs and my PGI environment

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
``plink2 --pfile {inPlinkFile} --hwe 0.000001 0.001 --make-pgen --out {outPlinkFile}``

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

### Imputation

Since AoU has diverse ancestry samples, we fill in any missing variant values (we are not imputing variants that were not genotypes) that were set that way by AoU quality filtering. This involves selecting the statistically phased genomic regions that overlap with out data. This is done in with plink, bcftools, and python.

## Files - Other

- added function definitions where reasonable TODO:
- cyadd.pyx: changed variable type definition to work with updated numpy version
- bridge.py was refactored to use argparse and reusbale functions

## Classes

- merge all separate classes into one python file TODO:

## DataProcess using Datatools

- merge all separate files into one python file? TODO:
- plink2pkl.py

  - This implementation's result matches the older version.
  - Changed to use pgen file format with --export A option in plink2.
  - Will need to change to loading with pgenlib to make it cleaner and so that I don't have to save a large genotype file as a raw text file. TODO:
  - sparseness of the genotype file should be assessed to see whether or not it would be worth saveing as a sparse array. TODO:
- bindataa.py

  - This implementation's result matches the older version.
  - redundantly saves SNPdata class. Instead, run the code in this file whenever a dominant or recessive data type is needed. TODO:
- bpmind.py

  - This implementation's result matches the older version.
  - spmatrix was refactored so that there are no values larger than 1, these checks aren't needed anymore
  - wpmdata size does not divide by two like you would for an n choose k problem where k=2. This may be an error or it could be accounted for later on. This will need to be confirmed. TODO:
- imputesnp.py

  - no longer needed as imputation should be done outside of BridGE, as detailed in my AoU repo. This is to account for the fact that the All of Us data has a diverse ancestry and basic imputation would only work for samples of the same ancestry.
- mapsnp2gene.py

  - This implementation's result matches the older version.
  - adjustments for pgen file and variant ids is renamed since we can't use rsIDs for whole genome SNPs. This change needs to be propagated throughout the code and classes. TODO:
  - replaced lambda filtering for numpy boolean arrays to speed up computations.
  - snp-gene matrix is saved using bools instead of int.
- msigdb2pkl.py

  - This implementation's result matches the older version.
  - jagged csv files are read differently. I keep 3 columns, of which, the gene name column holds a list of genes that are in each pathway.
  - a binary (boolean) matrix is created and used that fills entries array-wise based on genes in each pathway.
  - Optionally, I'm thinking of adding a Jaccard filtering criteria to the pathways. TODO:
- snppathway.py

  - This implementation's result matches the older version.
  - speed improvements with numpy array broadcasting when testing if pathway size is between 10 and 300
  - create a snp to pathway matrix using sparse dot product which speeds up this calculation tremendously
  - checks again now if pathways size is between 10 and 300 for SNPs this time.
  - removing any SNPs not in pathways, and any pathways with no SNPs.

TODO: after confirming the modules after this are correct implementations of the previous BridGE version, recheck that these datatools are also working as intended and match the older version's results.

## ComputeInteraction

- This implementation's result matches the older version.
- matrix_operations_par.py
  - change numpy array data type to float32 instead of float64 to reduce RAM usage. TODO:
  - kept the splitting of jobs implementation, however, I noticed that numpy uses all available CPUs for mat mul calculations. Therefore, instead of running split jobs simultaneously across workers (which would could also increase RAM usage with a large number of SNPs), I split jobs with n_jobs and n_workers are used within each job. This means that we can adjust how big the total SNP-SNP interaction computation is (n_jobs, reduce RAM usage) while still using many workers to run all the hypergeometric tests.
    - n_jobs won't make this module run any faster, but helps to keep RAM usage down if you have a system with that restriction
    - n_workers will reduce the time it takes to run this module
    - This could be advantageous for the VM options given by AoU, such as using the high-cpu VMs.
  - Using sparse arrays for interaction network to save space. Need to convert to save just as coo and convert to csr when needed? TODO:
  - Claude found a way to reduce 12 mat muls to 2, so instead of computing g10/g01/g00/x10/x01/x00 separately for both protective and risk networks, these are derived from row/column sums of g11/x11. This also removes dense intermediate arrays.
  - removed multiprocessing initialization of args and global variables.
  - updated random seed generation of permuted pheno index. Wondering if this will need to be enhanced with ancestry information for AoU? TODO:
  - parallel pool is creatd in bridge.py so that we don't have to keep creating and closing workers, especially if we can use --R=5 to run all of the random networks sequentially, which is more doable with the efficient computations that have been implemented.
- hygetest.py
  - combined with HygeCache.py into one file.
  - increased maxsize lru_cache from 100,000 to 2,000,000. No noticible increase in RAM with test data. Will need to test if I could instead use functools.cache (lighter weight) and see if RAM takes a big hit vs. computation time TODO:
  - Switched from computing cumulative distribution function `1 - hypergeom.cdf()` to equivalent survival function `hypergeom.df()` as this will give greater numeric accuracy.
  - double checked and confirmed that even though the arg order is different to `_hyge_single()`, it is equivalent to the previous version, but this ordering made more sense to me.
- HygeCache.py
  - added in functionality that splits the hypergeom tests into chunks per number of workers in a multiprocessing pool. More workers means this part will run faster and it won't increase RAM usage dramatically.

## ComputeStats

- bpmind.py saves wpmsize as (n^2 - n) TODO:
  - since it's a within pathway there are duplicates within this symmetric matrix and wpmsize instead should be (n^2 - n) / 2.
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

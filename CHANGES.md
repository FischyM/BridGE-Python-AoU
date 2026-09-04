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
- redid all shell scripts to change everything to plink2

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
  - add in a seed arg to control what random permutations get seeded with.
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

# Running BridGE 3 with Slurm

This document provides instructions for running BridGE 3 on a Slurm-managed cluster. It includes example Slurm job scripts and explanations for key steps and parameters.

## Prerequisites

Run the same installation steps as outlined in the README.md file. Then, download and set up example data as outlined in the README.md file.

## Slurm Job Scripts

Additionally, your project directory will need a 'slurm' subdirectory for storing Slurm job scripts and output files.

```bash
cd BridGE-Python-AoU
mkdir -p example/slurm
```

Below are the Slurm job scripts for each step of the BridGE 3 workflow. Do note that you will need to adjust the scripts according to your cluster's configuration, your environment, and the specific requirements of your project. These scripts are not meant to be universal, rather, they provide a complete starting point for you to edit and customize as needed.

Each script uses the following Slurm job specific args, in the following order:

- walltime in hours
- walltime in minutes
- number of CPU cores
- amount of RAM in GB
- node to run on (e.g., msismall)

### 0: Preprocess GWAS data

We recommend preprocessing the data interactively before submitting it as a job, however, you can also submit the preprocessing step using the provided script below. Since this is using the example data, we already know ahead what args work and are needed for the script.
  
BridGE args after Slurm args:

- project directory
- plink file name (without extensions)
- output file name (without extensions)
- prj1000 file name
- population ID file name
- x1 coordinate for outlier removal
- x2 coordinate for outlier removal
- y1 coordinate for outlier removal
- y2 coordinate for outlier removal

```bash
./slurm_scripts/0-preprocess.sh 0 10 8 16 msismall example gwas_subset.hg38 gwas_final ALL.shapeit2_integrated_v1a.GRCh38.20181129.phased.rsid allpopid.txt 0.07 0.12 0.10 0.15
```

### 1: Process data for BridGE

The arg 'nWorkers' will use the same value as the number of CPU cores specified in the Slurm job.

BridGE args after Slurm args:

- project directory
- input file name (without extensions)
- gene list bed file name (glist-hg38-50000.bed)
- pathway/gene-set file name (c2.cp.v2026.1.Hs)
- similarity measure
- jaccard similarity threshold
- overlap similarity threshold

```bash
./slurm_scripts/1-process_data.sh 0 10 8 16 msismall example gwas_final glist-hg38-50000.bed c2.cp.v2026.1.Hs either 0.5 0.5
```

### 2: SNP interactions

ComputeInteractions uses a cache to speed up SNP interaction score computations, however, this is designed to be used within a single job. Running multiple jobs like we are here, each job will maintain its own cache whose impact will be overall minimal on the total computation time. The arg 'nWorkers' will use the same value as the number of CPU cores specified in the Slurm job.

BridGE args after Slurm args:

- project directory
- number of random networks, in addition to the real network
- number of chunks to splits the total work into

```bash
./slurm_scripts/2-interactions.sh 0 30 32 32 msismall example 10 4
```

### 3: Pathway stats

When using a large number of pathways (greater than 2000), it is recommended to increase the number of jobs and the walltime to ensure that the computation completes successfully. For this example, we are using 949 pathways, which runs quite fast and does not require additional adjustments. The arg 'nWorkers' will use the same value as the number of CPU cores specified in the Slurm job.

BridGE args after Slurm args:

- project directory
- number of random networks, in addition to the real network
- number of chunks to splits the total work into
- number of SNP permutations

```bash
./slurm_scripts/3-stats.sh 0 10 16 16 msismall example 10 1 10000
```

### 4: False discovery rate

BridGE args after Slurm args:

- project directory
- number of random networks, in addition to the real network
- p-value threshold

```bash
./slurm_scripts/4-fdr.sh 0 10 8 16 msismall example 10 0.05
```

### 5: Summarize results

BridGE args after Slurm args:

- project directory
- false discovery rate threshold

```bash
./slurm_scripts/5-summarize.sh 0 10 8 16 msismall example 0.25
```

## Alternative Slurm Job Scripts

Above separated each BridGE module to a separate job script. This allows you to run each step independently, verify input and output, and manage resources more efficiently.

Alternatively, you can combine some steps together. This can be useful if you are rerunning with slightly different parameters and want to streamline the process.

### Process data for BridGE

Run the same preprocessing scripts above to prepare the data for BridGE.

```bash
./slurm_scripts/0-preprocess.sh 0 10 8 16 msismall example gwas_subset.hg38 gwas_final ALL.shapeit2_integrated_v1a.GRCh38.20181129.phased.rsid allpopid.txt 0.07 0.12 0.10 0.15

./slurm_scripts/1-process_data.sh 0 10 8 16 msismall example gwas_final glist-hg38-50000.bed c2.cp.v2026.1.Hs either 0.5 0.5
```

### Compute heavy work

This combines steps 2 and 3 into one script, since we can run one network (real or random) sequentially for both SNP interactions and pathway statistics.

BridGE args after Slurm args:

- project directory
- number of random networks, in addition to the real network
- ComputeInteraction's number of chunks to split the total work into
- PathwayStats' number of chunks to split the total work into
- number of SNP permutations

```bash
./slurm_scripts/computations.sh 0 45 32 32 msismall example 10 4 1 10000
```

### Results

This script combines steps 4 and 5 into one, handling the false discovery rate calculation and summarizing the results in a single job. Both are quite quick, but this at least simplifies the workflow.

BridGE args after Slurm args:

- project directory
- number of random networks, in addition to the real network
- p-value threshold
- false discovery rate threshold

```bash
./slurm_scripts/results.sh 0 30 1 32 msismall example 10 0.05 0.25
```

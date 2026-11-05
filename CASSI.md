# CASSI for BridGE

This document provides instructions for running CASSI within the context of BridGE. As CASSI has been well documented, we refer to the original documentation for detailed usage and options (see README.txt and cassi.pdf).

## Preparing data

We suggest to use the preprocessed data that has been prepared for BridGE, as it ensures consistency and compatibility with the CASSI workflow. Please refer to the BridGE documentation for details on how to prepare the data. We will assume the following before we begin:

- the current working directory is the root of the BridGE-Python-AoU project
- the bridge3 python environment is activated
- the 'example' directory has been created
- the example data have been downloaded
- the example data has been preprocessed up to BridGE's DataProcess module

CASSI works on plink1.9 files, so we will first convert our data to the appropriate format using plink2.

```bash
plink2 --pfile example/raw/gwas_final --make-bed --out example/raw/gwas_final_cassi
```

Optionally, you can compile a machine specific CASSI executable

## Running CASSI

Run the real network.

```bash
python cassi_run.py --plinkFile=example-2.0/raw/gwas_final_cassi --GI=lr --pvalue=0.05 --i=0
```

Run a single random network.

```bash
python cassi_run.py --plinkFile=example-2.0/raw/gwas_final_cassi --GI=lr --pvalue=0.05 --i=1
```

Run multiple networks in parallel. The number of networks to run is specified by the `--R` option, and the number of workers is specified by the `--nWorker` option. Note that when R=10, this is 11 networks: one real network and 10 random networks.

```bash
python cassi_run.py --plinkFile=example-2.0/raw/gwas_final_cassi --GI=lr --pvalue=0.05 --R=10 --nWorker=11
```

## Running BridGE with CASSI output

The DataProcess module of BridGE has already been run and we can skip the ComputeInteraction module as we jsut used CASSI to calculate our SNP interactions. The following commands will run the ComputeStats, ComputeFDR, and Summarize modules of BridGE using the CASSI output.

```bash
python bridge.py --projectDir=example-2.0 --module=ComputeStats --model=cassi_LR --nWorker=30 --nJobs=1 --snpPerms=10000 --seed=42 --R=10
python bridge.py --projectDir=example-2.0 --module=ComputeFDR --model=cassi_LR --pvalueCutoff=0.05 --R=10
python bridge.py --projectDir=example-2.0 --module=Summarize --model=cassi_LR --fdrCutoff=0.25
```

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

## Running CASSI

Run a real network.

```bash
./cassi-run.sh --plinkFile=example/raw/gwas_final_cassi --GI=lr --pvalue=0.05 --i=0 --seed=42

cassi -i example/raw/gwas_final_cassi.bed -lr -lr-th 0.05 -max 0 -o example/intermediate/cassi_LR-R0.txt
python -m cassi_ssm example/raw/gwas_final_cassi.bim example/intermediate/cassi_LR-R0.txt lr example/intermediate/snp_interaction_networks-cassi_LR-R0.npz
```

Run a single random network.

```bash
./cassi-run.sh --plinkFile=example/raw/gwas_final_cassi --GI=lr --pvalue=0.05 --i=0 --seed=42

cassi -i example/raw/gwas_final_cassi.bed -lr -lr-th 0.05 -max 0 -o example/intermediate/cassi_LR-R0.txt
python -m cassi_ssm example/raw/gwas_final_cassi.bim example/intermediate/cassi_LR-R0.txt lr example/intermediate/snp_interaction_networks-cassi_LR-R0.npz
```

Run multiple networks.

```bash
./cassi-run.sh --plinkFile=example/raw/gwas_final_cassi --GI=lr --pvalue=0.05 --R=10 --seed=42
```

## Running BridGE with CASSI output



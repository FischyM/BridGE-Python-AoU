# Run Bridging Gene sets with Epistasis (BridGE)

This README assumes that you also have cloned the BridGE-Python-AoU repo that goes along with this repo.

BridGE analysis works best from the command line, so this file will serve as the document to reproduce the subsequent BridGE analysis.

For more information or for troubleshooting, see the BridGE Nature Protocol paper here <https://www.nature.com/articles/s41596-024-00954-8>

## Setting up BridGE

Create the conda environment used in BridGE.

```bash
# starting from the home directory
cd repos/BridGE-Python-AoU/
conda env create -f BridGE-env.yml
conda activate BridGE-env
# install jupyterlab, ipython
```

Set up directories and data

```bash
mkdir -p height_analysis/raw
cp refdata/c2.cp.v7.1.entrez.gmt height_analysis/raw/
cp refdata/c2.cp.v7.1.symbols.gmt height_analysis/raw/
cp refdata/glist-hg38 height_analysis/raw/
mv /home/jupyter/filtered.genes* height_analysis/raw/
# now we will rename the files for ease of use
```

## Running BridGE

Those familiar with BridGE know that there are data processing steps used to ensure the sample data comes from the same ancestry group. AoU has already computed ancestries and the associated probabilities for us. This analysis also differs in that this version will use the wide range of ancestries available to us, so we will not be removing and data outliers based on ancestry. We have also already preprocessed our data in our "prepare_data" directory, meaning that the following scripts are skipped.

- data_checkpopulations.sh
- data_removeoutlier.sh
- preprocessgwas.sh

Now, run the following command which enables BridGE to run. This is required each time a new terminal session is created.

```bash
source setup.sh
```

### Process genotype and reference data

```bash
python bridge.py --projectDir=height_analysis --job=DataProcess --plinkFile=gwas_data_final --geneAnnotation=glist-hg38 --genesets=c2.cp.v7.1
```

### Compute genetic interaction networks

```bash
python bridge.py --projectDir=testdata --job=ComputeInteraction --model=combined --nWorker=30 --R=5
```

### Generate BPM/WPM/PATH statistics

```bash
python bridge.py --projectDir=testdata --job=ComputeStats --model=combined --nWorker=10 --snpPerms=100 --minPath=10 --R=5
```

### Compute false discovery rates

```bash
python bridge.py --projectDir=testdata --job=ComputeFDR --model=combined --pvalueCutoff=0.05 --minPath=10 --samplePerms=5
```

### Summarize and report results

```bash
python bridge.py --projectDir=testdata --job=Summarize --model=combined --fdrcut=0.25 --snpPathFile=snp_pathway_min10_max300.pkl
```

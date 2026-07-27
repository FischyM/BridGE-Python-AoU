# To Do for BridGE-Python-AoU

- update all packages to latest versions (python 3.12)
- implement loading in pgen file directly with plink2-python

## cyadd.pyx

small change to variable type in cyadd.pyx to account for newer Numpy api changes.

## datatools

- merge all datatools files into one python file

## classes

- merge all classes into one python file

## corefuncs

- find what is the slowest and use claude to find faster alternatives (GPU?)
- ComputeInteraction - matrix_operations_par.py
  - snp-to-pathway matrix by my implementation reduces the SNPs used, however, it appears that we compute snp-snp interaction pairs on all the SNPs from the plink file, even if we can't associate the snp to a gene and then to a pathway.

## environment

Figure out what is needed to update the BridGE-env environment without breaking BridGE, OR rather fix what breaks.

this is what seems to be needed in the python version of BridGE

```bash
# use miniforge to replace future conda enviornments 
conda create -name bridge-aou -c conda-forge python=3.12 matplotlib networkx numpy pandas scipy seaborn cython
```

however, I would add the following:

```bash
conda create -name bridge-aou -c conda-forge python=3.12 matplotlib networkx numpy pandas scipy seaborn cython jupyterlab ipython scikit-learn polars bioconda::pgenlib
pip install polars-bio
conda env export > updated-environment.yml
# conda env create -f updated-environment.yml
```

TODO: Then, find a consensus between what the updated BridGE environment needs and my PGI environment

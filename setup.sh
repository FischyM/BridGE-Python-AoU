#!/bin/bash

# seting up permissions
chmod u+x scripts/*.sh
chmod u+x scripts/plink2
chmod u+x cassi/cassi

CURRENTDIR=$(pwd)
export CURRENTDIR
export PYTHONPATH=$CURRENTDIR:$CURRENTDIR/scripts:$CURRENTDIR/src
export PATH=$CURRENTDIR:$PATH
export PATH=$CURRENTDIR/scripts:$PATH
export PATH=$CURRENTDIR/cassi:$PATH
alias plink2='${CURRENTDIR}/scripts/plink2'

if [ -z "$CONDA_PREFIX" ]
then 
    echo "No conda environment activated"
    source activate bridge-aou
else 
    echo "Conda environment is already activated: $CONDA_PREFIX"
fi

python -m py_compile bridge.py

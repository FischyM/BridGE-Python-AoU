#!/bin/bash


# seting up permissions
chmod u+x scripts/*.sh
chmod u+x scripts/plink2
chmod u+x cassi/cassi

# Set up environment variables and paths for the project
CURRENTDIR=$(pwd)
export CURRENTDIR
export PYTHONPATH=$CURRENTDIR:$CURRENTDIR/scripts:$CURRENTDIR/src
export PATH=$CURRENTDIR:$PATH
export PATH=$CURRENTDIR/scripts:$PATH
export PATH=$CURRENTDIR/cassi:$PATH
alias plink2='${CURRENTDIR}/scripts/plink2'

# pycache compiling driver file bridge.py which needs the environment to be activated
if [ -z "$CONDA_PREFIX" ]
then 
    echo "No conda environment activated. Activating that for you if it exists."
    source activate bridge-aou
else 
    echo "Conda environment is already activated: $CONDA_PREFIX"
fi

python -m py_compile bridge.py

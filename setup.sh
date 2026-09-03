#!/bin/bash

# seting up permissions
chmod u+x scripts/*.sh
chmod u+x scripts/plink
chmod u+x scripts/plink2
chmod u+x cassi/cassi

CURRENTDIR=$(pwd)
export CURRENTDIR
export PYTHONPATH=$CURRENTDIR:$CURRENTDIR/scripts:$CURRENTDIR/corefuns
export PATH=$CURRENTDIR:$PATH
export PATH=$CURRENTDIR/scripts:$PATH
export PATH=$CURRENTDIR/cassi:$PATH
alias plink='${CURRENTDIR}/scripts/plink'
alias plink2='${CURRENTDIR}/scripts/plink2'

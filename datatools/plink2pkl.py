import pickle

import pandas as pd

from datatools import imputesnp as isnp
from classes import SNPdataclass as snpc


def assess_sparseness(df):
    """Sparsity as percentage of missing/null values"""
    
    sparsity = df.isnull().sum().sum() / (len(df) * len(df.columns))
    print(f"Sparsity (missing/nulls): {sparsity:.2%}")

    # Count zeros as sparse (common in genomics)
    sparsity = (df == 0).sum().sum() / (len(df) * len(df.columns))
    print(f"Sparsity (zeros): {sparsity:.2%}")

    # Or combine zeros and nulls
    sparsity = ((df == 0) | df.isnull()).sum().sum() / (len(df) * len(df.columns))
    print(f"Sparsity (zeros + missing): {sparsity:.2%}")
    
def plink2pkl(rawFile, pvarFile, psamFile, outputFile):
    """Convert plink .pgen file to pickle file format.
        
    This function extracts all information from the .pgen file and 
    separates the genotype information from the rest. It saves all 
    into a <outputFile>.pkl file.
    
    INPUTS:
    rawFile - plink.raw file
    pvarFile - plink.pvar file that associated with rawFile
    psamFile - plink.psam file that associated with rawFile
    outputFile - name for output pickle file
    
    OUTPUTS:
    <outputFile>.pkl
    The .pkl file uses an SNPdata class with the following fields:
    - rsid: snp names
    - data: genotype data
    - chr: chromosome id
    - loc: physical location
    - pheno: sample's phenotype
    - fid: sample's family id
    - pid: sample id
    - sex: sample sex
    """
    
    # Creating headers for columns reading files into dataframes.
    pvar_header = ['chrom', 'pos', 'id', 'ref', 'alt']
    var_df = pd.read_csv(pvarFile, sep=r"\s+", header=0, names=pvar_header, engine='python')

    psam_header = ['fid', 'iid', 'sex', 'pheno']
    sam_df = pd.read_csv(psamFile, sep=r"\s+", header=0, names=psam_header, engine='python')

    geno_df = pd.read_csv(rawFile, sep=r"\s+", header=0, engine='python')

    # need to flip 0 and 2 counts, since plink's --export A counts the ref alleles
    data = 2 - geno_df[geno_df.columns[6:]]
    assess_sparseness(data)

    # remove ref allele from the end of the rsIDs
    prev_cols = data.columns.tolist()
    new_cols = [col.split('_')[0] for col in prev_cols]
    data.columns = new_cols

    # Structuring data to be saved into pickle format.
    SNPdata = snpc.SNPclass(
        data, 
        var_df.id, var_df.chrom, var_df.pos,
        sam_df.pheno-1, sam_df.fid, sam_df.iid, sam_df.sex,
        )

    # Save data to pickle file.
    with open(outputFile, 'wb') as file:
        pickle.dump(SNPdata, file, protocol=pickle.HIGHEST_PROTOCOL)

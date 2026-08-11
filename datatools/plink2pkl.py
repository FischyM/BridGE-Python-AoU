import pickle

import pandas as pd

from classes import SNPclass


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
    
def plink2pkl(pgenFile, pvarFile, psamFile, outputFile):
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
    # var_header = ['pos', 'id', 'ref', 'alt', 'qual', 'filter', 'info', 'format', 'cm']
    var_df = pd.read_csv(pvarFile, sep=r"\s+", header=0, names=pvar_header, engine='python')

    psam_header = ['fid', 'iid', 'sex', 'pheno']
    # psam_header = ['iid', 'sid', 'pat', 'mat', 'sex', 'pheno']
    sam_df = pd.read_csv(psamFile, sep=r"\s+", header=0, names=psam_header, engine='python')

    # TODO: read this in with Pgenlib
    geno_df = pd.read_csv(pgenFile, sep=r"\s+", header=0, engine='python')

    # TODO: this may not be needed with Pgenlib
    # need to flip 0 and 2 counts, since plink's --export A counts the ref alleles
    data = 2 - geno_df[geno_df.columns[6:]]
    assess_sparseness(data)

    # remove ref allele from the end of the rsIDs
    prev_cols = data.columns.tolist()
    new_cols = [col.split('_')[0] for col in prev_cols]
    data.columns = new_cols

    # Structuring data to be saved into pickle format.
    snp_data = SNPclass(
        data=data, 
        varid=var_df.id,
        chrom=var_df.chrom,
        pos=var_df.pos,
        pheno=sam_df.pheno-1,
        fid=sam_df.fid,
        iid=sam_df.iid,
        sex=sam_df.sex,
        )

    # Save data to pickle file.
    with open(outputFile, 'wb') as file:
        pickle.dump(snp_data, file)

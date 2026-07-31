import pickle

from classes import SNPdata

def bindataa(project_dir, dataFile, expr):
    """Binarize 012 format SNP data based on dominant/recessive assumptions.

    INPUTS:
    project_dir: directory of all the project files
    dataFile - name of the data file. This .mat file consists
        a structure array SNPdata with the following fields:
        - rsid:snp names
        - data:genotype data
        - chr: chromosome id
        - loc: physical location
        - pheno: sample's phenotype
        - fid: sample's family id
        - pid: sample id
        - gender
    expr - flag used to designate dominant ('d'/'D') or recessive ('r'/'R')

    OUTPUTS:
    a pickle file SNPdataA(D or R).pkl
    """
    
    # Reading in SNPdata datafile
    with open(dataFile, "rb") as file:
        snp_data: SNPdata = pickle.load(file)

    # Checking expression flag to proceed as dominant or recessive (D or R).
    if expr == 'r' or expr == 'R':
        # If recessive, set 1s to 0s, 2s to 1s, and set appropriate filename.
        filename = f"{project_dir}/intermediate/SNPdataAR.pkl"
        replace_dict = {1: 0, 2: 1}
        snp_data.data = snp_data.data.replace(replace_dict)
        
    elif expr == 'd' or expr == 'D':
        # If dominant, set 1s to 1s, 2s to 1s, and set appropriate filename.
        filename = f"{project_dir}/intermediate/SNPdataAD.pkl"
        replace_dict = {2: 1}
        snp_data.data = snp_data.data.replace(replace_dict)
        
    else:
        # Default case where expression provided was neither D or R
        print("Provide 'd'/'D' or 'r'/'R' to designate dominant/recessive.")
        return
    
    # Saving updated SNPdata in output pickle file.
    with open(filename, 'wb') as file:
        pickle.dump(snp_data, file)

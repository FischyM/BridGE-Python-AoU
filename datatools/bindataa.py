import pickle


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
    
    # Reading in pickle datafile
    pklin = open(dataFile, "rb")
    SNPdata = pickle.load(pklin)
    pklin.close()

    # Checking expression flag to proceed as dominant or recessive (D or R).
    if expr == 'r' or expr == 'R':
        # If recessive, set 1s to 0s, 2s to 1s, and set appropriate filename.
        filename = f"{project_dir}/intermediate/SNPdataAR.pkl"
        replace_dict = {1: 0, 2: 1}
        SNPdata.data = SNPdata.data.replace(replace_dict)
        
    elif expr == 'd' or expr == 'D':
        # If dominant, set 1s to 1s, 2s to 1s, and set appropriate filename.
        filename = f"{project_dir}/intermediate/SNPdataAD.pkl"
        replace_dict = {2: 1}
        SNPdata.data = SNPdata.data.replace(replace_dict)
        
    else:
        # Default case where expression provided was neither D or R
        print("Provide 'd'/'D' or 'r'/'R' to designate dominant/recessive.")
        return

    # TODO: this redundantly saves SNPdata class, however, it would be easy to simply
    # change the data when either dominant or recessive is needed.
    # Will need to make sure that there aren't other changes to the SNPdata class other than the data
    
    # Saving updated SNPdata in output pickle file.
    final = open(filename, 'wb')
    pickle.dump(SNPdata, final, protocol=pickle.HIGHEST_PROTOCOL)
    final.close()

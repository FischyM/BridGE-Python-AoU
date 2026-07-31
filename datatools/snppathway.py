import pickle

import numpy as np
import pandas as pd
from scipy.sparse import csr_array

from classes import SNPset, SNPdata, GeneSet


def snppathway(dataFile, sgmFile, genesets, minPath, maxPath):
    """Creates a snp-to-pathway mapping a snpset class object with following fields:
        - pathways: List of pathway names
        - spmatrix: Matrix of snp-pathway mapping (Numpy 2d array)
        - geneset: Path to geneset file in .pkl format.

    Args:
        dataFile (str): Path to the file with genotype data in the Pickle format.
        sgmFile (str): SNP to gene mapping file in the Pickle format.
        genesets (str): Gene-set file in pickle format.
        minPath (int): Minimum size for a pathway to be in the mapping.
        maxPath (int): Maximum size for a pathway to be in the mapping.

    Returns:
        str: Path to the output pickle file containing the snp-to-pathway mapping.
    """
    
    # find project directory
    p_dir = dataFile.split('/')
    project_dir = "/".join(p_dir[0:-1])

    # Reading in data files
    with open(dataFile, "rb") as file:
        snp_data: SNPdata = pickle.load(file)
    with open(sgmFile, "rb") as file:
        sgm: pd.DataFrame = pickle.load(file)  # snps are rows, genes are columns
    with open(genesets, "rb") as file:
        geneset: GeneSet = pickle.load(file)

    # find the snps in SNPdata (plink data) that are also in the snp-gene matrix
    # since the snp-gene matrix was created from the plink data, this is simply a sanity check that runs fast
    tmp_ids = np.intersect1d(snp_data.varid, sgm.index)
    ind_ids = sgm.index.isin(tmp_ids)
    tmp_sgm = sgm.loc[ind_ids, :]

    # keep only pathways with total genes less than upper limit and more than lower limit
    ind = (np.sum(geneset.gpmatrix, axis=0) <= maxPath) & (np.sum(geneset.gpmatrix, axis=0) >= minPath)
    tmp_gpm = geneset.gpmatrix.loc[:, ind]

    # keep genes that are in both snp-gene and gene-pathway matrices
    keep_genes = np.intersect1d(tmp_gpm.index, tmp_sgm.columns)
    tmp2_sgm = tmp_sgm.loc[:, keep_genes]
    tmp2_gpm = tmp_gpm.loc[keep_genes, :]

    # make snp-pathway matrix with dot product of sparse arrays (near instant computation)
    sg_sparse = csr_array(tmp2_sgm.to_numpy())
    gp_sparse = csr_array(tmp2_gpm.to_numpy())
    tmp_sgp = sg_sparse.dot(gp_sparse).toarray()

    # after matrix multiplication (dot product) there will be values greater than 1
    # set data type to bool and then back to int
    tmp_sgp = tmp_sgp.astype(bool).astype(int)
    tmp_sgp_df = pd.DataFrame(tmp_sgp,
                                index=pd.Series(tmp2_sgm.index, name='var_id'),
                                columns=pd.Series(tmp2_gpm.columns, name='pathway'))

    # remove pathways with total SNPs more than upper limit and less than lower limit
    ind = (np.sum(tmp_sgp_df, axis=0) <= maxPath) & (np.sum(tmp_sgp_df, axis=0) >= minPath)
    tmp_sgp_df = tmp_sgp_df.loc[:, ind]

    # remove snps (rows) that aren't in a pathway
    ind_rows = (np.sum(tmp_sgp_df, axis=1) == 0)
    remove_rows = tmp_sgp_df.index[ind_rows]
    tmp_sgp_df = tmp_sgp_df.drop(remove_rows, axis=0)

    # remove pathways (columns) that aren't in any snps
    ind_cols = (np.sum(tmp_sgp_df, axis=0) == 0)
    remove_cols = tmp_sgp_df.columns[ind_cols]
    tmp_sgp_df = tmp_sgp_df.drop(remove_cols, axis=1)

    # check again the SNP limit (mostly just for lower bound, but we'll keep in upper bound too)
    ind = (np.sum(tmp_sgp_df, axis=0) <= maxPath) & (np.sum(tmp_sgp_df, axis=0) >= minPath)
    tmp_sgp_df = tmp_sgp_df.loc[:, ind]

    # Preparing data and filename for pickle storage.
    pathways = tmp_sgp_df.sum(axis=0)
    snp_set = SNPset(pathways, tmp_sgp_df, genesets)
    outfilename = f"{project_dir}/snp_pathway_min{minPath}_max{maxPath}.pkl"

    # Saving data to pickle file.
    with open(outfilename, 'wb') as file:
        pickle.dump(snp_set, file)

    # Returning the name of the output file to be used by other modules.
    return outfilename

import pickle
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.sparse import csr_array

from classes import snpsetclass, bpmindclass, genesetclass, SNPclass


def imputesnp(data):
    """Impute SNP data (0,1,2 format)

    This function imputes missing alleles in SNP dataset by replacing them with major-major.

    Args:
        data (data): SNP data matrix (0,1,2 format, -1 is missing value) with rows are samples and columns are SNPs

    Returns:
        _type_: new SNP data matrix
    """
    pd.options.mode.chained_assignment = None
    
    # Iterating over SNPs.
    for column in data:
        # Count occurences of 0s, 1s, and 2s.
        sum0 = (data[column].values == 0).sum()
        sum1 = (data[column].values == 1).sum()
        sum2 = (data[column].values == 2).sum()

        # Replace nans (missing values) with most frequent value. (0, 1, or 2)
        gen = np.argmax([sum0, sum1, sum2])
        data[column] = data[column].replace(to_replace=(np.nan), value=gen)

    # Returns imputed data.
    return data


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
        snp_data = pickle.load(file)

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


def bpmind(snpPathwayFile):
    """Exctracts SNP indices for BPM/WPM sets. Saves a BPMind.pkl file with a bpmindclass class with fields:
        bpm - DataFrame with all BPM data (pathway names, pathway inices, SNPs in pathaways(redundants removed))
        wpm - DataFrame with all WPM data (pathway names, pathway inices, SNPs in pathaways)

    Args:
        snpPathwayFile (str): SNP-pathway mapping file in pickle format (.pkl), containing a matrix
    """

    # find project directory
    p_dir = snpPathwayFile.split('/')
    project_dir = "/".join(p_dir[0:-1])

    # Reading in data files
    with open(snpPathwayFile, "rb") as file:
        snp_set: snpsetclass = pickle.load(file)

    # Retrieving pathways list from snp_set
    pathways = snp_set.pathways
    snpmat = snp_set.spmatrix

    # Finding all possible combinations of pairs for pathway names and sizes.
    combnames = np.array(list(combinations(pathways.index, 2)))

    # Finding nonzero WPM indices
    WPMind = [ np.nonzero(snpmat[column])[0].tolist() for column in snpmat.columns ]
    wpmdata = {
        'pathway': pathways.index,
        'indsize': pathways.values,
        'ind': WPMind,
        'size': ((pathways.to_numpy() * pathways.to_numpy()) - pathways.to_numpy()),
        }
    wpm = pd.DataFrame(wpmdata)

    # Finding BPM indices
    BPMind1, BPMind2, ind1size, ind2size = [], [], [], []
    for i in range(len(snpmat.columns)):
        p1 = snpmat.iloc[:, i].to_numpy()
        
        for j in range(i + 1, len(snpmat.columns)):
            p2 = snpmat.iloc[:, j].to_numpy()
            
            # snps in pathway 1 but not in pathway 2
            d1 = p1 - p2
            ind1 = np.where(d1 == 1)[0].tolist()
            
            # snps in pathway 2 but not in pathway 1
            d2 = p2 - p1
            ind2 = np.where(d2 == 1)[0].tolist()
            
            BPMind1.append(ind1)
            BPMind2.append(ind2)
            
            ind1size.append(len(ind1))
            ind2size.append(len(ind2))

    # Getting between pathway sizes by multiplying combination available pairs.
    if (len(pathways) > 1):
        size = np.array(ind1size) * np.array(ind2size)
        # Orienting bpm/wpm data and converting to dataframes.
        bpmdata = {
            'path1names': combnames[:, 0], 'ind1': BPMind1, 'ind1size': ind1size, 
            'path2names': combnames[:, 1], 'ind2': BPMind2, 'ind2size': ind2size, 
            'size': size,
            }
    else:
        bpmdata = {
            'path1names': [], 'ind1size': [],
            'path2names': [], 'ind2size': [],
            'size': [],
            }
    bpm = pd.DataFrame(bpmdata)

    # Reading bpm and wpm models into bpmind class for pickle storage.
    bpmobj = bpmindclass(bpm, wpm)

    # Saving bpmind data to pickle file.
    with open(f"{project_dir}/BPMind.pkl", 'wb') as file:
        pickle.dump(bpmobj, file)


def mapsnp2gene(pvarFile, geneAnnotation, mappingDistance, option, outfile):
    """Creates snp to gene matrix in the DataFrame format and saves it to a pickle file.

    Args:
        pvarFile (str): path to Plink variant file in .pvar format.
        geneAnnotation (str): path to gene annotation file.
        mappingDistance (int): snp to gene mapping distance.
        option (str): saving mode for snp-gene map.
        outfile (str): file name for saving the results.
    """
    # # PVAR
    # # Creating SNP dataframe from snp annotation file.
    # pvar_header = ['chrom', 'pos', 'var_id', 'ref', 'alt']
    # var_df = pd.read_csv(pvarFile, sep=r"\s+", header=0, names=pvar_header, engine='python')  # has header row
    # var_df['chrom'] = pd.to_numeric(var_df['chrom'])
    # TODO: change this when moving to .pvar file
    # BIM
    # Creating SNP dataframe from snp annotation file.
    snph = ['chrom', 'var_id', 'tmp1', 'pos', 'ref', 'alt']
    var_df = pd.read_csv(pvarFile, sep=r"\s+", names=snph, engine='python')
    var_df['chrom'] = pd.to_numeric(var_df['chrom'])

    # Creating gene dataframe from gene annotation file.
    gene_header = ['chrom', 'geneloc1', 'geneloc2', 'genes']
    gdf = pd.read_csv(geneAnnotation, sep=r"\s+", names=gene_header, engine='python')  # does not have a header
    gdf = gdf[gdf.chrom.apply(lambda x: x.isnumeric())]
    gdf['chrom'] = pd.to_numeric(gdf['chrom'])
    gdf.sort_values(by='chrom', inplace=True)

    # Expanding gene window by subtracting and adding from start and end loci.
    gdf['geneloc1'] = gdf['geneloc1'] - mappingDistance
    gdf['geneloc2'] = gdf['geneloc2'] + mappingDistance

    # Doing an outer join to get all genes and snp listed by chromosome.
    cdf = gdf.merge(var_df, how='outer', on='chrom')

    # keep only snps that are located between start and end loci adjusted by mappingDistance.
    cdf = cdf[(cdf['pos'] >= cdf['geneloc1']) & (cdf['pos'] <= cdf['geneloc2'])]
    
    # Creating list of unique rsids from filtered results.
    snplist = cdf['var_id'].drop_duplicates()

    # Option chosen to save to snplist.
    if (option == 'snplist'):

        # Saving SNPlist to pickle file.
        with open(outfile, 'wb') as final:
            pickle.dump(snplist, final)

    # Option chosen to save to matrix.
    elif (option == 'matrix'):

        # Getting list of unique and genes from filtered results.
        genelist = cdf['genes'].drop_duplicates()

        # Creating dataframe of appropriate size, and setting labels.
        sgm = pd.DataFrame(np.zeros((len(snplist), len(genelist))),
                                index=snplist, columns=genelist, dtype=bool)

        # Setting snp-gene matrix values to true if snp is within gene window.
        for row in cdf.itertuples():
            sgm.loc[row.var_id, row.genes] = True

        # Saving snp-gene matrix to pickle file.
        with open(outfile, 'wb') as final:
            pickle.dump(sgm, final)

    else:
        # Output option not recognized.
        print("Return option error, valid options are 'snplist', or 'matrix'")


def msigdb2pkl(symbolsFile, entrezFile):
    """Convert MsigDB gene set file (.gmt) to pickle file (Python pkl).
        
    Args:
        symbolsFile: MsigDB gene set file using gene symbols (.symbols.gmt).
        entrezFile: MsigDB gene set file using gene entrez ids (.entrez.gmt).
        
    OUTPUTS:
        <symbolsFile>.pkl - This pickle file uses a geneset class with fields:
            geneset.entrezids - gene {symbol: entrezID} lookup dictionary
            geneset.gpmatrix - gene pathway binary dataframe
    """
    
    # load pathway files
    symbols_df = pd.read_csv(symbolsFile, header=None)
    symbols_df = symbols_df[0].str.split('\t', expand=True, n=2)
    symbols_df.columns = ['pathway_names', "url", "gene_names"]
    symbols_df['gene_names'] = symbols_df['gene_names'].str.split('\t')

    entrez_df = pd.read_csv(entrezFile, header=None)
    entrez_df = entrez_df[0].str.split('\t', expand=True, n=2)
    entrez_df.columns = ['pathway_names', "url", "entrez_ids"]
    entrez_df['entrez_ids'] = entrez_df['entrez_ids'].str.split('\t')

    # make gene by pathway binary matrix
    pathway_list = symbols_df['pathway_names'].tolist()
    gene_list = list(set([gene for sublist in symbols_df['gene_names'].tolist() for gene in sublist]))
    gpm = pd.DataFrame(np.zeros((len(gene_list), len(pathway_list))),
                            index=pd.Series(gene_list, name='genes'),
                            columns=pd.Series(pathway_list, name='pathway'),
                            dtype=bool)
    
    # fill out binary matrix
    for pathway in pathway_list:
        pathway_mask: pd.Series = symbols_df['pathway_names'] == pathway
        genes_in_pathway = symbols_df.loc[pathway_mask, 'gene_names'].tolist()[0]
        gpm.loc[genes_in_pathway, pathway] = True
    
    # Creating dictionary for easy lookup of entrezID by symbol.
    symboldict = {}
    for symbol_genes, entrez_ids in zip(symbols_df['gene_names'].tolist(), entrez_df['entrez_ids'].tolist()):
        for symbol, entrez_id in zip(symbol_genes, entrez_ids):
            symboldict[symbol] = int(entrez_id)

    # 6/25/26 MF - confirmed this gene by pathway matrix is correct and matches the original implementation
    # Converting data to pickle storage file with geneset class.
    geneset = genesetclass(symboldict, gpm)
    symbols_pkl_file = symbolsFile.replace(".symbols.gmt", ".pkl")
    symbols_pkl_file = symbols_pkl_file.replace("raw/", "intermediate/")
    with open(symbols_pkl_file, 'wb') as file:
        pickle.dump(geneset, file)

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
        snp_data: SNPclass = pickle.load(file)
    with open(sgmFile, "rb") as file:
        sgm: pd.DataFrame = pickle.load(file)  # snps are rows, genes are columns
    with open(genesets, "rb") as file:
        geneset: genesetclass = pickle.load(file)

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
    snp_set = snpsetclass(pathways, tmp_sgp_df, genesets)
    outfilename = f"{project_dir}/snp_pathway_min{minPath}_max{maxPath}.pkl"

    # Saving data to pickle file.
    with open(outfilename, 'wb') as file:
        pickle.dump(snp_set, file)

    # Returning the name of the output file to be used by other modules.
    return outfilename

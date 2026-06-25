import pickle

import numpy as np
import pandas as pd

from classes import genesetdataclass as gsc


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
    geneset = gsc.genesetclass(symboldict, gpm)
    symbols_pkl_file = symbolsFile.replace(".symbols.gmt", ".pkl")
    symbols_pkl_file = symbols_pkl_file.replace("raw/", "intermediate/")
    final = open(symbols_pkl_file, 'wb')
    pickle.dump(geneset, final, protocol=pickle.HIGHEST_PROTOCOL)
    final.close()

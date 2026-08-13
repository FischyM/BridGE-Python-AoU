import pickle
from itertools import combinations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.sparse import csr_array, coo_array
from pgenlib import PgenReader

from classes import SNPclass, genesetclass, snpgeneclass, snpsetclass, bpmindclass


def plink2pkl(pgen_file, pvar_file, psam_file, output_file):
    """Convert plink files to pickle file format.

    This function extracts all information from the genotype file and 
    separates the genotype information from the rest. It saves all 
    into a <outputFile>.pkl file.
    
    Args:
        pgen_file (str): genotype file (either .pgen or .bed)
        pvar_file (str): variant file (either .pvar or .bim)
        psam_file (str): sample file (either .psam or .fam)
        output_file (str): output file name
    """
    
    # https://www.cog-genomics.org/plink/2.0/formats#pvar
    # this file has a header, so we can read it directly
    # but since it may have comments at the top, we need to skip those lines
    with open(pvar_file) as f:
        skip = sum(1 for line in f if line.startswith("##"))
    variant_df = pd.read_csv(pvar_file, sep="\t", header=0, skiprows=skip)
    
    # https://www.cog-genomics.org/plink/2.0/formats#psam
    # this file has a header, so we can read it directly
    # but since it may have comments at the top, we need to skip those lines
    with open(psam_file) as f:
        skip = sum(1 for line in f if line.startswith("##"))
    sample_df = pd.read_csv(psam_file, sep="\t", header=0, skiprows=skip)

    print(f"Phenotype composition: {np.unique(sample_df.PHENO1, return_counts=True)}")
    print(f"Sex composition: {np.unique(sample_df.SEX, return_counts=True)}")

    # read genotypes with pgenlib
    with PgenReader(pgen_file.encode()) as pgr:
        m = pgr.get_variant_ct()
        n = pgr.get_raw_sample_ct()
        assert m == len(variant_df), f"Variant count mismatch: {m} vs {len(variant_df)}"
        assert n == len(sample_df), f"Sample count mismatch: {n} vs {len(sample_df)}"
        
        # G[i] corresponds to line i of the .pvar, and G[i, j] to line j of the .psam
        # NOTE: once the data is read, missing data is coded as -9
        G = np.empty((m, n), dtype=np.int8)  # variants x samples
        pgr.read_range(0, m, G)
        
    G = G.T  # samples x variants
    # code missing values as NaN for proper handling, which means int8 -> float64
    # G = G.T.astype(np.float64)
    # G[G == -9] = np.nan
    
    print(f"Genotype matrix composition: {np.unique(G, return_counts=True)}")
    print(f"    Percentage of zero values: {(G.size - np.count_nonzero(G)) / G.size:.2%}")
    print(f"    Percentage of missing values: {( np.sum(G == -9) / G.size ):.2%}")

    valid = (G != -9)
    freq  = np.where(valid, G, 0).sum(0) / (2 * valid.sum(0))
    maf = np.minimum(freq, 1 - freq)
    plt.hist(maf, bins=50)
    plt.xlabel("MAF"); plt.ylabel("variants"); plt.show()
    
    # Structuring data to be saved into pickle format.
    snp_data = SNPclass(
        data=G, 
        varid=variant_df.ID,
        chrom=variant_df['#CHROM'],
        pos=variant_df.POS,
        pheno=sample_df.PHENO1 - 1,
        iid=sample_df.IID,
        sex=sample_df.SEX,
        )

    # Save data to pickle file.
    with open(output_file, 'wb') as f:
        pickle.dump(snp_data, f)


def msigdb2pkl(symbols_file, entrez_file, output_file):
    """Convert MsigDB gene set file (.gmt) to pickle file (Python pkl).
        
    Args:
        symbols_file (str): MsigDB gene set file using gene symbols (.symbols.gmt).
        entrez_file (str): MsigDB gene set file using gene entrez ids (.entrez.gmt).
        output_file (str): Output pickle file for the gene set.
    """
    
    # load pathway files
    symbols_df = pd.read_csv(symbols_file, header=None)
    symbols_df = symbols_df[0].str.split('\t', expand=True, n=2)
    symbols_df.columns = ['pathway_names', "url", "gene_names"]
    symbols_df['gene_names'] = symbols_df['gene_names'].str.split('\t')

    entrez_df = pd.read_csv(entrez_file, header=None)
    entrez_df = entrez_df[0].str.split('\t', expand=True, n=2)
    entrez_df.columns = ['pathway_names', "url", "entrez_ids"]
    entrez_df['entrez_ids'] = entrez_df['entrez_ids'].str.split('\t')

    # make gene by pathway binary matrix
    pathway_list = symbols_df['pathway_names'].tolist()
    gene_list = list(set([gene for sublist in symbols_df['gene_names'].tolist() for gene in sublist]))
    gene_pathway_df = pd.DataFrame(np.zeros((len(gene_list), len(pathway_list))),
                            index=pd.Series(gene_list, name='genes'),
                            columns=pd.Series(pathway_list, name='pathway'),
                            dtype=bool)
    
    # fill out binary matrix
    for pathway in pathway_list:
        pathway_mask: pd.Series = symbols_df['pathway_names'] == pathway
        genes_in_pathway = symbols_df.loc[pathway_mask, 'gene_names'].tolist()[0]
        gene_pathway_df.loc[genes_in_pathway, pathway] = True
    
    # Creating dictionary for easy lookup of entrezID by symbol.
    symboldict = {}
    for symbol_genes, entrez_ids in zip(symbols_df['gene_names'].tolist(), entrez_df['entrez_ids'].tolist()):
        for symbol, entrez_id in zip(symbol_genes, entrez_ids):
            symboldict[symbol] = int(entrez_id)

    # Converting data to pickle storage file with geneset class.
    geneset = genesetclass(entrezids=symboldict, gpmatrix=gene_pathway_df)
    with open(output_file, 'wb') as f:
        pickle.dump(geneset, f)


def mapsnp2gene(pvar_file, gene_annotation_file, mapping_distance, snp_gene_pkl):
    """Creates snp to gene matrix in the DataFrame format and saves it to a pickle file.

    Args:
        pvar_file (str): path to Plink variant file in .pvar format.
        gene_annotation (str): path to gene annotation file.
        mapping_distance (int): snp to gene mapping distance.
        snp_gene_pkl (str): file name for saving the results.
    """
    
    # Creating SNP dataframe from snp annotation file.
    with open(pvar_file) as f:
        skip = sum(1 for line in f if line.startswith("##"))
    variant_df = pd.read_csv(pvar_file, sep="\t", header=0, skiprows=skip)
    variant_df['#CHROM'] = pd.to_numeric(variant_df['#CHROM'])

    # Creating gene dataframe from gene annotation file.
    gene_header = ['#CHROM', 'geneloc1', 'geneloc2', 'genes']
    gene_df = pd.read_csv(gene_annotation_file, sep=r"\s+", names=gene_header)
    gene_df = gene_df[gene_df["#CHROM"].apply(lambda x: x.isnumeric())]
    gene_df['#CHROM'] = pd.to_numeric(gene_df['#CHROM'])
    gene_df.sort_values(by='#CHROM', inplace=True)

    # Expanding gene window by subtracting and adding from start and end loci.
    gene_df['geneloc1_expanded'] = gene_df['geneloc1'] - mapping_distance
    gene_df['geneloc2_expanded'] = gene_df['geneloc2'] + mapping_distance

    # replacement to avoid a large outer join of SNPs to genes by chrom and improve performance
    snp_ids_matched = []
    genes_matched = []

    variant_groups = dict(tuple(variant_df.groupby('#CHROM', sort=False)))

    for chrom, gene_sub in gene_df.groupby('#CHROM', sort=False):
        variant_sub = variant_groups.get(chrom)
        if variant_sub is None or gene_sub.empty:
            continue

        # Sort this chromosome's SNPs by position once.
        order = np.argsort(variant_sub['POS'].to_numpy())
        pos_sorted = variant_sub['POS'].to_numpy()[order]
        ids_sorted = variant_sub['ID'].to_numpy()[order]

        starts = gene_sub['geneloc1_expanded'].to_numpy()
        ends = gene_sub['geneloc2_expanded'].to_numpy()
        genes = gene_sub['genes'].to_numpy()

        # Vectorized binary search: for every gene window at once,
        # find the slice of sorted SNPs that falls inside [start, end].
        left = np.searchsorted(pos_sorted, starts, side='left')
        right = np.searchsorted(pos_sorted, ends, side='right')

        for lo, hi, gene in zip(left, right, genes):
            if hi > lo:
                snp_ids_matched.append(ids_sorted[lo:hi])
                genes_matched.append(np.full(hi - lo, gene))

    all_ids = np.concatenate(snp_ids_matched)
    all_genes = np.concatenate(genes_matched)

    # get unique SNP IDs and gene names
    snplist = pd.Series(all_ids).drop_duplicates().reset_index(drop=True)
    genelist = pd.Series(all_genes).drop_duplicates().reset_index(drop=True)

    # use sparse array to quickly create snp to gene mapping matrix
    snp_idx = {v: i for i, v in enumerate(snplist)}
    gene_idx = {v: i for i, v in enumerate(genelist)}
    rows = pd.Series(all_ids).map(snp_idx).to_numpy()
    cols = pd.Series(all_genes).map(gene_idx).to_numpy()
    data = np.ones(len(all_ids), dtype=bool)
    sparse_array = coo_array((data, (rows, cols)), shape=(len(snplist), len(genelist)), dtype=bool).tocsr()
    snp_gene_df = pd.DataFrame.sparse.from_spmatrix(sparse_array, index=snplist, columns=genelist)

    # Saving snp-gene matrix to pickle file.
    snp_gene_class = snpgeneclass(sgmatrix=snp_gene_df, mapping_dist=mapping_distance)
    with open(snp_gene_pkl, 'wb') as f:
        pickle.dump(snp_gene_class, f)


def snppathway(snp_data_pkl, snp_gene_pkl, geneset_pkl, min_path, max_path, output_file):
    """Creates a snp-to-pathway mapping.

    Args:
        snp_data_pkl (str): Path to the file with genotype data in the Pickle format.
        snp_gene_pkl (str): Path to the SNP to gene mapping file in the Pickle format.
        geneset_pkl (str): Path to the gene-set file in pickle format.
        min_path (int): Minimum size for a pathway to be in the mapping.
        max_path (int): Maximum size for a pathway to be in the mapping.
        output_file (str): Path to the output pickle file containing the snp-to-pathway mapping.
    """
    
    # Reading in data files
    with open(snp_data_pkl, "rb") as f:
        snp_data: SNPclass = pickle.load(f)
    with open(snp_gene_pkl, "rb") as f:
        snp_gene_mapping: snpgeneclass = pickle.load(f)
        snp_gene_df = snp_gene_mapping.sgmatrix  # snps are rows, genes are columns
    with open(geneset_pkl, "rb") as f:
        gene_pathway_mapping: genesetclass = pickle.load(f)
        gene_pathway_df = gene_pathway_mapping.gpmatrix  # genes are rows, pathways are columns

    # find the snps in SNPdata (plink data) that are also in the snp-gene matrix
    # since the snp-gene matrix was created from the plink data, this is simply a sanity check that runs fast
    tmp_ids = np.intersect1d(snp_data.varid, snp_gene_df.index)
    ind_ids = snp_gene_df.index.isin(tmp_ids)
    tmp_sgm = snp_gene_df.loc[ind_ids, :]
    
    # keep only pathways with total genes less than upper limit and more than lower limit
    ind = (np.sum(gene_pathway_df, axis=0) <= max_path) & (np.sum(gene_pathway_df, axis=0) >= min_path)
    tmp_gpm = gene_pathway_df.loc[:, ind]

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
                                index=pd.Series(tmp2_sgm.index, name='varid'),
                                columns=pd.Series(tmp2_gpm.columns, name='pathway'))

    # remove pathways with total SNPs more than upper limit and less than lower limit
    ind = (np.sum(tmp_sgp_df, axis=0) <= max_path) & (np.sum(tmp_sgp_df, axis=0) >= min_path)
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
    ind = (np.sum(tmp_sgp_df, axis=0) <= max_path) & (np.sum(tmp_sgp_df, axis=0) >= min_path)
    tmp_sgp_df = tmp_sgp_df.loc[:, ind]

    # Save data to pickle file.
    pathways = tmp_sgp_df.sum(axis=0)
    snp_set = snpsetclass(pathways=pathways, spmatrix=tmp_sgp_df)
    with open(output_file, 'wb') as f:
        pickle.dump(snp_set, f)


def bpmind(snp_pathway_pkl, output_file):
    """Exctracts SNP indices for BPM/WPM sets. Saves a BPMind.pkl file with a bpmindclass class.
    
    Args:
        snp_pathway_pkl (str): Path to the SNP to pathway mapping file in the Pickle format.
        output_file (str): Path to the output pickle file containing the BPM/WPM indices.
    """

    # Reading in data files
    with open(snp_pathway_pkl, "rb") as f:
        snp_set: snpsetclass = pickle.load(f)

    # Retrieving pathways list from snp_set
    pathways = snp_set.pathways
    snpmat = snp_set.spmatrix

    # Finding all possible combinations of pairs for pathway names and sizes.
    combnames = np.array(list(combinations(pathways.index, 2)))

    # Finding WPM indices
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

    # Saving bpmind data to pickle file.
    bpmobj = bpmindclass(bpm=bpm, wpm=wpm)
    with open(output_file, 'wb') as f:
        pickle.dump(bpmobj, f)

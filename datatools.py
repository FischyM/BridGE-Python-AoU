import pickle, sys
from itertools import combinations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.sparse import csr_array, csc_array, coo_array
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

    print(f"    Phenotype composition: {np.unique(sample_df.PHENO1, return_counts=True)}")
    print(f"    Sex composition: {np.unique(sample_df.SEX, return_counts=True)}")

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
    
    print(f"    Genotype matrix composition: {np.unique(G, return_counts=True)}")
    print(f"        number of samples: {n}, number of variants: {m}")
    print(f"        Percentage of zero values: {(G.size - np.count_nonzero(G)) / G.size:.2%}")
    print(f"        Percentage of missing values: {( np.sum(G == -9) / G.size ):.2%}")
    
    print("    Setting any missing values to zero")
    G[G == -9] = 0

    # maf = np.mean(G, axis=0) / 2
    # plt.hist(maf, bins=50)
    # plt.xlabel("MAF")
    # plt.ylabel("variants")
    # plt.savefig(output_file.replace('.pkl', '.maf.png'))
    
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


def msigdb2pkl(symbols_file, entrez_file, sim_measure, jaccard_cutoff, overlap_cutoff, min_size, max_size, output_file):
    """Convert MsigDB gene set file (.gmt) to pickle file (Python pkl).
        
    Args:
        symbols_file (str): MsigDB gene set file using gene symbols (.symbols.gmt).
        entrez_file (str): MsigDB gene set file using gene entrez ids (.entrez.gmt).        
        sim_measure (str): The similarity measure to use ("jaccard", "overlap", or "either").
        jaccard_cutoff (float): Cutoff for the jaccard similarity measure.
        overlap_cutoff (float): Cutoff for the overlap measure.
        min_size (int): Minimum size of gene sets to include.
        max_size (int): Maximum size of gene sets to include.
        output_file (str): Output pickle file for the gene set.
    """
    
    def jaccard_sim(a, b, inter):
        return inter / len(a | b) if inter else 0.0

    def overlap_sim(a, b, inter):
        return inter / min(len(a), len(b)) if inter else 0.0

    def similarity(a, b, sim_measure):
        inter = len(a & b)
        if sim_measure == "jaccard":
            return jaccard_sim(a, b, inter)
        return overlap_sim(a, b, inter)  # overlap coefficient

    def too_similar(gs, gi, sim_measure, jaccard_cutoff, overlap_cutoff):
        """True if gi should be dropped for being too similar to already-kept gs."""
        if sim_measure == "either":
            inter = len(gs & gi)
            # OR on the drop condition: dropping requires only one measure to flag
            # redundancy, not agreement from both. See module docstring.
            return jaccard_sim(gs, gi, inter) >= jaccard_cutoff or overlap_sim(gs, gi, inter) >= overlap_cutoff
        return similarity(gs, gi, sim_measure) >= jaccard_cutoff
    
    if sim_measure == "jaccard":
        print(f"    rule: drop a pathway if jaccard >= {jaccard_cutoff} vs. any already-kept pathway")
        tmp_str = f" using jaccard >= {jaccard_cutoff}"
    elif sim_measure == "overlap":
        print(f"    rule: drop a pathway if overlap >= {overlap_cutoff} vs. any already-kept pathway")
        tmp_str = f" using overlap >= {overlap_cutoff}"
    else:
        print(f"    rule: drop a pathway if jaccard >= {jaccard_cutoff} OR overlap >= {overlap_cutoff} vs. any already-kept pathway")
        tmp_str = f" using jaccard >= {jaccard_cutoff} OR overlap >= {overlap_cutoff}"
        
    # load pathway files
    symbols_df = pd.read_csv(symbols_file, header=None)
    symbols_df = symbols_df[0].str.split('\t', expand=True, n=2)
    symbols_df.columns = ['pathway_names', "url", "gene_names"]
    symbols_df['gene_names'] = symbols_df['gene_names'].str.split('\t')

    entrez_df = pd.read_csv(entrez_file, header=None)
    entrez_df = entrez_df[0].str.split('\t', expand=True, n=2)
    entrez_df.columns = ['pathway_names', "url", "entrez_ids"]
    entrez_df['entrez_ids'] = entrez_df['entrez_ids'].str.split('\t')
    
    # filter pathways based in size and similarity
    if not symbols_df['pathway_names'].equals(entrez_df['pathway_names']):
        sys.exit(
            f"Error: pathway names/order in {symbols_file} and {entrez_file} do not match. "
            "The two files must describe the same gene sets in the same row order."
        )
        
    # size filter, applied before redundancy filtering (see module docstring).
    candidates = [row.Index for row in symbols_df.itertuples() if min_size <= len(row.gene_names) <= max_size]
    print(f"    size filter [{min_size}, {max_size}]: {len(candidates)} / {len(symbols_df)} pathways")

    # greedy redundancy filter, smallest pathway first.
    ordered = sorted(candidates, key=lambda i: len(symbols_df.loc[i, 'gene_names']))
    keep_pathway_inds = []  # indices into the original (unsorted) arrays, in size-ascending order
    keep_gene_sets = []
    for i in ordered:
        query_gene_set = set(symbols_df.loc[i, 'gene_names'])
        keep = True
        for gene_set in keep_gene_sets:
            if too_similar(gene_set, query_gene_set, sim_measure, jaccard_cutoff, overlap_cutoff):
                keep = False
                break
        if keep:
            keep_pathway_inds.append(i)
            keep_gene_sets.append(query_gene_set)
    print(f"    kept {len(keep_pathway_inds)} / {len(candidates)} size-filtered pathways{tmp_str}")
    
    # filter the original dataframes to only include the kept pathways
    symbols_df = symbols_df.loc[keep_pathway_inds].reset_index(drop=True)
    entrez_df = entrez_df.loc[keep_pathway_inds].reset_index(drop=True)
    
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
    
    # save filtered dataframes to csv for inspection
    with open(symbols_file.replace(".symbols.gmt", ".filtered.symbols.gmt"), 'w') as f:
        for _, row in symbols_df.iterrows():
            f.write(f"{row['pathway_names']}\t{row['url']}\t" + "\t".join(row['gene_names']) + "\n")
    with open(entrez_file.replace(".entrez.gmt", ".filtered.entrez.gmt"), 'w') as f:
        for _, row in entrez_df.iterrows():
            f.write(f"{row['pathway_names']}\t{row['url']}\t" + "\t".join(map(str, row['entrez_ids'])) + "\n")


def mapsnp2gene(pvar_file, gene_annotation_file, mapping_distance, output_file):
    """Creates snp to gene matrix in the DataFrame format and saves it to a pickle file.

    Args:
        pvar_file (str): path to Plink variant file in .pvar format.
        gene_annotation (str): path to gene annotation file.
        mapping_distance (int): snp to gene mapping distance.
        output_file (str): file name for saving the results.
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
    data = np.ones(len(all_ids))
    sparse_array = coo_array((data, (rows, cols)), shape=(len(snplist), len(genelist)))
    snp_gene_df = pd.DataFrame(sparse_array.toarray(), index=snplist, columns=genelist, dtype=bool)

    # Saving snp-gene matrix to pickle file.
    snp_gene_class = snpgeneclass(sgmatrix=snp_gene_df, mapping_dist=mapping_distance)
    with open(output_file, 'wb') as f:
        pickle.dump(snp_gene_class, f)
    

def snppathway(project_dir, min_path, max_path, output_file):
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
    with open(f"{project_dir}/intermediate/snp_data.pkl", "rb") as f:
        snp_data: SNPclass = pickle.load(f)
        
    with open(f"{project_dir}/intermediate/snp_gene_mapping.pkl", "rb") as f:
        snp_gene_mapping: snpgeneclass = pickle.load(f)
        snp_gene_df = snp_gene_mapping.sgmatrix.astype(np.int64)
        # snps are rows, genes are columns, bool values converted to int64 for matrix multiplication
        
    with open(f"{project_dir}/intermediate/gene_pathway_mapping.pkl", "rb") as f:
        gene_pathway_mapping: genesetclass = pickle.load(f)
        gene_pathway_df = gene_pathway_mapping.gpmatrix.astype(np.int64)
        # genes are rows, pathways are columns, bool values converted to int64 for matrix multiplication

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
    tmp_sgp_df = pd.DataFrame(tmp_sgp.astype(bool).astype(int),
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
    snp_set = snpsetclass(pathways=pathways, spmatrix=tmp_sgp_df, min_path=min_path, max_path=max_path)
    with open(output_file, 'wb') as f:
        pickle.dump(snp_set, f)

def bpmind(project_dir, min_path, output_file):
    """Extracts SNP indices for BPM/WPM sets.

    Option C: computes the surviving pathway pairs and their sizes with Option B's
    sparse-matmul core, then expands them into the original bpm format in a second
    pass. Output is byte-identical to the original, including index labels.

    The phase split means the exact memory cost of the ind1/ind2 columns is known
    before a single list is allocated.
    """

    # Reading in data files
    with open(f"{project_dir}/intermediate/snp_pathway_mapping.pkl", "rb") as f:
        snp_set: snpsetclass = pickle.load(f)

    # Retrieving pathways list from snp_set
    pathways = snp_set.pathways
    snpmat = snp_set.spmatrix

    n_snp, n_path = snpmat.shape

    # Single conversion of the dense frame to a sparse column matrix.
    S = csc_array((snpmat.to_numpy() != 0).astype(np.int32))
    S.sort_indices()

    # Finding WPM indices -- the per-column SNP row indices are the CSC structure.
    WPMind = [S.indices[S.indptr[j]:S.indptr[j + 1]] for j in range(n_path)]
    wpmdata = {
        'pathway': pathways.index,
        'indsize': pathways.values,
        'ind': [a.tolist() for a in WPMind],
        'size': ((pathways.to_numpy() * pathways.to_numpy()) - pathways.to_numpy()),
        }
    wpm = pd.DataFrame(wpmdata)

    # Pathway sizes as counted in the matrix, not pathways.values.
    colsum = np.diff(S.indptr).astype(np.int64)

    # All pairwise intersection counts in one sparse matmul: C[i, j] = |P_i & P_j|.
    C = (S.T @ S).tocsr()
    C.sort_indices()

    # Phase 1: surviving pairs and their sizes. No per-pair Python iteration.
    path1_blocks, path2_blocks, s1_blocks, s2_blocks = [], [], [], []

    for i in range(n_path - 1):
        j = np.arange(i + 1, n_path, dtype=np.int32)

        lo, hi = C.indptr[i], C.indptr[i + 1]
        cols = C.indices[lo:hi]
        k0 = np.searchsorted(cols, i + 1)
        ov = np.zeros(j.size, dtype=np.int32)
        ov[cols[k0:] - (i + 1)] = C.data[lo + k0:hi]

        ind1size = colsum[i] - ov
        ind2size = colsum[j] - ov

        # filter out pathways that are too small after removing SNPs that are in
        # both pathways of a BPM
        keep = (ind1size >= min_path) & (ind2size >= min_path)
        n_keep = int(keep.sum())
        if n_keep == 0:
            continue

        path1_blocks.append(np.full(n_keep, i, dtype=np.int32))
        path2_blocks.append(j[keep])
        s1_blocks.append(ind1size[keep].astype(np.int32))
        s2_blocks.append(ind2size[keep].astype(np.int32))

    orig_size = n_path * (n_path - 1) // 2

    if path1_blocks:
        pairs = {
            'path1': np.concatenate(path1_blocks),
            'path2': np.concatenate(path2_blocks),
            'ind1size': np.concatenate(s1_blocks),
            'ind2size': np.concatenate(s2_blocks),
            }
    else:
        empty = np.array([], dtype=np.int32)
        pairs = {'path1': empty, 'path2': empty, 'ind1size': empty, 'ind2size': empty}

    n_bpm = len(pairs['path1'])
    print(f"    Total number of WPMs: {len(wpm)}")
    print(f"    Total number of BPMs: {orig_size}")
    print(f"    Total BPMs filtered with min_path={min_path}: {orig_size - n_bpm}")
    # print(f"    ind1/ind2 columns will need {materialized_bytes(pairs) / 2**30:.1f} GB")

    # Phase 2: expand into the original bpm format.
    bpm = materialize_bpm(pairs, wpm, n_path, n_snp)

    # Saving bpmind data to pickle file.
    bpmobj = bpmindclass(bpm=bpm, wpm=wpm)
    with open(output_file, 'wb') as f:
        pickle.dump(bpmobj, f)

def materialized_bytes(pairs):
    """Exact Python memory the ind1/ind2 columns will occupy, before building them.

    A list of n ints costs 56 bytes for the list plus 8 (slot) + 28 (int object)
    per element. Only computable because phase 1 already produced the sizes.
    """
    n_ints = int(pairs['ind1size'].sum()) + int(pairs['ind2size'].sum())
    return 2 * 56 * len(pairs['path1']) + 36 * n_ints

def materialize_bpm(pairs, wpm, n_path, n_snp, start=0, stop=None):
    """Expand rows [start, stop) of the phase-1 pair table into the original format.

    Rows arrive grouped by path1, so the pathway-1 membership mask is set once per
    pathway rather than once per pair. Concatenating the frames from consecutive
    [start, stop) slices reproduces the full frame exactly.
    """
    stop = len(pairs['path1']) if stop is None else stop

    i_col = pairs['path1'][start:stop].astype(np.int64)
    j_col = pairs['path2'][start:stop].astype(np.int64)
    ind1size = pairs['ind1size'][start:stop].astype(np.int64)
    ind2size = pairs['ind2size'][start:stop].astype(np.int64)

    names = wpm['pathway'].to_numpy()
    ind_arrays = [np.asarray(a, dtype=np.int64) for a in wpm['ind']]

    BPMind1, BPMind2 = [], []
    mask_i = np.zeros(n_snp, dtype=bool)
    mask_j = np.zeros(n_snp, dtype=bool)

    if len(i_col):
        starts = np.flatnonzero(np.r_[True, i_col[1:] != i_col[:-1]])
        ends = np.r_[starts[1:], len(i_col)]
    else:
        starts = ends = np.array([], dtype=np.int64)

    for lo, hi in zip(starts, ends):
        p1 = ind_arrays[i_col[lo]]
        mask_i[p1] = True

        for k in range(lo, hi):
            p2 = ind_arrays[j_col[k]]
            mask_j[p2] = True

            # snps in pathway 1 but not in pathway 2
            BPMind1.append(p1[~mask_j[p1]].tolist())
            # snps in pathway 2 but not in pathway 1
            BPMind2.append(p2[~mask_i[p2]].tolist())

            mask_j[p2] = False

        mask_i[p1] = False

    # Row-major position in the full upper triangle, reproducing the index labels
    # the original's boolean filter leaves behind.
    pos = i_col * (n_path - 1) - (i_col * (i_col - 1)) // 2 + (j_col - i_col - 1)

    # Orienting bpm data and converting to a dataframe.
    bpmdata = {
        'path1names': names[i_col], 'ind1': BPMind1, 'ind1size': ind1size,
        'path2names': names[j_col], 'ind2': BPMind2, 'ind2size': ind2size,
        'size': ind1size * ind2size,
        }
    return pd.DataFrame(bpmdata, index=pos)

def bpmind_old(project_dir, min_path, output_file):
    """Exctracts SNP indices for BPM/WPM sets."""

    # Reading in data files
    with open(f"{project_dir}/intermediate/snp_pathway_mapping.pkl", "rb") as f:
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
    size = np.array(ind1size) * np.array(ind2size)
    
    # Orienting bpm/wpm data and converting to dataframes.
    bpmdata = {
        'path1names': combnames[:, 0], 'ind1': BPMind1, 'ind1size': ind1size, 
        'path2names': combnames[:, 1], 'ind2': BPMind2, 'ind2size': ind2size, 
        'size': size,
        }
    bpm = pd.DataFrame(bpmdata)
    orig_size = len(bpm)
    
    # filter out pathways that are too small after removing SNPs that are in both pathways of a BPM
    bpm = bpm[(bpm['ind1size'] >= min_path) & (bpm['ind2size'] >= min_path)]
    print(f"    Total number of WPMs: {len(wpm)}")
    print(f"    Total number of BPMs: {orig_size}")
    print(f"    Total BPMs filtered with min_path={min_path}: {orig_size - len(bpm)}")

    # Saving bpmind data to pickle file.
    bpmobj = bpmindclass(bpm=bpm, wpm=wpm)
    with open(output_file, 'wb') as f:
        pickle.dump(bpmobj, f)


def bpmind_optimized(project_dir, min_path, output_file):
    """Extracts SNP indices for BPM/WPM sets.

    Option B: stores per-pathway SNP indices once (in wpm) and only pathway-pair
    positions plus sizes in bpm. ind1/ind2 are reconstructed on demand via
    bpm_pair_indices(). Memory drops from O(n_path^2 * mean_pathway_size) to
    O(n_path^2) small integers, and the per-pair Python loop disappears entirely.
    
    NOTE: there could be a tangible speedup by implementing this logic in genstats_perm.py
    if someone wanted to try speeding that module up.
    """

    # Reading in data files
    with open(f"{project_dir}/intermediate/snp_pathway_mapping.pkl", "rb") as f:
        snp_set: snpsetclass = pickle.load(f)

    # Retrieving pathways list from snp_set
    pathways = snp_set.pathways
    snpmat = snp_set.spmatrix

    n_path = snpmat.shape[1]

    # Single conversion of the dense frame to a sparse column matrix.
    S = csc_array((snpmat.to_numpy() != 0).astype(np.int32))
    S.sort_indices()

    # Finding WPM indices -- these are now the only stored SNP index lists.
    WPMind = [S.indices[S.indptr[j]:S.indptr[j + 1]] for j in range(n_path)]
    wpmdata = {
        'pathway': pathways.index,
        'indsize': pathways.values,
        'ind': [a.tolist() for a in WPMind],
        'size': ((pathways.to_numpy() * pathways.to_numpy()) - pathways.to_numpy()),
        }
    wpm = pd.DataFrame(wpmdata)

    # Pathway sizes as counted in the matrix, not pathways.values.
    colsum = np.diff(S.indptr).astype(np.int64)

    # All pairwise intersection counts in one sparse matmul: C[i, j] = |P_i & P_j|.
    C = (S.T @ S).tocsr()
    C.sort_indices()

    # Finding BPM pairs and sizes. No per-pair Python iteration.
    path1_blocks, path2_blocks, s1_blocks, s2_blocks = [], [], [], []

    for i in range(n_path - 1):
        j = np.arange(i + 1, n_path, dtype=np.int32)

        lo, hi = C.indptr[i], C.indptr[i + 1]
        cols = C.indices[lo:hi]
        k0 = np.searchsorted(cols, i + 1)
        ov = np.zeros(j.size, dtype=np.int32)
        ov[cols[k0:] - (i + 1)] = C.data[lo + k0:hi]

        ind1size = colsum[i] - ov
        ind2size = colsum[j] - ov

        # filter out pathways that are too small after removing SNPs that are in
        # both pathways of a BPM
        keep = (ind1size >= min_path) & (ind2size >= min_path)
        n_keep = int(keep.sum())
        if n_keep == 0:
            continue

        path1_blocks.append(np.full(n_keep, i, dtype=np.int32))
        path2_blocks.append(j[keep])
        s1_blocks.append(ind1size[keep].astype(np.int32))
        s2_blocks.append(ind2size[keep].astype(np.int32))

    orig_size = n_path * (n_path - 1) // 2

    if path1_blocks:
        path1 = np.concatenate(path1_blocks)
        path2 = np.concatenate(path2_blocks)
        ind1size = np.concatenate(s1_blocks)
        ind2size = np.concatenate(s2_blocks)
    else:
        path1 = path2 = np.array([], dtype=np.int32)
        ind1size = ind2size = np.array([], dtype=np.int32)

    # Orienting bpm/wpm data and converting to dataframes. path1/path2 are
    # positional indices into wpm; names come from wpm['pathway'].
    bpmdata = {
        'path1': path1, 'ind1size': ind1size,
        'path2': path2, 'ind2size': ind2size,
        'size': ind1size.astype(np.int64) * ind2size.astype(np.int64),
        }
    bpm = pd.DataFrame(bpmdata)

    print(f"Total number of BPMs: {orig_size}")
    print(f"Total BPMs filtered with min_path={min_path}: {orig_size - len(bpm)}")

    # Saving bpmind data to pickle file.
    bpmobj = bpmindclass(bpm=bpm, wpm=wpm)
    with open(output_file, 'wb') as f:
        pickle.dump(bpmobj, f)


# ---------------------------------------------------------------------------
# Downstream accessors. These replace direct reads of bpm['ind1'] / bpm['ind2'].
# ---------------------------------------------------------------------------

def bpm_pair_indices(bpm, wpm, row):
    """Reconstruct (ind1, ind2) for one BPM row as sorted int arrays.

    O(mean_pathway_size) per call. Equivalent to the original
    bpm['ind1'].iloc[row] / bpm['ind2'].iloc[row].
    """
    # dtype is pinned because np.asarray([]) returns float64, which silently
    # yields float index arrays for empty pathways.
    p1 = np.asarray(wpm['ind'].iat[int(bpm['path1'].iat[row])], dtype=np.int64)
    p2 = np.asarray(wpm['ind'].iat[int(bpm['path2'].iat[row])], dtype=np.int64)
    ind1 = np.setdiff1d(p1, p2, assume_unique=True)
    ind2 = np.setdiff1d(p2, p1, assume_unique=True)
    return ind1, ind2


def bpm_pair_names(bpm, wpm):
    """Recover the original path1names / path2names columns as numpy arrays."""
    names = wpm['pathway'].to_numpy()
    return names[bpm['path1'].to_numpy()], names[bpm['path2'].to_numpy()]


def bpm_indices_batched(bpm, wpm, rows, n_snp):
    """Reconstruct ind1/ind2 for many rows, reusing one scratch mask.

    Faster than repeated bpm_pair_indices when iterating a block of BPMs, which
    is the access pattern in genstats_perm.py. Returns two lists of int arrays.
    """
    # dtype is pinned because np.asarray([]) returns float64, which cannot be
    # used to index the scratch mask for an empty pathway.
    ind_arrays = [np.asarray(a, dtype=np.int64) for a in wpm['ind']]
    p1_col = bpm['path1'].to_numpy()
    p2_col = bpm['path2'].to_numpy()

    mask = np.zeros(n_snp, dtype=bool)
    out1, out2 = [], []

    for r in rows:
        p1 = ind_arrays[p1_col[r]]
        p2 = ind_arrays[p2_col[r]]

        mask[p2] = True
        out1.append(p1[~mask[p1]])
        mask[p2] = False

        mask[p1] = True
        out2.append(p2[~mask[p2]])
        mask[p1] = False

    return out1, out2

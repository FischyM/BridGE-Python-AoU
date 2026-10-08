import sys
from itertools import combinations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from scipy.sparse import csr_array, coo_array
from pgenlib import PgenReader


def imputesnp(data: np.ndarray) -> np.ndarray:
    # Count 0s, 1s, and 2s per SNP (column) -> shape (3, n_snps)
    counts = np.stack([(data == g).sum(axis=0) for g in (0, 1, 2)])

    # Most frequent genotype per SNP (ties go to the lower value, as before)
    mode = counts.argmax(axis=0)

    # Replace missing values in each column with that column's mode
    missing = data == -9
    data[missing] = mode[np.nonzero(missing)[1]]
    return data

def convert_plink(pgen_file, pvar_file, psam_file, output_file, compressed=False):
    """Convert plink files to numpy's .npz file format.

    This function extracts all information from the genotype file and 
    separates the genotype information from the rest. It saves all 
    into a <outputFile>.npz file.
    
    Args:
        pgen_file (str): genotype file (.pgen)
        pvar_file (str): variant file (.pvar)
        psam_file (str): sample file (.psam)
        output_file (str): output file name, ending in .npz
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
        G = np.empty((m, n), dtype=np.int8)  # variants x samples
        pgr.read_range(0, m, G)
        
    G = G.T  # samples x variants
    
    print(f"    Genotype matrix composition: {np.unique(G, return_counts=True)}")
    print(f"        number of samples: {n}, number of variants: {m}")
    print(f"        Percentage of zero values: {(G.size - np.count_nonzero(G)) / G.size:.2%}")
    print(f"        Percentage of missing values: {( np.sum(G == -9) / G.size ):.2%}")
    
    # TODO: decide what to do with missing values: set to zero or impute with mode
    # print("    Setting any missing values to zero")
    # G[G == -9] = 0
    print("    Setting any missing values to the most frequent value (mode) for that SNP")
    G = imputesnp(G)

    maf = np.mean(G, axis=0) / 2
    plt.hist(maf, bins=50)
    plt.xlabel("MAF")
    plt.ylabel("variants")
    plt.savefig(output_file.replace('.npz', '.maf.png'))
    
    # compressed takes longer, but saves space. Consider using compressed for large datasets.
    if compressed:
        np.savez_compressed(
            file=output_file, allow_pickle=False,
            data=G,
            varid=variant_df.ID.to_numpy().astype(str),
            chrom=variant_df['#CHROM'],
            pos=variant_df.POS,
            pheno=sample_df.PHENO1 - 1,
            iid=sample_df.IID,
            sex=sample_df.SEX,
            )
    else:
        np.savez(
            file=output_file, allow_pickle=False,
            data=G,
            varid=variant_df.ID.to_numpy().astype(str),
            chrom=variant_df['#CHROM'],
            pos=variant_df.POS,
            pheno=sample_df.PHENO1 - 1,
            iid=sample_df.IID,
            sex=sample_df.SEX,
            )


def gene2pathway(symbols_file, entrez_file, sim_measure, jaccard_cutoff, overlap_cutoff, min_size, max_size, output_file):
    """Convert MSigDB gene set (pathway) files to a parquet file after filtering for 
    pathway size and optional similarity.
        
    Args:
        symbols_file (str): MsigDB gene set file using gene symbols (.symbols.gmt).
        entrez_file (str): MsigDB gene set file using gene entrez ids (.entrez.gmt).        
        sim_measure (str): The similarity measure to use ("jaccard", "overlap", or "either").
        jaccard_cutoff (float): Cutoff for the jaccard similarity measure.
        overlap_cutoff (float): Cutoff for the overlap measure.
        min_size (int): Minimum size of gene sets to include.
        max_size (int): Maximum size of gene sets to include.
        output_file (str): Output parquet file for the gene set.
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
    elif sim_measure == "either":
        print(f"    rule: drop a pathway if jaccard >= {jaccard_cutoff} OR overlap >= {overlap_cutoff} vs. any already-kept pathway")
        tmp_str = f" using jaccard >= {jaccard_cutoff} OR overlap >= {overlap_cutoff}"
    else:
        print(f"    rule: no similarity measure applied (options: jaccard, overlap, either, or none)")
        tmp_str = ""
        
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
    
    # greedy redundancy filter, smallest pathway first
    ordered = sorted(candidates, key=lambda i: len(symbols_df.loc[i, 'gene_names']))
    if sim_measure in ('jaccard', 'overlap', 'either'):
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
        
    else:
        symbols_df = symbols_df.loc[ordered].reset_index(drop=True)
        entrez_df = entrez_df.loc[ordered].reset_index(drop=True)
        print(f"    kept {len(ordered)} / {len(candidates)} size-filtered pathways: no similarity measure applied")

    
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
    
    # # Creating dictionary for easy lookup of entrezID by symbol.
    # symboldict = {}
    # for symbol_genes, entrez_ids in zip(symbols_df['gene_names'].tolist(), entrez_df['entrez_ids'].tolist()):
    #     for symbol, entrez_id in zip(symbol_genes, entrez_ids):
    #         symboldict[symbol] = int(entrez_id)

    print(f"    {len(gene_pathway_df)} genes, {len(gene_pathway_df.columns)} pathways")
    gene_pathway_df.to_parquet(output_file, index=True, compression='zstd')
    # gene_pathway_df = pd.read_parquet(output_file)
    
    # save filtered dataframes to csv for inspection
    with open(symbols_file.replace(".symbols.gmt", ".filtered.symbols.gmt"), 'w') as f:
        for _, row in symbols_df.iterrows():
            f.write(f"{row['pathway_names']}\t{row['url']}\t" + "\t".join(row['gene_names']) + "\n")
    with open(entrez_file.replace(".entrez.gmt", ".filtered.entrez.gmt"), 'w') as f:
        for _, row in entrez_df.iterrows():
            f.write(f"{row['pathway_names']}\t{row['url']}\t" + "\t".join(map(str, row['entrez_ids'])) + "\n")


def snp2gene(pvar_file, gene_annotation_file, output_file):
    """Creates snp to gene matrix in the DataFrame format and saves it to a parquet file.

    Args:
        pvar_file (str): path to Plink variant file in .pvar format.
        gene_annotation_file (str): path to gene annotation file.
        output_file (str): file name for saving the results.
    """
    
    # Creating SNP dataframe from snp annotation file.
    with open(pvar_file) as f:
        skip = sum(1 for line in f if line.startswith("##"))
    variant_df = pd.read_csv(pvar_file, sep="\t", header=0, skiprows=skip)
    variant_df['#CHROM'] = pd.to_numeric(variant_df['#CHROM'])

    # Creating gene dataframe from gene annotation file.
    gene_header = ['#CHROM', 'geneloc1', 'geneloc2', 'gene']
    gene_df = pd.read_csv(gene_annotation_file, sep=r"\s+", names=gene_header)
    gene_df = gene_df[gene_df["#CHROM"].apply(lambda x: x.isnumeric())]
    gene_df['#CHROM'] = pd.to_numeric(gene_df['#CHROM'])
    gene_df.sort_values(by='#CHROM', inplace=True)

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

        starts = gene_sub['geneloc1'].to_numpy()
        ends = gene_sub['geneloc2'].to_numpy()
        genes = gene_sub['gene'].to_numpy()

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

    # keep every input SNP (mapped or not) as a row; genes are those with >=1 SNP
    snplist = variant_df['ID'].reset_index(drop=True)
    genelist = pd.Series(all_genes).drop_duplicates().reset_index(drop=True)
    
    # use sparse array to quickly create snp to gene mapping matrix
    snp_idx = {v: i for i, v in enumerate(snplist)}
    gene_idx = {v: i for i, v in enumerate(genelist)}
    rows = pd.Series(all_ids).map(snp_idx).to_numpy()
    cols = pd.Series(all_genes).map(gene_idx).to_numpy()
    data = np.ones(len(all_ids))
    sparse_array = coo_array((data, (rows, cols)), shape=(len(snplist), len(genelist)))
    snp_gene_df = pd.DataFrame(sparse_array.toarray(), index=snplist, columns=genelist, dtype=bool)
    print(f"    removed {len(variant_df) - len(snplist)} SNPs")
    print(f"    {len(snplist)} SNPs, {len(genelist)} genes")
    snp_gene_df.to_parquet(output_file, index=True, compression='zstd')
    # snp_gene_df = pd.read_parquet(output_file)
    

def snp2pathway(project_dir, output_file):
    """Creates a snp-to-pathway mapping from snp-to-gene and gene-to-pathway mapping dataframes.

    Args:
        project_dir (str): Path to the project directory.
        output_file (str): Path to the output parquet file.
    """
    
    # snps are rows, genes are columns, bool values converted to int64 for matrix multiplication
    snp_gene_df = pd.read_parquet(f"{project_dir}/intermediate/snp_gene_mapping.parquet").astype(np.int64)
    print(f"    {snp_gene_df.shape[0]} SNPs, {snp_gene_df.shape[1]} genes") 
    
    # genes are rows, pathways are columns, bool values converted to int64 for matrix multiplication
    gene_pathway_df = pd.read_parquet(f"{project_dir}/intermediate/gene_pathway_mapping.parquet").astype(np.int64)
    print(f"    {gene_pathway_df.shape[0]} genes, {gene_pathway_df.shape[1]} pathways")

    # keep genes that are in both snp-gene and gene-pathway matrices
    keep_genes = np.intersect1d(gene_pathway_df.index, snp_gene_df.columns)
    print(f"    {len(keep_genes)} overlapping genes in both snp-gene and gene-pathway matrices")
    tmp_sgm = snp_gene_df.loc[:, keep_genes]
    tmp_gpm = gene_pathway_df.loc[keep_genes, :]

    # make snp-pathway matrix with dot product of sparse arrays (near instant computation)
    sg_sparse = csr_array(tmp_sgm.to_numpy(), dtype=np.int64)
    gp_sparse = csr_array(tmp_gpm.to_numpy(), dtype=np.int64)
    tmp_sgp = sg_sparse.dot(gp_sparse).todense()  # type: ignore

    # after matrix multiplication (dot product) there will be values greater than 1
    # set data type to bool and then back to int
    snp_pathway_df = pd.DataFrame(tmp_sgp.astype(bool).astype(int),
                                index=pd.Series(tmp_sgm.index, name='varid'),
                                columns=pd.Series(tmp_gpm.columns, name='pathway'))
    print(f"    {snp_pathway_df.shape[0]} SNPs, {snp_pathway_df.shape[1]} pathways")
    snp_pathway_df.to_parquet(output_file, index=True, compression='zstd')
    # snp_pathway_df = pd.read_parquet(output_file)


def _bpm_row(snp_pathway_np, i):
    """Computes SNP set differences between pathway i and every pathway j > i.

    Each row's index arrays are concatenated into one flat array (with sizes
    to split it back) so workers send a few large objects to the parent
    instead of thousands of small ones.
    """
    p1 = snp_pathway_np[:, i]
    ind1_row, ind2_row, ind1size_row, ind2size_row = [], [], [], []
    for j in range(i + 1, snp_pathway_np.shape[1]):
        p2 = snp_pathway_np[:, j]

        # snps in pathway 1 but not in pathway 2
        ind1 = np.where((p1 - p2) == 1)[0]

        # snps in pathway 2 but not in pathway 1
        ind2 = np.where((p2 - p1) == 1)[0]

        ind1_row.append(ind1)
        ind2_row.append(ind2)
        ind1size_row.append(len(ind1))
        ind2size_row.append(len(ind2))

    # The last pathway has no j > i, and np.concatenate fails on an empty list.
    if not ind1_row:
        empty = np.empty(0, dtype=np.intp)
        return empty, empty, empty, empty

    return (
        np.concatenate(ind1_row),
        np.concatenate(ind2_row),
        np.array(ind1size_row, dtype=np.intp),
        np.array(ind2size_row, dtype=np.intp),
    )

def pathway_indices(project_dir, min_path, n_workers, output_file):
    """Exctracts SNP indices for BPM/WPM sets.

    Args:
        project_dir (str): path to the project directory
        min_path (int): minimum path size
        n_workers (int): number of workers
        output_file (str): path to the output file, which is separated into a WPM and 
            a BPM parquet file.
    """

    # Reading in data files
    snp_pathway_df = pd.read_parquet(f"{project_dir}/intermediate/snp_pathway_mapping.parquet")
    pathways = snp_pathway_df.sum(axis=0)
    snp_pathway_np = snp_pathway_df.to_numpy()
    pathways_np = pathways.to_numpy()
    len_columns = len(snp_pathway_df.columns)

    # Finding WPM indices
    WPMind = [ np.nonzero(snp_pathway_np[:, i])[0] for i in range(len_columns) ]
    wpm = pd.DataFrame({
        'pathway': pathways.index,
        'indsize': pathways.values,
        'ind': WPMind,
        'size': pathways_np * pathways_np - pathways_np,
        })
    
    # filter out pathways that are too small given the available SNPs in the WPM
    # TODO: change back?
    wpm = wpm[(wpm['indsize'] >= min_path)].reset_index(drop=True)
    print(f"    Total WPMs after filtering with min_path={min_path}: {len(wpm):,} / {len(pathways):,}")
    wpm.to_parquet(f"{output_file}-wpm.parquet", index=True, compression='snappy')

    # Finding all possible combinations of pairs for pathway names and sizes.
    # combnames = np.array(list(combinations(pathways.index, 2))) TODO: change back?
    combnames = np.array(list(combinations(wpm['pathway'], 2)))
    snp_pathway_np_subset = snp_pathway_df.loc[:, wpm['pathway'].to_list()].to_numpy()  # TODO: remove?

    # Finding BPM indices: one task per pathway i covers all pairs (i, j > i).
    # Parallel returns results in input order, so the flattened lists line up with combnames.
    if n_workers is None:
        n_workers = -1
    rows = Parallel(n_jobs=n_workers)(
        delayed(_bpm_row)(snp_pathway_np_subset, i) for i in range(len(wpm))
    )
    BPMind1, BPMind2, ind1size, ind2size = [], [], [], []
    for flat1, flat2, row_ind1size, row_ind2size in rows:  # type: ignore
        # The last pathway has no pairs; np.split would still return one empty array.
        if len(row_ind1size) == 0:
            continue
        # Split each row's flat array back into one index array per pair.
        BPMind1.extend(np.split(flat1, np.cumsum(row_ind1size)[:-1]))
        BPMind2.extend(np.split(flat2, np.cumsum(row_ind2size)[:-1]))
        ind1size.extend(row_ind1size.tolist())
        ind2size.extend(row_ind2size.tolist())

    # Orienting bpm data and converting to dataframes.
    bpm = pd.DataFrame({
        'path1names': combnames[:, 0], 'ind1': BPMind1, 'ind1size': ind1size,
        'path2names': combnames[:, 1], 'ind2': BPMind2, 'ind2size': ind2size,
        'size': np.array(ind1size) * np.array(ind2size),  # Getting BPM sizes by multiplying combination available pairs.
        })

    # filter out pathways that are too small after removing SNPs that are in both pathways of a BPM
    # bpm = bpm[(bpm['ind1size'] >= min_path) & (bpm['ind2size'] >= min_path)].reset_index(drop=True)  TODO: change back?
    print(f"    Total BPMs after filtering with min_path={min_path}: {len(bpm):,} / {len(combnames):,}")
    bpm.to_parquet(f"{output_file}-bpm.parquet", index=True, compression='zstd')


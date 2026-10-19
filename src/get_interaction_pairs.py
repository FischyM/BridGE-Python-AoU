import numpy as np
import pandas as pd
from scipy.sparse import csr_array, load_npz
from scipy.stats import hypergeom


np.seterr(divide='ignore', invalid='ignore')

DRIVER_LIMIT = 20
GI_FOLD_CUTOFF = 1.2
GI_HYGE_CUTOFF = 0.05
PAIR_CHUNK = 2048  # pairs per genotype-product block (bounds temporary memory)

TABLE_COLUMNS = ['path1', 'path2', 'snp1', 'chr1', 'loc1', 'gene1',
                 'snp2', 'chr2', 'loc2', 'gene2', 'GI type', 'case frequency',
                 'control frequency', 'GI', 'OR', 'effect', 'LD']

GT_LABELS = {'RR': 'recessive', 'DD': 'dominant', 'RD': 'recessive_dominant'}
COMBINED_IDS = {1: 'RR', 2: 'DD', 3: 'RD'}


# Formats LD values: fixed point down to 0.001, scientific below that.
def _format_ld(x):
    return '{:.3f}'.format(x) if x >= 0.001 else '{:.3e}'.format(x)


def _hygetest(n, d, k, m):
    """Vectorized form of the original hygetest: sf(k-1; n, m, d).
    """
    return hypergeom.sf(k - 1, n, m, d)


def _sort_desc(values):
    """Row order of pandas sort_values(ascending=False): quicksort, NaN last.

    Reproduces pandas' nargsort exactly, so ties keep the original order.
    """
    idx = np.arange(len(values))
    nan = np.isnan(values)
    non_nan = values[~nan][::-1]
    non_nan_idx = idx[~nan][::-1]
    order = non_nan_idx[non_nan.argsort(kind='quicksort')][::-1]
    return np.concatenate([order, idx[nan]])


def _snp_genes(snp_ids, pathway_genes, sg_matrix, sg_genes, snp_row, gene_col):
    """Slash-joined gene names for each SNP, '/' where there is no mapping.

    Only genes that are both in the pathway and in the snp-gene matrix count;
    names are joined in sorted order, as np.intersect1d gave the original.
    """
    kept_genes = np.intersect1d(pathway_genes, sg_genes)
    cols = np.array([gene_col[g] for g in kept_genes], dtype=np.intp)
    genes = []
    for snp in snp_ids:
        row = snp_row.get(snp)
        if row is None:
            genes.append('/')
            continue
        hits = kept_genes[sg_matrix[row, cols]]
        genes.append('/'.join(hits) if len(hits) > 0 else '/')
    return np.array(genes)


def _pair_counts(weights, X1, X2):
    """weights @ (X1 * X2), column by column, in blocks of PAIR_CHUNK pairs.

    Equals the original sum((w*a) * (w*b)) for each pair, since the genotype
    columns are 0/1 and every term is an exact integer in float64.
    """
    out = np.empty(X1.shape[1])
    for s in range(0, X1.shape[1], PAIR_CHUNK):
        out[s:s + PAIR_CHUNK] = weights @ (X1[:, s:s + PAIR_CHUNK] * X2[:, s:s + PAIR_CHUNK])
    return out


def _driver_string(snps, genes, snp_mean_gi, snp_mean_gi_bg, n_bg, n_other, in_int, all_int):
    """Semicolon-joined 'snp_gene_foldX_hygeY' summary of the top driver SNPs."""
    gi_fold = np.divide(snp_mean_gi, snp_mean_gi_bg)
    gi_fold[np.isnan(gi_fold)] = 0
    gi_hyge = _hygetest(n_bg, n_other, in_int, all_int)
    gi_hyge_log = -1 * np.log10(gi_hyge)

    # significance filter, then sort by fold (descending)
    keep = np.flatnonzero((gi_hyge <= GI_HYGE_CUTOFF) & (gi_fold > GI_FOLD_CUTOFF))
    keep = keep[_sort_desc(gi_fold[keep])]

    # original: rows with gi_fold > 1, then dropna, then the top DRIVER_LIMIT
    numeric = np.column_stack([snp_mean_gi[keep], snp_mean_gi_bg[keep], gi_fold[keep],
                               gi_hyge[keep], gi_hyge_log[keep]])
    keep = keep[(gi_fold[keep] > 1) & ~np.isnan(numeric).any(axis=1)][:DRIVER_LIMIT]

    # .tolist() gives Python floats, so round() matches the original's iterrows values
    return ';'.join(
        f"{s}_{g}_fold{round(f, 2)}_hyge{round(h, 2)}"
        for s, g, f, h in zip(snps[keep], genes[keep],
                              gi_fold[keep].tolist(), gi_hyge_log[keep].tolist()))


def load_sparse_array(path):
    with np.load(path) as f:
        names = {k.split("__")[0] for k in f.files}
        return {n: csr_array((f[f"{n}__data"], f[f"{n}__indices"], f[f"{n}__indptr"]),
                                shape=tuple(f[f"{n}__shape"])) for n in names}


def get_interaction_pair(project_dir, snp_int_file, model, n, path1, path2, effects, path_ids, fdrcutoff, imported_ssm, densitycutoff=None):
    """Finds driver SNPs and genes for a set of BPMs or WPMs.

    Whether a row is a BPM or a WPM is decided by comparing path1 and path2.

    Args:
        project_dir (str): Path to the project directory.
        snp_int_file (str): Interaction network .npz (sparse arrays).
        model (str): The model used for the analysis.
        n (int): Number of BPMs/WPMs supplied.
        path1 (DataFrame): One column, pathway-1 name per module.
        path2 (DataFrame): One column, pathway-2 name per module. Same as path1 for WPMs.
        effects (DataFrame): One column of 'risk'/'protective' per module.
        path_ids (dict): Pathway name to its index in the dataset, built by collectresults.
        fdrcutoff (float): FDR threshold, used only to name the output file.
        imported_ssm (bool): True when the network was imported rather than
            computed from genotypes; drops the genotype-derived columns.
        densitycutoff (float, optional): Quantile used to pick the interaction
            score cutoff. Defaults to a fixed cutoff of 0.2.

    Returns:
        list: [bpm_path1_drivers, bpm_path2_drivers, wpm_path_drivers], each a
            one-column DataFrame indexed by global module index. Only the
            relevant one or two are populated; the others are empty.

    Side effect:
        Writes <project_dir>/results/interaction_list-{bpm,wpm}-<model>-<fdr>.xlsx
    """
    int_network = load_sparse_array(snp_int_file)

    bpm = pd.read_parquet(f"{project_dir}/intermediate/pathway_indices-bpm.parquet")
    wpm = pd.read_parquet(f"{project_dir}/intermediate/pathway_indices-wpm.parquet")
    gene_pathway_df = pd.read_parquet(f"{project_dir}/intermediate/gene_pathway_mapping.parquet")
    snp_gene_df = pd.read_parquet(f"{project_dir}/intermediate/snp_gene_mapping.parquet")
    
    snp_data = np.load(f"{project_dir}/intermediate/snp_data.npz")
    G = snp_data['data']
    varid = snp_data['varid']
    chrom = snp_data['chrom']
    pos = snp_data['pos']
    pheno = snp_data['pheno']

    # load ld_file
    ld_sparse_csr = load_npz(f"{project_dir}/raw/plink.ld.r2.npz").tocsr()
    ld_vars = pd.read_csv(f"{project_dir}/raw/plink.ld.r2.vars", names=['varid'], dtype=str)
    assert np.array_equal(ld_vars['varid'].to_numpy(dtype=str), varid.astype(str))

    # genotype coding, applied per selected column rather than to all of G
    dom_map = np.array([0, 1, 1], dtype=np.int8)  # dominant:  0->0, 1->1, 2->1
    rec_map = np.array([0, 0, 1], dtype=np.int8)  # recessive: 0->0, 1->0, 2->1

    # find score cutoffs if densitycutoff provided
    if densitycutoff is None:
        pos_cutoff = 0.2
        neg_cutoff = 0.2
    else:
        if densitycutoff <= 0 or densitycutoff >= 1:
            densitycutoff = 0.1
        pos_cutoff = np.quantile(int_network['protective'].toarray(), 1 - densitycutoff)
        neg_cutoff = np.quantile(int_network['risk'].toarray(), 1 - densitycutoff)

    # --- pandas -> numpy, once, before the per-module loop -------------------
    names1 = path1.iloc[:, 0].to_numpy()
    names2 = path2.iloc[:, 0].to_numpy()
    effect_arr = effects.iloc[:, 0].to_numpy()

    # parquet row labels are 0..n-1, so positional == label lookup
    pathway_size = wpm.shape[0]
    n_pairs = int(pathway_size * (pathway_size - 1) / 2)
    bpm_ind1 = bpm['ind1'].to_numpy()
    bpm_ind2 = bpm['ind2'].to_numpy()
    wpm_ind = wpm['ind'].to_numpy()

    gp_matrix = gene_pathway_df.to_numpy() == 1
    gp_genes = gene_pathway_df.index.to_numpy()
    gp_col = {name: c for c, name in enumerate(gene_pathway_df.columns)}

    sg_matrix = snp_gene_df.to_numpy() == 1
    sg_genes = snp_gene_df.columns.to_numpy()
    snp_row = {snp: r for r, snp in enumerate(snp_gene_df.index)}
    gene_col = {gene: c for c, gene in enumerate(sg_genes)}

    pheno_size = pheno.shape[0]
    res_pheno = np.ones(pheno.shape) - pheno
    pheno_nnz = np.sum(pheno)
    pheno_nz = pheno_size - pheno_nnz
    pheno_sq = (pheno * pheno).astype(np.float64)  # (mask*a)*(mask*b) == mask**2 * a*b
    res_sq = res_pheno * res_pheno

    bpm_path1_drivers, bpm_path2_drivers, wpm_path_drivers = [], [], []
    path_index = []
    pair_tables = []
    wpm_flag = False

    for i_path in range(n):
        pathname1 = names1[i_path]
        pathname2 = names2[i_path]
        effect = effect_arr[i_path]

        if effect == 'protective':
            ssm = int_network['protective']
            max_id = int_network.get('protective_max_id')  # combined model only
            score_cutoff = pos_cutoff
        else:
            ssm = int_network['risk']
            max_id = int_network.get('risk_max_id')
            score_cutoff = neg_cutoff

        # find pathway index in wpm and find snp ids
        p_id1 = path_ids[pathname1]
        p_id2 = path_ids[pathname2]
        if p_id2 < p_id1:
            p_id1, p_id2 = p_id2, p_id1

        if p_id1 == p_id2:
            wpm_flag = True
            ind1 = wpm_ind[p_id1]
            ind2 = ind1
            rem = 0 if effect == 'protective' else wpm.shape[0]
            path_index.append(p_id1 + rem)
        else:
            bpm_id = int(n_pairs - (pathway_size - p_id1) * (pathway_size - p_id1 - 1) / 2 + p_id2 - p_id1 - 1)
            rem = 0 if effect == 'protective' else bpm.shape[0]
            path_index.append(bpm_id + rem)
            ind1 = bpm_ind1[bpm_id]
            ind2 = bpm_ind2[bpm_id]

        ind1 = np.asarray(ind1, dtype=np.intp)
        ind2 = np.asarray(ind2, dtype=np.intp)

        # get snp ids
        ind1_snp = varid[ind1]
        ind2_snp = varid[ind2]

        # genes in each pathway, then per-SNP gene names
        ind1_gp = gp_genes[gp_matrix[:, gp_col[pathname1]]]
        ind2_gp = gp_genes[gp_matrix[:, gp_col[pathname2]]]
        ind1_gene = _snp_genes(ind1_snp, ind1_gp, sg_matrix, sg_genes, snp_row, gene_col)
        if p_id1 == p_id2:
            ind2_gene = ind1_gene
        else:
            ind2_gene = _snp_genes(ind2_snp, ind2_gp, sg_matrix, sg_genes, snp_row, gene_col)

        # slice the sparse network first, then densify only the module block
        ssm_dis = ssm[ind1][:, ind2].toarray()
        if model == 'combined':
            m_id = max_id[ind1][:, ind2].toarray()  # type: ignore

        # For a WPM both sides are the same pathway, so only one triangle counts.
        tmp_dis = np.tril(ssm_dis) if p_id1 == p_id2 else ssm_dis

        bin_int_index = np.argwhere(tmp_dis > score_cutoff)
        i = bin_int_index[:, 0]
        j = bin_int_index[:, 1]
        a = ind1[i]  # genome-wide column of each pair's first SNP
        b = ind2[j]  # ... and second SNP
        interaction_pairs = i.shape[0]

        GI = ssm_dis[i, j]
        freq_case = np.zeros(interaction_pairs)
        freq_control = np.zeros(interaction_pairs)
        snp_pair = np.zeros((pheno_size, interaction_pairs))

        # per-pair genotype model ('combined' picks it from max_id)
        if model == 'combined':
            tmp_model = np.array([COMBINED_IDS.get(dm, 'combined') for dm in m_id[i, j].tolist()], dtype=object)  # type: ignore
            if np.any(tmp_model == 'combined'):
                # the original would fail here too (GI type list shorter than the table)
                raise ValueError('combined network has an interaction with no max_id model (expected 1, 2 or 3)')
        else:
            tmp_model = np.full(interaction_pairs, model, dtype=object)

        for gt in ('RR', 'DD'):
            sel = np.flatnonzero(tmp_model == gt)
            if sel.size == 0:
                continue
            gmap = rec_map if gt == 'RR' else dom_map
            X1 = gmap[G[:, a[sel]]]
            X2 = gmap[G[:, b[sel]]]
            freq_case[sel] = _pair_counts(pheno_sq, X1, X2) / pheno_nnz
            freq_control[sel] = _pair_counts(res_sq, X1, X2) / pheno_nz
            snp_pair[:, sel] = X1 * X2

        sel = np.flatnonzero(tmp_model == 'RD')
        if sel.size > 0:
            R1 = rec_map[G[:, a[sel]]]
            D1 = dom_map[G[:, a[sel]]]
            R2 = rec_map[G[:, b[sel]]]
            D2 = dom_map[G[:, b[sel]]]
            # orientation 1: recessive on pathway 1, dominant on pathway 2
            freq_case_1 = _pair_counts(pheno_sq, R1, D2) / pheno_nnz
            freq_control_1 = _pair_counts(res_sq, R1, D2) / pheno_nz
            # orientation 2: dominant on pathway 1, recessive on pathway 2
            freq_case_2 = _pair_counts(pheno_sq, D1, R2) / pheno_nnz
            freq_control_2 = _pair_counts(res_sq, D1, R2) / pheno_nz

            freq_control_1[freq_control_1 == 0] = 1
            freq_control_2[freq_control_2 == 0] = 1
            freq_R1 = freq_case_1 / freq_control_1
            freq_R2 = freq_case_2 / freq_control_2

            # Protective keeps the weaker ratio, risk keeps the stronger one.
            take_first = (freq_R1 > freq_R2) != (effect == 'protective')
            freq_case[sel] = np.where(take_first, freq_case_1, freq_case_2)
            freq_control[sel] = np.where(take_first, freq_control_1, freq_control_2)
            snp_pair[:, sel] = np.where(take_first, R1 * D2, D1 * R2)

        # base position, chromosome and LD for every pair at once
        chr1 = chrom[a].astype(np.float64)
        chr2 = chrom[b].astype(np.float64)
        pos1 = [str(p) for p in pos[a]]
        pos2 = [str(p) for p in pos[b]]
        ld = np.full(interaction_pairs, 'NA', dtype=object)
        same = np.flatnonzero(chr1 == chr2)
        if not imported_ssm and same.size > 0:
            r2 = np.asarray(ld_sparse_csr[a[same], b[same]]).ravel()
            ld[same] = [_format_ld(x) for x in r2]

        # compute odds ratio here based on the frequencies
        odds_ratio = np.divide(np.multiply(1 - freq_control, freq_case),
                               np.multiply(1 - freq_case, freq_control))

        if model in ('RR', 'RD', 'DD', 'combined'):
            gt_type = np.array([GT_LABELS[m] for m in tmp_model], dtype=object)
        else:
            gt_type = np.full(interaction_pairs, 'imported', dtype=object)

        order = _sort_desc(GI)
        pair_tables.append({
            'path1': np.full(interaction_pairs, pathname1, dtype=object),
            'path2': np.full(interaction_pairs, pathname2, dtype=object),
            'snp1': ind1_snp[i][order], 'chr1': chr1[order],
            'loc1': np.array(pos1, dtype=object)[order], 'gene1': ind1_gene[i][order],
            'snp2': ind2_snp[j][order], 'chr2': chr2[order],
            'loc2': np.array(pos2, dtype=object)[order], 'gene2': ind2_gene[j][order],
            'GI type': gt_type[order],
            'case frequency': freq_case[order], 'control frequency': freq_control[order],
            'GI': GI[order], 'OR': odds_ratio[order],
            'effect': np.full(interaction_pairs, effect, dtype=object),
            'LD': ld[order]})

        # --- driver SNPs -----------------------------------------------------
        hits = ssm_dis > score_cutoff
        in_int1 = np.sum(hits, axis=1)
        all_int1 = np.asarray((ssm[ind1] > score_cutoff).sum(axis=1)).ravel()
        path1_drivers = _driver_string(
            snps=ind1_snp, genes=ind1_gene,
            snp_mean_gi=in_int1 / ind2.shape[0],
            snp_mean_gi_bg=all_int1 / ssm.shape[1],
            n_bg=ssm.shape[1], n_other=ind2.shape[0],
            in_int=in_int1, all_int=all_int1)

        if p_id1 == p_id2:
            wpm_path_drivers.append(path1_drivers)
        else:
            # snp_mean_gi_bg sums over rows of ssm while all_int sums over columns.
            # These agree only if ssm is symmetric; worth confirming.
            in_int2 = np.sum(hits, axis=0)
            path2_drivers = _driver_string(
                snps=ind2_snp, genes=ind2_gene,
                snp_mean_gi=in_int2 / ind1.shape[0],
                snp_mean_gi_bg=np.asarray((ssm[ind2] > score_cutoff).sum(axis=1)).ravel() / ssm.shape[0],
                n_bg=ssm.shape[1], n_other=ind1.shape[0],
                in_int=in_int2,
                all_int=np.asarray((ssm[:, ind2] > score_cutoff).sum(axis=0)).ravel())

            bpm_path1_drivers.append(path1_drivers)
            bpm_path2_drivers.append(path2_drivers)

    bpm_path1_drivers = pd.DataFrame(bpm_path1_drivers, columns=['bpm_path1_drivers'], index=path_index if bpm_path1_drivers else None)
    bpm_path2_drivers = pd.DataFrame(bpm_path2_drivers, columns=['bpm_path2_drivers'], index=path_index if bpm_path2_drivers else None)
    wpm_path_drivers = pd.DataFrame(wpm_path_drivers, columns=['wpm_path_drivers'], index=path_index if wpm_path_drivers else None)

    # write interaction list (one DataFrame, built once)
    if pair_tables:
        interaction_table = pd.DataFrame(
            {c: np.concatenate([t[c] for t in pair_tables]) for c in TABLE_COLUMNS},
            columns=TABLE_COLUMNS)
    else:
        interaction_table = pd.DataFrame(columns=TABLE_COLUMNS)

    # remove non-relevant columns if ssm is imported
    if imported_ssm:
        interaction_table = interaction_table.drop(
            labels=['OR', 'GI type', 'case frequency', 'control frequency'], axis=1)

    kind = 'wpm' if wpm_flag else 'bpm'
    list_file = f'{project_dir}/results/interaction_list-{kind}-{model}-{fdrcutoff:.2f}.xlsx'

    with pd.ExcelWriter(list_file, engine='openpyxl') as writer:  # iter_cols below is openpyxl API
        interaction_table.to_excel(writer, index=False, sheet_name='Sheet1')
        ws = writer.sheets['Sheet1']
        for col in ws.iter_cols(min_col=12, max_col=16, min_row=2):  # L:P, skip header
            for cell in col:
                cell.number_format = '0.000'

    return [bpm_path1_drivers, bpm_path2_drivers, wpm_path_drivers]

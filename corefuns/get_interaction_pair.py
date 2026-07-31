import os
import pickle

import numpy as np
import pandas as pd

from classes import SNPdata, GeneSet, SNPset, BPMind, InteractionNetwork
from corefuns.HygeCache import _hyge_single


np.seterr(divide='ignore', invalid='ignore')

DRIVER_LIMIT = 20
GI_FOLD_CUTOFF = 1.2
GI_HYGE_CUTOFF = 0.05

TABLE_COLUMNS = ['path1', 'path2', 'snp1', 'chr1', 'loc1', 'gene1',
                 'snp2', 'chr2', 'loc2', 'gene2', 'GI type', 'case frequency',
                 'control frequency', 'GI', 'OR', 'effect', 'LD']

SNP_STAT_COLUMNS = ['snps', 'genes', 'snp_mean_gi', 'snp_mean_gi_bg',
                    'gi_fold', 'gi_hyge', 'gi_hyge_log']

# Formats LD values: fixed point down to 0.001, scientific below that.
def _format_ld(x):
    return '{:.3f}'.format(x) if x >= 0.001 else '{:.3e}'.format(x)


def hygetest_caller(input_row):
    return _hyge_single(input_row[0], input_row[1], input_row[2], input_row[3])


def _snp_genes(snp2gene, snp_ids, kept_snps, kept_genes):
    """Slash-joined gene names for each SNP, '/' where there is no mapping."""
    mapped = snp2gene.loc[kept_snps, kept_genes]
    genes = []
    for snp in snp_ids:
        if snp not in kept_snps:
            genes.append('/')
            continue
        row = mapped.loc[snp]
        hits = row[row == 1].index.values
        genes.append('/'.join(hits) if len(hits) > 0 else '/')
    return np.array(genes)


def _driver_string(snp_table):
    """Semicolon-joined 'snp_gene_foldX_hygeY' summary of the top driver SNPs."""
    rows = snp_table[snp_table['gi_fold'] > 1].dropna()
    if rows.empty:
        return ''
    return ';'.join(
        f"{r['snps']}_{r['genes']}"
        f"_fold{round(r['gi_fold'], 2)}_hyge{round(r['gi_hyge_log'], 2)}"
        for _, r in rows.head(DRIVER_LIMIT).iterrows())


def _pair_freq(mask, col_a, col_b, denom):
    """Fraction of the masked samples carrying both genotypes.

    Keeps the original's form of applying `mask` to each column before
    multiplying, rather than assuming the mask is strictly 0/1.
    """
    return np.sum(np.multiply(np.multiply(mask, col_a),
                              np.multiply(mask, col_b))) / denom


def _snp_stat_table(snps, genes, snp_mean_gi, snp_mean_gi_bg, in_int, all_int, n_bg, n_other):
    """Per-SNP interaction enrichment, filtered to the significant drivers."""
    gi_fold = np.divide(snp_mean_gi, snp_mean_gi_bg)
    gi_fold[np.isnan(gi_fold)] = 0

    in_int = np.reshape(in_int, (in_int.shape[0], 1))
    all_int = np.reshape(all_int, (all_int.shape[0], 1))
    N = np.broadcast_to(n_bg, in_int.shape)
    D = np.broadcast_to(n_other, in_int.shape)
    hype_in = np.concatenate((N, D, in_int, all_int), axis=1)
    gi_hyge = np.apply_along_axis(hygetest_caller, 1, hype_in)

    table = pd.DataFrame(
        {'snps': snps, 'genes': genes,
         'snp_mean_gi': snp_mean_gi, 'snp_mean_gi_bg': snp_mean_gi_bg,
         'gi_fold': gi_fold, 'gi_hyge': gi_hyge,
         'gi_hyge_log': -1 * np.log10(gi_hyge)},
        columns=SNP_STAT_COLUMNS)
    table = table[(table['gi_hyge'] <= GI_HYGE_CUTOFF)
                  & (table['gi_fold'] > GI_FOLD_CUTOFF)]
    return table.sort_values('gi_fold', ascending=False)


def get_interaction_pair(n, path1, path2, effects, ssmfile, bpmfile, snp2pathwayfile,
                         snp2genefile, path_ids, fdrcutoff, imported_ssm,
                         densitycutoff=None):
    """Finds driver SNPs and genes for a set of BPMs or WPMs.

    Whether a row is a BPM or a WPM is decided by comparing path1 and path2.

    Args:
        n (int): Number of BPMs/WPMs supplied.
        path1 (DataFrame): One column, pathway-1 name per module.
        path2 (DataFrame): One column, pathway-2 name per module. Same as path1
            for WPMs.
        effects (DataFrame): One column of 'risk'/'protective' per module.
        ssmfile (str): Interaction network pickle.
        bpmfile (str): Pickle with the SNP ids for each BPM/WPM.
        snp2pathwayfile (str): Pickle mapping SNPs to pathways.
        snp2genefile (str): Pickle mapping SNPs to genes.
        path_ids (dict): Pathway name to its index in the dataset, built by
            collectresults.
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
        Writes <project_dir>/results/interaction_list_{bpm,wpm}_<model>_<fdr>.xlsx
    """
    with open(snp2genefile, 'rb') as fh:
        snp2gene: pd.DataFrame = pickle.load(fh)

    with open(snp2pathwayfile, 'rb') as fh:
        snp2path: SNPset = pickle.load(fh)
    
    with open(bpmfile, 'rb') as fh:
        bpm_ind: BPMind = pickle.load(fh)
    
    with open(snp2path.geneset, 'rb') as fh:
        geneset: GeneSet = pickle.load(fh)
        
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(snp2pathwayfile)))
    
    with open(os.path.join(project_dir, 'intermediate', 'SNPdataAD.pkl'), 'rb') as fh:
        snpdataAD: SNPdata = pickle.load(fh)
    
    with open(os.path.join(project_dir, 'intermediate', 'SNPdataAR.pkl'), 'rb') as fh:
        snpdataAR: SNPdata = pickle.load(fh)
    
    with open(ssmfile, 'rb') as fh:
        int_network: InteractionNetwork = pickle.load(fh)

    # load ld_file
    ld_file = os.path.join(project_dir, 'intermediate', 'plink.ld')
    try:
        ld_data = pd.read_csv(ld_file, header=None, index_col=False, sep='\t').to_numpy()
        ld_provided = True
    except FileNotFoundError:
        ld_data = None
        ld_provided = False

    # find score cutoffs if densitycutoff provided
    if densitycutoff is None:
        pos_cutoff = 0.2
        neg_cutoff = 0.2
    else:
        if densitycutoff <= 0 or densitycutoff >= 1:
            densitycutoff = 0.1
        pos_cutoff = np.quantile(int_network.protective, 1 - densitycutoff)
        neg_cutoff = np.quantile(int_network.risk, 1 - densitycutoff)

    # Loop-invariant lookups, hoisted out of the per-module loop.
    model = os.path.splitext(os.path.basename(ssmfile))[0].split('_')[-2]
    pathway_size = bpm_ind.wpm['pathway'].shape[0]
    n_pairs = int(pathway_size * (pathway_size - 1) / 2)
    bpm_ind1, bpm_ind2 = bpm_ind.bpm['ind1'], bpm_ind.bpm['ind2']
    wpm_ind = bpm_ind.wpm['ind']
    all_genes = snp2gene.columns.values
    all_snps = snp2gene.index.values

    pheno = np.asarray(snpdataAD.pheno)
    pheno_size = pheno.shape[0]
    res_pheno = np.ones(pheno.shape) - pheno
    pheno_nnz = np.sum(pheno)
    pheno_nz = pheno_size - pheno_nnz
    # NOTE: deliberately not hoisting snpdataAD.data / snpdataAR.data to numpy
    # here. That would materialise the whole samples x SNPs genotype matrix for
    # the duration of the call; the per-module slices below stay small.

    bpm_path1_drivers, bpm_path2_drivers, wpm_path_drivers = [], [], []
    path_index = []
    pair_tables = []
    wpm_flag = False

    for i_path in range(n):
        pathname1 = path1.iloc[i_path, 0]
        pathname2 = path2.iloc[i_path, 0]
        effect = effects.iloc[i_path, 0]

        if effect == 'protective':
            ssm = int_network.protective
            max_id = int_network.protective_max_id
            score_cutoff = pos_cutoff
        else:
            ssm = int_network.risk
            max_id = int_network.risk_max_id
            score_cutoff = neg_cutoff

        ## find pathway index in wpm and find snp ids
        p_id1 = path_ids[pathname1]
        p_id2 = path_ids[pathname2]
        if p_id2 < p_id1:
            p_id1, p_id2 = p_id2, p_id1

        if p_id1 == p_id2:
            wpm_flag = True
            ind1 = wpm_ind[p_id1]
            ind2 = ind1
            rem = 0 if effect == 'protective' else bpm_ind.wpm.shape[0]
            path_index.append(p_id1 + rem)
        else:
            bpm_id = int(n_pairs - (pathway_size - p_id1) * (pathway_size - p_id1 - 1) / 2
                         + p_id2 - p_id1 - 1)
            rem = 0 if effect == 'protective' else bpm_ind.bpm.shape[0]
            path_index.append(bpm_id + rem)
            ind1 = bpm_ind1[bpm_id]
            ind2 = bpm_ind2[bpm_id]

        ## get snp rsids
        ind1_snp = snpdataAR.rsid[ind1].values
        ind2_snp = snpdataAR.rsid[ind2].values

        ## find all genes for the 2 pathways from the geneset(index)
        tmp = geneset.gpmatrix[pathname1]
        ind1_gp = tmp[tmp == 1].index.values
        tmp = geneset.gpmatrix[pathname2]
        ind2_gp = tmp[tmp == 1].index.values

        ## keep only genes and snps present in the snp2gene matrix
        ind1_gene = _snp_genes(snp2gene, ind1_snp,
                              np.intersect1d(ind1_snp, all_snps),
                              np.intersect1d(ind1_gp, all_genes))
        if p_id1 == p_id2:
            ind2_gene = ind1_gene
        else:
            ind2_gene = _snp_genes(snp2gene, ind2_snp,
                                  np.intersect1d(ind2_snp, all_snps),
                                  np.intersect1d(ind2_gp, all_genes))

        ind1 = np.asarray(ind1)
        ind2 = np.asarray(ind2)

        ssm_dis = ssm[ind1, :][:, ind2]
        if model == 'combined':
            m_id = max_id[ind1, :][:, ind2]
        # For a WPM both sides are the same pathway, so only one triangle counts.
        tmp_dis = np.tril(ssm_dis) if p_id1 == p_id2 else ssm_dis

        bin_int_index = np.argwhere(tmp_dis > score_cutoff)
        i = bin_int_index[:, 0]
        j = bin_int_index[:, 1]
        snps1 = ind1_snp[i]
        snps2 = ind2_snp[j]
        genes1 = ind1_gene[i]
        genes2 = ind2_gene[j]

        interaction_pairs = i.shape[0]
        GI = np.zeros(interaction_pairs)
        GT_type = []
        freq_case = np.zeros(interaction_pairs)
        freq_control = np.zeros(interaction_pairs)
        chr1 = np.zeros(interaction_pairs)
        chr2 = np.zeros(interaction_pairs)
        pos1, pos2, ld = [], [], []

        AD1 = snpdataAD.data.iloc[:, ind1].to_numpy()
        AD2 = snpdataAD.data.iloc[:, ind2].to_numpy()
        AR1 = snpdataAR.data.iloc[:, ind1].to_numpy()
        AR2 = snpdataAR.data.iloc[:, ind2].to_numpy()
        snp_pair = np.zeros((pheno_size, interaction_pairs))

        for k in range(interaction_pairs):
            GI[k] = ssm_dis[i[k], j[k]]
            tmp_model = model
            if model == 'combined':
                # disease model with maximum interaction in combined model
                dm = m_id[i[k], j[k]]
                if dm == 1:
                    tmp_model = 'RR'
                elif dm == 2:
                    tmp_model = 'DD'
                elif dm == 3:
                    tmp_model = 'RD'

            if tmp_model == 'RR':
                GT_type.append('recessive')
                freq_case[k] = _pair_freq(pheno, AR1[:, i[k]], AR2[:, j[k]], pheno_nnz)
                freq_control[k] = _pair_freq(res_pheno, AR1[:, i[k]], AR2[:, j[k]], pheno_nz)
                snp_pair[:, k] = np.multiply(AR1[:, i[k]], AR2[:, j[k]])
            elif tmp_model == 'DD':
                GT_type.append('dominant')
                freq_case[k] = _pair_freq(pheno, AD1[:, i[k]], AD2[:, j[k]], pheno_nnz)
                freq_control[k] = _pair_freq(res_pheno, AD1[:, i[k]], AD2[:, j[k]], pheno_nz)
                snp_pair[:, k] = np.multiply(AD1[:, i[k]], AD2[:, j[k]])
            elif tmp_model == 'RD':
                GT_type.append('recessive_dominant')
                ## orientation 1: recessive on pathway 1, dominant on pathway 2
                freq_case_1 = _pair_freq(pheno, AR1[:, i[k]], AD2[:, j[k]], pheno_nnz)
                freq_control_1 = _pair_freq(res_pheno, AR1[:, i[k]], AD2[:, j[k]], pheno_nz)
                ## orientation 2: dominant on pathway 1, recessive on pathway 2
                freq_case_2 = _pair_freq(pheno, AD1[:, i[k]], AR2[:, j[k]], pheno_nnz)
                freq_control_2 = _pair_freq(res_pheno, AD1[:, i[k]], AR2[:, j[k]], pheno_nz)

                if freq_control_1 == 0:
                    freq_control_1 = 1
                if freq_control_2 == 0:
                    freq_control_2 = 1
                freq_R1 = freq_case_1 / freq_control_1
                freq_R2 = freq_case_2 / freq_control_2

                # Protective keeps the weaker ratio, risk keeps the stronger one.
                # The snp_pair product follows whichever orientation is chosen:
                # orientation 1 is AR1*AD2, orientation 2 is AD1*AR2.
                take_first = (freq_R1 > freq_R2) != (effect == 'protective')
                if take_first:
                    freq_case[k], freq_control[k] = freq_case_1, freq_control_1
                    snp_pair[:, k] = np.multiply(AR1[:, i[k]], AD2[:, j[k]])
                else:
                    freq_case[k], freq_control[k] = freq_case_2, freq_control_2
                    snp_pair[:, k] = np.multiply(AD1[:, i[k]], AR2[:, j[k]])

            # find base position and chromosome here
            chr1[k] = snpdataAD.chr.iloc[ind1[i[k]]]
            chr2[k] = snpdataAD.chr.iloc[ind2[j[k]]]
            pos1.append(str(snpdataAD.loc[ind1[i[k]]]['loc']))
            pos2.append(str(snpdataAD.loc[ind2[j[k]]]['loc']))
            if ld_provided and chr1[k] == chr2[k]:
                ld.append(_format_ld(ld_data[ind1[i[k]], ind2[j[k]]]))
            else:
                ld.append('NA')

        # compute odds ratio here based on the frequencies
        odds_ratio = np.divide(np.multiply(1 - freq_control, freq_case),
                               np.multiply(1 - freq_case, freq_control))

        genotype_models = ('RR', 'RD', 'DD', 'combined')
        output_pair = pd.DataFrame(
            {'path1': pathname1, 'path2': pathname2,
             'snp1': snps1, 'chr1': chr1, 'loc1': pos1, 'gene1': genes1,
             'snp2': snps2, 'chr2': chr2, 'loc2': pos2, 'gene2': genes2,
             'GI type': GT_type if model in genotype_models else 'imported',
             'case frequency': freq_case, 'control frequency': freq_control,
             'GI': GI, 'OR': odds_ratio, 'effect': effect, 'LD': ld},
            columns=TABLE_COLUMNS)
        pair_tables.append(output_pair.sort_values('GI', ascending=False))

        ## preparing output for pathway 1
        output_path1_snp = _snp_stat_table(
            snps=ind1_snp, genes=ind1_gene,
            snp_mean_gi=np.sum(ssm_dis > score_cutoff, axis=1) / ind2.shape[0],
            snp_mean_gi_bg=np.sum(ssm[ind1, :] > score_cutoff, axis=1) / ssm.shape[1],
            in_int=np.sum(ssm_dis > score_cutoff, axis=1),
            all_int=np.sum(ssm[ind1, :] > score_cutoff, axis=1),
            n_bg=ssm.shape[1], n_other=ind2.shape[0])

        if p_id1 == p_id2:
            wpm_path_drivers.append(_driver_string(output_path1_snp))
        else:
            ## preparing output for pathway 2
            # NOTE: snp_mean_gi_bg sums over rows of ssm while all_int sums over
            # columns. These agree only if ssm is symmetric; worth confirming.
            output_path2_snp = _snp_stat_table(
                snps=ind2_snp, genes=ind2_gene,
                snp_mean_gi=np.sum(ssm_dis > score_cutoff, axis=0) / ind1.shape[0],
                snp_mean_gi_bg=np.sum(ssm[ind2, :] > score_cutoff, axis=1) / ssm.shape[0],
                in_int=np.sum(ssm_dis > score_cutoff, axis=0),
                all_int=np.sum(ssm[:, ind2] > score_cutoff, axis=0),
                n_bg=ssm.shape[1], n_other=ind1.shape[0])

            bpm_path1_drivers.append(_driver_string(output_path1_snp))
            bpm_path2_drivers.append(_driver_string(output_path2_snp))

    bpm_path1_drivers = pd.DataFrame(bpm_path1_drivers, columns=['bpm_path1_drivers'], index=path_index if bpm_path1_drivers else None)
    bpm_path2_drivers = pd.DataFrame(bpm_path2_drivers, columns=['bpm_path2_drivers'], index=path_index if bpm_path2_drivers else None)
    wpm_path_drivers = pd.DataFrame(wpm_path_drivers, columns=['wpm_path_drivers'], index=path_index if wpm_path_drivers else None)

    # write interaction list
    if pair_tables:
        interaction_table = pd.concat(pair_tables, ignore_index=True)
    else:
        interaction_table = pd.DataFrame(columns=TABLE_COLUMNS)

    # remove non-relevant columns if ssm is imported
    if imported_ssm:
        interaction_table = interaction_table.drop(
            labels=['OR', 'GI type', 'case frequency', 'control frequency'], axis=1)

    results_dir = os.path.join(project_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    kind = 'wpm' if wpm_flag else 'bpm'
    list_file = os.path.join(
        results_dir, f'interaction_list_{kind}_{model}_{float(fdrcutoff):.2f}.xlsx')

    with pd.ExcelWriter(list_file, engine='xlsxwriter') as writer:
        interaction_table.to_excel(writer, index=False, sheet_name='Sheet1')
        number_format = writer.book.add_format({'num_format': '0.000'})
        writer.sheets['Sheet1'].set_column('L:P', None, number_format)

    return [bpm_path1_drivers, bpm_path2_drivers, wpm_path_drivers]

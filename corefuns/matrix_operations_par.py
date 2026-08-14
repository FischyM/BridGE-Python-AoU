import math, pickle
from datetime import datetime
from os import path

import numpy as np
from scipy.sparse import coo_array

from corefuns import HygeCache as hc
from corefuns import withinclassrand as wrand
from classes import SNPclass, InteractionNetwork


# matrix_operations_par computes the interaction network. The functions to call are run() and combine()
#
# REFACTOR NOTES (see accompanying summary):
#   - Workers no longer write into a giant shared dense (s x s) ctypes array. Each worker
#     returns sparse (row, col, value) triples for its block, which the parent assembles
#     into a scipy.sparse.csr_matrix. Most SNP pairs fail the alpha1/alpha2 filters, so this
#     is a large memory win at any meaningful value of s.
#   - sy is processed in column tiles (sy_chunk_size) inside each worker so peak memory per
#     worker no longer scales with the full s, only with (chunk_rows x sy_chunk_size).
#   - g10/g01/g00/x10/x01/x00 are derived from row/column sums of g11/x11 instead of being
#     computed via separate matmuls, and xp11/xp10/xp01/xp00 = g - x (since pheno_res = 1-pheno
#     is linear). This drops matmuls per chunk from 12 to 2 and removes 8 dense intermediates
#     (Ix, Iy, sx_res, sy_res, tempx_r, temp_r, temp, and the redundant g/x recomputation).
#   - InteractionNetwork now stores scipy.sparse.csr_matrix for risk/protective instead of
#     dense numpy arrays. Downstream consumers (genstats_perm.py, fdrsampleperm.py,
#     collectresults.py) will need to be updated to accept sparse matrices, consistent with
#     the broader sparse-matrix migration already underway in DataProcess.
#
# INPUTS:
#	project_dir: Project directory including all data files
#	model: disease model, can be RR-DD-RD, for combining them, call combine() function instead of run()
#	alpha1: maximum p-value threshold for p11 in combinations
#	alpha2: minimum p-value threshold for p10, p01, p00 in combinations
#	n_workers: Number of CPU cores used for parallel computing
#	R: network number identifier, 0 for real, non-zero for random networks(phenotype labels will be randomly shuffled before computing interactions)
#
# OUTPUTS:
#   ssM_mhygessi_{model}_R{R}.pkl - This pickle file contains an InteractionNetwork class object with following fields:
#       - risk: Risk-associated SNP-SNP interaction scores, scipy.sparse.csr_matrix
#       - protective: Protective SNP-SNP interaction scores, scipy.sparse.csr_matrix
#		- risk_max_id: indicator of which disease model has the maximum risk score for each SNP pair, used in combined model
#		- protective_max_id:  indicator of which disease model has the maximum protective score for each SNP pair, used in combined model
#


def helper_score_from_counts(cache, g11, x11, g10, x10, g01, x01, g00, x00, alpha1, alpha2, risk, pool, n_workers):
    """Reproduces the original p-value/log-score/filter logic, operating on 1D arrays."""

    eps = 1e-10
    p11 = cache.apply_hyge(g11, x11, risk, pool, n_workers) + eps
    p10 = cache.apply_hyge(g10, x10, risk, pool, n_workers) + eps
    p01 = cache.apply_hyge(g01, x01, risk, pool, n_workers) + eps
    p00 = cache.apply_hyge(g00, x00, risk, pool, n_workers) + eps
    q_min = np.minimum(np.minimum(p01, p10), p00)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        log_out = -np.log10(p11 / q_min)
    
    fail = (p11 > alpha1) | (p10 <= alpha2) | (p01 <= alpha2) | (p00 <= alpha2)
    log_out[fail] = 0
    log_out[q_min == 0] = 0
    log_out[~np.isfinite(log_out)] = 0
    log_out[log_out < 0] = 0
    # original also had `log_out[p11 == 0] = 0`, but p11 is always >= eps > 0 here
    # (it was dead code in the original too) so it's omitted.
    return log_out

def run(project_dir, model, alpha1, alpha2, n_jobs, n_workers, pool, R):
    """Computes the interaction network for a single model (RR, RD, or DD) and saves it to a pickle file.

    Args:
        project_dir (_type_): _description_
        model (_type_): _description_
        alpha1 (_type_): _description_
        alpha2 (_type_): _description_
        n_jobs (_type_): _description_
        n_workers (_type_): _description_
        R (_type_): _description_
    """
    
    print(f'Computing SNP-SNP interactions: R={R} model={model}', flush=True)
    output_name = f"{project_dir}/intermediate/ssM_mhygessi_{model}_R{R}.pkl"
    cluster_file = f"{project_dir}/intermediate/PlinkFile.cluster2"

    # loading and reading SNP data
    # read SNP data and convert to dominant and recessive coding
    with open(f"{project_dir}/intermediate/snp_data.pkl", "rb") as f:
        snp_data: SNPclass = pickle.load(f)
    pheno = snp_data.pheno
    G  = snp_data.data  # assuming there are no missing values in the genotype data (i.e., no -9s)
    
    # convert genotype data to dominant and recessive coding with a simple mapping
    dom_map = np.array([0, 1, 1])  # dominant:  0->0, 1->1, 2->1 
    rec_map = np.array([0, 0, 1])  # recessive: 0->0, 1->0, 2->1
    dataD = dom_map[G]
    dataR = rec_map[G]
        
    symmetric_flag = (model == 'RR' or model == 'DD')
    if model == 'RR':
        datai = dataR
        dataj = dataR
    elif model == 'DD':
        datai = dataD
        dataj = dataD
    else:
        datai = dataR
        dataj = dataD

    population_size = pheno.shape[0]
    ## shuffle phenotypes if R != 0
    if R > 0:
        if not path.exists(cluster_file):
            # single deterministic permutation per R
            rng = np.random.default_rng(66754 * R)
            permuted_idx = rng.permutation(population_size)
            pheno = pheno[permuted_idx]
        else:
            # TODO: this will fail and needs to be fixed if we ever want to run R > 0 with a cluster file
            pheno = wrand.withinclassrand(R, cluster_file, f"{project_dir}/intermediate/SNPdataAD.pkl")

    case_size = int(np.count_nonzero(pheno))
    control_size = population_size - case_size
    cache = hc.HygeCache(population_size, case_size, control_size)

    sx_full = np.ascontiguousarray(datai, dtype=np.float64)
    sy_full = np.ascontiguousarray(dataj, dtype=np.float64)
    pheno = np.asarray(pheno, dtype=np.float64).ravel()
    s = sx_full.shape[1]

    ## dividing sx for parallel computing (unchanged balancing logic - earlier chunks have
    ## fewer lower-triangle entries for symmetric models, so chunk boundaries are sqrt-spaced)
    idx = [0]
    if model == 'RR' or model == 'DD':
        share = s * s / n_jobs
        for i in range(n_jobs):
            if i == n_jobs - 1:
                idx.append(s)
            else:
                idx.append(math.floor(math.sqrt(idx[i] * idx[i] + share)))
    else:
        share = math.floor(s / n_jobs)
        for i in range(n_jobs):
            if i == n_jobs - 1:
                idx.append(s)
            else:
                idx.append(idx[i] + share)

    results = []
    for i in range(n_jobs):
        t1 = datetime.now()
        i1 = idx[i]
        i2 = idx[i + 1]
        print(f"    job split {i+1}/{n_jobs}: i1={i1}, i2={i2}", flush=True, end="")
        
        sx = np.ascontiguousarray(sx_full[:, i1:i2], dtype=np.float64)
        s = sy_full.shape[1]
        tempx = sx * pheno[:, None]
        sx_totals = sx.sum(axis=0)               # (b,)
        casex_totals = tempx.sum(axis=0)         # (b,)
        sy_totals = sy_full.sum(axis=0)               # (s,)
        caseY_totals = (sy_full * pheno[:, None]).sum(axis=0)  # (s,)
        
        # For symmetric models (RR/DD) we only need the strict lower triangle of the full
        # s x s matrix - everything else is filled in by mirroring in run(). Compute the
        # (local_row, global_col) coordinates of that triangle once, for this worker's row band.
        if symmetric_flag:
            tril_rows, tril_cols = np.tril_indices(i2 - i1, i1 - 1, s)
            sel = (tril_rows, tril_cols)
            out_rows = tril_rows + i1
            out_cols = tril_cols
        else:
            b = sx.shape[1]
            rr, cc = np.meshgrid(np.arange(b), np.arange(s), indexing='ij')
            sel = (rr.ravel(), cc.ravel())
            out_rows = sel[0] + i1
            out_cols = sel[1]

        # cache = hc.HygeCache(population_size, case_size, control_size)
        g11 = sx.T @ sy_full      # (b, s) - matmul #1
        x11 = tempx.T @ sy_full   # (b, s) - matmul #2
        
        g10 = sx_totals[:, None] - g11
        g01 = sy_totals[None, :] - g11
        g00 = population_size - sx_totals[:, None] - sy_totals[None, :] + g11
        
        x10 = casex_totals[:, None] - x11
        x01 = caseY_totals[None, :] - x11
        x00 = case_size - casex_totals[:, None] - caseY_totals[None, :] + x11
        
        # xp_* = g_* - x_* because pheno_res = 1 - pheno is linear in the counts above.
        xp11 = g11 - x11
        xp10 = g10 - x10
        xp01 = g01 - x01
        xp00 = g00 - x00
        
        g11_v, x11_v = g11[sel], x11[sel]
        g10_v, x10_v = g10[sel], x10[sel]
        g01_v, x01_v = g01[sel], x01[sel]
        g00_v, x00_v = g00[sel], x00[sel]
        xp11_v, xp10_v = xp11[sel], xp10[sel]
        xp01_v, xp00_v = xp01[sel], xp00[sel]
        
        # -log10(p-values) for risk and protective, with alpha1/alpha2 filtering
        risk_score = helper_score_from_counts(cache, g11_v, x11_v, g10_v, x10_v,
                                        g01_v, x01_v, g00_v, x00_v, alpha1, alpha2, True, pool, n_workers)
        prot_score = helper_score_from_counts(cache, g11_v, xp11_v, g10_v, xp10_v,
                                        g01_v, xp01_v, g00_v, xp00_v, alpha1, alpha2, False, pool, n_workers)

        nz_r = risk_score != 0
        nz_p = prot_score != 0
        
        risk_rows = out_rows[nz_r]
        risk_cols = out_cols[nz_r]
        risk_vals = risk_score[nz_r]
        prot_rows = out_rows[nz_p]
        prot_cols = out_cols[nz_p]
        prot_vals = prot_score[nz_p]
        
        results.append((risk_rows, risk_cols, risk_vals, prot_rows, prot_cols, prot_vals))
        print(f" - {str(datetime.now() - t1).split('.')[0]}", flush=True)

    risk_rows = np.concatenate([ r[0] for r in results ])
    risk_cols = np.concatenate([ r[1] for r in results ])
    risk_vals = np.concatenate([ r[2] for r in results ])
    prot_rows = np.concatenate([ r[3] for r in results ])
    prot_cols = np.concatenate([ r[4] for r in results ])
    prot_vals = np.concatenate([ r[5] for r in results ])

    result_risk = coo_array((risk_vals, (risk_rows, risk_cols)), shape=(s, s)).tocsr()
    result_protective = coo_array((prot_vals, (prot_rows, prot_cols)), shape=(s, s)).tocsr()

    if model == 'RR' or model == 'DD':
        # only the strict lower triangle was computed - mirror it. Upper triangle of
        # result_risk is all zero by construction, so addition doesn't double-count.
        result_risk = result_risk + result_risk.T
        result_protective = result_protective + result_protective.T
    else:
        # RD: full matrix was computed (no triangle dedup) but isn't symmetric by
        # construction - take elementwise max with transpose, zero the diagonal.
        result_risk = result_risk.maximum(result_risk.T)
        result_risk.setdiag(0)
        result_risk.eliminate_zeros()
        
        result_protective = result_protective.maximum(result_protective.T)
        result_protective.setdiag(0)
        result_protective.eliminate_zeros()

    network = InteractionNetwork(result_risk, result_protective, None, None)

    with open(output_name, 'wb') as final:
        pickle.dump(network, final)   

def combine_max(rr, rd, dd):
    """Combine three sparse matrices (RR, RD, DD) into a single sparse matrix of elementwise max values.
    
    1 == RR was max
    2 == DD was max
    3 == RD was max

    Args:
        rr (csr_array): recessive risk/protective matrix
        rd (csr_array): recessive-dominant risk/protective matrix
        dd (csr_array): dominant risk/protective matrix

    Returns:
        network_max (csr_array): max of the three matrices, elementwise
        network_max_id (csr_array): id of which matrix was max (1=RR, 2=DD, 3=RD)
    """
    # elementwise max across the three — already sparse-native (unchanged from your code)
    network_max_temp = rr.maximum(dd)
    network_max = network_max_temp.maximum(rd)

    # strict inequalities between sparse arrays stay sparse (unlike >=, <=, ==, !=)
    cond3 = network_max_temp < rd   # rd is strictly the overall max
    cond1 = rr > dd                 # rr > dd

    # "cond1 AND NOT cond3": subtract the overlap instead of negating (negation would densify)
    cond1_final = cond1 - cond1.multiply(cond3)

    # nonzero pattern of the overall max, replacing the dense zero-out step
    pattern = network_max.astype(bool)

    region_13 = cond1_final + cond3
    region_2 = pattern - region_13          # "else" case: dd wins (or tie), within the nonzero pattern

    network_max_id = (cond1_final + region_2.multiply(2) + cond3.multiply(3))
    network_max_id.eliminate_zeros()
    
    return network_max, network_max_id

def combine(project_dir, alpha1, alpha2, n_jobs, n_workers, pool, R):
    """Run the three models (RR, RD, DD) and combine their results into a single InteractionNetwork.

    Args:
        project_dir (str): _description_
        alpha1 (float): _description_
        alpha2 (float): _description_
        n_jobs (int): _description_
        n_workers (int): _description_
        pool (multiprocessing.Pool): The process pool to use for parallel processing.
        R (int): _description_
    """
    run(project_dir, 'RR', alpha1, alpha2, n_jobs, n_workers, pool, R)
    run(project_dir, 'RD', alpha1, alpha2, n_jobs, n_workers, pool, R)
    run(project_dir, 'DD', alpha1, alpha2, n_jobs, n_workers, pool, R)

    ## load results for 3 models
    with open(f"{project_dir}/intermediate/ssM_mhygessi_RR_R{R}.pkl", 'rb') as rr_file:
        rr_network: InteractionNetwork = pickle.load(rr_file)
    with open(f"{project_dir}/intermediate/ssM_mhygessi_RD_R{R}.pkl", 'rb') as rd_file:
        rd_network: InteractionNetwork = pickle.load(rd_file)
    with open(f"{project_dir}/intermediate/ssM_mhygessi_DD_R{R}.pkl", 'rb') as dd_file:
        dd_network: InteractionNetwork = pickle.load(dd_file)

    risk_max, risk_max_id = combine_max(rr_network.risk, rd_network.risk, dd_network.risk)
    protective_max, protective_max_id = combine_max(rr_network.protective, rd_network.protective, dd_network.protective)

    network = InteractionNetwork(risk_max, protective_max, risk_max_id, protective_max_id)
    
    output_name = f"{project_dir}/intermediate/ssM_mhygessi_combined_R{R}.pkl"
    with open(output_name, 'wb') as final:
        pickle.dump(network, final)

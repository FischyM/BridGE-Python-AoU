import math
import sys
import pickle
import multiprocessing as mp
from os import path

import numpy as np
import scipy.sparse

from corefuns import HygeCache as hc
from corefuns import withinclassrand as wrand
from classes import InteractionNetwork


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


class job_quota:
    """Class used as a parameter holder for passing parameters to the threads"""
    
    def __init__(self, population_size, case_size, control_size):
        self.population_size = population_size
        self.case_size = case_size
        self.control_size = control_size
        self.symmetric = True
        self.sx = None
        self.sy = None
        self.i1 = 0
        self.i2 = 0
        self.pheno = None
        self.alpha1 = 0
        self.alpha2 = 0


def _score_from_counts(cache, g11, x11, g10, x10, g01, x01, g00, x00, alpha1, alpha2, risk):
    """Reproduces the original p-value/log-score/filter logic, operating on 1D arrays."""
    
    eps = 1e-10
    g11 = np.rint(g11).astype(np.int64)
    x11 = np.rint(x11).astype(np.int64)
    g10 = np.rint(g10).astype(np.int64)
    x10 = np.rint(x10).astype(np.int64)
    g01 = np.rint(g01).astype(np.int64)
    x01 = np.rint(x01).astype(np.int64)
    g00 = np.rint(g00).astype(np.int64)
    x00 = np.rint(x00).astype(np.int64)
 
    p11 = np.asarray(cache.apply_hyge(g11, x11, risk), dtype=np.float64) + eps
    p10 = np.asarray(cache.apply_hyge(g10, x10, risk), dtype=np.float64) + eps
    p01 = np.asarray(cache.apply_hyge(g01, x01, risk), dtype=np.float64) + eps
    p00 = np.asarray(cache.apply_hyge(g00, x00, risk), dtype=np.float64) + eps
 
    q_min = np.minimum(np.minimum(p01, p10), p00)
 
    with np.errstate(divide='ignore', invalid='ignore'):
        log_out = -np.log10(p11 / q_min)
 
    fail = (p11 > alpha1) | (p10 <= alpha2) | (p01 <= alpha2) | (p00 <= alpha2)
    log_out[fail] = 0
    log_out[q_min == 0] = 0
    log_out[~np.isfinite(log_out)] = 0
    log_out[log_out < 0] = 0
    # NOTE: original also had `log_out[p11 == 0] = 0`, but p11 is always >= eps > 0 here
    # (it was dead code in the original too) so it's omitted.
    return log_out
 
def parallel_run(job_arg):
    """Function for running the computation job in parallel.
    
    Each worker computes its block of the interaction matrix and returns sparse 
    (row, col, value) triples for risk and protective scores.
    
    Args:
        job_arg (job_quota): _description_

    Returns:
        tuple: _description_
    """
    pheno = np.asarray(job_arg.pheno, dtype=np.float32).ravel()
    sx = np.asarray(job_arg.sx, dtype=np.float32)   # (n, b)
    sy = np.asarray(job_arg.sy, dtype=np.float32)   # (n, s)
    i1 = job_arg.i1
    i2 = job_arg.i2
    s = sy.shape[1]
    symmetric_flag = job_arg.symmetric
    alpha1 = job_arg.alpha1
    alpha2 = job_arg.alpha2
 
    population_size = job_arg.population_size
    case_size = job_arg.case_size
 
    print('in the parallel run: i1 = %d, i2 = %d' % (i1, i2))
    sys.stdout.flush()
 
    tempx = sx * pheno[:, None]
    sx_totals = sx.sum(axis=0)               # (b,)
    casex_totals = tempx.sum(axis=0)         # (b,)
    sy_totals = sy.sum(axis=0)               # (s,)
    caseY_totals = (sy * pheno[:, None]).sum(axis=0)  # (s,)
 
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
 
    cache = hc.HygeCache(population_size, case_size, job_arg.control_size)
 
    g11 = sx.T @ sy      # (b, s) - matmul #1
    x11 = tempx.T @ sy   # (b, s) - matmul #2
 
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
 
    risk_score = _score_from_counts(cache, g11_v, x11_v, g10_v, x10_v,
                                     g01_v, x01_v, g00_v, x00_v, alpha1, alpha2, True).astype(np.float32)
    prot_score = _score_from_counts(cache, g11_v, xp11_v, g10_v, xp10_v,
                                     g01_v, xp01_v, g00_v, xp00_v, alpha1, alpha2, False).astype(np.float32)
 
    nz_r = risk_score != 0
    nz_p = prot_score != 0
 
    risk_rows = out_rows[nz_r]
    risk_cols = out_cols[nz_r]
    risk_vals = risk_score[nz_r]
    prot_rows = out_rows[nz_p]
    prot_cols = out_cols[nz_p]
    prot_vals = prot_score[nz_p]
 
    return (
        risk_rows.astype(np.int64), risk_cols.astype(np.int64), risk_vals,
        prot_rows.astype(np.int64), prot_cols.astype(np.int64), prot_vals,
    )

def run(project_dir, model, alpha1, alpha2, n_workers, R):
    """Computes the interaction network for a single model (RR, RD, or DD) and saves it to a pickle file.

    Args:
        project_dir (str): _description_
        model (str): _description_
        alpha1 (float): _description_
        alpha2 (float): _description_
        n_workers (int): _description_
        R (int): _description_
    """
        
    print('computing interaction. R=' + str(R) + ' model = ' + model)
    output_name = f"{project_dir}/intermediate/ssM_mhygessi_{model}_R{R}.pkl"
    cluster_file = f"{project_dir}/intermediate/PlinkFile.cluster2"

    # loading and reading SNP data - skipped if caller already loaded these (e.g. looping over R)
    with open(f"{project_dir}/intermediate/SNPdataAD.pkl", "rb") as pkl_d:
        snpdata_d = pickle.load(pkl_d)
    with open(f"{project_dir}/intermediate/SNPdataAR.pkl", "rb") as pkl_r:
        snpdata_r = pickle.load(pkl_r)

    pheno = snpdata_r.pheno
    dataR = snpdata_r.data
    dataD = snpdata_d.data

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
            # single deterministic permutation per R, instead of discarding R-1 throwaway
            # permutations from a shared RNG stream.
            rng = np.random.RandomState(66754 * R)
            permuted_idx = rng.permutation(population_size)
            pheno = pheno[permuted_idx]
        else:
            pheno = wrand.withinclassrand(R, cluster_file, f"{project_dir}/intermediate/SNPdataAD.pkl")

    case_size = int(np.count_nonzero(pheno))
    control_size = population_size - case_size

    sx_full = np.ascontiguousarray(datai, dtype=np.float32)
    sy_full = np.ascontiguousarray(dataj, dtype=np.float32)
    pheno = np.asarray(pheno, dtype=np.float32).ravel()
    s = sx_full.shape[1]

    ## dividing sx for parallel computing (unchanged balancing logic - earlier chunks have
    ## fewer lower-triangle entries for symmetric models, so chunk boundaries are sqrt-spaced)
    idx = [0]
    if model == 'RR' or model == 'DD':
        share = s * s / n_workers
        for i in range(n_workers):
            if i == n_workers - 1:
                idx.append(s)
            else:
                idx.append(math.floor(math.sqrt(idx[i] * idx[i] + share)))
    else:
        share = math.floor(s / n_workers)
        for i in range(n_workers):
            if i == n_workers - 1:
                idx.append(s)
            else:
                idx.append(idx[i] + share)

    job_args = []
    for i in range(n_workers):
        job_arg = job_quota(population_size, case_size, control_size)
        job_arg.alpha1 = alpha1
        job_arg.alpha2 = alpha2
        job_arg.symmetric = (model == 'RR' or model == 'DD')
        job_arg.i1 = idx[i]
        job_arg.i2 = idx[i + 1]
        job_arg.sx = np.ascontiguousarray(sx_full[:, job_arg.i1:job_arg.i2])
        job_arg.sy = sy_full
        job_arg.pheno = pheno
        job_args.append(job_arg)

    pool = mp.Pool(processes=n_workers)
    results = pool.map(parallel_run, job_args)
    pool.close()
    pool.join()

    risk_rows = np.concatenate([r[0] for r in results]) if results else np.array([], dtype=np.int64)
    risk_cols = np.concatenate([r[1] for r in results]) if results else np.array([], dtype=np.int64)
    risk_vals = np.concatenate([r[2] for r in results]) if results else np.array([], dtype=np.float64)
    prot_rows = np.concatenate([r[3] for r in results]) if results else np.array([], dtype=np.int64)
    prot_cols = np.concatenate([r[4] for r in results]) if results else np.array([], dtype=np.int64)
    prot_vals = np.concatenate([r[5] for r in results]) if results else np.array([], dtype=np.float64)

    result_risk = scipy.sparse.coo_matrix((risk_vals, (risk_rows, risk_cols)), shape=(s, s)).tocsr()
    result_protective = scipy.sparse.coo_matrix((prot_vals, (prot_rows, prot_cols)), shape=(s, s)).tocsr()

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

    network = InteractionNetwork.InteractionNetwork(result_risk, result_protective, None, None)

    with open(output_name, 'wb') as final:
        pickle.dump(network, final, protocol=pickle.HIGHEST_PROTOCOL)

def _combine_max(a, b, c):
    """Find the max score and model-id for each SNP pair across the three models.
    
    Elementwise max of three sparse matrices (a=RR, b=DD, c=RD) plus a model-id matrix
    (1=RR, 2=DD, 3=RD), restricted to the union of nonzero positions - all three inputs are
    nonnegative -log10(p) scores, so a+b+c is zero exactly where all three are zero, and this
    sum's sparsity pattern is a cheap way to get that union without ever going dense.
    
    Args:
        a (scipy.sparse.csr_matrix): The RR model results.
        b (scipy.sparse.csr_matrix): The DD model results.
        c (scipy.sparse.csr_matrix): The RD model results.

    Returns:
        tuple: A tuple containing the combined max matrix and the model-id matrix.
    """
    
    support = (a + b + c).tocoo()
    rows, cols = support.row, support.col
    if len(rows) == 0:
        shape = a.shape
        empty = scipy.sparse.csr_matrix(shape)
        return empty, empty.copy()
 
    a_v = np.asarray(a[rows, cols]).ravel()
    b_v = np.asarray(b[rows, cols]).ravel()
    c_v = np.asarray(c[rows, cols]).ravel()
 
    id_v = np.full(len(rows), 2, dtype=np.float64)
    id_v[a_v > b_v] = 1
    ab_max = np.maximum(a_v, b_v)
    id_v[ab_max < c_v] = 3
    max_v = np.maximum(ab_max, c_v)
 
    max_mat = scipy.sparse.coo_matrix((max_v, (rows, cols)), shape=a.shape).tocsr()
    id_mat = scipy.sparse.coo_matrix((id_v, (rows, cols)), shape=a.shape).tocsr()
    return max_mat, id_mat


def combine(project_dir, alpha1, alpha2, n_workers, R):
    """Run the three models (RR, RD, DD) and combine their results into a single InteractionNetwork.

    Args:
        project_dir (str): _description_
        alpha1 (float): _description_
        alpha2 (float): _description_
        n_workers (int): _description_
        R (int): _description_
    """
    
    run(project_dir, 'RR', alpha1, alpha2, n_workers, R)
    run(project_dir, 'RD', alpha1, alpha2, n_workers, R)
    run(project_dir, 'DD', alpha1, alpha2, n_workers, R)

    ## load results for 3 models
    with open(f"{project_dir}/intermediate/ssM_mhygessi_RR_R{R}.pkl", 'rb') as rr_file:
        rr_network = pickle.load(rr_file)
    with open(f"{project_dir}/intermediate/ssM_mhygessi_RD_R{R}.pkl", 'rb') as rd_file:
        rd_network = pickle.load(rd_file)
    with open(f"{project_dir}/intermediate/ssM_mhygessi_DD_R{R}.pkl", 'rb') as dd_file:
        dd_network = pickle.load(dd_file)

    risk_max, risk_max_id = _combine_max(rr_network.risk, rd_network.risk, dd_network.risk)
    protective_max, protective_max_id = _combine_max(rr_network.protective, rd_network.protective, dd_network.protective)

    network = InteractionNetwork.InteractionNetwork(risk_max, protective_max, risk_max_id, protective_max_id)
    
    output_name = f"{project_dir}/intermediate/ssM_mhygessi_combined_R{R}.pkl"
    with open(output_name, 'wb') as final:
        pickle.dump(network, final, protocol=pickle.HIGHEST_PROTOCOL)

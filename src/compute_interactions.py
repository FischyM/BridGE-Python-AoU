import math, sys
from datetime import datetime
from os import path

import numpy as np
import pandas as pd
from scipy.sparse import coo_array, csr_array

from src import hyge_cache as hc

# dtype of the genotype matrices and pair counts in run_single. All counts are whole numbers
# no larger than the number of samples, so float32 is exact below 2**24 (~16.7M) samples,
# and its matmuls are about 2x faster than float64 on typical x86 CPUs (AVX2/AVX-512).
COUNT_DTYPE = np.float32


def run_pheno_perm(rng, cluster_file, pheno, iid):
    """Permute phenotypes within genetic similarity clusters.

    Every cluster keeps its number of cases and controls; only which samples
    in the cluster carry the case label changes.

    Args:
        rng (numpy.random.Generator): Random number generator
        cluster_file (str): TSV with columns IID, PHENO, CLUSTER
        pheno (array): 0/1 phenotypes in snp_data.npz row order
        iid (array): sample IDs in snp_data.npz row order

    Returns:
        phenonew (np.ndarray): permuted phenotypes, same order as `pheno`
    """
    # load cluster file
    cluster_data = pd.read_csv(cluster_file, sep=r"\s+", comment="#", dtype={'IID': str})
    if not cluster_data['IID'].is_unique:
        sys.exit(f"duplicate IIDs in {cluster_file}")

    # align cluster file rows to the snp_data order: row i of the result is sample iid[i]
    idx = pd.Index(cluster_data['IID']).get_indexer(iid)
    if (idx < 0).any():
        sys.exit(f"{(idx < 0).sum()} sample(s) in snp_data.npz absent from {cluster_file}, "
                 f"e.g. {iid[idx < 0][0]}")
    labels = cluster_data['CLUSTER'].to_numpy()[idx]
    if not np.array_equal(cluster_data['PHENO'].to_numpy()[idx], pheno):
        sys.exit(f"phenotypes in {cluster_file} do not match snp_data.npz")

    # shuffle phenotypes among the samples of each cluster
    phenonew = pheno.copy()
    for cl in np.unique(labels):
        ind = np.flatnonzero(labels == cl)
        phenonew[ind] = pheno[rng.permutation(ind)]

    return phenonew

def save_sparse_array(path, arrays, compressed=False):
    out = {}
    for name, A in arrays.items():
        A = A.tocsr()
        out[f"{name}__data"] = A.data
        out[f"{name}__indices"] = A.indices
        out[f"{name}__indptr"] = A.indptr
        out[f"{name}__shape"] = np.array(A.shape)
    (np.savez_compressed if compressed else np.savez)(path, allow_pickle=False, **out)

def load_sparse_array(path):
    with np.load(path) as f:
        names = {k.split("__")[0] for k in f.files}
        return {n: csr_array((f[f"{n}__data"], f[f"{n}__indices"], f[f"{n}__indptr"]),
                                shape=tuple(f[f"{n}__shape"])) for n in names}

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

def run_single(project_dir, model, alpha1, alpha2, n_jobs, n_workers, pool, R, seed, compressed=False):
    """Computes the interaction network for a single model (RR, RD, or DD) and saves it to a pickle file.

    Args:
        project_dir (str): _description_
        model (str): _description_
        alpha1 (float): _description_
        alpha2 (float): _description_
        n_jobs (int): _description_
        n_workers (int): _description_
        R (int): _description_
    """
    t1 = datetime.now()
    print(f'    R={R}, model={model}', end="", flush=True)

    # loading and reading SNP data
    # read SNP data and convert to dominant and recessive coding
    snp_data = np.load(f"{project_dir}/intermediate/snp_data.npz")
    pheno = snp_data['pheno']
    G = snp_data['data']  # no missing data (-9), int8, samples x SNPs (Fortran-ordered as saved by data_tools)

    # # filter out SNPs that didn't make it through SNP to pathway mapping
    # with open(f"{project_dir}/intermediate/snp_pathway_mapping.pkl", "rb") as f:
    #     snp_pathway_mapping: snpsetclass = pickle.load(f)
        
    # # subset and reorder the SNP data to match the order of SNPs in the snp to pathway mapping
    # varid_subset = snp_pathway_mapping.spmatrix.index
    # sorter = np.argsort(snp_data.varid)
    # mapper_subset_to_full_varid = sorter[np.searchsorted(snp_data.varid, varid_subset, sorter=sorter)]
    # G_subset = snp_data.data[:, mapper_subset_to_full_varid]
    # print(f"n SNPs={len(snp_pathway_mapping.spmatrix.index)}", end="", flush=True)
    
    # convert genotype data to dominant and recessive coding with a simple mapping.
    # The maps are COUNT_DTYPE, so the lookup directly gives the matrices used by the
    # matmuls (no int64 intermediate). Only the coding(s) this model needs are built, and
    # RR/DD use one matrix for both sides. The lookup keeps G's memory order (Fortran), so
    # column slices of sx_full/sy_full below are contiguous views.
    dom_map = np.array([0, 1, 1], dtype=COUNT_DTYPE)  # dominant:  0->0, 1->1, 2->1
    rec_map = np.array([0, 0, 1], dtype=COUNT_DTYPE)  # recessive: 0->0, 1->0, 2->1

    symmetric_flag = (model == 'RR' or model == 'DD')
    if model == 'RR':
        sx_full = rec_map[G]
        sy_full = sx_full
    elif model == 'DD':
        sx_full = dom_map[G]
        sy_full = sx_full
    else:
        sx_full = rec_map[G]
        sy_full = dom_map[G]
    del G

    population_size = pheno.shape[0]
    if population_size >= 2 ** (np.finfo(COUNT_DTYPE).nmant + 1):
        sys.exit(f"{population_size} samples is too many for exact counts in {np.dtype(COUNT_DTYPE).name}; "
                 f"set COUNT_DTYPE = np.float64 in compute_interactions")
    # shuffle phenotypes if R != 0
    if R > 0:
        # single deterministic permutation per R
        rng = np.random.default_rng(seed * R) if seed != 0 else np.random.default_rng(R)
        cluster_file = f"{project_dir}/raw/plink.cluster.tsv"
        if not path.exists(cluster_file):
            permuted_idx = rng.permutation(population_size)
            pheno = pheno[permuted_idx]
        else:
            pheno = run_pheno_perm(rng, cluster_file, pheno, snp_data['iid'])

    case_size = int(np.count_nonzero(pheno))
    control_size = population_size - case_size
    cache = hc.HygeCache(population_size, case_size, control_size)

    pheno = np.asarray(pheno, dtype=COUNT_DTYPE).ravel()
    s = sx_full.shape[1]

    # y-side totals don't change between chunks, so compute them once.
    # pheno @ sy_full equals (sy_full * pheno[:, None]).sum(axis=0) without an N x s temporary.
    sy_totals = sy_full.sum(axis=0)
    caseY_totals = pheno @ sy_full

    # dividing sx for parallel computing (unchanged balancing logic - earlier chunks have
    # fewer lower-triangle entries for symmetric models, so chunk boundaries are sqrt-spaced)
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
        i1 = idx[i]
        i2 = idx[i + 1]
        
        sx = np.ascontiguousarray(sx_full[:, i1:i2], dtype=COUNT_DTYPE)
        s = sy_full.shape[1]
        tempx = sx * pheno[:, None]
        sx_totals = sx.sum(axis=0)
        casex_totals = tempx.sum(axis=0)

        # For symmetric models (RR/DD) we only need the strict lower triangle of the full
        # s x s matrix - everything else is filled in by mirroring below. This band's rows are
        # global SNPs i1..i2-1, so no needed column is >= i2 and the y side is cut to [:, :i2].
        # RD needs every (row, col) entry of the band.
        b = i2 - i1
        ncol = i2 if symmetric_flag else s
        sy = sy_full[:, :ncol]   # view, no copy

        # g11/x11 are the only counts that need a matmul. The others are derived from them and
        # the row totals (sx_totals, casex_totals) / column totals (sy_totals, caseY_totals)
        # entry by entry, so they are computed only for the entries kept, not on full blocks.
        # ri/ci line the totals up with g11/x11:
        #   RR/DD: g11/x11 are 1D over the lower-triangle entries -> ri/ci are each entry's
        #          local row and global column, and the totals are gathered at them
        #   RD:    g11/x11 are (b, ncol) blocks -> ri/ci broadcast the totals as a column / row
        if symmetric_flag:
            ri, ci = np.tril_indices(b, i1 - 1, ncol)
            g11 = (sx.T @ sy)[ri, ci]      # matmul #1; the (b, ncol) block is freed after the gather
            x11 = (tempx.T @ sy)[ri, ci]   # matmul #2
        else:
            ri, ci = np.s_[:, None], np.s_[None, :]
            g11 = sx.T @ sy      # (b, ncol) - matmul #1
            x11 = tempx.T @ sy   # (b, ncol) - matmul #2

        g10 = sx_totals[ri] - g11
        g01 = sy_totals[ci] - g11
        g00 = population_size - sx_totals[ri] - sy_totals[ci] + g11

        x10 = casex_totals[ri] - x11
        x01 = caseY_totals[ci] - x11
        x00 = case_size - casex_totals[ri] - caseY_totals[ci] + x11

        # helper_score_from_counts takes 1D arrays; for RD this flattens the blocks (views, no copy)
        g11, g10, g01, g00 = g11.ravel(), g10.ravel(), g01.ravel(), g00.ravel()
        x11, x10, x01, x00 = x11.ravel(), x10.ravel(), x01.ravel(), x00.ravel()

        # -log10(p-values) for risk, with alpha1/alpha2 filtering
        risk_score = helper_score_from_counts(cache, g11, x11, g10, x10,
                                        g01, x01, g00, x00, alpha1, alpha2, True, pool, n_workers)

        # xp_* = g_* - x_* because pheno_res = 1 - pheno is linear in the counts above.
        # x_* aren't needed after the risk score, so xp_* are written into their memory.
        xp11 = np.subtract(g11, x11, out=x11)
        xp10 = np.subtract(g10, x10, out=x10)
        xp01 = np.subtract(g01, x01, out=x01)
        xp00 = np.subtract(g00, x00, out=x00)
        del x11, x10, x01, x00   # these names would now refer to the xp_* values

        # -log10(p-values) for protective, with alpha1/alpha2 filtering
        prot_score = helper_score_from_counts(cache, g11, xp11, g10, xp10,
                                        g01, xp01, g00, xp00, alpha1, alpha2, False, pool, n_workers)

        # (row, col) coordinates are only built for the nonzero scores
        nz_r = risk_score != 0
        if symmetric_flag:
            risk_rows, risk_cols = ri[nz_r] + i1, ci[nz_r]
        else:
            flat_r = np.flatnonzero(nz_r)
            risk_rows, risk_cols = flat_r // ncol + i1, flat_r % ncol
        risk_vals = risk_score[nz_r]

        nz_p = prot_score != 0
        if symmetric_flag:
            prot_rows, prot_cols = ri[nz_p] + i1, ci[nz_p]
        else:
            flat_p = np.flatnonzero(nz_p)
            prot_rows, prot_cols = flat_p // ncol + i1, flat_p % ncol
        prot_vals = prot_score[nz_p]

        results.append((risk_rows, risk_cols, risk_vals, prot_rows, prot_cols, prot_vals))

        # free this chunk's arrays before the next chunk allocates its own
        del sx, tempx, ri, ci, g11, g10, g01, g00, xp11, xp10, xp01, xp00
        del risk_score, prot_score, nz_r, nz_p
    del sx_full, sy_full

    risk_rows = np.concatenate([ r[0] for r in results ])
    risk_cols = np.concatenate([ r[1] for r in results ])
    risk_vals = np.concatenate([ r[2] for r in results ])
    prot_rows = np.concatenate([ r[3] for r in results ])
    prot_cols = np.concatenate([ r[4] for r in results ])
    prot_vals = np.concatenate([ r[5] for r in results ])

    result_risk: csr_array = coo_array((risk_vals, (risk_rows, risk_cols)), shape=(s, s)).tocsr()
    result_protective: csr_array = coo_array((prot_vals, (prot_rows, prot_cols)), shape=(s, s)).tocsr()

    if model == 'RR' or model == 'DD':
        # only the strict lower triangle was computed - mirror it. Upper triangle of
        # result_risk is all zero by construction, so addition doesn't double-count.
        result_risk = result_risk + result_risk.T
        result_protective = result_protective + result_protective.T
    else:
        # RD: full matrix was computed (no triangle dedup) but isn't symmetric by
        # construction - take elementwise max with transpose, zero the diagonal.
        result_risk = result_risk.maximum(result_risk.T)  # type: ignore
        result_risk.setdiag(0)
        result_risk.eliminate_zeros()
        
        result_protective = result_protective.maximum(result_protective.T)  # type: ignore
        result_protective.setdiag(0)
        result_protective.eliminate_zeros()

    output_name = f"{project_dir}/intermediate/snp_interaction_networks-{model}-R{R}.npz"
    save_sparse_array(output_name, {
        'risk': result_risk,
        'protective': result_protective
    }, compressed=compressed)

    print(f" - {str(datetime.now() - t1).split('.')[0]}", flush=True)

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

def run_combined(project_dir, alpha1, alpha2, n_jobs, n_workers, pool, R, seed, compressed=False):
    """Run the three models (RR, RD, DD) and combine their results into a single InteractionNetwork.

    Args:
        project_dir (str): _description_
        alpha1 (float): _description_
        alpha2 (float): _description_
        n_jobs (int): _description_
        n_workers (int): _description_
        pool (multiprocessing.Pool): The process pool to use for parallel processing.
        R (int): _description_
        seed (int): _description_
    """
    run_single(project_dir, 'RR', alpha1, alpha2, n_jobs, n_workers, pool, R, seed, compressed=compressed)
    run_single(project_dir, 'RD', alpha1, alpha2, n_jobs, n_workers, pool, R, seed, compressed=compressed)
    run_single(project_dir, 'DD', alpha1, alpha2, n_jobs, n_workers, pool, R, seed, compressed=compressed)

    ## load results for 3 models
    rr_network = load_sparse_array(f"{project_dir}/intermediate/snp_interaction_networks-RR-R{R}.npz")
    rd_network = load_sparse_array(f"{project_dir}/intermediate/snp_interaction_networks-RD-R{R}.npz")
    dd_network = load_sparse_array(f"{project_dir}/intermediate/snp_interaction_networks-DD-R{R}.npz")

    risk_max, risk_max_id = combine_max(rr_network['risk'], rd_network['risk'], dd_network['risk'])
    protective_max, protective_max_id = combine_max(rr_network['protective'], rd_network['protective'], dd_network['protective'])

    output_name = f"{project_dir}/intermediate/snp_interaction_networks-combined-R{R}.npz"
    save_sparse_array(output_name, {
        'risk': risk_max,
        'protective': protective_max,
        'risk_max_id': risk_max_id,
        'protective_max_id': protective_max_id
    }, compressed=compressed)
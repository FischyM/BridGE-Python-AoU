import math
import pickle
import multiprocessing as mp
from datetime import datetime

import numpy as np
from scipy.sparse import csr_array, issparse
from scipy.stats import chi2, norm, rankdata

from classes import Stats, GenStats
np.seterr(divide='ignore', invalid='ignore')

# genstats() computes BPM/WPM/PATH statistics. Can be run parallel.
#
# REFACTOR NOTES (mirrors the approach taken in matrix_operations_par.py):
#   - The interaction network is kept as a scipy.sparse.csr_array end to end. Nothing in this
#     module ever materializes a dense (s x s) array, and the old dense sharedctypes.RawArray
#     (plus the np.copy of it made inside every worker) is gone. This is the dominant RAM win:
#     the previous code held one dense s x s copy in the parent plus one per worker.
#   - All of the "sum a submatrix block" loops (bpmgi, wpmgi, bpmsum, wpmsum, and the same
#     sums repeated inside every permutation) are replaced by the indicator-matrix identity
#         sum(mm[ind1, :][:, ind2]) == u1.T @ mm @ u2   with u1/u2 0-1 indicator columns
#     evaluated for many BPMs at once as (mm @ U2).multiply(U1).sum(axis=0). One sparse
#     product now does what was a Python loop over bpm_size fancy-indexed slices. cyadd is
#     no longer needed.
#   - Permutations no longer permute the network. Permuting the columns of mm and then summing
#     a block is identical to leaving mm alone and permuting the *rows of the indicator matrix*
#     on the column side, so each permutation costs one sparse product instead of a rebuild of
#     mm plus bpm_size slice-and-sum calls. The permutation SEQUENCE is reproduced exactly -
#     see snp_permutation_parallel().
#   - PATH degree used to call mannwhitneyu(dist_in, dist_out) once per pathway per permutation
#     on length-s dense vectors. Because dist_in/dist_out always partition the same vector
#     (sumMM, or a permutation of it), the midranks and the tie correction can be computed once
#     and reused: the per-pathway statistic is then just a rank sum, i.e. P.T @ ranks. This is
#     exact, not an approximation.
#   - call_chi2 is a closed-form vectorized 2x2 chi-square instead of bpm_size calls into
#     scipy.stats.chi2_contingency.
#   - The old `tr` list was tested with `if i in tr` inside a loop over bpm_size, i.e. O(n^2).
#     It is now the boolean mask `tr_mask`, derived directly from bpm['ind1size']/['ind2size'].
#   - Dead code removed: pre_comp1/pre_comp2/xs1/xs2 (built, shared, read by the workers, then
#     never used - and xs2 was built from pre_comp1 by copy/paste), the unused `arr`, the bare
#     `wpm_local_pv` expression statement, the datetime/psutil timing scaffolding, and the
#     unused `denisty_wpm` typo'd local.
#   - n_jobs / n_workers: n_jobs splits work into that many *sequential* subsets to cap peak
#     RAM, n_workers is the pool width. In the permutation stage n_jobs subsets the BPMs, not
#     the permutations: the permuted index q is produced exactly as the original produced it
#     and then fed to each subset in turn, so peak RAM falls with n_jobs while the permutation
#     sequence is untouched.
#   - Worker data sharing is by fork() copy-on-write: publish_shared() installs the read-only
#     sparse structures on the module before the pool is created. This is the same platform
#     assumption the previous sharedctypes/RawArray code already made.
#
# BEHAVIOUR NOTES (differences from the old file):
#   - Empirical p-values are REPRODUCIBLE: for a given snpPerms they do not depend on n_workers
#     or n_jobs, so a result can be reproduced on any machine. The original did not have this
#     property - it seeded one RNG stream per worker (PERM_SEED + proc*1000, each of length
#     snpPerms/n_workers), so changing the worker count changed the set of permutations drawn
#     and hence the p-values.
#     The canonical stream here is the original's n_workers=1 stream: one legacy MT19937 seeded
#     PERM_SEED, snpPerms draws, COMPOUNDING (the original reassigned mmtmp, so iteration k acts
#     on the composition of every draw so far). So this matches an original run at n_workers=1
#     exactly, and will differ from an original run at any other worker count -- but those runs
#     were not reproducible to begin with.
#   - Tables that are not valid contingency tables return p = 1 from call_chi2 rather than
#     raising (as chi2_contingency did) or reporting spurious significance. This covers a zero
#     marginal and, importantly, any negative count - see call_chi2().
#   - density_wpm for *non-kept* WPMs still carries the value computed from the binarized
#     network even when binary_flag is False. That is what the original did (the non-binary
#     branch reassigned a misspelled local) and downstream code may depend on it, so it is
#     preserved deliberately rather than "fixed".
#   - ind2keep_bpm follows the original exactly, including the non-binary branch's narrowing
#     to (bpm_local >= -log10(0.05)) and the recomputation of bpmind1/bpmind2 from that
#     narrowed mask. Note the thresholds here differ from the serial genstats.py sibling
#     (that one uses -log10(0.05) and `> minPath` at the chi2 stage); this file follows
#     genstats_perm.py: -log10(0.1) and `>= minPath`.
#
# INPUTS:
#   ssmFile: Interaction networks file in the pickle format.
#   bpmfile: files containing SNP ids for BPM/WPMs in pickle format.
#   binary_flag: If True, interaction scores are binarized for computing BPM/WPM/PATH significances
#   snpPerms: Number of snp permutations used for computing empirical p-values
#   minPath: minimum size for a pathway to be considered as WPM and in BPM.
#   n_jobs: number of sequential chunks the BPM passes are split into (lower peak RAM)
#   n_workers: number of parallel cpu cores the program shoud use (higher throughput)
#
# OUTPUTS:
#   genstats_<ssmFile without extension>.pkl - This pickle file contains a GenstasOut class, which itself contains 2 Stats class oject
#       - protective_stats: Statistics for protective network including ranksum scores,empirical p-values, expected density for BPM/WPMs
#       - risk_stats: Statistics for risk network including ranksum scores,empirical p-values, expected density for BPM/WPMs


PERM_SEED = 349898398

class perm_args:
    def __init__(self, skip, count):
        self.skip = skip
        self.count = count

class par_rank_args:
    def __init__(self, id, rows):
        self.id = id
        self.rows = np.asarray(rows, dtype=np.int64)


# ---------------------------------------------------------------------------
# worker data sharing
# ---------------------------------------------------------------------------
# Replaces init_worker()/init_worker_perm(). Called in the *parent* before the pool is
# created; children inherit the objects through fork() copy-on-write, so nothing large is
# pickled per job and nothing is copied per worker.

_SHARED = {}

def publish_shared(**kwargs):
    _SHARED.update(kwargs)

def clear_shared():
    _SHARED.clear()


# ---------------------------------------------------------------------------
# sparse helpers
# ---------------------------------------------------------------------------

def as_sparse(mm):
    """Coerce an interaction network to a float64 csr_array with no stored zeros."""
    if issparse(mm):
        out = csr_array(mm)
    else:
        out = csr_array(np.asarray(mm, dtype=np.float64))
    if out.data.dtype != np.float64:
        out.data = out.data.astype(np.float64)
    out.eliminate_zeros()
    return out

def binarize(mm, threshold):
    """Sparse equivalent of `mm[mm>=threshold] = 1; mm[mm<1] = 0`."""
    out = mm.copy()
    out.data = (out.data >= threshold).astype(np.float64)
    out.eliminate_zeros()
    return out

def sparse_quantile(mm, q):
    """np.quantile(dense_mm, q) computed from the stored values alone.

    Assumes every stored value is > 0, which holds for these -log10 score matrices.
    """
    total = int(mm.shape[0]) * int(mm.shape[1])
    data = np.sort(mm.data)
    n_zero = total - data.size

    def value_at(k):
        return 0.0 if k < n_zero else float(data[k - n_zero])

    pos = q * (total - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    v_lo = value_at(lo)
    return v_lo + (pos - lo) * (value_at(hi) - v_lo)

def indicator_matrix(index_lists, s):
    """(s x len(index_lists)) 0-1 csr_array; column j marks the SNPs in index_lists[j]."""
    n = len(index_lists)
    if n == 0:
        return csr_array((s, 0), dtype=np.float64)
    parts = [np.asarray(x, dtype=np.int64).ravel() for x in index_lists]
    lengths = np.fromiter((p.size for p in parts), dtype=np.int64, count=n)
    if lengths.sum() == 0:
        return csr_array((s, n), dtype=np.float64)
    rows = np.concatenate(parts)
    cols = np.repeat(np.arange(n, dtype=np.int64), lengths)
    data = np.ones(rows.size, dtype=np.float64)
    return csr_array((data, (rows, cols)), shape=(s, n))

def block_sums(mm, u1, u2):
    """Column-wise sum(mm[ind1_j, :][:, ind2_j]) for paired indicator columns u1/u2."""
    if u1.shape[1] == 0:
        return np.zeros(0)
    return np.asarray((mm @ u2).multiply(u1).sum(axis=0)).ravel()

def tiled_block_sums(mm, u1_tiles, u2_tiles, out, row_perm=None):
    """block_sums over pre-tiled indicator columns, optionally permuting the column side.

    row_perm is applied to the rows of the u2 tiles, which is equivalent to permuting the
    columns of mm (see refactor notes) but costs a reindex instead of rebuilding mm.
    """
    off = 0
    for u1t, u2t in zip(u1_tiles, u2_tiles):
        k = u1t.shape[1]
        u2p = u2t if row_perm is None else u2t[row_perm, :]
        out[off:off + k] = block_sums(mm, u1t, u2p)
        off += k
    return out

def tile_indicators(u, n_parts):
    """Split the indicator columns into n_parts subsets of BPMs.

    This is what n_jobs controls in the permutation stage: each sparse product then handles
    1/n_parts of the BPMs, so the (s x subset) intermediate - the peak allocation - shrinks
    proportionally. The permutation itself is untouched; the same q feeds every subset.
    """
    k = u.shape[1]
    if k == 0:
        return []
    tile = max(1, math.ceil(k / max(int(n_parts), 1)))
    return [u[:, lo:lo + tile] for lo in range(0, k, tile)]


# ---------------------------------------------------------------------------
# statistics helpers
# ---------------------------------------------------------------------------

def call_chi2(table):
    """Vectorized 2x2 chi-square, no continuity correction.

    Input format (per row): f11(bpm interactions) - f10(non-bpm interactions) -
    f01(bpm non-interactions) - f00(non-bpm non-interactions), i.e. [[f11,f10],[f01,f00]].
    Matches scipy.stats.chi2_contingency(obs, correction=False) for well-formed tables.

    Rows that are not a valid contingency table return p = 1, i.e. "no evidence of
    enrichment", which is the only defensible answer for a test of whether interactions
    differ from expectation. Three cases:
      - f11 == 0: the original short-circuited to p = 1 here, so this is unchanged.
      - a zero row/column marginal: chi2_contingency raised ValueError; the closed form
        divides by zero and yields inf/nan. Only reachable if a region is fully saturated.
      - ANY NEGATIVE COUNT: chi2_contingency also raised here. The closed form would happily
        return a tiny p-value (a negative cell inflates the |ad - bc| numerator while the
        denominator stays positive), reporting a malformed table as maximally significant.
        That is meaningless, so these are forced to p = 1 and reported. A negative count
        signals an upstream size/convention bug - e.g. wpmnotgi = wpmsize - wpmgi going
        negative if wpmsize counts unordered pairs while wpmgi (a full block sum) counts
        ordered ones. bpmnotgi is clamped upstream; wpmnotgi is not.
    """
    table = np.asarray(table, dtype=np.float64)
    a, b, c, d = table[:, 0], table[:, 1], table[:, 2], table[:, 3]
    n = a + b + c + d
    with np.errstate(divide='ignore', invalid='ignore'):
        stat = n * (a * d - b * c) ** 2 / ((a + b) * (c + d) * (a + c) * (b + d))
    results = chi2.sf(stat, 1)

    negative = np.any(table < 0, axis=1)
    invalid = ~np.isfinite(stat) | (a == 0) | negative
    results[invalid] = 1.0

    n_neg = int(np.count_nonzero(negative))
    if n_neg:
        print(f"\twarning: {n_neg} chi2 table row(s) contained a negative count and were set "
              f"to p=1; check the size vs interaction-count pair conventions upstream")
    return results

def tie_sum(values):
    """sum(t^3 - t) over tie groups, in float64 to survive very large groups."""
    if values.size == 0:
        return 0.0
    counts = np.unique(values, return_counts=True)[1]
    counts = counts[counts > 1].astype(np.float64)
    return float(np.sum(counts ** 3 - counts))

def mw_greater(rank_sum_in, n_in, n_out, ties):
    """Normal-approximation Mann-Whitney p-value, alternative='greater', continuity corrected.

    Same formula as the original ranksum() helper and as
    scipy.stats.mannwhitneyu(..., use_continuity=True, alternative='greater'), but driven by a
    precomputed midrank sum so the ranking can be shared across pathways/permutations.
    Accepts scalars or arrays.
    """
    n_in = np.asarray(n_in, dtype=np.float64)
    n_out = np.asarray(n_out, dtype=np.float64)
    n = n_in + n_out
    u = np.asarray(rank_sum_in, dtype=np.float64) - n_in * (n_in + 1.0) / 2.0
    with np.errstate(divide='ignore', invalid='ignore'):
        var = n_in * n_out * (n + 1.0 - ties / (n * (n - 1.0))) / 12.0
        z = (u - n_in * n_out / 2.0 - 0.5) / np.sqrt(var)
    p = norm.sf(z)
    p = np.where(np.isfinite(p) & (var > 0), p, 1.0)
    return p if p.ndim else float(p)

def mw_greater_sparse(nz_in, n_in, nz_out, n_out):
    """Mann-Whitney for two groups whose unstored entries are all zeros.

    nz_in/nz_out are the *stored* (nonzero, positive) values; n_in/n_out are the true group
    sizes. Zeros form one big tie group at the bottom of the ranking, so the statistic is
    exact without ever materializing the zeros.
    """
    z_in = float(n_in) - nz_in.size
    z_out = float(n_out) - nz_out.size
    z_tot = z_in + z_out

    pooled = np.concatenate((nz_in, nz_out)) if nz_out.size else nz_in
    if pooled.size:
        ranks = rankdata(pooled) + z_tot
        rank_sum_in = float(ranks[:nz_in.size].sum())
    else:
        rank_sum_in = 0.0
    rank_sum_in += z_in * (z_tot + 1.0) / 2.0

    ties = tie_sum(pooled)
    if z_tot > 1:
        ties += z_tot ** 3 - z_tot
    return mw_greater(rank_sum_in, n_in, n_out, ties)

def ranksum(x, y):
    """Kept for API compatibility: x is in the bpm, y is out of the bpm."""
    pooled = np.concatenate((y, x))
    ranks = rankdata(pooled)
    return mw_greater(float(ranks[y.shape[0]:].sum()), x.shape[0], y.shape[0], tie_sum(pooled))

def split_indices(rows, n_parts):
    """Split into at most n_parts non-empty contiguous pieces."""
    return [part for part in np.array_split(np.asarray(rows), max(int(n_parts), 1)) if part.size]


# ---------------------------------------------------------------------------
# parallel workers
# ---------------------------------------------------------------------------

def bpm_chi2_parallel(job_arg):
    """bpmgi / path1bggi / path2bggi for a slice of BPMs (binarized network)."""
    mm = _SHARED['mm']
    sumMM = _SHARED['sumMM']
    ind1 = _SHARED['ind1']
    ind2 = _SHARED['ind2']
    keep = _SHARED['tr_keep']

    rows = job_arg.rows
    bpmgi = np.zeros(rows.size)
    path1bggi = np.zeros(rows.size)
    path2bggi = np.zeros(rows.size)

    valid = keep[rows]
    if valid.any():
        sel = rows[valid]
        u1 = indicator_matrix([ind1[i] for i in sel], mm.shape[0])
        u2 = indicator_matrix([ind2[i] for i in sel], mm.shape[0])
        gi = block_sums(mm, u1, u2)
        bpmgi[valid] = gi
        path1bggi[valid] = np.asarray(u1.T @ sumMM).ravel() - gi
        path2bggi[valid] = np.asarray(u2.T @ sumMM).ravel() - gi
    return bpmgi, path1bggi, path2bggi

def parallel_ranksum(job_arg):
    """bpmsum + ranksum p-value for a slice of the kept BPMs (non-binary network)."""
    mm = _SHARED['mm']
    bpmind1 = _SHARED['bpmind1']
    bpmind2 = _SHARED['bpmind2']
    s = mm.shape[1]

    rows = job_arg.rows
    bpmsum_tmp = np.zeros(rows.size)
    bpm_local_tmp = np.ones(rows.size)
    mask = np.zeros(s, dtype=bool)

    for k, i in enumerate(rows):
        id1 = np.asarray(bpmind1[i], dtype=np.int64)
        id2 = np.asarray(bpmind2[i], dtype=np.int64)
        if id1.size < 5 or id2.size < 5:
            continue  # bpmsum 0 / p-value 1, as the old `tr` bookkeeping did

        block = mm[id1, :]
        mask[id2] = True
        inside = mask[block.indices]
        mask[id2] = False

        nz_in = block.data[inside]
        nz_out = block.data[~inside]
        n_in = id1.size * id2.size
        n_out = id1.size * (s - id2.size)

        bpmsum_tmp[k] = nz_in.sum()
        bpm_local_tmp[k] = mw_greater_sparse(nz_in, n_in, nz_out, n_out)
    return bpmsum_tmp, bpm_local_tmp

def snp_permutation_parallel(perm_args):
    """Run `share` SNP permutations and return exceedance counts for BPM/WPM/PATH.

    The permutation SEQUENCE reproduces the original bit for bit. Two details make that work:

    1. The RNG is the legacy global MT19937, seeded per worker as PERM_SEED + id * 1000, drawing
       exactly one np.random.permutation(s) per iteration. Switching to np.random.default_rng
       would draw a different sequence even from the same nominal seed.

    2. The original wrote `mmtmp = mmtmp[:, np.random.permutation(...)]`, REASSIGNING mmtmp, so
       the permutations compound: iteration k operates on mm[:, q_k] where q_k = q_{k-1}[p_k]
       and q_0 = arange(s). Each q_k is still marginally uniform (so the sampled distribution
       was never wrong), but the sequence is a random walk on the symmetric group rather than
       k independent draws. Carrying q forward here reproduces it; drawing fresh permutations
       agrees only on the first iteration.

    Given q, the column-permuted block sum is obtained by gathering the rows of the
    column-side indicator matrix with argsort(q), and the permuted column sums are just the
    original column sums reindexed by q.
    """
    mm = _SHARED['mm']
    u1_tiles = _SHARED['u1_tiles']
    u2_tiles = _SHARED['u2_tiles']
    pw = _SHARED['pw']
    ppath = _SHARED['ppath']
    bpmsum_obs = _SHARED['bpmsum_obs']
    wpmsum_obs = _SHARED['wpmsum_obs']
    path_obs = _SHARED['path_obs']
    col_ranks = _SHARED['col_ranks']
    col_ties = _SHARED['col_ties']
    path_lens = _SHARED['path_lens']
    s = mm.shape[0]

    ## ONE canonical stream, seeded the same way regardless of n_workers or n_jobs, so the
    ## permutations - and therefore the empirical p-values - are reproducible on any hardware.
    ## This is exactly the stream the original produced when run with n_workers=1.
    np.random.seed(PERM_SEED)

    count_bpm = np.zeros(bpmsum_obs.size)
    count_wpm = np.zeros(wpmsum_obs.size)
    count_path = np.zeros(path_obs.size)

    bpmsum_tmp = np.empty(bpmsum_obs.size)
    n_out_path = s - path_lens

    q = np.arange(s)  # cumulative permutation, matching the original's reassignment of mmtmp

    ## Advance to this piece's start by drawing and composing the permutations it is not
    ## evaluating. A draw is well under 1% of an evaluated iteration, so this is close to free,
    ## and it means the stream is identical to a worker that ran the whole share end to end.
    for _ in range(perm_args.skip):
        q = q[np.random.permutation(s)]

    for perm in range(perm_args.count):
        q = q[np.random.permutation(s)]
        inv = np.empty(s, dtype=np.int64)      # inverse by scatter: O(s), not O(s log s)
        inv[q] = np.arange(s, dtype=np.int64)

        # BPM: permuting mm's columns == permuting the rows of the column-side indicators
        tiled_block_sums(mm, u1_tiles, u2_tiles, bpmsum_tmp, row_perm=inv)
        count_bpm += bpmsum_tmp > bpmsum_obs

        # WPM: same block on both sides, so the same trick applies
        if wpmsum_obs.size:
            wpmsum_tmp = block_sums(mm, pw, pw[inv, :])
            count_wpm += wpmsum_tmp > wpmsum_obs

        # PATH degree: the permuted column sums are a permutation of the original ones, so
        # the midranks are known up front and the statistic reduces to a rank sum.
        if path_obs.size:
            rank_in = np.asarray(ppath.T @ col_ranks[q]).ravel()
            p = mw_greater(rank_in, path_lens, n_out_path, col_ties)
            count_path += (-1 * np.log10(p)) > path_obs

    return count_bpm, count_wpm, count_path


# ---------------------------------------------------------------------------
# main routine
# ---------------------------------------------------------------------------

def rungenstats(input_network, bpm, wpm, minPath, binary_flag, snpPerms, n_jobs, n_workers):
    ## inputs:
    ## - input_network: scipy.sparse interaction network (csr_array)
    ## - bpm: bpm dataframe
    ## - wpm: wpm dataframe
    ## - minPath: minimum number of snps in a pathway
    ## - binary_flag: flag to make the interaction network binary
    ## - n_jobs: sequential work chunks (RAM), n_workers: pool width (speed)

    n_jobs = max(int(n_jobs), 1)
    n_workers = max(int(n_workers), 1)
    ctx = mp.get_context('fork')  # workers read the sparse structures copy-on-write

    mm_scores = as_sparse(input_network)
    s = mm_scores.shape[0]

    bpm_size = bpm['size'].values.shape[0]
    bpmsize = bpm['size'].values
    ind1 = bpm['ind1'].values
    ind2 = bpm['ind2'].values
    bpmind1size = bpm['ind1size'].values
    bpmind2size = bpm['ind2size'].values

    wpm_size = wpm['size'].values.shape[0]
    wpmsize = wpm['size'].values
    wpmindsize = wpm['indsize'].values
    ind = wpm['ind'].values

    ## ?Binary  -- mm is the binarized network used for the chi2 stage; mm_scores is kept
    ## alongside it instead of being np.copy()'d (both are sparse, so this is cheap).
    # TODO: should this threshold be tunable or changed?
    if binary_flag:
        # if true, then the network was already binarized with present or not present
        mm = mm_scores
    else:
        # else, binarize with a 0.2 cutoff, but since this is -log10(pvalues) then it is equivalent to a pvalue of 0.63
        # since 0.1 pvalue threshold is used with chi2 marginal significance, maybe that should be used here too?
        # -1.0 * log10(0.1) = 1.0, so maybe use this instead?
        mm = binarize(mm_scores, 0.2)

    sumMM = np.asarray(mm.sum(axis=1)).ravel()

    ## pathway indicator matrix: column a marks the SNPs of pathway a. Reused for every WPM
    ## and PATH statistic below, observed and permuted.
    path_lists = [ np.asarray(x, dtype=np.int64).ravel() for x in ind ]
    path_lens = np.fromiter((p.size for p in path_lists), dtype=np.int64, count=wpm_size).astype(np.float64)
    pmat = indicator_matrix(path_lists, s)

    ### BPM binary chi2
    print("\tBPM chi2: ", end="")
    t1 = datetime.now()
    # bpm genetic interaction counts + background interactions, in n_jobs sequential chunks
    # of n_workers parallel slices. `tr` is now a mask instead of an O(n^2) `in` test.
    tr_mask = (bpmind1size < 5) | (bpmind2size < 5)
    publish_shared(mm=mm, sumMM=sumMM, ind1=ind1, ind2=ind2, tr_keep=~tr_mask)

    bpmgi = np.zeros(bpm_size)
    path1bggi = np.zeros(bpm_size)
    path2bggi = np.zeros(bpm_size)

    with ctx.Pool(processes=n_workers) as pool:
        for chunk in split_indices(np.arange(bpm_size), n_jobs):
            job_args = [par_rank_args(i, part) for i, part in enumerate(split_indices(chunk, n_workers))]
            for j_arg, res in zip(job_args, pool.map(bpm_chi2_parallel, job_args)):
                bpmgi[j_arg.rows] = res[0]
                path1bggi[j_arg.rows] = res[1]
                path2bggi[j_arg.rows] = res[2]
    clear_shared()

    # bpm non interaction
    bpmnotgi = bpmsize - bpmgi
    bpmnotgi[bpmnotgi < 0] = 0
    # non-bpm non-interation
    path1bgsize = bpmind1size * s
    path2bgsize = bpmind2size * s

    path1notgi = path1bgsize - path1bggi - bpmsize
    path2notgi = path2bgsize - path2bggi - bpmsize

    # call chi2
    ## build the tables
    table1 = np.stack((bpmgi, path1bggi, bpmnotgi, path1notgi)).transpose()
    table1[tr_mask, :] = 5
    table2 = np.stack((bpmgi, path2bggi, bpmnotgi, path2notgi)).transpose()
    table2[tr_mask, :] = 5
    ## call chi2
    chi2_bpm_1 = np.log10(call_chi2(table1)) * -1.0
    chi2_bpm_2 = np.log10(call_chi2(table2)) * -1.0
    chi2_bpm_1[tr_mask] = 0
    chi2_bpm_2[tr_mask] = 0

    ## consider under-enriched chi2s
    under1 = bpmgi / (bpmgi + bpmnotgi) < path1bggi / (path1bggi + path1notgi)
    under2 = bpmgi / (bpmgi + bpmnotgi) < path2bggi / (path2bggi + path2notgi)
    chi2_bpm_1[under1] = -1 * chi2_bpm_1[under1]
    chi2_bpm_2[under2] = -1 * chi2_bpm_2[under2]

    ## compute densitites
    density_bpm_local_1 = (bpmgi + path1bggi) / (path1notgi + path1bggi + bpmsize)
    density_bpm_local_2 = (bpmgi + path2bggi) / (path2notgi + path2bggi + bpmsize)

    ## choose the denser (or lower chi2 value)
    dense_index = np.zeros(bpm_size)
    dense_index[chi2_bpm_1 < chi2_bpm_2] = 1
    dense_index[chi2_bpm_1 > chi2_bpm_2] = 2
    dense_index[(dense_index == 0) & (density_bpm_local_1 > density_bpm_local_2)] = 1
    dense_index[(dense_index == 0) & (density_bpm_local_1 < density_bpm_local_2)] = 2

    ## finalize bpm local
    chi2_bpm_local = np.zeros(bpm_size)
    chi2_bpm_local[dense_index == 1] = chi2_bpm_1[dense_index == 1]
    chi2_bpm_local[dense_index == 2] = chi2_bpm_2[dense_index == 2]

    ## keeping track of significant bpms
    ind2keep_bpm = (chi2_bpm_local >= (-1.0 * np.log10(0.1))) & (bpmind1size >= minPath) & (bpmind2size >= minPath)

    ## keeping denser pathway in ind1_new
    swap = dense_index == 2
    ind1_new = np.where(swap, ind2, ind1)
    ind2_new = np.where(swap, ind1, ind2)
    ind1size_new = np.where(swap, bpmind2size, bpmind1size)

    ## pairs to keep
    bpmind1 = ind1_new[ind2keep_bpm]
    bpmind2 = ind2_new[ind2keep_bpm]
    print(f"{ind2keep_bpm.sum()} passed - {str(datetime.now() - t1).split('.')[0]}")

    ###WPM Chi2
    print("\tWPM chi2: ", end="")
    t1 = datetime.now()
    ## one sparse product replaces the per-pathway loop; the diagonal of P.T @ mm @ P is
    ## exactly sum(mm[ind[i], :][:, ind[i]]).
    wpmgi = block_sums(mm, pmat, pmat)
    wpmnotgi = wpmsize - wpmgi
    density_wpm = wpmgi / wpmsize

    ## WPM background size and interactions
    pathbggi = np.asarray(pmat.T @ sumMM).ravel() - wpmgi
    pathbgsize = wpmindsize * s
    pathbgnotgi = pathbgsize - pathbggi - wpmsize

    wpm_table = np.stack((wpmgi, pathbggi, wpmnotgi, pathbgnotgi)).transpose()

    ## call chi2
    chi2_wpm = np.log10(call_chi2(wpm_table)) * -1

    ## consider under-enriched chi2s
    under_wpm = wpmgi / (wpmgi + wpmnotgi) < pathbggi / (pathbggi + pathbgnotgi)
    chi2_wpm[under_wpm] = -1 * chi2_wpm[under_wpm]
    ind2keep_wpm = (chi2_wpm >= -1 * np.log10(0.1))
    print(f"{ind2keep_wpm.sum()} passed - {str(datetime.now() - t1).split('.')[0]}")

    ##### mutual binary - non-binary ends here

    if binary_flag:
        ## compute bpm interaction count and density for the remaining
        bpmsum = np.zeros(bpm_size)
        density_bpm = np.zeros(bpm_size)

        u1_keep = indicator_matrix([np.asarray(x, dtype=np.int64) for x in bpmind1], s)
        u2_keep = indicator_matrix([np.asarray(x, dtype=np.int64) for x in bpmind2], s)
        bpmsum_tmp = np.zeros(bpmind1.shape[0])
        tiled_block_sums(mm, tile_indicators(u1_keep, n_jobs), tile_indicators(u2_keep, n_jobs), bpmsum_tmp)

        density_bpm[ind2keep_bpm] = bpmsum_tmp / bpmsize[ind2keep_bpm]
        bpmsum[ind2keep_bpm] = bpmsum_tmp
        bpm_local = chi2_bpm_local  ## output

        ### WPM density
        wpm_local = chi2_wpm
        wpmsum = np.zeros(wpm_size)
        density_wpm = np.zeros(wpm_size)
        pw = pmat[:, ind2keep_wpm]
        wpmsum_tmp = block_sums(mm, pw, pw)
        density_wpm[ind2keep_wpm] = wpmsum_tmp / wpmsize[ind2keep_wpm]
        wpmsum[ind2keep_wpm] = wpmsum_tmp

    else:
        ## restore non-binary mm
        mm = mm_scores
        sumMM = np.asarray(mm.sum(axis=1)).ravel()
        ## ranksum test
        print("\tBPM ranksum: ", end="")
        t1 = datetime.now()
        bpmsum = np.zeros(bpm_size)
        density_bpm = np.zeros(bpm_size)
        n_keep = bpmind1.shape[0]
        bpmsum_tmp = np.zeros(n_keep)
        bpm_local_tmp = np.ones(n_keep)

        # parallel run for computing ranksum, in n_jobs sequential chunks
        publish_shared(mm=mm, bpmind1=bpmind1, bpmind2=bpmind2)
        with ctx.Pool(processes=n_workers) as pool:
            for chunk in split_indices(np.arange(n_keep), n_jobs):
                job_args = [par_rank_args(i, part) for i, part in enumerate(split_indices(chunk, n_workers))]
                for j_arg, res in zip(job_args, pool.map(parallel_ranksum, job_args)):
                    bpmsum_tmp[j_arg.rows] = res[0]
                    bpm_local_tmp[j_arg.rows] = res[1]
        clear_shared()

        density_bpm[ind2keep_bpm] = bpmsum_tmp / bpmsize[ind2keep_bpm]
        bpm_local = np.zeros(bpm_size)
        bpm_local[ind2keep_bpm] = -1 * np.log10(bpm_local_tmp)
        bpmsum[ind2keep_bpm] = bpmsum_tmp
        ## update ind2keep_bpm
        ind2keep_bpm = (bpm_local >= -1 * np.log10(0.05))
        print(f"{ind2keep_bpm.sum()} passed - {str(datetime.now() - t1).split('.')[0]}")

        ### wpm ranksum
        print("\tWPM ranksum: ", end="")
        t1 = datetime.now()
        ## NOTE: density_wpm deliberately keeps its binarized values outside ind2keep_wpm,
        ## matching the original (which assigned to a misspelled `denisty_wpm` here).
        # denisty_wpm = np.zeros(wpm_size)  # original had this typo
        density_wpm = np.zeros(wpm_size)  # this is the correct one
        wpmsum = np.zeros(wpm_size)
        wpm_local_tmp = np.ones(wpm_size)
        kept_wpm = np.flatnonzero(ind2keep_wpm)
        mask = np.zeros(s, dtype=bool)
        for a in kept_wpm:
            id1 = path_lists[a]
            block = mm[id1, :]
            mask[id1] = True
            inside = mask[block.indices]
            mask[id1] = False
            nz_in = block.data[inside]
            nz_out = block.data[~inside]
            wpmsum[a] = nz_in.sum()
            wpm_local_tmp[a] = mw_greater_sparse(nz_in, id1.size * id1.size,
                                                 nz_out, id1.size * (s - id1.size))
        density_wpm[ind2keep_wpm] = wpmsum[ind2keep_wpm] / wpmsize[ind2keep_wpm]
        wpm_local = np.zeros(wpm_size)
        wpm_local[ind2keep_wpm] = -1 * np.log10(wpm_local_tmp[ind2keep_wpm])
        ind2keep_wpm = (wpm_local >= -1 * np.log10(0.05))
        print(f"{ind2keep_wpm.sum()} passed - {str(datetime.now() - t1).split('.')[0]}")

    print("\tComputing expected densities ", end="")
    t1 = datetime.now()
    ## Recomputed here, at the same point the original does it, so that the non-binary branch's
    ## narrowed ind2keep_bpm is reflected in the pairs used from here on (the permutation stage
    ## below in particular). Nothing between the branch and this line uses them.
    bpmind1 = ind1_new[ind2keep_bpm]
    bpmind2 = ind2_new[ind2keep_bpm]
    ## compute expected bpm density -- vectorized per chunk instead of one gather per BPM
    density_bpm_expected = np.zeros(bpm_size)
    for chunk in split_indices(np.arange(bpm_size), n_jobs):
        u = indicator_matrix([np.asarray(ind1_new[i], dtype=np.int64) for i in chunk], s)
        lens = ind1size_new[chunk].astype(np.float64)
        totals = np.asarray(u.T @ sumMM).ravel()
        with np.errstate(divide='ignore', invalid='ignore'):
            vals = totals / (s * lens)
        density_bpm_expected[chunk] = np.where(lens > 0, vals, 0.0)

    ## compute expected wpm density
    with np.errstate(divide='ignore', invalid='ignore'):
        density_wpm_expected = np.asarray(pmat.T @ sumMM).ravel() / (s * path_lens)
    density_wpm_expected[path_lens == 0] = 0.0

    ## path degree -- dist_in/dist_out always partition sumMM, so rank once and reuse
    row_ranks = rankdata(sumMM)
    row_ties = tie_sum(sumMM)
    rank_in = np.asarray(pmat.T @ row_ranks).ravel()
    path_degree = -1 * np.log10(mw_greater(rank_in, path_lens, s - path_lens, row_ties))
    ind2keep_path = (path_degree >= -1 * np.log10(0.1))
    print(f"- {str(datetime.now() - t1).split('.')[0]}")

    ## random snp permutation to compute emirical p-value for the significant bpms
    print("\tSNP permutation ", end="")
    t1 = datetime.now()
    np.random.seed(PERM_SEED)  # inert (every worker reseeds), but the original set it here
    bpm_local_pv = np.ones(bpm_size)
    wpm_local_pv = np.ones(wpm_size)
    path_degree_pv = np.ones(wpm_size)

    ## bpmind1/bpmind2 already reflect the final ind2keep_bpm (recomputed above, as in the
    ## original), so the kept pairs, bpmsum baseline and count_bpm all have the same length.
    u1_keep = indicator_matrix([np.asarray(x, dtype=np.int64) for x in bpmind1], s)
    u2_keep = indicator_matrix([np.asarray(x, dtype=np.int64) for x in bpmind2], s)

    ## permuted block sums are compared against the observed ones on the same network
    bpmsum_obs = np.zeros(bpmind1.shape[0])
    tiled_block_sums(mm, tile_indicators(u1_keep, n_jobs), tile_indicators(u2_keep, n_jobs), bpmsum_obs)
    pw = pmat[:, ind2keep_wpm]
    wpmsum_obs = block_sums(mm, pw, pw)

    ## the permutation compares against permuted *column* sums, so rank those
    col_sums = np.asarray(mm.sum(axis=0)).ravel()

    publish_shared(
        mm=mm,
        u1_tiles=tile_indicators(u1_keep, n_jobs),
        u2_tiles=tile_indicators(u2_keep, n_jobs),
        pw=pw,
        ppath=pmat[:, ind2keep_path],
        bpmsum_obs=bpmsum_obs,
        wpmsum_obs=wpmsum_obs,
        path_obs=path_degree[ind2keep_path],
        col_ranks=rankdata(col_sums),
        col_ties=tie_sum(col_sums),
        path_lens=path_lens[ind2keep_path],
    )

    count_bpm = np.zeros(bpmind1.shape[0])
    count_wpm = np.zeros(int(np.sum(ind2keep_wpm)))
    count_path = np.zeros(int(np.sum(ind2keep_path)))

    ## assign parallel job args -- the original split, preserved exactly: proc 0 takes the
    ## remainder, every other proc takes floor(snpPerms/n_workers). n_jobs is NOT applied here;
    ## any other split changes each worker's share and therefore its seed.
    ## snpPerms permutations are drawn from a single stream, so the work is split by simply
    ## cutting that stream into contiguous pieces. A piece advances to its start by drawing (not
    ## evaluating) the permutations it skips, which is why the split can be chosen freely: every
    ## permutation still comes from the same stream at the same position no matter how many
    ## pieces there are. n_workers and n_jobs therefore change only the speed, never the result.
    ##
    ## Pieces are equal in evaluated count, which is also the minimum total fast-forward. The
    ## last piece skips the whole stream, but a draw is a fraction of a percent of an evaluated
    ## iteration, and that skip runs while the other workers are doing real work.
    pieces = [pc for pc in np.array_split(np.arange(snpPerms), n_workers) if pc.size]
    job_args = [perm_args(int(pc[0]), int(pc.size)) for pc in pieces]

    with ctx.Pool(processes=n_workers) as pool:
        results = pool.map(snp_permutation_parallel, job_args)
    # combine results
    for res in results:
        count_bpm = count_bpm + res[0]
        count_wpm = count_wpm + res[1]
        count_path = count_path + res[2]
    clear_shared()
    print(f"- {str(datetime.now() - t1).split('.')[0]}")

    bpm_local_pv[ind2keep_bpm] = (count_bpm + 1) / snpPerms
    wpm_local_pv[ind2keep_wpm] = (count_wpm + 1) / snpPerms
    path_degree_pv[ind2keep_path] = (count_path + 1) / snpPerms

    return bpm_local, bpm_local_pv, density_bpm, density_bpm_expected, dense_index, wpm_local, wpm_local_pv, density_wpm, density_wpm_expected, path_degree, path_degree_pv

def genstats(ssmfile, bpmfile, binary_flag, snpPerms, minPath, n_jobs, n_workers, netDensity=None):
    ### load bpmfile
    with open(bpmfile, 'rb') as pklin:
        bpm_obj = pickle.load(pklin)
    bpm = bpm_obj.bpm
    wpm = bpm_obj.wpm
    print(f"\tloaded {bpm.shape[0]:,} BPMs and {wpm.shape[0]} WPMs")

    ### load interaction network
    with open(ssmfile, 'rb') as pklin:
        network = pickle.load(pklin)
    p_network = as_sparse(network.protective)
    r_network = as_sparse(network.risk)

    print(f"\t{p_network.shape[0] * p_network.shape[1]:,} entries in the SNP-SNPinteraction network")
    p_density = p_network.nnz / (p_network.shape[0] * p_network.shape[1]) * 100
    print(f"\t{p_density:.2f}% of the entries are nonzero in protective network")
    r_density = r_network.nnz / (r_network.shape[0] * r_network.shape[1]) * 100
    print(f"\t{r_density:.2f}% of the entries are nonzero in risk network")

    if binary_flag:
        if netDensity is None:
            ## every stored value is > 0, so this is just "set the stored values to 1"
            p_network = binarize(p_network, 0)
            r_network = binarize(r_network, 0)
        else:
            p_cutoff = sparse_quantile(p_network, 1 - netDensity)
            r_cutoff = sparse_quantile(r_network, 1 - netDensity)
            ## a cutoff of 0 would binarize the zeros too, i.e. densify to an all-ones s x s
            ## matrix. That is unrepresentable sparsely (and almost certainly not intended),
            ## so fall back to keeping the stored entries and say so.
            for name, cutoff in (('protective', p_cutoff), ('risk', r_cutoff)):
                if cutoff <= 0:
                    print(f"\twarning: netDensity={netDensity} puts the {name} cutoff at "
                          f"{cutoff}; keeping all nonzero entries instead of densifying")
            p_network = binarize(p_network, max(p_cutoff, np.finfo(np.float64).tiny))
            r_network = binarize(r_network, max(r_cutoff, np.finfo(np.float64).tiny))

    print(f"running genstats on protective network")
    p_results = rungenstats(p_network, bpm, wpm, minPath, binary_flag, snpPerms, n_jobs, n_workers)
    p_stats = Stats(*p_results)

    print(f"running genstats on risk network")
    r_results = rungenstats(r_network, bpm, wpm, minPath, binary_flag, snpPerms, n_jobs, n_workers)
    r_stats = Stats(*r_results)
    print()

    out_obj = GenStats(p_stats, r_stats)
    tmp = ssmfile.split('/')
    tmp[-1] = 'genstats_' + tmp[-1]
    outputfile = '/'.join(tmp)
    with open(outputfile, 'wb') as final:
        pickle.dump(out_obj, final)

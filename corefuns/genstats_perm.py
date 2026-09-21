import math, pickle, signal, sys
import multiprocessing as mp
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.sparse import csr_array, issparse
from scipy.stats import chi2, norm, rankdata

from classes import bpmindclass, InteractionNetwork, Stats, GenstatsOut
np.seterr(divide='ignore', invalid='ignore')

# genstats() computes BPM/WPM/PATH statistics. Can be run parallel.
#
# REFACTOR NOTES
#   The interaction network mm is a scipy.sparse.csr_array end to end; nothing here ever
#   materializes a dense (s x s) array.
#
#   Everything in the permutation stage is driven by one derived matrix:
#
#       A[r, j] = sum_{i in R_r} mm[i, j]        R_r = the r-th distinct BPM ROW SET
#
#   i.e. "total interaction weight between row set R_r and SNP j". A is DENSE, and it is
#   indexed by distinct row set rather than by pathway, because a BPM's block is over SET
#   DIFFERENCES - bpmindclass stores ind1 = P_a \ P_b and ind2 = P_b \ P_a - so a
#   pathway-indexed matrix gives the wrong block for any pair whose pathways overlap. Pairs
#   with disjoint pathways have ind1 == P_a and collapse onto a single shared row, so A only
#   grows with the number of OVERLAPPING kept pairs. The identity that matters:
#
#       sum(mm[:, q][R_r, :][:, C_t]) == sum_{j in C_t} A[r, q[j]]
#
#   So a permutation costs one gather of length sum_t |C_t| over the kept BPMs plus a segment
#   sum, and A is built once per network rather than once per permutation. The column side is
#   unconstrained - the gather takes each BPM's own ind2 verbatim - so only the row side has to
#   be a row of A.
#
#   This is the whole reason the permutation stage is fast. The previous version evaluated
#   (mm @ U2).multiply(U1) per permutation with one indicator column PER BPM; that recomputed
#   mm @ (indicator) once for every BPM and allocated an (s x n_bpm) sparse intermediate each
#   time. Cost per permutation drops from O(n_bpm * nnz_per_column) to O(sum of kept column-set
#   sizes) - roughly 10^9 sparse multiply-adds down to ~10^7 element reads.
#
#   The OBSERVED (one-pass) BPM sums are hybrid. Disjoint pairs - about three quarters of them
#   in practice - are read straight off the pathway pair matrix pair = A_path @ pmat, which
#   gives every pathway pair's block sum in one sparse-times-dense product. Only overlapping
#   pairs go through the per-BPM sparse product on their stored set-difference lists. WPM and
#   PATH statistics always use whole pathways (wpm['ind']), so they use A_path throughout.
#
#   BPM/WPM ranksum: for a fixed row pathway a, the block mm[P_a, :] and therefore the midranks
#   of its stored values and its tie correction do not depend on the partner pathway b. They are
#   computed ONCE per row pathway and reused across all of its partners; only the in/out split
#   changes. See block_rank_context() / block_mw().
#
#   PATH degree: dist_in/dist_out always partition the same vector (sumMM, or a permutation of
#   it), so the midranks and tie correction are computed once and the per-pathway statistic is
#   just a rank sum, i.e. P.T @ ranks. Exact, not an approximation.
#
#   call_chi2 is a closed-form vectorized 2x2 chi-square instead of bpm_size calls into
#   scipy.stats.chi2_contingency, and takes the four count columns directly so the (bpm_size x 4)
#   stacked tables are never materialized.
#
#   Worker data sharing is by fork() copy-on-write: publish_shared() installs the read-only
#   arrays on the module before the pool is created, so A is shared, not copied per worker.
#
# BEHAVIOUR NOTES
#   - BPM pathway ids come from path1names/path2names matched against wpm['pathway'], which is
#     exact and vectorized. Row position cannot be used: the BPM list is a subset of the upper
#     triangle once pairs have been filtered upstream by size.
#   - A BPM's ind1size is the size of a SET DIFFERENCE, not of a pathway. ind1size ==
#     wpm['indsize'][idx1] is therefore the test for "these two pathways do not overlap", and
#     is what selects the fast path for the observed sums.
#   - Empirical p-values are reproducible and prefix-stable. Permutation k is drawn from
#     SeedSequence(seed, spawn_key=(k,)), so it depends only on the seed and on k - not on
#     snpPerms, n_workers or n_jobs. A 20-permutation run reproduces the first 10 permutations
#     of a 10-permutation run with the same seed. Permutations do NOT compound (each is an
#     independent draw), unlike the original, which reassigned mmtmp and so walked the symmetric
#     group. Each draw was marginally uniform either way, so the sampled distribution is
#     unchanged; only the sequence differs.
#   - Tables that are not valid contingency tables return p = 1 from call_chi2 rather than
#     raising (as chi2_contingency did) or reporting spurious significance. See call_chi2().
#   - net_density now actually applies. The quantile cutoffs were previously computed, used only
#     for a warning, and then discarded (both networks were binarized at 0 regardless, which is
#     a no-op when every stored value is positive).
#
# INPUTS:
#   ssmFile: Interaction networks file in the pickle format.
#   binary_flag: If True, interaction scores are binarized for computing BPM/WPM/PATH significances
#   snp_perms: Number of snp permutations used for computing empirical p-values
#   n_jobs: number of sequential chunks the big passes are split into (lower peak RAM)
#   n_workers: number of parallel cpu cores the program shoud use (higher throughput)
#   seed: RNG seed for the SNP permutation stage
#
# OUTPUTS:
#   genstats_<ssmFile without extension>.pkl - This pickle file contains a GenstasOut class, which itself contains 2 Stats class oject
#       - protective_stats: Statistics for protective network including ranksum scores,empirical p-values, expected density for BPM/WPMs
#       - risk_stats: Statistics for risk network including ranksum scores,empirical p-values, expected density for BPM/WPMs


class perm_args:
    def __init__(self, lo, hi):
        self.lo = lo
        self.hi = hi

class par_rank_args:
    def __init__(self, id, rows=None, groups=None):
        self.id = id
        self.rows = None if rows is None else np.asarray(rows, dtype=np.int64)
        self.groups = groups


# ---------------------------------------------------------------------------
# worker data sharing
# ---------------------------------------------------------------------------
# Called in the *parent* before the pool is created; children inherit the objects through
# fork() copy-on-write, so nothing large is pickled per job and nothing is copied per worker.

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
    """Set every stored value >= threshold to 1 and drop the rest.

    Only the STORED entries are touched, so a structural zero stays zero whatever the
    threshold is. This is deliberately NOT the dense `mm[mm>=threshold] = 1` for a threshold
    of 0 or less: that expression would promote the zeros too and densify to an all-ones
    (s x s) matrix. For a positive threshold the two agree exactly, which is the case that
    matters for the 0.2 score cutoff and for the netDensity quantile cutoffs.

    Explicitly stored zeros are removed first, so `threshold=0` means "keep every nonzero
    entry" regardless of whether the caller has already canonicalized the matrix.
    """
    out = mm.copy()
    out.eliminate_zeros()
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

def as_index(x):
    """A SNP index list as a contiguous int64 array (a no-op when it already is one)."""
    return np.asarray(x, dtype=np.int64).ravel()

def block_sums(mm, u1, u2):
    """Column-wise sum(mm[R_j, :][:, C_j]) for paired indicator columns u1/u2."""
    if u1.shape[1] == 0:
        return np.zeros(0)
    return np.asarray((mm @ u2).multiply(u1).sum(axis=0)).ravel()

def set_totals(values, sets, n_jobs):
    """sum(values[S]) for each set S, in n_jobs sequential chunks."""
    out = np.zeros(len(sets))
    for chunk in split_indices(np.arange(len(sets)), n_jobs):
        u = indicator_matrix([sets[k] for k in chunk], values.size)
        out[chunk] = np.asarray(u.T @ values).ravel()
    return out

def dedupe_sets(sets):
    """Map each SNP index list to an id over the DISTINCT lists.

    Pairs with disjoint pathways share a row set (the pathway itself), so this is what keeps A
    to roughly n_path + (number of overlapping kept pairs) rows instead of one row per BPM.
    """
    lut = {}
    ids = np.empty(len(sets), dtype=np.int64)
    distinct = []
    for k, x in enumerate(sets):
        key = x.tobytes()
        r = lut.get(key)
        if r is None:
            r = len(distinct)
            lut[key] = r
            distinct.append(x)
        ids[k] = r
    return ids, distinct

def row_set_sums(mm, row_sets, n_jobs):
    """A[r, j] = sum_{i in row_sets[r]} mm[i, j], dense (len(row_sets) x s).

    Built in n_jobs sequential chunks so the sparse intermediate - which is close to fully
    dense at these network densities - never exists for all row sets at once.
    """
    n = len(row_sets)
    s = mm.shape[1]
    A = np.empty((n, s), dtype=np.float64)
    for chunk in split_indices(np.arange(n), n_jobs):
        u = indicator_matrix([row_sets[r] for r in chunk], s)
        A[chunk, :] = (u.T @ mm).toarray()
    return A

def flat_gather_plan(row_ids, col_sets, s):
    """Index arrays that turn "sum_{j in C_t} A[r_t, q[j]]" into a take + reduceat.

    Lays each BPM's column SNP list out end to end alongside the flat offset r_t*s of its row
    in A. A permuted block sum is then A.ravel().take(row_base + q.take(col_idx)) reduced at
    `starts`. The column lists are used verbatim, so set differences need no special handling.
    """
    n = len(col_sets)
    if n == 0:
        z = np.zeros(0, dtype=np.int64)
        return z, z, z
    lens = np.fromiter((c.size for c in col_sets), dtype=np.int64, count=n)
    if np.any(lens == 0):
        raise ValueError("flat_gather_plan: a kept BPM/WPM has an empty column set; "
                         "np.add.reduceat cannot express an empty segment")
    starts = np.zeros(n, dtype=np.int64)
    np.cumsum(lens[:-1], out=starts[1:])
    col_idx = np.concatenate(col_sets)
    row_base = np.repeat(np.asarray(row_ids, dtype=np.int64) * s, lens)
    return col_idx, row_base, starts

def permuted_block_sums(A_flat, col_idx, row_base, starts, q=None):
    """Block sums for the pairs described by a flat_gather_plan, under SNP permutation q."""
    if starts.size == 0:
        return np.zeros(0)
    cols = col_idx if q is None else q.take(col_idx)
    return np.add.reduceat(A_flat.take(row_base + cols), starts)


# ---------------------------------------------------------------------------
# statistics helpers
# ---------------------------------------------------------------------------

def call_chi2(a, b, c, d, ignore=None):
    """Vectorized 2x2 chi-square, no continuity correction.

    Arguments are the four count columns: a = f11 (bpm interactions), b = f10 (non-bpm
    interactions), c = f01 (bpm non-interactions), d = f00 (non-bpm non-interactions), i.e.
    [[a,b],[c,d]]. Matches scipy.stats.chi2_contingency(obs, correction=False) for well-formed
    tables. `ignore` marks rows whose result is discarded downstream, so they are excluded from
    the negative-count warning only.

    Rows that are not a valid contingency table return p = 1, i.e. "no evidence of enrichment",
    which is the only defensible answer for a test of whether interactions differ from
    expectation. Three cases:
      - a == 0: the original short-circuited to p = 1 here, so this is unchanged.
      - a zero row/column marginal: chi2_contingency raised ValueError; the closed form divides
        by zero and yields inf/nan. Only reachable if a region is fully saturated.
      - ANY NEGATIVE COUNT: chi2_contingency also raised here. The closed form would happily
        return a tiny p-value (a negative cell inflates the |ad - bc| numerator while the
        denominator stays positive), reporting a malformed table as maximally significant.
        That is meaningless, so these are forced to p = 1 and reported. A negative count signals
        an upstream size/convention bug - e.g. wpmnotgi = wpmsize - wpmgi going negative if
        wpmsize counts unordered pairs while wpmgi (a full block sum) counts ordered ones.
        bpmnotgi is clamped upstream; wpmnotgi is not.
    """
    n = a + b + c + d
    with np.errstate(divide='ignore', invalid='ignore'):
        stat = n * (a * d - b * c) ** 2 / ((a + b) * (c + d) * (a + c) * (b + d))
    results = chi2.sf(stat, 1)

    negative = (a < 0) | (b < 0) | (c < 0) | (d < 0)
    invalid = ~np.isfinite(stat) | (a == 0) | negative
    results[invalid] = 1.0

    reportable = negative if ignore is None else (negative & ~ignore)
    n_neg = int(np.count_nonzero(reportable))
    if n_neg:
        print(f"\twarning: {n_neg} chi2 table row(s) contained a negative count and were set "
              f"to p=1; check the size vs interaction-count pair conventions upstream", flush=True)
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

    Same formula as scipy.stats.mannwhitneyu(..., use_continuity=True, alternative='greater'),
    but driven by a precomputed midrank sum so the ranking can be shared across pathways and
    permutations. Accepts scalars or arrays.
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

def block_rank_context(block, s):
    """Rank the stored values of mm[P_a, :] once, for reuse across every partner pathway.

    The in-group and out-group always partition the same block, whose population is |P_a| * s
    regardless of the partner. So the unstored entries are one tie group of zeros at the bottom
    of the ranking, and both the midranks and the tie correction are partner-independent.
    """
    data = block.data
    z_tot = float(block.shape[0]) * float(s) - data.size
    ranks = rankdata(data) + z_tot
    ties = tie_sum(data)
    if z_tot > 1:
        ties += z_tot ** 3 - z_tot
    return ranks, ties, z_tot

def block_mw(ranks, ties, z_tot, inside, n_in, n_out):
    """Mann-Whitney p-value for one partner, given a block_rank_context and the in-group mask."""
    rank_sum_in = float(ranks[inside].sum())
    z_in = float(n_in) - float(np.count_nonzero(inside))
    rank_sum_in += z_in * (z_tot + 1.0) / 2.0
    return mw_greater(rank_sum_in, n_in, n_out, ties)

def split_indices(rows, n_parts):
    """Split into at most n_parts non-empty contiguous pieces."""
    return [part for part in np.array_split(np.asarray(rows), max(int(n_parts), 1)) if part.size]

def group_by_row(row_ids):
    """Group positions by their row pathway: [(a, positions), ...], one entry per distinct a."""
    if row_ids.size == 0:
        return []
    order = np.argsort(row_ids, kind='stable')
    sorted_ids = row_ids[order]
    edges = np.flatnonzero(np.r_[True, sorted_ids[1:] != sorted_ids[:-1], True])
    return [(int(sorted_ids[edges[k]]), order[edges[k]:edges[k + 1]])
            for k in range(edges.size - 1)]


# ---------------------------------------------------------------------------
# parallel workers
# ---------------------------------------------------------------------------

def bpm_block_parallel(job_arg):
    """Block sum and the two sumMM totals for a slice of BPMs, from their stored SNP lists.

    Used only for pairs whose pathways OVERLAP, where the block is over set differences and the
    pathway pair matrix does not apply.
    """
    mm = _SHARED['mm']
    sumMM = _SHARED['sumMM']
    rows = _SHARED['rows']
    cols = _SHARED['cols']

    sel = job_arg.rows
    u1 = indicator_matrix([rows[i] for i in sel], mm.shape[0])
    u2 = indicator_matrix([cols[i] for i in sel], mm.shape[0])
    gi = block_sums(mm, u1, u2)
    return gi, np.asarray(u1.T @ sumMM).ravel(), np.asarray(u2.T @ sumMM).ravel()

def parallel_ranksum(job_arg):
    """bpmsum + ranksum p-value for a set of row-set groups (non-binary network).

    One mm[R, :] slice and one ranking serve every BPM sharing the row set R, which for disjoint
    pathway pairs is every BPM built on that pathway.
    """
    mm = _SHARED['mm']
    row_sets = _SHARED['row_sets']
    cols = _SHARED['cols']
    s = mm.shape[1]

    positions = []
    sums = []
    pvals = []
    mask = np.zeros(s, dtype=bool)

    for r, pos in job_arg.groups:
        id1 = row_sets[r]
        if id1.size < 5:
            continue                      # bpmsum 0 / p-value 1, as the old `tr` bookkeeping did
        block = mm[id1, :]
        ranks, ties, z_tot = block_rank_context(block, s)
        indices = block.indices
        for t in pos:
            id2 = cols[t]
            if id2.size < 5:
                continue
            mask[id2] = True
            inside = mask[indices]
            mask[id2] = False
            positions.append(t)
            sums.append(block.data[inside].sum())
            pvals.append(block_mw(ranks, ties, z_tot, inside,
                                  id1.size * id2.size, id1.size * (s - id2.size)))

    return (np.asarray(positions, dtype=np.int64),
            np.asarray(sums, dtype=np.float64),
            np.asarray(pvals, dtype=np.float64))

def snp_permutation_parallel(job_arg):
    """Run permutations [lo, hi) and return exceedance counts for BPM/WPM/PATH.

    Permutation k comes from SeedSequence(seed, spawn_key=(k,)), so it is addressed by index:
    a worker jumps straight to its slice with no fast-forward, the split across workers is
    free to change, and extending snp_perms leaves the earlier permutations untouched.
    """
    A_flat = _SHARED['A_flat']
    bpm_plan = _SHARED['bpm_plan']
    wpm_plan = _SHARED['wpm_plan']
    ppath = _SHARED['ppath']
    bpmsum_obs = _SHARED['bpmsum_obs']
    wpmsum_obs = _SHARED['wpmsum_obs']
    path_obs = _SHARED['path_obs']
    col_ranks = _SHARED['col_ranks']
    col_ties = _SHARED['col_ties']
    path_lens = _SHARED['path_lens']
    seed = _SHARED['seed']
    s = _SHARED['s']

    count_bpm = np.zeros(bpmsum_obs.size)
    count_wpm = np.zeros(wpmsum_obs.size)
    count_path = np.zeros(path_obs.size)
    n_out_path = s - path_lens

    for k in range(job_arg.lo, job_arg.hi):
        rng = np.random.default_rng(np.random.SeedSequence(entropy=seed, spawn_key=(k,)))
        q = rng.permutation(s)

        # BPM: permuting mm's columns is a gather on the BPM's own column SNP list
        if bpmsum_obs.size:
            count_bpm += permuted_block_sums(A_flat, *bpm_plan, q=q) > bpmsum_obs

        # WPM: whole pathway on both sides, so the same gather applies
        if wpmsum_obs.size:
            count_wpm += permuted_block_sums(A_flat, *wpm_plan, q=q) > wpmsum_obs

        # PATH degree: the permuted column sums are a permutation of the original ones, so the
        # midranks are known up front and the statistic reduces to a rank sum.
        if path_obs.size:
            rank_in = np.asarray(ppath.T @ col_ranks[q]).ravel()
            p = mw_greater(rank_in, path_lens, n_out_path, col_ties)
            count_path += (-1 * np.log10(p)) > path_obs

    return count_bpm, count_wpm, count_path


# ---------------------------------------------------------------------------
# main routine
# ---------------------------------------------------------------------------

# provide a more elegant way to cancel the worker pool on Ctrl+C
def init_worker():
    # Ignore SIGINT in worker processes; only the main process should handle it
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def bpm_pathway_ids(bpm, wpm):
    """Pathway index pair for every BPM row, from path1names/path2names.

    Row position cannot be used: once pairs have been filtered upstream by size the BPM list is
    a subset of np.triu_indices, not the whole of it. Matching names is exact and vectorized
    (~0.3 s at a few million rows).
    """
    for column in ('path1names', 'path2names'):
        if column not in bpm:
            raise ValueError(f"the BPM table has no '{column}' column, so its pathway pairs "
                             f"cannot be identified; columns present: {list(bpm.columns)}")
    names = pd.Index(wpm['pathway'].values)
    if not names.is_unique:
        raise ValueError("wpm['pathway'] contains duplicate names, so a BPM's pathways cannot "
                         "be identified unambiguously")
    idx1 = names.get_indexer(bpm['path1names'].values)
    idx2 = names.get_indexer(bpm['path2names'].values)
    missing = (idx1 < 0) | (idx2 < 0)
    if missing.any():
        raise ValueError(f"{int(missing.sum()):,} BPM row(s) name a pathway that is not in the "
                         f"WPM table; the BPM and WPM tables must come from the same BPMind file")
    return idx1.astype(np.int32), idx2.astype(np.int32)


def rungenstats(input_network, bpm, wpm, binary_flag, snp_perms, n_jobs, n_workers, seed):
    # inputs:
    # - input_network: scipy.sparse interaction network (csr_array)
    # - bpm: bpm dataframe (ind1/ind2 are SET DIFFERENCES: P_a \ P_b and P_b \ P_a)
    # - wpm: wpm dataframe (ind is the whole pathway)
    # - binary_flag: flag to make the interaction network binary
    # - n_jobs: sequential work chunks (RAM), n_workers: pool width (speed)
    # - seed: RNG seed for the SNP permutation stage

    n_jobs = max(int(n_jobs), 1)
    n_workers = max(int(n_workers), 1)
    ctx = mp.get_context('fork')  # workers read the shared arrays copy-on-write

    mm_scores = as_sparse(input_network)
    s = mm_scores.shape[0]

    bpm_size = bpm['size'].values.shape[0]
    bpmsize = bpm['size'].values
    ind1 = [as_index(x) for x in bpm['ind1'].values]
    ind2 = [as_index(x) for x in bpm['ind2'].values]
    bpmind1size = bpm['ind1size'].values
    bpmind2size = bpm['ind2size'].values

    wpm_size = wpm['size'].values.shape[0]
    wpmsize = wpm['size'].values
    wpmindsize = wpm['indsize'].values

    # pathway SNP lists, and the pathway pair each BPM row refers to
    path_lists = [as_index(x) for x in wpm['ind'].values]
    path_lens = np.fromiter((p.size for p in path_lists), dtype=np.int64,
                            count=wpm_size).astype(np.float64)
    pmat = indicator_matrix(path_lists, s)
    idx1, idx2 = bpm_pathway_ids(bpm, wpm)

    # A BPM whose stored lists are the full pathways has no overlap to correct for, so its
    # block sum is just the pathway pair's. Everything else needs its own lists.
    disjoint = (bpmind1size == path_lens[idx1]) & (bpmind2size == path_lens[idx2])

    # ?Binary  -- mm is the binarized network used for the chi2 stage; mm_scores is kept
    # alongside it instead of being copied (both are sparse, so this is cheap).
    if binary_flag:
        # if true, then the network was already binarized with present or not present
        mm = mm_scores
    else:
        # else, binarize with a 0.2 cutoff, but since this is -log10(pvalues) then it is equivalent to a pvalue of 0.63
        # since 0.1 pvalue threshold is used with chi2 marginal significance, maybe that should be used here too?
        # -1.0 * log10(0.1) = 1.0, so maybe use this instead?
        mm = binarize(mm_scores, 0.2)

    sumMM = np.asarray(mm.sum(axis=1)).ravel()

    # -------------------------------------------------
    # BPM binary chi2
    # -------------------------------------------------
    print("\tBPM chi2: ", end="", flush=True)
    t1 = datetime.now()
    # `tr` is a mask instead of an O(n^2) `in` test
    tr_mask = (bpmind1size < 5) | (bpmind2size < 5)

    # One pathway-by-pathway product gives every DISJOINT pair's block sum at once; the
    # diagonal of the same product is the WPM block sum.
    A_path = row_set_sums(mm, path_lists, n_jobs)
    pair = np.asarray(A_path @ pmat)
    path_sumMM = np.asarray(pmat.T @ sumMM).ravel()

    bpmgi = np.zeros(bpm_size)
    tot1 = np.zeros(bpm_size)
    tot2 = np.zeros(bpm_size)

    fast = disjoint & ~tr_mask
    bpmgi[fast] = pair[idx1[fast], idx2[fast]]
    tot1[fast] = path_sumMM[idx1[fast]]
    tot2[fast] = path_sumMM[idx2[fast]]

    # overlapping pairs: the block is over set differences, so use the stored lists
    slow = np.flatnonzero(~disjoint & ~tr_mask)
    publish_shared(mm=mm, sumMM=sumMM, rows=ind1, cols=ind2)
    pool = ctx.Pool(processes=n_workers, initializer=init_worker)
    try:
        for chunk in split_indices(slow, n_jobs):
            job_args = [par_rank_args(i, part)
                        for i, part in enumerate(split_indices(chunk, n_workers))]
            for j_arg, res in zip(job_args, pool.map(bpm_block_parallel, job_args)):
                bpmgi[j_arg.rows] = res[0]
                tot1[j_arg.rows] = res[1]
                tot2[j_arg.rows] = res[2]
        pool.close()
        pool.join()
        clear_shared()

    except KeyboardInterrupt:
        print("\nCtrl+C received — terminating worker pool...")
        pool.terminate()
        pool.join()
        sys.exit()

    path1bggi = tot1 - bpmgi
    path2bggi = tot2 - bpmgi

    # bpm non interaction
    bpmnotgi = bpmsize - bpmgi
    bpmnotgi[bpmnotgi < 0] = 0
    # non-bpm non-interation
    path1notgi = bpmind1size * s - path1bggi - bpmsize
    path2notgi = bpmind2size * s - path2bggi - bpmsize

    # call chi2 -- the four count columns are passed directly rather than stacked into
    # (bpm_size x 4) tables, which at this bpm_size saves several hundred MB per table
    chi2_bpm_1 = np.log10(call_chi2(bpmgi, path1bggi, bpmnotgi, path1notgi, ignore=tr_mask)) * -1.0
    chi2_bpm_2 = np.log10(call_chi2(bpmgi, path2bggi, bpmnotgi, path2notgi, ignore=tr_mask)) * -1.0
    chi2_bpm_1[tr_mask] = 0
    chi2_bpm_2[tr_mask] = 0

    # consider under-enriched chi2s
    under1 = bpmgi / (bpmgi + bpmnotgi) < path1bggi / (path1bggi + path1notgi)
    under2 = bpmgi / (bpmgi + bpmnotgi) < path2bggi / (path2bggi + path2notgi)
    chi2_bpm_1[under1] = -1 * chi2_bpm_1[under1]
    chi2_bpm_2[under2] = -1 * chi2_bpm_2[under2]

    # compute densitites
    density_bpm_local_1 = (bpmgi + path1bggi) / (path1notgi + path1bggi + bpmsize)
    density_bpm_local_2 = (bpmgi + path2bggi) / (path2notgi + path2bggi + bpmsize)

    # choose the denser (or lower chi2 value)
    dense_index = np.zeros(bpm_size)
    dense_index[chi2_bpm_1 < chi2_bpm_2] = 1
    dense_index[chi2_bpm_1 > chi2_bpm_2] = 2
    dense_index[(dense_index == 0) & (density_bpm_local_1 > density_bpm_local_2)] = 1
    dense_index[(dense_index == 0) & (density_bpm_local_1 < density_bpm_local_2)] = 2

    # finalize bpm local
    chi2_bpm_local = np.zeros(bpm_size)
    chi2_bpm_local[dense_index == 1] = chi2_bpm_1[dense_index == 1]
    chi2_bpm_local[dense_index == 2] = chi2_bpm_2[dense_index == 2]
    del chi2_bpm_1, chi2_bpm_2, density_bpm_local_1, density_bpm_local_2, under1, under2

    # keeping track of significant bpms
    ind2keep_bpm = (chi2_bpm_local >= (-1.0 * np.log10(0.1)))

    # keeping denser pathway first. The SNP lists are swapped as references, and the pathway
    # ids and sumMM totals are swapped alongside them so both stay consistent.
    swap = dense_index == 2
    row_sets = [ind2[i] if swap[i] else ind1[i] for i in range(bpm_size)]
    col_sets = [ind1[i] if swap[i] else ind2[i] for i in range(bpm_size)]
    idx1_new = np.where(swap, idx2, idx1)
    ind1size_new = np.where(swap, bpmind2size, bpmind1size).astype(np.float64)
    print(f"{ind2keep_bpm.sum():,} passed - {str(datetime.now() - t1).split('.')[0]}", flush=True)

    # -------------------------------------------------
    # WPM Chi2
    # -------------------------------------------------
    print("\tWPM chi2: ", end="", flush=True)
    t1 = datetime.now()
    wpmgi = np.ascontiguousarray(np.diagonal(pair))
    wpmnotgi = wpmsize - wpmgi
    density_wpm = wpmgi / wpmsize

    # WPM background size and interactions
    pathbggi = path_sumMM - wpmgi
    pathbgnotgi = wpmindsize * s - pathbggi - wpmsize

    chi2_wpm = np.log10(call_chi2(wpmgi, pathbggi, wpmnotgi, pathbgnotgi)) * -1

    # consider under-enriched chi2s
    under_wpm = wpmgi / (wpmgi + wpmnotgi) < pathbggi / (pathbggi + pathbgnotgi)
    chi2_wpm[under_wpm] = -1 * chi2_wpm[under_wpm]
    ind2keep_wpm = (chi2_wpm >= -1 * np.log10(0.1))
    del pair
    print(f"{ind2keep_wpm.sum():,} passed - {str(datetime.now() - t1).split('.')[0]}", flush=True)

    # -------------------------------------------------
    # mutual binary - non-binary ends here
    # -------------------------------------------------
    if binary_flag:
        # compute bpm interaction count and density for the remaining
        bpmsum = np.zeros(bpm_size)
        density_bpm = np.zeros(bpm_size)
        keep_pos = np.flatnonzero(ind2keep_bpm)

        # the swapped block sum for the kept pairs, from their stored lists
        bpmsum_tmp = np.zeros(keep_pos.size)
        publish_shared(mm=mm, sumMM=sumMM, rows=row_sets, cols=col_sets)
        pool = ctx.Pool(processes=n_workers, initializer=init_worker)
        try:
            for chunk in split_indices(np.arange(keep_pos.size), n_jobs):
                parts = split_indices(chunk, n_workers)
                job_args = [par_rank_args(i, keep_pos[part]) for i, part in enumerate(parts)]
                for part, res in zip(parts, pool.map(bpm_block_parallel, job_args)):
                    bpmsum_tmp[part] = res[0]
            pool.close()
            pool.join()
            clear_shared()
        except KeyboardInterrupt:
            print("\nCtrl+C received — terminating worker pool...")
            pool.terminate()
            pool.join()
            sys.exit()

        density_bpm[ind2keep_bpm] = bpmsum_tmp / bpmsize[ind2keep_bpm]
        bpmsum[ind2keep_bpm] = bpmsum_tmp
        bpm_local = chi2_bpm_local  # output

        # WPM density
        wpm_local = chi2_wpm
        wpmsum = np.zeros(wpm_size)
        density_wpm = np.zeros(wpm_size)
        pw = pmat[:, ind2keep_wpm]
        wpmsum_tmp = block_sums(mm, pw, pw)
        density_wpm[ind2keep_wpm] = wpmsum_tmp / wpmsize[ind2keep_wpm]
        wpmsum[ind2keep_wpm] = wpmsum_tmp

    else:
        # restore non-binary mm
        mm = mm_scores
        sumMM = np.asarray(mm.sum(axis=1)).ravel()
        path_sumMM = np.asarray(pmat.T @ sumMM).ravel()
        # -------------------------------------------------
        # BPM ranksum test
        # -------------------------------------------------
        print("\tBPM ranksum: ", end="", flush=True)
        t1 = datetime.now()
        bpmsum = np.zeros(bpm_size)
        density_bpm = np.zeros(bpm_size)
        keep_pos = np.flatnonzero(ind2keep_bpm)
        bpmsum_tmp = np.zeros(bpm_size)
        bpm_local_tmp = np.ones(bpm_size)

        # Grouped by ROW SET so each mm[R, :] slice and its ranking is built once and reused
        # across every BPM sharing it (all disjoint pairs on a given pathway). Groups are dealt
        # out round robin, which balances the pool better than contiguous slices when set sizes
        # vary widely.
        set_ids, distinct_rows = dedupe_sets([row_sets[i] for i in keep_pos])
        groups = [(r, keep_pos[pos]) for r, pos in group_by_row(set_ids)]
        publish_shared(mm=mm, row_sets=distinct_rows, cols=col_sets)
        pool = ctx.Pool(processes=n_workers, initializer=init_worker)
        try:
            job_args = [par_rank_args(w, groups=groups[w::n_workers]) for w in range(n_workers)]
            job_args = [j for j in job_args if j.groups]
            for res in pool.map(parallel_ranksum, job_args):
                bpmsum_tmp[res[0]] = res[1]
                bpm_local_tmp[res[0]] = res[2]
            pool.close()
            pool.join()
            clear_shared()

        except KeyboardInterrupt:
            print("\nCtrl+C received — terminating worker pool...")
            pool.terminate()
            pool.join()
            sys.exit()

        density_bpm[ind2keep_bpm] = bpmsum_tmp[ind2keep_bpm] / bpmsize[ind2keep_bpm]
        bpm_local = np.zeros(bpm_size)
        bpm_local[ind2keep_bpm] = -1 * np.log10(bpm_local_tmp[ind2keep_bpm])
        bpmsum[ind2keep_bpm] = bpmsum_tmp[ind2keep_bpm]
        # update ind2keep_bpm
        ind2keep_bpm = (bpm_local >= -1 * np.log10(0.05))
        print(f"{ind2keep_bpm.sum():,} passed - {str(datetime.now() - t1).split('.')[0]}", flush=True)

        # -------------------------------------------------
        # WPM ranksum
        # -------------------------------------------------
        print("\tWPM ranksum: ", end="", flush=True)
        t1 = datetime.now()
        density_wpm = np.zeros(wpm_size)
        wpmsum = np.zeros(wpm_size)
        wpm_local_tmp = np.ones(wpm_size)
        mask = np.zeros(s, dtype=bool)
        for a in np.flatnonzero(ind2keep_wpm):
            id1 = path_lists[a]
            block = mm[id1, :]
            ranks, ties, z_tot = block_rank_context(block, s)
            mask[id1] = True
            inside = mask[block.indices]
            mask[id1] = False
            wpmsum[a] = block.data[inside].sum()
            wpm_local_tmp[a] = block_mw(ranks, ties, z_tot, inside,
                                        id1.size * id1.size, id1.size * (s - id1.size))
        density_wpm[ind2keep_wpm] = wpmsum[ind2keep_wpm] / wpmsize[ind2keep_wpm]
        wpm_local = np.zeros(wpm_size)
        wpm_local[ind2keep_wpm] = -1 * np.log10(wpm_local_tmp[ind2keep_wpm])
        ind2keep_wpm = (wpm_local >= -1 * np.log10(0.05))
        print(f"{ind2keep_wpm.sum():,} passed - {str(datetime.now() - t1).split('.')[0]}", flush=True)

    # -------------------------------------------------
    # expected densities
    # -------------------------------------------------
    print("\tComputing expected densities ", end="", flush=True)
    t1 = datetime.now()
    # sum(sumMM[row set]) for EVERY bpm - the original computes expected density for all of
    # them, including the ones excluded from the chi2 test - against the final sumMM, which the
    # non-binary branch has restored to the score network. Disjoint pairs read the pathway
    # total directly; the rest are summed from their stored lists.
    tot1_new = np.where(disjoint, path_sumMM[idx1_new], 0.0)
    slow_new = np.flatnonzero(~disjoint)
    for chunk in split_indices(slow_new, n_jobs):
        tot1_new[chunk] = set_totals(sumMM, [row_sets[i] for i in chunk], 1)

    with np.errstate(divide='ignore', invalid='ignore'):
        density_bpm_expected = tot1_new / (s * ind1size_new)
        density_wpm_expected = path_sumMM / (s * path_lens)
    density_bpm_expected[ind1size_new == 0] = 0.0
    density_wpm_expected[path_lens == 0] = 0.0

    # path degree -- dist_in/dist_out always partition sumMM, so rank once and reuse
    row_ranks = rankdata(sumMM)
    row_ties = tie_sum(sumMM)
    rank_in = np.asarray(pmat.T @ row_ranks).ravel()
    path_degree = -1 * np.log10(mw_greater(rank_in, path_lens, s - path_lens, row_ties))
    ind2keep_path = (path_degree >= -1 * np.log10(0.1))
    print(f"- {str(datetime.now() - t1).split('.')[0]}", flush=True)

    # -------------------------------------------------
    # SNP permutations
    # -------------------------------------------------
    # random snp permutation to compute emirical p-value for the significant bpms
    print("\tSNP permutation ", end="", flush=True)
    t1 = datetime.now()
    bpm_local_pv = np.ones(bpm_size)
    wpm_local_pv = np.ones(wpm_size)
    path_degree_pv = np.ones(wpm_size)

    # ind2keep_bpm is read here, after the non-binary branch may have narrowed it, so the kept
    # pairs, the observed baseline and count_bpm all have the same length - as in the original,
    # which recomputed bpmind1/bpmind2 at this point.
    keep_pos = np.flatnonzero(ind2keep_bpm)
    kept_wpm = np.flatnonzero(ind2keep_wpm)

    # One dense A over the DISTINCT row sets of everything being permuted. Disjoint pairs share
    # their pathway's row, so this is roughly n_path + (overlapping kept pairs) rows, not one
    # row per BPM.
    perm_rows = [row_sets[i] for i in keep_pos] + [path_lists[a] for a in kept_wpm]
    row_ids, distinct_rows = dedupe_sets(perm_rows)
    gb = len(distinct_rows) * s * 8 / 1e9
    # print(f"[A: {len(distinct_rows):,} row sets x {s:,} SNPs = {gb:.1f} GB] ", end="", flush=True)
    A_rows = row_set_sums(mm, distinct_rows, n_jobs)
    A_flat = A_rows.ravel()

    bpm_plan = flat_gather_plan(row_ids[:keep_pos.size], [col_sets[i] for i in keep_pos], s)
    wpm_plan = flat_gather_plan(row_ids[keep_pos.size:], [path_lists[a] for a in kept_wpm], s)

    # permuted block sums are compared against the observed ones on the same network, computed
    # through the same gather so the two sides are numerically identical
    bpmsum_obs = permuted_block_sums(A_flat, *bpm_plan)
    wpmsum_obs = permuted_block_sums(A_flat, *wpm_plan)

    # the permutation compares against permuted *column* sums, so rank those
    col_sums = np.asarray(mm.sum(axis=0)).ravel()

    publish_shared(
        A_flat=A_flat,
        bpm_plan=bpm_plan,
        wpm_plan=wpm_plan,
        ppath=pmat[:, ind2keep_path],
        bpmsum_obs=bpmsum_obs,
        wpmsum_obs=wpmsum_obs,
        path_obs=path_degree[ind2keep_path],
        col_ranks=rankdata(col_sums),
        col_ties=tie_sum(col_sums),
        path_lens=path_lens[ind2keep_path],
        seed=seed,
        s=s,
    )

    count_bpm = np.zeros(keep_pos.size)
    count_wpm = np.zeros(kept_wpm.size)
    count_path = np.zeros(int(np.sum(ind2keep_path)))

    # Permutations are addressed by index, so the split across workers is arbitrary: piece
    # boundaries change nothing about which permutation is drawn at position k. n_workers and
    # n_jobs therefore change only the speed, never the result.
    pieces = [pc for pc in np.array_split(np.arange(snp_perms), n_workers) if pc.size]
    job_args = [perm_args(int(pc[0]), int(pc[-1]) + 1) for pc in pieces]

    pool = ctx.Pool(processes=n_workers, initializer=init_worker)
    try:
        results = pool.map(snp_permutation_parallel, job_args)
        # combine results
        for res in results:
            count_bpm = count_bpm + res[0]
            count_wpm = count_wpm + res[1]
            count_path = count_path + res[2]
        pool.close()
        pool.join()
        clear_shared()
    except KeyboardInterrupt:
        print("\nCtrl+C received — terminating worker pool...")
        pool.terminate()
        pool.join()
        sys.exit()

    print(f"- {str(datetime.now() - t1).split('.')[0]}", flush=True)

    bpm_local_pv[ind2keep_bpm] = (count_bpm + 1) / snp_perms
    wpm_local_pv[ind2keep_wpm] = (count_wpm + 1) / snp_perms
    path_degree_pv[ind2keep_path] = (count_path + 1) / snp_perms

    stats_obj = Stats(
        bpm_local=bpm_local,
        bpm_local_pv=bpm_local_pv,
        density_bpm=density_bpm,
        density_bpm_expected=density_bpm_expected,
        dense_index=dense_index,
        wpm_local=wpm_local,
        wpm_local_pv=wpm_local_pv,
        density_wpm=density_wpm,
        density_wpm_expected=density_wpm_expected,
        path_degree=path_degree,
        path_degree_pv=path_degree_pv
    )

    return stats_obj

def genstats(project_dir, ssmfile, binary_flag, net_density, snp_perms, n_jobs, n_workers, seed):

    # load pathway indices
    with open(f"{project_dir}/intermediate/pathway_indices.pkl", 'rb') as f:
        pathway_indices: bpmindclass = pickle.load(f)
    bpm = pathway_indices.bpm
    wpm = pathway_indices.wpm
    print(f"\tloaded {bpm.shape[0]:,} BPMs and {wpm.shape[0]} WPMs", flush=True)

    # load interaction network
    with open(ssmfile, 'rb') as f:
        network: InteractionNetwork = pickle.load(f)
    p_network: csr_array = as_sparse(network.protective)
    r_network: csr_array = as_sparse(network.risk)

    print(f"\tloaded protective and risk networks with {p_network.shape[0]:,} SNPs", flush=True)
    print(f"\t{p_network.shape[0] * p_network.shape[1]:,} entries in the SNP-SNP interaction network", flush=True)

    p_density = p_network.nnz / (p_network.shape[0] * p_network.shape[1]) * 100
    print(f"\t{p_density:.2f}% of the entries are nonzero in protective network", flush=True)

    r_density = r_network.nnz / (r_network.shape[0] * r_network.shape[1]) * 100
    print(f"\t{r_density:.2f}% of the entries are nonzero in risk network", flush=True)

    if binary_flag:
        if net_density is None:
            # every stored value is > 0, so this is just "set the stored values to 1"
            p_network = binarize(p_network, 0)
            r_network = binarize(r_network, 0)
        else:
            p_cutoff = sparse_quantile(p_network, 1 - net_density)
            r_cutoff = sparse_quantile(r_network, 1 - net_density)
            # A cutoff of 0 or less would binarize the zeros too, i.e. densify to an all-ones
            # s x s matrix. That is unrepresentable sparsely (and almost certainly not
            # intended), so fall back to keeping the stored entries and say so.
            tiny = np.finfo(np.float64).tiny
            for name, cutoff in (('protective', p_cutoff), ('risk', r_cutoff)):
                if cutoff <= 0:
                    print(f"\twarning: net_density={net_density} puts the {name} cutoff at "
                          f"{cutoff}; keeping all nonzero entries instead of densifying", flush=True)
            p_network = binarize(p_network, max(p_cutoff, tiny))
            r_network = binarize(r_network, max(r_cutoff, tiny))

    print(f"running genstats on protective network", flush=True)
    protective_stats = rungenstats(p_network, bpm, wpm, binary_flag, snp_perms, n_jobs, n_workers, seed)

    print(f"running genstats on risk network", flush=True)
    risk_stats = rungenstats(r_network, bpm, wpm, binary_flag, snp_perms, n_jobs, n_workers, seed)

    print(flush=True)
    out_obj = GenstatsOut(protective_stats, risk_stats)

    output_file = f"{project_dir}/intermediate/genstats_{ssmfile.split('/')[-1]}"
    with open(output_file, 'wb') as f:
        pickle.dump(out_obj, f)

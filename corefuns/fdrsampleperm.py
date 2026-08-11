import pickle

import numpy as np
import pandas as pd

from classes import Stats, GenstatsOut, fdrrclass


# fdrsampleperm() computes False Discovery Rates for BPM/WPM/PATH modules
#
# INPUTS:
#   ssmFile: Interaction networks file(path to file) in the pickle format.
#   BPMindFile: file containing SNP ids for BPM/WPMs in pickle format.
#   pcut: p-value cutoff for BPM/WPM/PATH to be considered significant and to be in FDR computing process
#   minPath: minimum size for a pathway to be considered as WPM and in BPM.
#   N: Number of random networks
#
# OUTPUTS:
#   results_<ssmFile without extension>.pkl - This pickle file contains a fdrresultclass class with fields:
#       - bpm_pv: empirical p-values for BPMs
#       - wpm_pv: empirical p-values for WPMs
#       - path_pv: empirical p-values for PATHs
#       - bpm_ranksum: -log10 ranksum p-values for BPMs
#       - wpm_ranksum: -log10 ranksum p-values for WPMs
#       - path_ranksum: -log10 ranksum p-values for PATHs
#       - fdrbpm2: FDR for BPMs
#       - fdrwpm2: FDR for WPMs
#       - fdrpath2: FDR for PATHs
#
#
# REFACTOR NOTES
#   Statistics are carried as plain (n_rows, N+1) float arrays instead of DataFrames plus lists
#   of column names. Column 0 is the real network, columns 1..N the random ones. Row r is the
#   protective copy of module r and row r + n_modules the risk copy, exactly as the original
#   np.concatenate((protective, risk)) laid them out.
#
#   calculate_fdr() was O(k * n_rows * N) pandas comparisons: for each of the k significant
#   modules it rescanned every module in every network. At 2,707,670 BPM rows, N=20 and
#   k ~ 1.4e5 that is ~7e12 element comparisons through the pandas layer. All four counts
#   (m1/m2/n1/n2) are 2-D dominance counts, so they now come from one sweep over the distinct
#   query p-values -- see _dominance_counts(). The two monotonicity corrections, also written
#   as O(k^2) rescans, are a suffix minimum and a 2-D dominance minimum. Nothing is
#   approximated: all counts are exact, verified against the original loops.
#
#   No multiprocessing. After vectorization the stage is a handful of sorts and cumulative
#   sums; workers would cost more in pickling than they save. The three calculate_fdr() calls
#   are independent and are the place to fan out if profiling ever disagrees.
#
#   Bug fixed: the while-loop collecting significant rows advanced via
#   `vals.loc[valid_row][first_pv_col] = np.nan`, which is chained assignment. Under pandas
#   copy-on-write that writes to a throwaway temporary, the row is never cleared, and the
#   loop spins forever. It is a boolean mask now.
#
#   Bug fixed: the output path was `open('/'.join(outfilename), ...)`, but outfilename is
#   already a string, so join() inserted a slash between every character.


# ---------------------------------------------------------------------------
# helpers for calculate_fdr
# ---------------------------------------------------------------------------

def _frame(values, name):
    return pd.DataFrame({name: np.asarray(values, dtype=np.float64)})


def _stack(data):
    return np.column_stack(list(data.values())).astype(np.float64, copy=False)


def _dominance_counts(pv, s, qpv, qs):
    ## Exact 2-D dominance counts of a point cloud against the k queries:
    ##     m[i] = #{j : pv[j] <= qpv[i]}
    ##     n[i] = #{j : pv[j] <= qpv[i] and s[j] >= qs[i]}
    ##
    ## Sweeps the distinct query p-values in ascending order. Points are bucketed by the
    ## distinct query score thresholds, so once a level's points are folded into the
    ## histogram a single reverse cumulative sum answers every query at that level.
    ## Empirical p-values are multiples of 1/snpPerms, so the number of levels is bounded
    ## by pcut * snpPerms + 1 (~501 at pcut=0.05, snpPerms=10000).

    levels, level_of_query = np.unique(qpv, return_inverse=True)
    thresholds, bin_of_query = np.unique(qs, return_inverse=True)
    n_bins = thresholds.size + 1

    ## a point joins the sweep at the first level >= its p-value (points past the last level
    ## never join), and satisfies s >= thresholds[j] for every j below its own bucket.
    ## entry_level is narrowed because argsort's radix sort makes fewer passes on a smaller
    ## dtype -- worth ~2x on the sort at 5e7 points.
    level_dtype = np.int16 if levels.size < np.iinfo(np.int16).max else np.int32
    entry_level = np.searchsorted(levels, pv, side='left').astype(level_dtype, copy=False)
    bucket = np.searchsorted(thresholds, s, side='right')

    p_order = np.argsort(entry_level, kind='stable')
    p_start = np.searchsorted(entry_level[p_order], np.arange(levels.size + 1), side='left')
    q_order = np.argsort(level_of_query, kind='stable')
    q_start = np.searchsorted(level_of_query[q_order], np.arange(levels.size + 1), side='left')

    hist = np.zeros(n_bins, dtype=np.int64)
    m = np.empty(qpv.size, dtype=np.int64)
    n = np.empty(qpv.size, dtype=np.int64)
    total = 0

    for level in range(levels.size):
        joining = p_order[p_start[level]:p_start[level + 1]]
        if joining.size:
            hist += np.bincount(bucket[joining], minlength=n_bins)
            total += joining.size
        queries = q_order[q_start[level]:q_start[level + 1]]
        if queries.size:
            at_or_above = np.cumsum(hist[::-1])[::-1]
            m[queries] = total
            n[queries] = at_or_above[bin_of_query[queries] + 1]

    return m, n


def _suffix_min(pv, f):
    ## out[i] = min{f[j] : pv[j] >= pv[i]}, ties in pv included.
    ##
    ## Replaces the original's O(k^2) rescan. That rescan wrote results back into the array
    ## it was scanning, but because it walked the values in descending order of f every
    ## update only ever replaced a value with the minimum over a subset of the range being
    ## minimised, which leaves the plain suffix minimum below unchanged.
    order = np.argsort(pv, kind='stable')
    suffix = np.minimum.accumulate(f[order][::-1])[::-1]
    return suffix[np.searchsorted(pv[order], pv, side='left')]


def _dominance_min(pv, s, f):
    ## out[i] = min{f[j] : pv[j] >= pv[i] and s[j] <= s[i]}, ties on both axes included.
    ##
    ## Sweeps the distinct p-values from high to low, so every point with pv >= the current
    ## level is already folded into a per-bucket running minimum over s; the prefix minimum
    ## of that array then answers every query at the level. Same in-place-update argument as
    ## _suffix_min, so this matches the original O(k^2) loop exactly.
    levels, level_of = np.unique(pv, return_inverse=True)
    thresholds, bin_of = np.unique(s, return_inverse=True)

    order = np.argsort(level_of, kind='stable')
    start = np.searchsorted(level_of[order], np.arange(levels.size + 1), side='left')

    best = np.full(thresholds.size, np.inf)
    out = np.empty(f.size, dtype=np.float64)

    for level in range(levels.size - 1, -1, -1):
        group = order[start[level]:start[level + 1]]
        np.minimum.at(best, bin_of[group], f[group])
        out[group] = np.minimum.accumulate(best)[bin_of[group]]

    return out


# ---------------------------------------------------------------------------
# Main funcs for calculate_fdr
# ---------------------------------------------------------------------------

def calculate_fdr(sdf, pvdf, pcut, N, type):
    ## inputs:
    ## - sdf, pvdf: (n_rows, N+1) arrays of ranksum scores and empirical p-values,
    ##   column 0 = real network, columns 1..N = random networks
    ## - pcut: p-value cutoff for a module to enter the FDR computation
    ## - N: number of random networks
    ## - type: 'bpm', 'wpm' or 'path'; names the output columns and selects whether the
    ##   fdr2 correction compares raw or rounded ranksum scores
    ## returns two single-column DataFrames, f'{type}1' and f'{type}2', one row per module,
    ## 1.0 for modules that did not pass pcut

    sdf1 = sdf[:, 0]
    sdf_rest = sdf[:, 1:]
    pv1 = pvdf[:, 0]
    pv_rest = pvdf[:, 1:]

    ## significant modules, in row order -- replaces the first_valid_index() while loop
    vrows = np.flatnonzero(pv1 <= pcut)
    valid_pvs = pv1[vrows]
    vpv1 = sdf1[vrows]

    rfdr1 = np.ones(pv1.size)
    rfdr2 = np.ones(pv1.size)
    if vrows.size == 0:
        return _frame(rfdr1, type + '1'), _frame(rfdr2, type + '2')

    ## m1/m2: modules at least as significant by empirical p-value, real / random networks
    ## n1/n2: modules at least as significant by BOTH empirical p-value and ranksum score
    ## The original's `.ge(0)` filters are no-ops: scores are -log10(p) >= 0 and every
    ## threshold vpv1[i] is itself one of those scores, so `score >= vpv1[i]` implies it.
    ## The 'bpm' and 'wpm'/'path' branches of the original loop were byte-identical, so
    ## there is only one path here.
    m1, n1 = _dominance_counts(pv1, sdf1, valid_pvs, vpv1)
    m2, n2 = _dominance_counts(pv_rest.ravel(), sdf_rest.ravel(), valid_pvs, vpv1)

    with np.errstate(divide='ignore', invalid='ignore'):
        fdr1 = np.nan_to_num(m2 / (N * m1))
        fdr2 = np.nan_to_num(n2 / (N * n1))

    ## correct FDRs so BPM/WPM/PATH with lower p-vals does not have larger FDRs
    rfdr1[vrows] = _suffix_min(valid_pvs, fdr1)

    ## same for fdr2, but a module is also not allowed a larger FDR than any module that is
    ## worse on both axes. WPM/PATH scores are compared rounded to whole numbers here, and
    ## only here -- the counts above use the raw values, as in the original.
    key = vpv1 if type == 'bpm' else np.round(vpv1)
    rfdr2[vrows] = _dominance_min(valid_pvs, key, fdr2)

    return _frame(rfdr1, type + '1'), _frame(rfdr2, type + '2')


def fdrsampleperm(ssmFile, pcut, N):
    ## one entry per network, keyed exactly like the original DataFrame columns
    bpm_data, bpm_pv_data = {}, {}
    wpm_data, wpm_pv_data = {}, {}
    path_data, path_pv_data = {}, {}

    for i in range(0, N + 1):
        tssmFile = ssmFile.replace("_R0", "_R" + str(i))
        tssm_tmp = tssmFile.split('/')
        tssm_tmp[-1] = 'genstats_' + tssm_tmp[-1]
        genstatsfile = '/'.join(tssm_tmp)

        ## load genstats file
        with open(genstatsfile, "rb") as pklin:
            gs: GenstatsOut = pickle.load(pklin)

        prot: Stats = gs.protective_stats
        risk: Stats = gs.risk_stats

        ## retrieve bpm/wpm/path stats, protective followed by risk
        bpm_data["bpm" + str(i)] = np.concatenate((prot.bpm_local, risk.bpm_local))
        bpm_pv_data["bpm_pv" + str(i)] = np.concatenate((prot.bpm_local_pv, risk.bpm_local_pv))
        wpm_data["wpm" + str(i)] = np.concatenate((prot.wpm_local, risk.wpm_local))
        wpm_pv_data["wpm_pv" + str(i)] = np.concatenate((prot.wpm_local_pv, risk.wpm_local_pv))
        path_data["path" + str(i)] = np.concatenate((prot.path_degree, risk.path_degree))
        path_pv_data["path_pv" + str(i)] = np.concatenate((prot.path_degree_pv, risk.path_degree_pv))

    ## stack into (n_rows, N+1) arrays; column 0 is the real network
    bpm = _stack(bpm_data)
    bpm_pv = _stack(bpm_pv_data)
    wpm = _stack(wpm_data)
    wpm_pv = _stack(wpm_pv_data)
    path = _stack(path_data)
    path_pv = _stack(path_pv_data)

    # calling calculate_fdr() function to compute FDRs
    fdrBPM1, fdrBPM2 = calculate_fdr(bpm, bpm_pv, pcut, N, 'bpm')
    fdrWPM1, fdrWPM2 = calculate_fdr(wpm, wpm_pv, pcut, N, 'wpm')
    fdrPATH1, fdrPATH2 = calculate_fdr(path, path_pv, pcut, N, 'path')

    bpm_ranksum = _frame(bpm[:, 0], 'bpm_ranksum')
    wpm_ranksum = _frame(wpm[:, 0], 'wpm_ranksum')
    path_ranksum = _frame(path[:, 0], 'path_ranksum')
    bpm_pv = _frame(bpm_pv[:, 0], 'bpm_pv')
    wpm_pv = _frame(wpm_pv[:, 0], 'wpm_pv')
    path_pv = _frame(path_pv[:, 0], 'path_pv')

    ssm_tmp = ssmFile.split('/')
    ssm_tmp[-1] = 'results_' + ssm_tmp[-1]
    outfilename = '/'.join(ssm_tmp)
    save_obj = fdrrclass(
        bpm_pv, wpm_pv, path_pv,
        bpm_ranksum, wpm_ranksum, path_ranksum,
        fdrBPM1, fdrBPM2, fdrWPM1, fdrWPM2, fdrPATH1, fdrPATH2,
        )

    with open(outfilename, 'wb') as fh:
        pickle.dump(save_obj, fh)

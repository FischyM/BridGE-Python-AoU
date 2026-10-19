import math

import numpy as np
import pandas as pd


FDR_STEP = 0.05
SIM_CUTOFF = 0.25

def bpmsim(BPMind1x, BPMind2x, BPMind1y, BPMind2y):
    """Similarity between two sets of BPMs; same output as bpmsim.bpmsim.

    Entry (i, j) is max(|1x_i & 1y_j| * |2x_i & 2y_j|, |1x_i & 2y_j| * |2x_i & 1y_j|)
    divided by min(|1x_i| * |2x_i|, |1y_j| * |2y_j|). When the x and y sets are
    identical, only the upper triangle is computed, it is mirrored, and the
    diagonal is left at 0 (as scipy's squareform does in bpmsim.bpmsim).
    """
    nx = BPMind1x.shape[0]
    ny = BPMind1y.shape[0]

    sim = np.zeros((nx, ny))
    test = BPMind1x.equals(BPMind1y) and BPMind2x.equals(BPMind2y)

    # Convert once to sets so intersections use set semantics, as in bpmsim.bpmsim.
    s1x = [set(v) for v in BPMind1x]
    s2x = [set(v) for v in BPMind2x]
    s1y = s1x if test else [set(v) for v in BPMind1y]
    s2y = s2x if test else [set(v) for v in BPMind2y]
    # Sizes come from the original lists (bpmsim.bpmsim uses len() of the lists).
    nprod_x = [len(a) * len(b) for a, b in zip(BPMind1x, BPMind2x)]
    nprod_y = nprod_x if test else [len(a) * len(b) for a, b in zip(BPMind1y, BPMind2y)]

    for i in range(nx):
        for j in range(i + 1 if test else 0, ny):
            l = min(nprod_x[i], nprod_y[j])
            tmp1 = len(s1x[i] & s1y[j]) * len(s2x[i] & s2y[j])
            tmp2 = len(s1x[i] & s2y[j]) * len(s2x[i] & s1y[j])
            # np.divide keeps bpmsim.bpmsim's behavior for l == 0 (nan/inf, not an error).
            sim[i, j] = np.divide(max(tmp1, tmp2), l)

    if test:
        sim = sim + sim.T
    return sim

def pathsim(pathind):
    """Pairwise similarity between pathways (NumPy-only, vectorized).

    sim[i, j] = |set(p_i) & set(p_j)| / min(len(p_i), len(p_j)) for i != j,
    with a zero diagonal. Same output as the original double-loop version.

    Args:
        pathind (Series): one list/array of SNP indices per pathway.

    Returns:
        ndarray: (n, n) float64 symmetric similarity matrix.

    Raises:
        ZeroDivisionError: if there are 2+ pathways and any is empty (as the
            original loop did).
    """
    paths = list(pathind)
    n = len(paths)
    sim = np.zeros((n, n))
    if n < 2:
        return sim

    # Denominator uses list lengths (not set sizes), as in the original.
    sizes = np.array([len(p) for p in paths], dtype=np.int64)
    if (sizes == 0).any():
        raise ZeroDivisionError("pathsim: empty pathway")

    # Pathway x unique-SNP 0/1 matrix; duplicate SNPs within a pathway collapse to 1.
    rows = np.repeat(np.arange(n), sizes)
    _, cols = np.unique(np.concatenate([np.asarray(p).ravel() for p in paths]), return_inverse=True)
    member = np.zeros((n, cols.max() + 1), dtype=np.float32)
    member[rows, cols] = 1.0

    # All pairwise overlap counts in one matrix multiply (exact: counts < 2**24).
    overlap = (member @ member.T).astype(np.float64)
    sim = overlap / np.minimum(sizes[:, None], sizes[None, :])
    np.fill_diagonal(sim, 0.0)
    return sim

def _greedy_groups(fdrs, similar):
    """Assign redundancy groups walking from the most to the least significant module.

    Each module joins the group of the first more-significant module it is
    similar to, otherwise it starts a new group. This is deliberately not
    connected-components: for a chain A~B, B~C where A is not similar to C, C
    stays in B's group.

    Args:
        fdrs (Series): FDRs of the modules, indexed by global module index, in
            the same row order as `similar`.
        similar (ndarray): Boolean square matrix, True where two modules are
            redundant. Rows and columns follow the row order of `fdrs`.

    Returns:
        Series: group label per module, indexed by global module index, ordered
            by ascending FDR.
    """
    rank = fdrs.argsort(kind='stable').to_numpy()
    labels = np.zeros(rank.shape[0], dtype=np.int64)
    for x1 in range(1, rank.shape[0]):
        for x2 in range(x1 + 1):
            if x2 == x1:
                labels[x1] = labels.max() + 1
            elif similar[rank[x1], rank[x2]]:
                labels[x1] = labels[x2]
                break
    return pd.Series(labels, index=fdrs.index[rank])

def _bpm_similar(bpm, local_ind):
    # local_ind holds row POSITIONS in bpmind.bpm, not index labels: the FDR
    # frames are laid out in the row order of bpmind.bpm, whose index can have
    # gaps (pathway pairs dropped upstream). Label lookup would KeyError.
    ind1 = bpm['ind1'].iloc[local_ind]
    ind2 = bpm['ind2'].iloc[local_ind]
    return bpmsim(ind1, ind2, ind1, ind2) >= SIM_CUTOFF

def _wpm_similar(wpm, local_ind):
    ind = wpm['ind'].iloc[local_ind]
    return bpmsim(ind, ind, ind, ind) >= SIM_CUTOFF

def _path_similar(wpm, local_ind):
    return pathsim(wpm['ind'].iloc[local_ind]) >= SIM_CUTOFF

def _groups_at_threshold(fdr_frame, fdr_col, n_modules, fdrcut, pathway_ind_df, similar_fn):
    """Redundancy groups for one module type at one FDR threshold.

    Protective modules occupy global indices 0..n_modules-1 and risk modules
    n_modules..2*n_modules-1. The two directions are grouped independently, then
    the risk labels are offset so the two label sets do not collide.

    Returns:
        tuple[Series, int]: group label per module indexed by global module
            index, and the total number of distinct groups.
    """
    ind = np.asarray(fdr_frame[fdr_frame <= fdrcut].dropna().index)
    protective = ind[ind < n_modules]
    risk = ind[ind >= n_modules]

    parts, n_groups = [], 0
    for global_ind, local_ind in ((protective, protective), (risk, risk - n_modules)):
        if global_ind.size == 0:
            continue
        if global_ind.size == 1:
            labels = pd.Series([0], index=global_ind, dtype=np.int64)
        else:
            fdrs = fdr_frame.loc[global_ind][fdr_col]
            labels = _greedy_groups(fdrs, similar_fn(pathway_ind_df, local_ind))
        parts.append(labels + n_groups)
        n_groups += labels.nunique()

    if not parts:
        return pd.Series(dtype=np.int64), 0
    return pd.concat(parts), n_groups

def check_BPM_WPM_redundancy(project_dir, fdrBPM, fdrWPM, fdrPATH, FDRcut):
    """Groups redundant BPMs/WPMs/PATHs at every 0.05 FDR threshold up to FDRcut.

    Args:
        fdrBPM (DataFrame): single FDR column 'bpm2'. Protective modules are the
            first half of the rows, risk modules the second half.
        fdrWPM (DataFrame): as above, column 'wpm2'.
        fdrPATH (DataFrame): as above, column 'path2'.
        bpmindfile (str): pickle with the SNP ids for each BPM/WPM.
        FDRcut (float): highest FDR threshold to group at.

    Returns:
        tuple: six lists, each with one entry per 0.05 threshold from 0.05 up to
        FDRcut:
            - BPM_nosig_noRD, WPM_nosig_noRD, PATH_nosig_noRD (int): number of
              non-redundant modules at that threshold.
            - BPM_group, WPM_group, PATH_group (Series): group label per module,
              indexed by GLOBAL MODULE INDEX and ordered by ascending FDR within
              each effect direction. Callers must align by index, not position:
              the ordering is protective-then-risk, which is not the same as
              global FDR rank.
    """
    bpm = pd.read_parquet(f"{project_dir}/intermediate/pathway_indices-bpm.parquet")
    wpm = pd.read_parquet(f"{project_dir}/intermediate/pathway_indices-wpm.parquet")
    n_bpm = len(bpm['size'])
    n_wpm = len(wpm['size'])

    BPM_group, BPM_nosig_noRD = [], []
    WPM_group, WPM_nosig_noRD = [], []
    PATH_group, PATH_nosig_noRD = [], []

    for level in range(1, math.ceil(FDRcut / FDR_STEP) + 1):
        fdrcut = level * FDR_STEP

        labels, n_groups = _groups_at_threshold(fdrBPM, 'fdrbpm2', n_bpm, fdrcut, bpm, _bpm_similar)
        BPM_group.append(labels)
        BPM_nosig_noRD.append(n_groups)

        labels, n_groups = _groups_at_threshold(fdrWPM, 'fdrwpm2', n_wpm, fdrcut, wpm, _wpm_similar)
        WPM_group.append(labels)
        WPM_nosig_noRD.append(n_groups)

        labels, n_groups = _groups_at_threshold(fdrPATH, 'fdrpath2', n_wpm, fdrcut, wpm, _path_similar)
        PATH_group.append(labels)
        PATH_nosig_noRD.append(n_groups)

    return (BPM_nosig_noRD, WPM_nosig_noRD, PATH_nosig_noRD, BPM_group, WPM_group, PATH_group)

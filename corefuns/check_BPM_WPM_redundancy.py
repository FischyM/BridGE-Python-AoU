import math
import pickle

import numpy as np
import pandas as pd

from corefuns import bpmsim, pathsim

FDR_STEP = 0.05
SIM_CUTOFF = 0.25


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

def _bpm_similar(bpmind, local_ind):
    # local_ind holds row POSITIONS in bpmind.bpm, not index labels: the FDR
    # frames are laid out in the row order of bpmind.bpm, whose index can have
    # gaps (pathway pairs dropped upstream). Label lookup would KeyError.
    ind1 = bpmind.bpm['ind1'].iloc[local_ind]
    ind2 = bpmind.bpm['ind2'].iloc[local_ind]
    return bpmsim.bpmsim(ind1, ind2, ind1, ind2) >= SIM_CUTOFF

def _wpm_similar(bpmind, local_ind):
    ind = bpmind.wpm['ind'].iloc[local_ind]
    return bpmsim.bpmsim(ind, ind, ind, ind) >= SIM_CUTOFF

def _path_similar(bpmind, local_ind):
    return pathsim.pathsim(bpmind.wpm['ind'].iloc[local_ind]) >= SIM_CUTOFF

def _groups_at_threshold(fdr_frame, fdr_col, n_modules, fdrcut, bpmind, similar_fn):
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
            labels = _greedy_groups(fdrs, similar_fn(bpmind, local_ind))
        parts.append(labels + n_groups)
        n_groups += labels.nunique()

    if not parts:
        return pd.Series(dtype=np.int64), 0
    return pd.concat(parts), n_groups

def check_BPM_WPM_redundancy(fdrBPM, fdrWPM, fdrPATH, bpmindfile, FDRcut):
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
    with open(bpmindfile, 'rb') as fh:
        bpmind = pickle.load(fh)

    n_bpm = len(bpmind.bpm['size'])
    n_wpm = len(bpmind.wpm['size'])

    BPM_group, BPM_nosig_noRD = [], []
    WPM_group, WPM_nosig_noRD = [], []
    PATH_group, PATH_nosig_noRD = [], []

    for level in range(1, math.ceil(FDRcut / FDR_STEP) + 1):
        fdrcut = level * FDR_STEP

        labels, n_groups = _groups_at_threshold(fdrBPM, 'bpm2', n_bpm, fdrcut, bpmind, _bpm_similar)
        BPM_group.append(labels)
        BPM_nosig_noRD.append(n_groups)

        labels, n_groups = _groups_at_threshold(fdrWPM, 'wpm2', n_wpm, fdrcut, bpmind, _wpm_similar)
        WPM_group.append(labels)
        WPM_nosig_noRD.append(n_groups)

        labels, n_groups = _groups_at_threshold(fdrPATH, 'path2', n_wpm, fdrcut, bpmind, _path_similar)
        PATH_group.append(labels)
        PATH_nosig_noRD.append(n_groups)

    return (BPM_nosig_noRD, WPM_nosig_noRD, PATH_nosig_noRD, BPM_group, WPM_group, PATH_group)

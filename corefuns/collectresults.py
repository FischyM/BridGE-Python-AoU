import pickle

import numpy as np
import pandas as pd

from classes import fdrrclass, bpmindclass
from corefuns import check_BPM_WPM_redundancy as cbwr
from corefuns import get_interaction_pair as gpair
from corefuns import pathway_map as pmap


FDR_STEP = 0.05

def _stack(series):
    """Duplicate a per-module series: rows 0..n-1 protective, n..2n-1 risk.

    This is the row layout of the FDR / p-value / ranksum frames in the results
    pickle. Returns a one-column DataFrame with a fresh RangeIndex.
    """
    return pd.concat([series, series], ignore_index=True).to_frame()


def _reorder(frame, order):
    """Put a frame in FDR-sorted order and drop back to a RangeIndex."""
    return frame.reindex(index=order).reset_index(drop=True)


def _fdr_levels(fdrcut):
    """Number of FDR_STEP thresholds in fdrcut, which must be a multiple of it."""
    levels = fdrcut / FDR_STEP
    if abs(levels - round(levels)) > 1e-9:
        raise ValueError(f'fdrcut must be a multiple of {FDR_STEP}, got {fdrcut}.')
    return int(round(levels))


def _effect(ind, n_modules, column):
    """Label each significant module protective or risk.

    Protective modules occupy global indices 0..n_modules-1, risk modules
    n_modules..2*n_modules-1. Keeps ind's index so the labels reorder alongside
    every other column.
    """
    labels = np.where(np.asarray(ind.index) < n_modules, 'protective', 'risk')
    return pd.DataFrame({column: labels}, index=ind.index)


def _group_column(groups_per_level, order, label):
    """Redundant-group label for each reported module.

    check_BPM_WPM_redundancy returns one Series per FDR_STEP threshold indexed by
    global module index, so the last one is the level at exactly fdrcut and
    aligns to `order` by label rather than by position.
    """
    labels = groups_per_level[-1].reindex(order)
    missing = int(labels.isna().sum())
    if missing:
        raise ValueError(f'{label}: no redundancy group label for {missing} reported modules.')
    return labels.astype(np.int64).reset_index(drop=True).to_frame('group')


def collectresults(project_dir, ssmfile, model, fdrcut, imported_ssm, densitycutoff):
    """Collects BPM/WPM/PATH results and exports them to an Excel workbook.

    Also calls out to driver-gene discovery and redundant-module grouping.

    Args:
        project_dir (str): Path to the project directory.
        ssmfile (str): Path to the real-network interaction file (pickle).
        model (str): Model name, e.g. "combined".
        fdrcut (float): FDR threshold for keeping modules. Must be a multiple of 0.05.
        imported_ssm (bool): True when the interaction network was imported rather than computed from genotypes.
        densitycutoff (float, optional): Network density cutoff, passed through to get_interaction_pair.

    Returns:
        str: Path to the written workbook, which contains:
            - output_discovery_summary: module counts per FDR threshold.
            - output_noRD_discovery_summary: non-redundant counts per threshold.
            - output_bpm_table / output_wpm_table / output_path_table: the
              modules below fdrcut with their stats and driver genes.
    """
    results_file = f"{project_dir}/intermediate/results_{ssmfile.split('/')[-1]}"
    with open(results_file, 'rb') as f:
        results: fdrrclass = pickle.load(f)
    fdrBPM = results.fdrbpm2
    fdrWPM = results.fdrwpm2
    fdrPATH = results.fdrpath2

    ind_bpm = fdrBPM[fdrBPM <= fdrcut].dropna()
    ind_wpm = fdrWPM[fdrWPM <= fdrcut].dropna()
    ind_path = fdrPATH[fdrPATH <= fdrcut].dropna()

    # Narrow the stats frames to the significant modules. Label-based throughout;
    # ind_*.index holds labels from these same frames.
    if not ind_bpm.empty:
        fdrBPM = fdrBPM.loc[ind_bpm.index]
        bpm_pv = results.bpm_pv.loc[ind_bpm.index]
        bpm_ranksum = results.bpm_ranksum.loc[ind_bpm.index]

    if not ind_wpm.empty:
        fdrWPM = fdrWPM.loc[ind_wpm.index]
        wpm_pv = results.wpm_pv.loc[ind_wpm.index]
        wpm_ranksum = results.wpm_ranksum.loc[ind_wpm.index]

    if not ind_path.empty:
        fdrPATH = fdrPATH.loc[ind_path.index]
        path_pv = results.path_pv.loc[ind_path.index]
        path_ranksum = results.path_ranksum.loc[ind_path.index]

    # --- pathway names, sizes, effect direction, driver genes ---------------
    # snppathwayfile is not read here: bridge.py checks it exists and
    # get_interaction_pair loads it (and the geneset it points at) itself.
    
    pathway_inds_pkl = f"{project_dir}/intermediate/pathway_indices.pkl"
    if not (ind_bpm.empty and ind_wpm.empty and ind_path.empty):
        with open(pathway_inds_pkl, 'rb') as f:
            bpm_ind: bpmindclass = pickle.load(f)
        pathways = bpm_ind.wpm['pathway']
        path_ids = {name: i for i, name in enumerate(pathways)}
        n_bpm = len(bpm_ind.bpm.index)
        n_wpm = len(bpm_ind.wpm.index)

        if not ind_bpm.empty:
            path1 = _stack(bpm_ind.bpm['path1names']).loc[ind_bpm.index]
            path2 = _stack(bpm_ind.bpm['path2names']).loc[ind_bpm.index]
            bpm_size = _stack(bpm_ind.bpm['size']).loc[ind_bpm.index]
            eff_bpm = _effect(ind_bpm, n_bpm, 'eff_bpm')
            bpm_path1_drivers, bpm_path2_drivers, _ = gpair.get_interaction_pair(
                project_dir=project_dir,
                ssmfile=ssmfile,
                model=model,
                n=len(ind_bpm),
                path1=path1,
                path2=path2,
                effects=eff_bpm,
                path_ids=path_ids,
                fdrcutoff=fdrcut,
                imported_ssm=imported_ssm,
                densitycutoff=densitycutoff,
                )

        if not ind_wpm.empty:
            path_wpm = _stack(pathways).loc[ind_wpm.index]
            wpm_size = _stack(bpm_ind.wpm['size']).loc[ind_wpm.index]
            eff_wpm = _effect(ind_wpm, n_wpm, 'eff_wpm')
            _, _, wpm_path_drivers = gpair.get_interaction_pair(
                project_dir=project_dir,
                ssmfile=ssmfile,
                model=model,
                n=len(ind_wpm),
                path1=path_wpm,
                path2=path_wpm,
                effects=eff_wpm,
                path_ids=path_ids,
                fdrcutoff=fdrcut,
                imported_ssm=imported_ssm,
                densitycutoff=densitycutoff,
                )

        if not ind_path.empty:
            path_path = _stack(pathways).loc[ind_path.index]
            path_size = _stack(bpm_ind.wpm['indsize']).loc[ind_path.index]
            eff_path = _effect(ind_path, n_wpm, 'eff_path')

    # --- redundancy grouping and the pathway map ----------------------------
    (BPM_nosig_noRD, WPM_nosig_noRD, PATH_nosig_noRD,
     BPM_groups, WPM_groups, PATH_groups) = cbwr.check_BPM_WPM_redundancy(
         fdrBPM=fdrBPM,
         fdrWPM=fdrWPM,
         fdrPATH=fdrPATH,
         bpmindfile=pathway_inds_pkl,
         FDRcut=fdrcut,
         )

    pmap.draw_map(project_dir, fdrcut, results_file, BPM_groups, WPM_groups, PATH_groups)

    # --- output tables ------------------------------------------------------
    # `order` is a stable FDR sort. Group labels are matched to it by module
    # index, so the two orderings do not have to agree.
    # DOUBLE CHECK SORTING WHEN FIXING BPMSIM AND PATHSIM.
    output_bpm_table = output_wpm_table = output_path_table = None

    if not ind_bpm.empty:
        order = fdrBPM.sort_values(kind='stable', by='bpm2').index
        output_bpm_table = pd.concat([
            _reorder(path1, order),
            _reorder(path2, order),
            _group_column(BPM_groups, order, 'BPM'),
            _reorder(fdrBPM, order).round(2),
            _reorder(eff_bpm, order),
            _reorder(bpm_size, order),
            _reorder(bpm_pv, order),
            _reorder(bpm_ranksum, order).round(2),
            _reorder(bpm_path1_drivers, order),
            _reorder(bpm_path2_drivers, order),
        ], axis=1)
        output_bpm_table = output_bpm_table.rename(
            columns={'bpm2': 'fdrBPM', 'size': 'bpm_size'}
        ).sort_values(by=['fdrBPM', 'bpm_pv', 'bpm_ranksum'],
                      ascending=[True, True, False]).reset_index(drop=True)

    if not ind_wpm.empty:
        order = fdrWPM.sort_values(kind='stable', by='wpm2').index
        output_wpm_table = pd.concat([
            _reorder(path_wpm, order),
            _group_column(WPM_groups, order, 'WPM'),
            _reorder(fdrWPM, order).round(2),
            _reorder(eff_wpm, order),
            _reorder(wpm_size, order),
            _reorder(wpm_pv, order),
            _reorder(wpm_ranksum, order).round(2),
            _reorder(wpm_path_drivers, order),
        ], axis=1)
        output_wpm_table = output_wpm_table.rename(
            columns={'wpm2': 'fdrWPM', 'size': 'wpm_size'}
        ).sort_values(by=['fdrWPM', 'wpm_pv', 'wpm_ranksum'],
                      ascending=[True, True, False]).reset_index(drop=True)

    if not ind_path.empty:
        order = fdrPATH.sort_values(kind='stable', by='path2').index
        output_path_table = pd.concat([
            _reorder(path_path, order),
            _group_column(PATH_groups, order, 'PATH'),
            _reorder(fdrPATH, order).round(2),
            _reorder(eff_path, order),
            _reorder(path_size, order),
            _reorder(path_pv, order),
            _reorder(path_ranksum, order).round(2),
        ], axis=1)
        output_path_table = output_path_table.rename(
            columns={'path2': 'fdrPATH', 'indsize': 'path_size'}
        ).sort_values(by=['fdrPATH', 'path_pv', 'path_ranksum'],
                      ascending=[True, True, False]).reset_index(drop=True)

    # --- summary sheets -----------------------------------------------------
    n_levels = _fdr_levels(fdrcut)
    header = ['minfdr'] + [ f'fdr{int(round(k * FDR_STEP * 100)):02d}' for k in range(1, n_levels + 1) ]

    # One row per module type with results. PATH is checked on its own; the
    # original gated both the WPM and PATH rows on WPM being non-empty.
    rows = [('BPM', fdrBPM, 'bpm2', BPM_nosig_noRD)]
    if not fdrWPM.empty:
        rows.append(('WPM', fdrWPM, 'wpm2', WPM_nosig_noRD))
    if not fdrPATH.empty:
        rows.append(('PATH', fdrPATH, 'path2', PATH_nosig_noRD))

    index = [name for name, _, _, _ in rows]
    discovery = [[frame[col].min()]
                 + [int((frame[col] <= k * FDR_STEP).sum()) for k in range(1, n_levels + 1)]
                 for _, frame, col, _ in rows]
    noRD = [ [frame[col].min()] + nosig for _, frame, col, nosig in rows ]

    output_discovery_summary = pd.DataFrame(discovery, columns=header, index=index)
    output_noRD_discovery_summary = pd.DataFrame(noRD, columns=header, index=index)

    # --- write --------------------------------------------------------------
    out_file = f"{project_dir}/results/output_{results_file.split('/')[-1].split(".")[0]}.xlsx"
    sheets = {
        'output_discovery_summary': output_discovery_summary,
        'output_noRD_discovery_summary': output_noRD_discovery_summary,
        'output_bpm_table': output_bpm_table,
        'output_wpm_table': output_wpm_table,
        'output_path_table': output_path_table,
    }
    with pd.ExcelWriter(out_file) as writer:
        for name, table in sheets.items():
            if table is not None:
                table.to_excel(writer, sheet_name=name)

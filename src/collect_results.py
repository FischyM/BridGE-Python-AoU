from pathlib import Path

import numpy as np
import polars as pl

from src import get_interaction_pairs as gpair
from src import check_BPM_WPM_redundancy as cbwr
from src import plot_pathway as path_plot
from src.polars_io import to_workbook


FDR_STEP = 0.05

def _stack(values):
    """Duplicate a per-module column: rows 0..n-1 protective, n..2n-1 risk.

    This is the row layout of the FDR / p-value / ranksum arrays in the results
    file, so the result is indexed by global module index.
    """
    values = np.asarray(values)
    return np.concatenate([values, values])


def _fdr_levels(fdrcut):
    """Number of FDR_STEP thresholds in fdrcut, which must be a positive multiple of it."""
    levels = fdrcut / FDR_STEP
    if abs(levels - round(levels)) > 1e-9 or round(levels) < 1:
        raise ValueError(f'fdrcut must be a positive multiple of {FDR_STEP}, got {fdrcut}.')
    return int(round(levels))


def _effect(modules, n_modules):
    """Label each module protective or risk.

    Protective modules occupy global indices 0..n_modules-1, risk modules
    n_modules..2*n_modules-1.
    """
    return np.where(modules < n_modules, 'protective', 'risk')


def _by_module(frame, order, label=None):
    """frame's last column for each module in `order` (frame has columns 'module' and one value).

    Modules missing from frame give null, or a ValueError naming `label` if given.
    """
    values = pl.DataFrame({'module': order}).join(frame, on='module', how='left', maintain_order='left')
    values = values[frame.columns[-1]]
    missing = values.null_count()
    if label is not None and missing:
        raise ValueError(f'{label}: no redundancy group label for {missing} reported modules.')
    return values


def _group_column(groups_per_level, order, label):
    """Redundant-group label for each reported module.

    check_BPM_WPM_redundancy returns one frame per FDR_STEP threshold with columns
    'module' (global module index) and 'group', so the last one is the level at
    exactly fdrcut and is matched to `order` by module rather than by position.
    """
    return _by_module(groups_per_level[-1], order, label)


def _sort_table(table, by):
    """Sort a module table by FDR and p-value (ascending) then ranksum (descending).

    Stable, with NaN last in every key, as pandas' multi-column sort_values; then
    an unnamed 0..n-1 index column goes first, as pandas' reset_index wrote.
    """
    return (table.with_columns(pl.col(pl.Float64).fill_nan(None))
                 .sort(by, descending=[False, False, True], nulls_last=True, maintain_order=True)
                 .with_row_index(''))


def _significant(fdr, fdrcut):
    """Global module indices with FDR <= fdrcut (NaN never passes)."""
    return np.flatnonzero(fdr <= fdrcut)


def run_results(project_dir, snp_int_file, model, fdrcut, imported_ssm, densitycutoff):
    """Collects BPM/WPM/PATH results and exports them to an Excel workbook.

    Also calls out to driver-gene discovery and redundant-module grouping.

    Args:
        project_dir (str): Path to the project directory.
        snp_int_file (str): Path to the real-network interaction file (.npz).
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
    snp_int_file = Path(snp_int_file)
    # validate up front: redundancy grouping and the map need at least one FDR level
    n_levels = _fdr_levels(fdrcut)

    output_file = Path(f"{project_dir}/results/output-{'-'.join(snp_int_file.stem.split('-')[-2:])}.xlsx")

    result_file_name = f"results-{'-'.join(snp_int_file.stem.split('-')[-2:])}.npz"
    result_file = snp_int_file.with_name(result_file_name)
    results = np.load(result_file)

    # every stats array is indexed by global module index
    fdr_bpm_all = results['fdrbpm2'].ravel()
    fdr_wpm_all = results['fdrwpm2'].ravel()
    fdr_path_all = results['fdrpath2'].ravel()

    ind_bpm = _significant(fdr_bpm_all, fdrcut)
    ind_wpm = _significant(fdr_wpm_all, fdrcut)
    ind_path = _significant(fdr_path_all, fdrcut)
    print(f"    Found {len(ind_bpm)} BPMs, {len(ind_wpm)} WPMs, and {len(ind_path)} PATHs below FDR {fdrcut}.")

    # FDR frames (columns 'module' and the FDR), narrowed to the significant
    # modules when there are any; the stats arrays below are aligned with ind_*.
    fdrBPM = pl.DataFrame({'module': np.arange(fdr_bpm_all.size), 'fdrbpm2': fdr_bpm_all})
    fdrWPM = pl.DataFrame({'module': np.arange(fdr_wpm_all.size), 'fdrwpm2': fdr_wpm_all})
    fdrPATH = pl.DataFrame({'module': np.arange(fdr_path_all.size), 'fdrpath2': fdr_path_all})

    if ind_bpm.size:
        fdrBPM = fdrBPM[ind_bpm]
        bpm_pv = results['bpm_pv'].ravel()[ind_bpm]
        bpm_ranksum = results['bpm_ranksum'].ravel()[ind_bpm]

    if ind_wpm.size:
        fdrWPM = fdrWPM[ind_wpm]
        wpm_pv = results['wpm_pv'].ravel()[ind_wpm]
        wpm_ranksum = results['wpm_ranksum'].ravel()[ind_wpm]

    if ind_path.size:
        fdrPATH = fdrPATH[ind_path]
        path_pv = results['path_pv'].ravel()[ind_path]
        path_ranksum = results['path_ranksum'].ravel()[ind_path]

    # --- pathway names, sizes, effect direction, driver genes ---------------
    # get_interaction_pair loads the snp/gene/pathway mappings itself.
    if ind_bpm.size or ind_wpm.size or ind_path.size:
        print(f"    Finding driver SNP pairs for significant BPMs/WPMs")
        bpm = pl.read_parquet(f"{project_dir}/intermediate/pathway_indices-bpm.parquet",
                              columns=['path1names', 'path2names', 'size'])
        wpm = pl.read_parquet(f"{project_dir}/intermediate/pathway_indices-wpm.parquet",
                              columns=['pathway', 'indsize', 'size'])
        pathways = wpm['pathway'].to_numpy()
        path_ids = {name: i for i, name in enumerate(pathways)}
        n_bpm = bpm.height
        n_wpm = wpm.height

        if ind_bpm.size:
            path1 = _stack(bpm['path1names'].to_numpy())[ind_bpm]
            path2 = _stack(bpm['path2names'].to_numpy())[ind_bpm]
            bpm_size = _stack(bpm['size'].to_numpy())[ind_bpm]
            eff_bpm = _effect(ind_bpm, n_bpm)
            bpm_path1_drivers, bpm_path2_drivers, _ = gpair.get_interaction_pair(
                project_dir=project_dir,
                snp_int_file=snp_int_file,
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

        if ind_wpm.size:
            path_wpm = _stack(pathways)[ind_wpm]
            wpm_size = _stack(wpm['size'].to_numpy())[ind_wpm]
            eff_wpm = _effect(ind_wpm, n_wpm)
            _, _, wpm_path_drivers = gpair.get_interaction_pair(
                project_dir=project_dir,
                snp_int_file=snp_int_file,
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

        if ind_path.size:
            path_path = _stack(pathways)[ind_path]
            path_size = _stack(wpm['indsize'].to_numpy())[ind_path]
            eff_path = _effect(ind_path, n_wpm)
    else:
        print("    No driver SNP pairs saved as there were no significant BPMs/WPMs.")


    # --- redundancy grouping and the pathway map ----------------------------
    (BPM_nosig_noRD, WPM_nosig_noRD, PATH_nosig_noRD,
        BPM_groups, WPM_groups, PATH_groups) = cbwr.check_BPM_WPM_redundancy(
            project_dir=project_dir,
            fdrBPM=fdrBPM,
            fdrWPM=fdrWPM,
            fdrPATH=fdrPATH,
            FDRcut=fdrcut,
            )
    print(f"    Found {BPM_nosig_noRD[-1]} non-redundant BPMs, {WPM_nosig_noRD[-1]} non-redundant WPMs, and {PATH_nosig_noRD[-1]} non-redundant PATHs below FDR {fdrcut}.")

    path_plot.draw_map(project_dir, fdrcut, output_file, BPM_groups, WPM_groups, PATH_groups, fdr_bpm_all)


    # --- output tables ------------------------------------------------------
    # `rank` is a stable FDR sort of the significant modules and `order` their
    # module indices in that order. Group labels and driver strings are matched
    # to it by module index, so their own orderings do not have to agree.
    # DOUBLE CHECK SORTING WHEN FIXING BPMSIM AND PATHSIM.
    output_bpm_table = output_wpm_table = output_path_table = None

    if ind_bpm.size:
        fdr = fdrBPM['fdrbpm2'].to_numpy()
        rank = np.argsort(fdr, kind='stable')
        order = ind_bpm[rank]
        output_bpm_table = _sort_table(pl.DataFrame({
            'path1names': path1[rank],
            'path2names': path2[rank],
            'group': _group_column(BPM_groups, order, 'BPM'),
            'fdrBPM': np.round(fdr[rank], 2),
            'eff_bpm': eff_bpm[rank],
            'bpm_size': bpm_size[rank],
            'bpm_pv': bpm_pv[rank],
            'bpm_ranksum': np.round(bpm_ranksum[rank], 2),
            'bpm_path1_drivers': _by_module(bpm_path1_drivers, order),
            'bpm_path2_drivers': _by_module(bpm_path2_drivers, order),
        }), ['fdrBPM', 'bpm_pv', 'bpm_ranksum'])

    if ind_wpm.size:
        fdr = fdrWPM['fdrwpm2'].to_numpy()
        rank = np.argsort(fdr, kind='stable')
        order = ind_wpm[rank]
        output_wpm_table = _sort_table(pl.DataFrame({
            'pathway': path_wpm[rank],
            'group': _group_column(WPM_groups, order, 'WPM'),
            'fdrWPM': np.round(fdr[rank], 2),
            'eff_wpm': eff_wpm[rank],
            'wpm_size': wpm_size[rank],
            'wpm_pv': wpm_pv[rank],
            'wpm_ranksum': np.round(wpm_ranksum[rank], 2),
            'wpm_path_drivers': _by_module(wpm_path_drivers, order),
        }), ['fdrWPM', 'wpm_pv', 'wpm_ranksum'])

    if ind_path.size:
        fdr = fdrPATH['fdrpath2'].to_numpy()
        rank = np.argsort(fdr, kind='stable')
        order = ind_path[rank]
        output_path_table = _sort_table(pl.DataFrame({
            'pathway': path_path[rank],
            'group': _group_column(PATH_groups, order, 'PATH'),
            'fdrPATH': np.round(fdr[rank], 2),
            'eff_path': eff_path[rank],
            'path_size': path_size[rank],
            'path_pv': path_pv[rank],
            'path_ranksum': np.round(path_ranksum[rank], 2),
        }), ['fdrPATH', 'path_pv', 'path_ranksum'])

    # --- summary sheets -----------------------------------------------------
    header = ['minfdr'] + [ f'fdr{int(round(k * FDR_STEP * 100)):02d}' for k in range(1, n_levels + 1) ]

    # One row per module type with results. PATH is checked on its own; the
    # original gated both the WPM and PATH rows on WPM being non-empty.
    rows = [('BPM', fdrBPM['fdrbpm2'].to_numpy(), BPM_nosig_noRD)]
    if fdrWPM.height:
        rows.append(('WPM', fdrWPM['fdrwpm2'].to_numpy(), WPM_nosig_noRD))
    if fdrPATH.height:
        rows.append(('PATH', fdrPATH['fdrpath2'].to_numpy(), PATH_nosig_noRD))

    # first column is the unnamed index of row labels, as pandas wrote it
    index = {'': [name for name, _, _ in rows]}
    discovery = [[float(np.nanmin(fdr))] + [int((fdr <= k * FDR_STEP).sum()) for k in range(1, n_levels + 1)]
                 for _, fdr, _ in rows]
    noRD = [ [float(np.nanmin(fdr))] + nosig for _, fdr, nosig in rows ]

    output_discovery_summary = pl.DataFrame(index).hstack(pl.DataFrame(discovery, schema=header, orient='row'))
    output_noRD_discovery_summary = pl.DataFrame(index).hstack(pl.DataFrame(noRD, schema=header, orient='row'))

    # --- write --------------------------------------------------------------

    sheets = {
        'output_discovery_summary': output_discovery_summary,
        'output_noRD_discovery_summary': output_noRD_discovery_summary,
        'output_bpm_table': output_bpm_table,
        'output_wpm_table': output_wpm_table,
        'output_path_table': output_path_table,
    }
    sheets = {name: table for name, table in sheets.items() if table is not None}
    for name, table in sheets.items():
        print(f"    Writing {name} with {len(table)} rows")
    to_workbook(sheets).save(output_file)

    return str(output_file)

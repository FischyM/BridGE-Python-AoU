import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextToPath


LAYOUT_SEED = 0  # fixed so node positions are reproducible across runs

# Pathway key under the network (sizes in inches unless noted).
FIG_W = 12.0
KEY_MARGIN = 0.5          # left/right page margin
KEY_PAD = 0.25            # space above and below the key
KEY_HEADER_H = 0.3        # 'Pathways' heading
KEY_COL_GAP_PT = 18       # gap between key columns, points
KEY_MAX_FONT = 10         # points
KEY_MIN_FONT_2COL = 6     # below this, fall back to one full-width column
KEY_LINE_SPACING = 1.4    # line height as a multiple of font size


def _key_layout(pathways):
    """Font size, column count and rows for the key.

    Every entry uses one font size, shrunk from KEY_MAX_FONT until the longest
    "i: name" fits its column. Two columns by default; one if two would need a
    font smaller than KEY_MIN_FONT_2COL.
    """
    measure = TextToPath()
    prop = FontProperties(family='sans-serif', size=KEY_MAX_FONT)
    longest = max(measure.get_text_width_height_descent(f"{i}: {p}", prop, ismath=False)[0]
                  for i, p in enumerate(pathways))  # points at KEY_MAX_FONT
    for cols in (2, 1):
        col_w_pt = (FIG_W - 2 * KEY_MARGIN) * 72 / cols - KEY_COL_GAP_PT
        font = min(KEY_MAX_FONT, KEY_MAX_FONT * col_w_pt / longest)
        if font >= KEY_MIN_FONT_2COL or cols == 1:
            break
    rows = -(-len(pathways) // cols)
    return font, cols, rows


def _representatives(groups):
    """Global module index of one module per redundancy group, sorted by group label.

    `groups` is a group-label Series from check_BPM_WPM_redundancy, indexed by
    global module index and ordered by ascending FDR within each direction, so
    the first occurrence of each label is the lowest-FDR member of its group.
    """
    first = groups[~groups.duplicated()]
    return pd.Series(first.index, index=first.to_numpy()).sort_index().to_numpy()


def _split(i, size):
    """Global module index -> (row position, effect direction).

    Global indices 0..size-1 are protective and size..2*size-1 are risk.
    """
    return (i - size, 'risk') if i >= size else (i, 'protective')


def _build_graph(bpm, wpm, BPM_groups, WPM_groups, BPM_fdr, bpm_limit):
    """Pick the modules to draw and turn them into a pathway graph.

    Returns:
        tuple[nx.DiGraph, list[str]]: graph with nodes 0..n-1, and the pathway
            name of each node (sorted alphabetically).
    """
    wpm_names = wpm['pathway'].to_numpy()
    p1_names = bpm['path1names'].to_numpy()
    p2_names = bpm['path2names'].to_numpy()
    bpm_size = bpm.shape[0]

    # (pathway1, pathway2, direction); a WPM is a self pair.
    to_draw = []

    # WPMs: one per redundancy group.
    for i in _representatives(WPM_groups):
        pos, direction = _split(i, wpm.shape[0])
        to_draw.append((wpm_names[pos], wpm_names[pos], direction))

    # # PATHs (kept for future use): one per redundancy group, drawn as a
    # # node only. Re-enabling also needs PATH_groups passed in here and a
    # # colour per direction in draw_map.
    # for i in _representatives(PATH_groups):
    #     pos, direction = _split(i, wpm.shape[0])
    #     to_draw.append((wpm_names[pos], None, direction))

    # BPMs touching a WPM pathway take priority, both effect directions.
    wpm_pathways = np.unique([p for p, _, _ in to_draw])
    touching = np.flatnonzero(np.isin(p1_names, wpm_pathways) | np.isin(p2_names, wpm_pathways))
    significant = BPM_groups.index
    for pos in touching:
        if pos in significant:
            to_draw.append((p1_names[pos], p2_names[pos], 'protective'))
        if pos + bpm_size in significant:
            to_draw.append((p1_names[pos], p2_names[pos], 'risk'))

    # Then one BPM per redundancy group: the bpm_limit groups whose
    # representative has the lowest FDR, across both effect directions.
    # Ties keep group-label order (protective before risk).
    reps = _representatives(BPM_groups)
    reps = reps[np.argsort(BPM_fdr.loc[reps].to_numpy(), kind='stable')][:bpm_limit]
    for i in reps:
        pos, direction = _split(i, bpm_size)
        to_draw.append((p1_names[pos], p2_names[pos], direction))

    # Number pathways alphabetically, then collect edges. If the same pathway
    # pair is drawn more than once, the last entry decides the colour.
    pathways = sorted({p for p1, p2, _ in to_draw for p in (p1, p2) if p is not None})
    node = {p: i for i, p in enumerate(pathways)}
    edge_color = {}
    for p1, p2, direction in to_draw:
        if p2 is None:
            continue
        key = tuple(sorted((node[p1], node[p2])))
        edge_color[key] = 'orange' if direction == 'protective' else 'blue'

    G = nx.DiGraph()
    G.add_nodes_from(range(len(pathways)), color='lightblue')
    for (i, j), color in sorted(edge_color.items()):
        G.add_edge(i, j, color=color)
        G.add_edge(j, i, color=color)
    return G, pathways


def draw_map(project_dir, fdrcut, output_file, BPM_group_tmp, WPM_group_tmp, PATH_group_tmp, BPM_fdr, bpm_limit=20):
    """Draws the non-redundant pathway map at `fdrcut` to network_map-*.pdf.

    One page: the network (nodes are pathways, edges are BPMs, self loops are
    WPMs) with a key underneath giving the pathway name for each node number.
    Nothing is written when fewer than 3 pathways would be drawn.

    The group arguments are the per-threshold lists from
    check_BPM_WPM_redundancy; only the last entry (exactly fdrcut) is used.
    PATH_group_tmp is currently unused (see the PATH block in _build_graph).
    BPM_fdr is a Series of BPM FDRs indexed by global module index (must cover
    every module in BPM_group_tmp[-1]); it ranks groups for bpm_limit.
    """
    bpm = pd.read_parquet(f"{project_dir}/intermediate/pathway_indices-bpm.parquet")
    wpm = pd.read_parquet(f"{project_dir}/intermediate/pathway_indices-wpm.parquet")

    G, pathways = _build_graph(bpm, wpm, BPM_group_tmp[-1], WPM_group_tmp[-1], BPM_fdr, bpm_limit)
    if len(pathways) < 3:
        return

    fig_title = 'Non-redundant network map with FDR threshold=' + str(int(fdrcut*100))
    key_font, key_cols, key_rows = _key_layout(pathways)

    # Layout in inches: title strip, 12x12 network, then the key underneath.
    # The key grows with the number of pathways so it never overlaps the network.
    title_h, net_h = 0.6, 12.0
    key_line_h = key_font * KEY_LINE_SPACING / 72
    key_h = 2 * KEY_PAD + KEY_HEADER_H + key_rows * key_line_h
    fig_h = title_h + net_h + key_h

    with plt.rc_context({'font.family': 'sans-serif'}):
        fig = plt.figure(figsize=(FIG_W, fig_h))
        fig.suptitle(fig_title, fontsize=16, y=1 - 0.15 / fig_h, va='top')

        # network
        ax = fig.add_axes([0, key_h / fig_h, 1, net_h / fig_h])
        edges = list(G.edges())
        pos = nx.spring_layout(G, k=1.5, iterations=200, seed=LAYOUT_SEED)
        nx.draw(G, pos, ax=ax, edgelist=edges, edge_color=[G.edges[e]['color'] for e in edges], style='-',
                nodelist=list(G.nodes), node_color=[c for _, c in G.nodes(data='color')])
        nx.draw_networkx_labels(G, pos, ax=ax, labels={i: i for i in G.nodes}, font_size=10)
        ax.axis('off')
        ax.set_xlim([1.5*x for x in ax.get_xlim()])
        ax.set_ylim([1.5*y for y in ax.get_ylim()])
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='pathway', markerfacecolor='lightblue', markersize=15),
            # Line2D([0], [0], marker='o', color='w', label='protective PATH pwathway', markerfacecolor='yellow', markersize=15),
            # Line2D([0], [0], marker='o', color='w', label='risk PATH pwathway', markerfacecolor='blue', markersize=15),
            Line2D([0], [0], color='orange', lw=4, label='protective interaction'),
            Line2D([0], [0], color='blue', lw=4, label='risk interaction'),
            Line2D([0], [0], marker='o', color='w', label='self loops indicate WPM interactions',
                   markerfacecolor='w', markersize=1),
        ]
        ax.legend(handles=legend_elements, loc='lower left')

        # key: node number -> pathway name, filled column by column
        top = key_h - KEY_PAD
        fig.add_artist(Line2D([KEY_MARGIN / FIG_W, 1 - KEY_MARGIN / FIG_W], [top / fig_h] * 2,
                              color='0.75', lw=0.8))
        fig.text(KEY_MARGIN / FIG_W, (top - 0.08) / fig_h, 'Pathways', fontsize=key_font,
                 fontweight='bold', va='top')
        col_w = (FIG_W - 2 * KEY_MARGIN) / key_cols
        for i, p in enumerate(pathways):
            col, row = divmod(i, key_rows)
            fig.text((KEY_MARGIN + col * col_w) / FIG_W,
                     (top - KEY_HEADER_H - row * key_line_h) / fig_h,
                     f"{i}: {p}", fontsize=key_font, va='top')

        net_map_name = f"network_map-{'-'.join(output_file.stem.split('-')[-2:])}.pdf"
        fig.savefig(output_file.with_name(net_map_name), format='pdf')
    plt.close(fig)

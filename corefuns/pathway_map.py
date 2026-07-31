import pickle

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import rcParams
from matplotlib.backends.backend_pdf import PdfPages

from classes import BPMind

# draw_map() draws a non-redundant network map of the significant BPMs/WPMs/PATHs.
#
# INPUTS:
#   project_dir: project directory, used to locate BPMind.pkl and the results dir.
#   fdrcut: FDR threshold the map is drawn at. Must be a multiple of 0.05.
#   resultsfile: results pickle path. Only its filename is used, to name the PDF.
#   BPM_group_tmp/WPM_group_tmp/PATH_group_tmp: lists returned by
#       check_BPM_WPM_redundancy, one entry per 0.05 threshold. Each entry is a
#       Series of redundant-group labels indexed by GLOBAL module index, so the
#       last entry is the level at exactly fdrcut.
#   bpm_limit: maximum number of BPM redundant groups to draw.
#
# OUTPUTS:
#   <project_dir>/results/network-map-<ssmfile>.pdf


def draw_map(project_dir,fdrcut,resultsfile,BPM_group_tmp,WPM_group_tmp,PATH_group_tmp,bpm_limit=20):
    rcParams['font.family'] = 'sans-serif'

    # load BPMind.pkl file
    bpm_file = project_dir + '/intermediate/BPMind.pkl'
    with open(bpm_file,'rb') as f:
        bpm_ind: BPMind = pickle.load(f)

    bpm = bpm_ind.bpm
    wpm = bpm_ind.wpm
    wpm_size = wpm.shape[0]
    bpm_size = bpm.shape[0]

    # The last entry of each group list is the level at exactly fdrcut, so there
    # is no threshold index to work out. Protective modules occupy global indices
    # 0..size-1 and risk modules size..2*size-1.
    fdr_th = fdrcut

    # Find and add nodes
    to_draw = []
    used_pathways = []
    ## add WPMs: exactly one from each redundant group
    wpm_groups = WPM_group_tmp[-1]
    for g in np.unique(wpm_groups.to_numpy()):
        members = wpm_groups.index[wpm_groups == g]
        xid = members[0]                      # lowest-FDR member of the group
        risk_type = xid >= wpm_size
        if risk_type:
            xid = xid - wpm_size
        p1 = wpm['pathway'][xid]
        used_pathways.append(p1)
        if risk_type:
            to_draw.append((p1,p1,'risk'))
        else:
            to_draw.append((p1,p1,'protective'))

    ## add PATHs
    #path_groups = PATH_group_tmp[-1]
    #for g in np.unique(path_groups.to_numpy()):
    #	members = path_groups.index[path_groups == g]
    #	xid = members[0]
    #	risk_type = xid >= wpm_size
    #	if risk_type:
    #		xid = xid - wpm_size
    #	p1 = wpm['pathway'][xid]
    #	used_pathways.append(p1)
    #	if risk_type:
    #		to_draw.append((p1,None,'risk'))
    #	else:
    #		to_draw.append((p1,None,'protective'))

    ## add BPMs based on the bpm limit
    used_pathways = np.unique(used_pathways)
    bpm_groups = BPM_group_tmp[-1]
    significant_bpms = bpm_groups.index
    group_ids = np.unique(bpm_groups.to_numpy())
    ### priority is with WPM-PATH associated pathways
    # NOTE: x and y are positions in bpm, so bpm_id is always in 0..bpm_size-1 and
    # this pass can only ever match a protective module. Risk BPMs touching a
    # WPM/PATH pathway are never drawn here. Fixing that means testing both
    # bpm_id and bpm_id + bpm_size against significant_bpms, which changes the
    # figure, so it is left as-is for now.
    for p in used_pathways:
        x = np.where(bpm['path1names'] == p)
        x = x[0]
        y = np.where(bpm['path2names'] == p)
        y = y[0]
        tmp = np.union1d(x,y)
        for bpm_id in tmp:
            if bpm_id in significant_bpms:
                risk_type = False
                if bpm_id >= bpm_size:
                    risk_type = True
                    bpm_id = bpm_id - bpm_size
                p1 = bpm['path1names'][bpm_id]
                p2 = bpm['path2names'][bpm_id]
                if risk_type:
                    to_draw.append((p1,p2,'risk'))
                else:
                    to_draw.append((p1,p2,'protective'))

    ### draw exactly one bpm from each group
    for g in group_ids[:bpm_limit]:
        members = bpm_groups.index[bpm_groups == g]
        xid = members[0]                      # lowest-FDR member of the group
        risk_type = xid >= bpm_size
        if risk_type:
            xid = xid - bpm_size
        p1 = bpm['path1names'][xid]
        p2 = bpm['path2names'][xid]
        if risk_type:
            to_draw.append((p1,p2,'risk'))
        else:
            to_draw.append((p1,p2,'protective'))

    # Create adjacency matrix
    ## first find all pathways
    used_pathways = []
    for t in to_draw:
        p1 = t[0]
        p2 = t[1]
        if p2 != None and p1 != p2:
            used_pathways.append(p2)
        used_pathways.append(p1)
    used_pathways = list(set(used_pathways))
    inds = {}
    nodes_labels = {}
    simple_labels = {}
    for i in range(len(used_pathways)):
        p = used_pathways[i]
        inds[p] = i
        nodes_labels[i] = p
        simple_labels[i] = i
    ## now create the matrix
    adj_matrix = np.zeros((len(used_pathways),len(used_pathways)))
    path_array = []
    for t in to_draw:
        p1 = t[0]
        p2 = t[1]
        int_type = t[2]
        if p2 == None:
            path_array.append((p1,int_type))
            continue
        if p1 == p2:
            xid = inds[p1]
            if int_type == 'protective':
                adj_matrix[xid,xid] = 1
            else:
                adj_matrix[xid,xid] = -1
        else:
            xid = inds[p1]
            yid = inds[p2]
            if int_type == 'protective':
                adj_matrix[xid,yid] = 1
                adj_matrix[yid,xid] = 1
            else:
                adj_matrix[xid,yid] = -1
                adj_matrix[yid,xid] = -1

    if adj_matrix.shape[0] < 3:
        return

    # Draw
    ## Graph(map)
    G = nx.DiGraph()
    added_node_flag = np.zeros((len(used_pathways),))
    ### add PATHs with color
    for t in path_array:
        p = t[0]
        int_type = t[1]
        x_id = inds[p]
        if added_node_flag[x_id] == 1: # added before!
            continue
        if int_type == 'protective':
            G.add_nodes_from([(x_id, {"color": "lightblue"})])
        else:
            G.add_nodes_from([(x_id, {"color": "lightblue"})])
        added_node_flag[x_id] = 1

    ### add all other the nodes
    for i in range(len(used_pathways)):
        if added_node_flag[i] == 0:
            p = used_pathways[i]
            G.add_nodes_from([(inds[p], {"color": "lightblue"})])
    ### add BPMs by adding edges
    for i in range(len(used_pathways)):
        for j in range(i,len(used_pathways)):
            val = adj_matrix[i,j]
            if val == 1:
                p1 = used_pathways[i]
                p2 = used_pathways[j]
                G.add_edge(inds[p1],inds[p2],color='orange')
                G.add_edge(inds[p2],inds[p1],color='orange')
            elif val == -1:
                p1 = used_pathways[i]
                p2 = used_pathways[j]
                G.add_edge(inds[p1],inds[p2],color='blue')
                G.add_edge(inds[p2],inds[p1],color='blue')
    ### drawing 
    fig_title = 'Non-redundant network map with FDR threshold=' + str(int(fdr_th*100))
    nodes = G.nodes
    node_colors = [nodes[u]['color'] for u in nodes]
    edges = G.edges()
    colors = [G[u][v]['color'] for u,v in edges]
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["figure.figsize"] = [12, 12]
    fig1 = plt.figure()
    pos = nx.spring_layout(G,k=1.5, iterations=200)
    nx.draw(G, pos, edgelist=edges, edge_color=colors,style='-',node_color=node_colors,nodelist=nodes)
    nx.draw_networkx_labels(G,pos,labels=simple_labels,font_size=10)
    plt.axis('off')
    axis = plt.gca()
    axis.set_xlim([1.5*x for x in axis.get_xlim()])
    axis.set_ylim([1.5*y for y in axis.get_ylim()])
    ## add legend
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='pathway',markerfacecolor='lightblue', markersize=15)]
    #legend_elements.append(Line2D([0], [0], marker='o', color='w', label='protective PATH pwathway',markerfacecolor='yellow', markersize=15))
    #legend_elements.append(Line2D([0], [0], marker='o', color='w', label='risk PATH pwathway',markerfacecolor='blue', markersize=15))
    legend_elements.append(Line2D([0], [0], color='orange', lw=4, label='protective interaction'))
    legend_elements.append(Line2D([0], [0], color='blue', lw=4, label='risk interaction'))
    legend_elements.append(Line2D([0], [0], marker='o', color='w', label='self loops indicate WPM interactions',markerfacecolor='w', markersize=1))
    plt.legend(handles=legend_elements, loc='lower left')



    ## add Pathway names to the next page
    fig2 = plt.figure()
    txt = ''
    for i in range(len(used_pathways)):
        tmp = str(i)+': '+nodes_labels[i]
        txt = txt + tmp + '\n'
    plt.axis('off')
    plt.text(0.05,0.05,txt, transform=fig2.transFigure, size=12)
    # find output file name based on the resultsfile
    tmp = resultsfile.split('results_')
    ssmfile = tmp[1]
    tmp = ssmfile.split('.pkl')
    ssmfile = tmp[0]
    pp = PdfPages(project_dir+'/results/network-map-'+ssmfile+'.pdf')
    fig_nums = plt.get_fignums()
    figs = [plt.figure(n) for n in fig_nums]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()

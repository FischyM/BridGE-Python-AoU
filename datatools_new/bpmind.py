import pickle
from itertools import combinations

import numpy as np
import pandas as pd

from classes import bpmindclass as bpmc


def bpmind(snpPathwayFile):
    """Exctracts SNP indices for BPM/WPM sets. Saves a BPMind.pkl file with a bpmindclass class with fields:
        bpm - DataFrame with all BPM data (pathway names, pathway inices, SNPs in pathaways(redundants removed))
        wpm - DataFrame with all WPM data (pathway names, pathway inices, SNPs in pathaways)

    Args:
        snpPathwayFile (str): SNP-pathway mapping file in pickle format (.pkl), containing a matrix: Result of the snppathway function. 
    """

    # find project directory
    p_dir = snpPathwayFile.split('/')
    project_dir = "/".join(p_dir[0:-1])

    # Reading in pickle datafile
    with open(snpPathwayFile, "rb") as file:
        snpset = pickle.load(file)

    # Retrieving pathways list from snpset
    pathways = snpset.pathways
    snpmat = snpset.spmatrix

    # Finding all possible combinations of pairs for pathway names and sizes.
    combnames = np.array(list(combinations(pathways.index, 2)))

    # Finding nonzero WPM indices
    WPMind = [ np.nonzero(snpmat[column])[0].tolist() for column in snpmat.columns ]
    wpmdata = {
        'pathway': pathways.index,
        'indsize': pathways.values,
        'ind': WPMind,
        'size': ((pathways.values * pathways.values) - pathways.values),  # TODO: is divide by 2 needed here?
        }
    wpm = pd.DataFrame(wpmdata)

    # Finding BPM indices
    BPMind1, BPMind2, ind1size, ind2size = [], [], [], []
    for i in range(len(snpmat.columns)):
        p1 = snpmat.iloc[:, i].to_numpy()
        
        for j in range(i + 1, len(snpmat.columns)):
            p2 = snpmat.iloc[:, j].to_numpy()
            
            # snps in pathway 1 but not in pathway 2
            d1 = p1 - p2
            ind1 = np.where(d1 == 1)[0].tolist()
            
            # snps in pathway 2 but not in pathway 1
            d2 = p2 - p1
            ind2 = np.where(d2 == 1)[0].tolist()
            
            ind1size.append(len(ind1))
            ind2size.append(len(ind2))
            
            BPMind1.append(ind1)
            BPMind2.append(ind2)

    # Getting between pathway sizes by multiplying combination available pairs.
    if (len(pathways) > 1):
        size = np.array(ind1size) * np.array(ind2size)
        # Orienting bpm/wpm data and converting to dataframes.
        bpmdata = {
            'path1names': combnames[:, 0], 'ind1size': ind1size, 'ind1': BPMind1,
            'path2names': combnames[:, 1], 'ind2size': ind2size, 'ind2': BPMind2,
            'size': size,
            }
    else:
        bpmdata = {
            'path1names': [], 'ind1size': [],
            'path2names': [], 'ind2size': [],
            'size': [],
            }
    bpm = pd.DataFrame(bpmdata)

    # Reading bpm and wpm models into bpmind class for pickle storage.
    bpmobj = bpmc.bpmindclass(bpm, wpm)

    # Saving bpmind data to pickle file.
    with open(f"{project_dir}/BPMind.pkl", 'wb') as file:
        pickle.dump(bpmobj, file)

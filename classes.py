from dataclasses import dataclass

from numpy import ndarray
from pandas import DataFrame, Series
from scipy.sparse import csr_array


@dataclass
class SNPclass():
    """Class for SNP data, including genotype data and associated metadata.
    
    data (ndarray): genotype data with samples as rows and SNPs as columns
    varid (Series): SNP identifiers (e.g., rsIDs or chrom:pos:ref:alt string)
    chrom (Series): chromosome identifiers for each SNP
    pos (Series):   physical positions of each SNP on the chromosome
    pheno (Series): phenotype information for each sample
    iid (Series):   individual identifiers for each sample
    sex (Series):   sex information for each sample
    """
    data: ndarray
    varid: Series
    chrom: Series
    pos: Series
    pheno: Series
    iid: Series
    sex: Series

@dataclass
class genesetclass():
    """Class for gene set data, including gene to pathway mapping and entrez ID lookup.
    
    entrezids (dict):       gene {symbol: entrezID} lookup dictionary
    gpmatrix (DataFrame):   gene pathway binary dataframe
    """
    entrezids: dict
    gpmatrix: DataFrame

@dataclass
class snpgeneclass():
    """Class for SNP to gene mapping data.
    
    This class stores the mapping of SNPs to genes, along with the distance by which gene 
    boundaries are expanded for SNP to gene mapping.
    
    sgmatrix (DataFrame):   dictionary mapping SNPs to genes
    mapping_dist (int):     mapping distance in bp to expand gene boundaries by for SNP to gene mapping
    """
    sgmatrix: DataFrame
    mapping_dist: int
    
@dataclass
class snpsetclass():
    """Class for SNP set data, including SNP to pathway mapping and pathway sizes.
    
    pathways (Series):      number of SNPs in each pathway, includes the pathway names as the index
    spmatrix (DataFrame):   snp to pathway mapping DataFrame
    """
    pathways: Series
    spmatrix: DataFrame

@dataclass
class bpmindclass():
    """BPM and WPM extracted data needed for computing statistics and FDR.
    
    The file is saved as a pickle file called "BPMind.pkl" in the intermediate directory.
    
    bpm (DataFrame): DataFrame that has all BPM associated data 
        path1names: pathway 1 names
        ind1:       SNP indices in pathway 1 that are not in pathway 2
        ind1size:   size of ind1
        path2names: pathway 2 names
        ind2:       SNP indices in pathway 2 that are not in pathway 1
        ind2size:   size of ind2
        size:       size of the between pathway module (ind1size * ind2size) that is nonredundant
    wpm (DataFrame): DataFrame with all WPM associated data
        pathway:    pathway names
        ind:        SNP indices for each pathway
        indsize:    size of ind
        size:       size of the within pathway module ((indsize * indsize) - indsize) that is nonredundant.
        
    Note: wpm.size results in a symmetric matrix with redundant values (upper triangle == lower triangle). 
    This difference is accounted for in the BPM/WPM density calculations.
    """
    bpm: DataFrame
    wpm: DataFrame

@dataclass
class InteractionNetwork():
    """Class for interaction network data.
    
    This is a SNP-SNP Matrix of the modified hypergeometric scores for the SNP-SNP interactions.
    The saved file can be found in the intermediate directory and is called "ssM_mhygessi_<model>_R<r>.pkl" 
    where <model> is one of RR, RD, or DD and <r> is the randomization index.
    
    For the max_id matrices, the values indicate which of the three input matrices (RR, RD, DD)
    was the maximum value for each SNP pair (1=RR, 2=DD, 3=RD).
    
    risk (csr_array):                       sparse matrix of risk interaction scores between SNPs
    protective (csr_array):                 sparse matrix of protective interaction scores between SNPs
    risk_max_id (csr_array | None):         optional sparse matrix indicating the indices of the maximum risk interactions
    protective_max_id (csr_array | None):   optional sparse matrix indicating the indices of the maximum protective interactions
    """
    risk: csr_array
    protective: csr_array
    risk_max_id: csr_array | None
    protective_max_id: csr_array | None
    
@dataclass
class Stats():
    """Class for storing statistical results for BPMs, WPMs, and PATHs.
    
    p-values depends on if the InteractionNetwork is optionally binarized. 
    If binarized, p-values are chi2, otherwise they are ranksum.
    
    bpm_local (ndarray):            p-values for BPMs 
    bpm_local_pv (ndarray):         permuted empirical p-values for BPMs 
    density_bpm (ndarray):          observed densities for BPMs
    density_bpm_expected (ndarray): expected densities for BPMs
    dense_index (ndarray):          indices of dense BPMs
    wpm_local (ndarray):            p-values for WPMs
    wpm_local_pv (ndarray):         permuted empirical p-values for WPMs
    density_wpm (ndarray):          observed densities for WPMs
    density_wpm_expected (ndarray): expected densities for WPMs
    path_degree (ndarray):          p-values for PATHs
    path_degree_pv (ndarray):       permuted empirical p-values for PATHs
    """
    bpm_local: ndarray
    bpm_local_pv: ndarray
    density_bpm: ndarray
    density_bpm_expected: ndarray
    dense_index: ndarray
    wpm_local: ndarray
    wpm_local_pv: ndarray
    density_wpm: ndarray
    density_wpm_expected: ndarray
    path_degree: ndarray
    path_degree_pv: ndarray
    
@dataclass
class GenstatsOut():
    """Class for storing the output of the genetic statistics analysis.
    
    This class is saved in the intermediate directory as a pickle file called "genstats_<ssmFile>.pkl".
    
    protective_stats (Stats):   statistics for protective interactions
    risk_stats (Stats):         statistics for risk interactions
    """
    protective_stats: Stats
    risk_stats: Stats

@dataclass
class fdrrclass():
    """Class for storing the output of the FDR analysis.
    
    This class is saved in the intermediate directory as a pickle file called "results_<ssmFile>.pkl".
    
    FDR here is calculated for BPMs, WPMs, and PATHs, using the random networks and
    the observed data for both protective and risk networks.
    
    There are two FDR calculations for each module type.
    - case one: FDR for BPMs, WPMs, and PATHs based on empirical p-values (real / random networks).
    - case two: FDR for BPMs, WPMs, and PATHs based on both empirical p-values (from case one) and ranksum scores.
    
    The p-values used were calculated either with the ranksum test or the chi2 test depending on 
    if the InteractionNetwork is binarized or not.
    
    TODO: no matter the origin of the p-values, they are called ranksum here. 
    This should be fixed to avoid confusion.
    
    bpm_pv (DataFrame): permuted empirical p-values for BPMs
    wpm_pv (DataFrame): permuted empirical p-values for WPMs
    path_pv (DataFrame): permuted empirical p-values for PATHs
    bpm_ranksum (DataFrame): p-values for BPMs
    wpm_ranksum (DataFrame): p-values for WPMs
    path_ranksum (DataFrame): p-values for PATHs
    fdrbpm1 (DataFrame): case 1 FDR for BPM
    fdrbpm2 (DataFrame): case 2 FDR for BPM
    fdrwpm1 (DataFrame): case 1 FDR for WPM
    fdrwpm2 (DataFrame): case 2 FDR for WPM
    fdrpath1 (DataFrame): case 1 FDR for PATH
    fdrpath2 (DataFrame): case 2 FDR for PATH
    """
    bpm_pv: DataFrame
    wpm_pv: DataFrame
    path_pv: DataFrame
    bpm_ranksum: DataFrame
    wpm_ranksum: DataFrame
    path_ranksum: DataFrame
    fdrbpm1: DataFrame
    fdrbpm2: DataFrame
    fdrwpm1: DataFrame
    fdrwpm2: DataFrame
    fdrpath1: DataFrame
    fdrpath2: DataFrame


import threading, time, os
import psutil

class ResourceMonitor:
    """
    Usage:
    from classes import ResourceMonitor
    with ResourceMonitor(interval=0.2):
        mapsnp2gene(...)
    """
    def __init__(self, interval=0.2):
        self.interval = interval
        self.proc = psutil.Process(os.getpid())
        self._stop = threading.Event()
        self.peak_rss = 0
        self.peak_cpu = 0.0
        self.baseline_rss = 0

    def _procs(self):
        try:
            return [self.proc] + self.proc.children(recursive=True)
        except psutil.NoSuchProcess:
            return [self.proc]

    def _sample(self):
        rss = cpu = 0
        for p in self._procs():
            try:
                rss += p.memory_info().rss
                cpu += p.cpu_percent()  # % of one core; sums across processes
            except psutil.NoSuchProcess:
                pass
        return rss, cpu

    def _poll(self):
        self._sample()  # prime cpu_percent (first call is always 0)
        while not self._stop.wait(self.interval):
            rss, cpu = self._sample()
            self.peak_rss = max(self.peak_rss, rss)
            self.peak_cpu = max(self.peak_cpu, cpu)

    def __enter__(self):
        self.baseline_rss, _ = self._sample()
        self.peak_rss = self.baseline_rss
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        print(f"Peak RAM increase: {(self.peak_rss - self.baseline_rss) / 1e9:.2f} GB")
        print(f"Peak CPU usage: {self.peak_cpu:.0f}% (of a single core; "
              f"{psutil.cpu_count()} cores available)")

from dataclasses import dataclass

from numpy import ndarray
from pandas import DataFrame, Series
from scipy.sparse import csr_array


@dataclass
class SNPdata():
    """Class to store SNP data from plink2's pgen data format. Created in plink2pkl() and used in bindataa().
    
    data: DataFrame containing genotype data
    varid: Series containing SNP names as variant ID formmated as chrom:pos:ref:alt
    chrom: Series containing chromosome IDs od SNPs
    pos: Series containing physical locations of SNPs
    pheno: Series containing sample's phenotype
    fid: Series containing sample's family ID
    iid: Series containing sample's individual ID
    sex: Series containing sample's sex
    """
    __slots__ = ['data', 'varid', 'chrom', 'pos', 'pheno', 'fid', 'iid', 'sex']
    data: DataFrame
    varid: Series
    chrom: Series
    pos: Series
    pheno: Series
    fid: Series
    iid: Series
    sex: Series
    
@dataclass
class GeneSet():
    """Class to store gene set data (ie. gene to pathway mappings). Created in msigdb2pkl().
    
    entrezids: dictionary mapping gene symbols to entrez IDs
    gpmatrix: DataFrame representing genes mapping to pathways as a boolean matrix (dataframe)
    """
    __slots__ = ['entrezids', 'gpmatrix']
    entrezids: dict
    gpmatrix: DataFrame

@dataclass
class SNPset():
    """Class to store SNP set data (ie. SNP to pathway mappings). Created in snppathway().
    
    pathways: Series containing total number of SNPs in each pathway
    spmatrix: DataFrame containing the SNP to pathway mapping
    genefile: Path to the gene set file
    """
    __slots__ = ['pathways', 'spmatrix', 'genefile']
    pathways: Series
    spmatrix: DataFrame
    genefile: str
    
@dataclass
class BPMind():
    """Class to store BPM and WPM data.
    
    bpm: DataFrame containing all BPM data
        path1names  - pathway 1 names
        ind1size    - number of SNPs in pathway 1
        ind1        - indices for SNPset.spmatrix of SNPs in pathway 1 but not in pathway 2
        path2names  - pathway 2 names
        ind2size    - number of SNPs in pathway 2
        ind2        - indices for SNPset.spmatrix of SNPs in pathway 2 but not in pathway 1
        size        - number of SNP pairs between pathway 1 and pathway 2 (ind1size * ind2size)
    wpm: DataFrame containing all WPM data
        pathway     - pathway names
        indsize     - number of SNPs in each pathway
        ind         - indices for SNPset.spmatrix of SNPs in each pathway
        size        - number of SNP pairs within each pathway (n^2 - n)
        NOTE: size is not divided by 2 because this symmetric matrix is accounted for later on
    """
    __slots__ = ['bpm', 'wpm']
    bpm: DataFrame
    wpm: DataFrame

@dataclass
class InteractionNetwork():
    """Class to store the interaction network data.
    
    risk_network: CSR matrix of risk interactions
    protective_network: CSR matrix of protective interactions
    r_max_id: CSR matrix of the maximum risk interaction IDs. For a model other than combined this is None
    p_max_id: CSR matrix of the maximum protective interaction IDs. For a model other than combined this is None
    """
    __slots__ = ['risk_network', 'protective_network', 'r_max_id', 'p_max_id']
    risk_network: csr_array
    protective_network: csr_array
    r_max_id: csr_array | None
    p_max_id: csr_array | None

@dataclass
class Stats():
    """Class to store the statistics from the genstats function.
    
    bpm_local: ndarray of local BPM statistics
    bpm_local_pv: ndarray of local BPM p-values
    density_bpm: ndarray of the density of the BPM network
    density_bpm_expected: ndarray of the expected density of the BPM network
    dense_index: ndarray of indices of the dense BPM network
    wpm_local: ndarray of local WPM statistics
    wpm_local_pv: ndarray of local WPM p-values
    density_wpm: ndarray of the density of the WPM network
    density_wpm_expected: ndarray of the expected density of the WPM network
    path_degree: ndarray of the degree of each pathway in the network
    path_degree_pv: ndarray of the p-values for the degree of each pathway in the network
    """
    __slots__ = [
        'bpm_local', 'bpm_local_pv', 'density_bpm', 'density_bpm_expected', 'dense_index',
        'wpm_local', 'wpm_local_pv', 'density_wpm', 'density_wpm_expected',
        'path_degree', 'path_degree_pv'
        ]
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
class GenStats():
    """Class to store the protective and risk statistics from the genstats function."""
    __slots__ = ['protective_stats', 'risk_stats']
    protective_stats: Stats
    risk_stats: Stats

@dataclass
class FDRstats():
    __slots__ = [
        'bpm_pv', 'wpm_pv', 'path_pv',
        'bpm_ranksum', 'wpm_ranksum', 'path_ranksum',
        'fdrbpm1', 'fdrbpm2', 'fdrwpm1', 'fdrwpm2', 'fdrpath1', 'fdrpath2'
        ]
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

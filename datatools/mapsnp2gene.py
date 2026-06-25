import pickle

import numpy as np
import pandas as pd


def mapsnp2gene(pvarFile, geneAnnotation, mappingDistance, option, outfile):
    """Creates snp to gene matrix in the DataFrame format and saves it to a pickle file.

    Args:
        pvarFile (str): path to Plink variant file in .pvar format.
        geneAnnotation (str): path to gene annotation file.
        mappingDistance (int): snp to gene mapping distance.
        option (str): saving mode for snp-gene map.
        outfile (str): file name for saving the results.
    """

    # Creating SNP dataframe from snp annotation file.
    pvar_header = ['chrom', 'pos', 'var_id', 'ref', 'alt']
    var_df = pd.read_csv(pvarFile, sep=r"\s+", header=0, names=pvar_header, engine='python')  # has header row
    var_df['chrom'] = pd.to_numeric(var_df['chrom'])

    # Creating gene dataframe from gene annotation file.
    gene_header = ['chrom', 'geneloc1', 'geneloc2', 'genes']
    gdf = pd.read_csv(geneAnnotation, sep=r"\s+", names=gene_header, engine='python')  # does not have a header
    gdf = gdf[gdf.chrom.apply(lambda x: x.isnumeric())]
    gdf['chrom'] = pd.to_numeric(gdf['chrom'])
    gdf.sort_values(by='chrom', inplace=True)

    # Expanding gene window by subtracting and adding from start and end loci.
    gdf['geneloc1'] = gdf['geneloc1'] - mappingDistance
    gdf['geneloc2'] = gdf['geneloc2'] + mappingDistance

    # Doing an outer join to get all genes and snp listed by chromosome.
    cdf = gdf.merge(var_df, how='outer', on='chrom')

    # keep only snps that are located between start and end loci adjusted by mappingDistance.
    cdf = cdf[(cdf['pos'] >= cdf['geneloc1']) & (cdf['pos'] <= cdf['geneloc2'])]
    
    # Creating list of unique rsids from filtered results.
    snplist = cdf['var_id'].drop_duplicates()

    # Option chosen to save to snplist.
    if (option == 'snplist'):

        # Saving SNPlist to pickle file.
        final = open(outfile, 'wb')
        pickle.dump(snplist, final, protocol=pickle.HIGHEST_PROTOCOL)
        final.close()

    # Option chosen to save to matrix.
    elif (option == 'matrix'):

        # Getting list of unique and genes from filtered results.
        genelist = cdf['genes'].drop_duplicates()

        # Creating dataframe of appropriate size, and setting labels.
        sgm = pd.DataFrame(np.zeros((len(snplist), len(genelist))),
                                index=snplist, columns=genelist, dtype=bool)

        # Setting snp-gene matrix values to true if snp is within gene window.
        for row in cdf.itertuples():
            sgm.loc[row.var_id, row.genes] = True

        # Saving snp-gene matrix to pickle file.
        final = open(outfile, 'wb')
        pickle.dump(sgm, final, protocol=pickle.HIGHEST_PROTOCOL)
        final.close()

    else:
        # Output option not recognized.
        print("Return option error, valid options are 'snplist', or 'matrix'")

## main function of BridGE

import sys
from os import path
from datatools import plink2pkl as p2p
from datatools import bindataa as ba
from datatools import msigdb2pkl as msig2p
from datatools import mapsnp2gene as snp2gene
from datatools import snppathway as snpp
from datatools import bpmind as bpm
from corefuns import matrix_operations_par as ci
from corefuns import genstats_perm as gs
from corefuns import fdrsampleperm as fdr
from corefuns import collectresults as cl
import datetime


if __name__ == '__main__':
    # Default parameters defined
    job = ''
    plinkfile = ''
    project_dir = 'data'
    genesets = 'c2.cp.v7.1' 
    gene_annotation = 'glist-hg38' 
    mappingDistance = 50000
    minPath = 10
    maxPath = 300
    alpha1 = 0.05
    alpha2 = 0.05
    n_workers = 4
    sample_perms = 10
    binaryNetwork = False
    snpPerms = 10000
    i = -1
    r = -1
    pval_cutoff = 0.05
    fdrcut = 0.25
    densitycutoff = None
    ssmfile = None
    model = None
    snppathwayfile = 'snp_pathway_min10_max300.pkl'
 
    for arg in sys.argv:
        if '=' in arg and '--' in arg:
            o = arg.split('=')[0]
            a = arg.split('=')[1]
            if o == '--job':
                job = a
            elif o == '--plinkFile':
                plinkfile = a
            elif o == '--genesets':
                genesets = a
            elif o == '--geneAnnotation':
                gene_annotation = a
            elif o == '--mappingDistance':
                mappingDistance = int(a)
            elif o == '--maxPath':
                maxPath = int(a)
            elif o == '--minPath':
                minPath = int(a)
            elif o == '--model':
                model = a
            elif o == '--nWorker':
                n_workers = int(a)
            elif o == '--samplePerms':
                sample_perms = int(a)
            elif o == '--binaryNetwork':
                if int(a) == 1:
                    binaryNetwork = True
            elif o == '--snpPerms':
                snpPerms = int(a)
            elif o == '--pvalueCutoff':
                pval_cutoff = float(a)
            elif o == '--i':
                i = int(a)
            elif o == '--fdrcut':
                fdrcut = float(a)
            elif o == '--snpPathFile':
                snppathwayfile = a
            elif o == '--projectDir':
                project_dir = a
            elif o == '--densityCutoff':
                densitycutoff = float(a)
            elif o == '--ssmfile':
                ssmfile = a
            elif o == '--R':
                r = int(a) 

    if job == 'DataProcess':
        print('data processing...')
        sys.stdout.flush()

        # convert plinkfile to pickle
        if plinkfile == '':
            sys.exit('plinkFile not provided')
        rawfile = f"{project_dir}/intermediate/{plinkfile}.raw"
        bimfile = f"{project_dir}/intermediate/{plinkfile}.bim"
        famfile = f"{project_dir}/intermediate/{plinkfile}.fam"
        if not path.exists(rawfile) or not path.exists(bimfile) or not path.exists(famfile):
            sys.exit('plinkFiles do not exist')
        finalfile = f"{project_dir}/intermediate/{plinkfile}.pkl"
        p2p.plink2pkl(rawfile, bimfile, famfile, finalfile)

        # converting snp data assuming different disease models
        ba.bindataa(project_dir, finalfile, 'r')
        ba.bindataa(project_dir, finalfile, 'd')
        
        # prepare gene set information
        symbolsfile = f"{project_dir}/raw/{genesets}.symbols.gmt"
        entrezfile = f"{project_dir}/raw/{genesets}.entrez.gmt"
        if not path.exists(symbolsfile) or not path.exists(entrezfile):
            sys.exit(f'genesets do not exist: {symbolsfile}, {entrezfile}')
        msig2p.msigdb2pkl(symbolsfile, entrezfile)
        # TODO: add in functionality to reduce gene set based on Jaccard similarity?
        
        # build relationship between snps and genes
        gene_annotation_file = f"{project_dir}/raw/{gene_annotation}"
        if not path.exists(gene_annotation_file):
            sys.exit('gene annotation file not found')
        sgmfile = f"{project_dir}/intermediate/snpgenemapping_{int(mappingDistance/1000)}kb.pkl"
        snp2gene.mapsnp2gene(bimfile, gene_annotation_file, mappingDistance, 'matrix', sgmfile)

        # extract snp-pathway information
        geneset_pkl = f"{project_dir}/intermediate/{genesets}.pkl"
        outfile = snpp.snppathway(finalfile, sgmfile, geneset_pkl, minPath, maxPath)
        bpm.bpmind(outfile)
      
    elif job == 'ComputeInteraction':
        if not (model == 'RR' or model == 'RD' or model == 'DD' or model == 'combined'):
            sys.exit('wrong model')
            
        snpDataAD = f"{project_dir}/intermediate/SNPdataAD.pkl"
        if not path.exists(snpDataAD):
            sys.exit(snpDataAD + ' not found')
            
        snpDataAR = f"{project_dir}/intermediate/SNPdataAR.pkl"
        if not path.exists(snpDataAR):
            sys.exit(snpDataAR + ' not found')
            
        if r < 0 :
            if model == 'combined':
                ci.combine(project_dir, alpha1, alpha2, n_workers, i)
            else:
                ci.run(project_dir, model, alpha1, alpha2, n_workers, i)
        else:
            for i in range(r + 1):
                if model == 'combined':
                    ci.combine(project_dir, alpha1, alpha2, n_workers, i)
                else:
                    ci.run(project_dir, model, alpha1, alpha2, n_workers, i)

    elif job == 'ComputeStats':
        if not (model == 'RR' or model == 'RD' or model == 'DD' or model == 'combined' or ssmfile != None):
            sys.exit('wrong model')
            
        bpmfile = f"{project_dir}/intermediate/BPMind.pkl"
        if not path.exists(bpmfile):
            sys.exit(f"{bpmfile} not found")
            
        snpDataAD = f"{project_dir}/intermediate/SNPdataAD.pkl"
        if not path.exists(snpDataAD):
            sys.exit(snpDataAD + ' not found')
            
        snpDataAR = f"{project_dir}/intermediate/SNPdataAR.pkl"
        if not path.exists(snpDataAR):
            sys.exit(snpDataAR + ' not found')
            
        if ssmfile == None:
            if r < 0:
                if model == 'combined':
                    ssmfile = f"{project_dir}/intermediate/ssM_mhygessi_combined_R{str(i)}.pkl"
                else:
                    ssmfile = f"{project_dir}/intermediate/ssM_mhygessi_{model}_R{str(i)}.pkl"
                gs.genstats(ssmfile, bpmfile, binaryNetwork, snpPerms, minPath, n_workers, densitycutoff)
                
            else:
                for i in range(r+1):
                    if model == 'combined':
                        ssmfile = f"{project_dir}/intermediate/ssM_mhygessi_combined_R{str(i)}.pkl"
                    else:
                        ssmfile = f"{project_dir}/intermediate/ssM_mhygessi_{model}_R{str(i)}.pkl"
                    gs.genstats(ssmfile, bpmfile, binaryNetwork, snpPerms, minPath, n_workers, densitycutoff)
                    
        else:
            ssmfile = f"{project_dir}/intermediate/{ssmfile}"
            gs.genstats(ssmfile, bpmfile, binaryNetwork, snpPerms, minPath, n_workers, densitycutoff)

    elif job == 'ComputeFDR':
        bpmfile = f"{project_dir}/intermediate/BPMind.pkl"
        if not path.exists(bpmfile):
            sys.exit(f"{bpmfile} not found")
            
        if ssmfile == None:
            if model == 'combined':
                ssmfile = f"{project_dir}/intermediate/ssM_mhygessi_combined_R0.pkl"
            else:
                ssmfile = f"{project_dir}/intermediate/ssM_mhygessi_{model}_R0.pkl"
        else:
            ssmfile = f"{project_dir}/intermediate/{ssmfile}"
        if not path.exists(ssmfile):
            sys.exit(f"{ssmfile} not found")
            
        fdr.fdrsampleperm(ssmfile, bpmfile, pval_cutoff, minPath, sample_perms)

    elif job == 'Summarize':
        bpmfile = f"{project_dir}/intermediate/BPMind.pkl"
        if not path.exists(bpmfile):
            sys.exit(f"bpm file not found at: {bpmfile}")
            
        snppathwayfile = f"{project_dir}/intermediate/{snppathwayfile}"
        if not path.exists(snppathwayfile):
            sys.exit(f"snp-pathway mapping file not found at: {snppathwayfile}")
            
        snpgenemappingfile = f"{project_dir}/intermediate/snpgenemapping_{int(mappingDistance/1000)}kb.pkl"
        if not path.exists(snpgenemappingfile):
            sys.exit(f"snpgenemappingfile not found at: {snpgenemappingfile}")
        
        if ssmfile == None:
            imported = False
            if model == 'combined':
                ssmfile = f"{project_dir}/intermediate/ssM_mhygessi_combined_R0.pkl"
                resultsfile = f"{project_dir}/intermediate/results_ssM_mhygessi_combined_R0.pkl"
            else:
                ssmfile = f"{project_dir}/intermediate/ssM_mhygessi_{model}_R0.pkl"
                resultsfile = f"{project_dir}/intermediate/results_ssM_mhygessi_{model}_R0.pkl"
        else:
            resultsfile = f"{project_dir}/intermediate/results_{ssmfile}"
            ssmfile = f"{project_dir}/intermediate/{ssmfile}"
            imported = True
        if not path.exists(ssmfile):
            sys.exit(f"interaction file not found at: {ssmfile}")
        if not path.exists(resultsfile):
            sys.exit(f"results file not found at: {resultsfile}")

        cl.collectresults(resultsfile, fdrcut, ssmfile, bpmfile, snppathwayfile, snpgenemappingfile, imported, densitycutoff)

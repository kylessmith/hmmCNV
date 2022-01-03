import pandas as pd
import numpy as np
import copy as cp
from . import hmm_utilities
import time
from intervalframe import IntervalFrame


def get_seq_info(genomeBuild = "hg19", genomeStyle = "NCBI"):
    """
    """

    # Read Chromosome lengths
    seq_info = {}
    for line in open(genomeBuild+"_chrom_sizes.txt", "r"):
        fields = line.strip().split("\t")

        # Manage chr's
        if genomeStyle == "NCBI":
            seq_info[fields[0][3:]] = int(fields[1])
        else:
            seq_info[fields[0]] = int(fields[1])

    return seq_info

def wigToSeries(wig_fn):
    """
    """

    # Set up recordings
    bin_value = defaultdict(list)
    bin_index = defaultdict(list)

    # Read wig file
    for line in open(wig_fn, "r"):
        fields = line.strip().split(" ")
        
        # Check for header
        if len(fields) > 1:
            chrom = fields[1].split("=")[1]
            start = int(fields[2].split("=")[1])
            end = int(fields[3].split("=")[1])
            bin_size = int(fields[4].split("=")[1])
        # Record bin value
        else:
            value = int(fields[0])
            bin_value[chrom].append(value)
            bin_index[chrom].append(start)
            start += bin_size

    # Convert lists to Series
    wig_results = {}
    for chrom in bin_value:
        wig_results = pd.Series(np.array(bin_value[chrom]),
                                index=bin_index[chrom])

    return wig_results, bin_size


def hmmCNV(tumour_copy, normal = [0.2, 0.5, 0.75], ploidy = [1, 2, 3], scStates = None,
            lambda_p = None, lambdaScaleHyperParam = 3, estimateNormal = True, estimatePloidy = True,
            estimateScPrevalence = True, maxFracCNASubclone = 0.7, maxFracGenomeSubclone = 0.5,
            minSegmentBins = 50, altFracThreshold = 0.05, coverage = None, maxCN = 7, txnE = 0.9999999,
            txnStrength = 1e7, normalizeMaleX = True, includeHOMD = False, fracReadsInChrYForMale = 0.001,
            chrXMedianForMale = -0.1, outDir = "./", libdir = None, plotFileType = "pdf", plotYLim = (-2,2),
            gender = None, genomeBuild = "hg19", genomeStyle = "UCSC",
            chrs = np.append(np.arange(1,23).astype(str), "X"), chrTrain = np.arange(1, 23).astype(str),
            chrNormalize = np.arange(1, 23).astype(str), verbose=False):
    """
    Run ichor HMM copy algorithm

    Parameters
    ----------
        tumour_copy : IntervalFrame

        normal : list of int

        ploidy : list of int

        scStates : 

        lambda_p : float

        lambdaScaleHyperParam : int

        estimateNormal : bool

        estimatePloidy : bool

        estimateScPrevalence : bool

        maxFracCNASubclone : float

        maxFracGenomeSubclone : float

        minSegmentBins : int

        altFracThreshold : float

        coverage : 

        maxCN : int

        txnE : float

        txnStrength : float

        normalizeMaleX : bool

        includeHOMD : bool

        fracReadsInChrYForMale : float

        chrXMedianForMale : float

        outDir : str

        libdir : str

        plotFileType : str

        plotYLim : tuple

        gender : dict

        genomeBuild : str

        genomeStyle : str

        chrs : array-like

        chrTrain : array-like

        chrNormalize : array-like

        verbose : bool

    Returns
    -------
        loglik : pandas.DataFrame
        
        results : dict
    """

    # Manage chr's
    if genomeStyle == "UCSC":
        chrTrain = np.array(["chr"+c for c in chrTrain])
        chrs = np.array(["chr"+c for c in chrs])

    normal_copy = None
    gender_mismatch = False
    numSamples = 1

    i = 0
    # Determine which chromosomes to use for training
    chrInd = tumour_copy.index.get_locs(chrTrain)

    ### RUN HMM ###
    start_time = time.perf_counter()

    results = {}
    loglik = pd.DataFrame([], columns=["init", "n_est", "phi_est", "BIC", "Frac_genome_subclonal",
                                    "Frac_CNA_subclonal", "loglik"],
                          index=np.arange(len(normal)* len(ploidy)))

    counter = 0
    compNames = np.repeat("NA", loglik.shape[0])
    mainName = np.repeat("NA", len(normal) * len(ploidy))

    #### restart for purity and ploidy values ####
    for n in normal:
        for p in ploidy:
            logR = tumour_copy.df.loc[:,"ratios"].copy(deep=True)

            param = hmm_utilities.getDefaultParameters(logR.loc[chrInd], maxCN=maxCN, includeHOMD=includeHOMD, 
                                        ct_sc=scStates, ploidy=p, e=txnE, e_sameState=50, strength=txnStrength)
            param["phi_0"] = np.repeat(p, numSamples)
            param["n_0"] = np.repeat(n, numSamples)

            ############################################
            ######## CUSTOM PARAMETER SETTINGS #########
            ############################################
            # 0.1x cfDNA #
            if lambda_p is None:
                logR_var = 1 / ((np.nanstd(logR, ddof=1, axis=0) / np.sqrt(len(param["ct"]))) **2)
                param["lambda"] = np.repeat(logR_var, len(param["ct"]))
                param["lambda"][np.in1d(param["ct"], np.array([2]))] = logR_var 
                param["lambda"][np.in1d(param["ct"], np.array([1,3]))] = logR_var 
                param["lambda"][param["ct"] >= 4] = logR_var / 5
                param["lambda"][param["ct"] == np.max(param["ct"])] = logR_var / 15
                param["lambda"][param["ct_sc_status"]] = logR_var / 10
            else:
                param["lambda"][np.in1d(param["ct"], np.array([2]))] = lambda_p[2]
                param["lambda"][np.in1d(param["ct"], np.array([1]))] = lambda_p[1]
                param["lambda"][np.in1d(param["ct"], np.array([3]))] = lambda_p[3]
                param["lambda"][param["ct"] >= 4] = lambda_p[4]
                param["lambda"][param["ct"] == np.max(param["ct"])] = lambda_p[2] / 15
                param["lambda"][param["ct_sc_status"]] = lambda_p[2] / 10
            param["alphaLambda"] = np.repeat(lambdaScaleHyperParam, len(param["ct"]))

            #############################################
            ################ RUN HMM ####################
            #############################################

            hmmResults_cor = hmm_utilities.HMMsegment(tumour_copy.copy(), None, dataType="ratios", 
                                                        param=param, chrTrain=chrTrain, maxiter=50,
                                                        estimateNormal=estimateNormal, estimatePloidy=estimatePloidy,
                                                        estimateSubclone=estimateScPrevalence, verbose=verbose)

            ########### CN correction ###################
            iteration = hmmResults_cor["results"]["iter"]

            # correct integer copy number based on estimated purity and ploidy
            correctedResults = hmm_utilities.correctIntegerCN(cn = hmmResults_cor["cna"].copy(),
                                                segs = hmmResults_cor["results"]["segs"].copy(), 
                                                purity = 1 - hmmResults_cor["results"]["n"][0][iteration],
                                                ploidy = hmmResults_cor["results"]["phi"][0][iteration],
                                                cellPrev = 1 - hmmResults_cor["results"]["sp"][0][iteration], 
                                                maxCNtoCorrect_autosomes = maxCN, maxCNtoCorrect_X = maxCN,
                                                minPurityToCorrect = 0.03, 
                                                gender = gender["gender"] if gender is not None else None,
                                                chrs = chrs,correctHOMD = includeHOMD)
            
            hmmResults_cor["results"]["segs"] = correctedResults["segs"]
            hmmResults_cor["cna"] = correctedResults["cn"]

            ## convert full diploid solution (of chrs to train) to have 1.0 normal or 0.0 purity
            ## check if there is an altered segment that has at least a minimum # of bins
            segsS = hmmResults_cor["results"]["segs"]
            segsS = segsS.loc[chrTrain, :]
            segAltInd = np.where(segsS.loc[:,"event"].values != "NEUT")[0]

            maxBinLength = -np.inf
            if np.sum(segAltInd) > 0:
                maxInd = np.argmax(segsS.ends()[segAltInd] - segsS.starts()[segAltInd] + 1)
                query = segsS.index[segAltInd[maxInd]]
                #query = query[query.unique_labels[0]]
                #subject = tumour_copy[pid]
                nhits = tumour_copy.nhits(query.start, query.end, query.label)
                maxBinLength = nhits
            
            ## check if there are proportion of total bins altered 
            # if segment size smaller than minSegmentBins, but altFrac > altFracThreshold, then still estimate TF
            cnaS = hmmResults_cor["cna"]
            altInd = cnaS.df.loc[np.in1d(cnaS.index.extract_labels(), chrTrain), "event"].values == "NEUT"
            altFrac = np.nansum(~altInd) / len(altInd)
            if maxBinLength <= minSegmentBins and altFrac <= altFracThreshold:
                hmmResults_cor["results"]["n"][0][iteration] = 1.0

            iteration = hmmResults_cor["results"]["iter"]
            results[counter] = hmmResults_cor
            loglik.loc[counter, "loglik"] = hmmResults_cor["results"]["loglik"][iteration]
            subClonalBinCount = pd.Series({0:hmmResults_cor["cna"].df.loc[:,"subclone_status"].sum()})
            fracGenomeSub = subClonalBinCount / pd.Series({0:hmmResults_cor["cna"].shape[0]})
            fracAltSub = subClonalBinCount / pd.Series({0:np.sum(hmmResults_cor["cna"].df.loc[:,"copy_number"].values != 2)})
            fracAltSub = pd.Series({i:0 if np.isnan(fracAltSub.loc[i]) else fracAltSub.loc[i] for i in fracAltSub.index.values})
            loglik.loc[counter, "Frac_genome_subclonal"] = ",".join(list(fracGenomeSub.values.astype(str)))
            loglik.loc[counter, "Frac_CNA_subclonal"] = ",".join(list(fracAltSub.values.astype(str)))
            loglik.loc[counter, "init"] = "n"+str(n)+"-p"+str(p)
            loglik.loc[counter, "n_est"] = ",".join(list(hmmResults_cor["results"]["n"][:, iteration].astype(str)))
            loglik.loc[counter, "phi_est"] = ",".join(list(hmmResults_cor["results"]["phi"][:, iteration].astype(str)))

            counter = counter + 1

    if verbose: print("Total ULP-WGS HMM Runtime:", ((time.perf_counter() - start_time) / 60), "min.")

    return loglik, results
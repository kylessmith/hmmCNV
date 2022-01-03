import numpy as np
import pandas as pd
from . import em_utilities
from .hmm import viterbi
from intervalframe import IntervalFrame
from ailist import LabeledIntervalArray


def getTransitionMatrix(K, e, strength):
    """


    Parameters
    ----------

    
    Returns
    -------


    """

    A = np.zeros((K,K), order="F")

    for j in range(K):
        A[j,:] = (1 - e) / (K - 1)
        A[j,j] = e

    A = normalize(A)
    A_prior = A
    dirPrior = A * strength

    return {"A":A, "dirPrior":dirPrior}


def normalize(A):
    """
    Normalize a given array to sum to 1

    Parameters
    ----------

    
    Returns
    -------


    """

    def vectorNormalize(x):
        return x / (np.sum(x, axis=0) + (np.sum(x, axis=0) == 0))

    if A.ndim < 2:
        M = vectorNormalize(A)
    else:
        M = vectorNormalize(A)

    return M


def getDefaultParameters(logR, maxCN = 5, ct_sc = None, ploidy = 2, e = 0.9999999,
                         e_sameState = 10, strength = 10000000, includeHOMD = False):
    """
    Determine default parameters
    
    Parameters
    ----------
        logR : pandas.Series
            Log ratios
        maxCN : int
            Maximum Copy Numbers
        ct_sc : int
            
        ploidy : int
            Assumed ploidy
        e : float
            
        e_sameState : int
            
        strength : int
            (default:10000000)
        includeHOMD : bool
            Whether to include homozygous deletions
        
    Returns
    -------
        param : dict
            Dictionary of HMM starting parameters
    
    """
    
    # Determine if HOMD is included
    if includeHOMD:
        ct = np.arange(0, maxCN+1)
    else:
        ct = np.arange(1, maxCN+1)
    
    # Initialize parameters
    length_ct_sc = 0 if ct_sc is None else len(ct_sc)
    param = {"strength" : strength, "e" : e,
    		"ct" : ct if ct_sc is None else np.append(ct, ct_sc),
    		"ct_sc_status" : np.append(np.repeat(False, len(ct)), np.repeat(True, length_ct_sc)),
    		"phi_0" : 2, "alphaPhi" : 4, "betaPhi" : 1.5,
    		"n_0" : 0.5, "alphaN" : 2, "betaN" : 2,
    		"sp_0" : 0.5, "alphaSp" : 2, "betaSp" : 2,
    		"lambda" : np.repeat(100, len(ct) + length_ct_sc).reshape(-1,1),
    		"nu" : 2.1,
    		"kappa" : np.repeat(75, len(ct)), 
    		"alphaLambda" : 5}
            
    K = len(param["ct"])
    
    # Initialize hyperparameters for precision using observed data
    param["numberSamples"] = 1
    betaLambdaVal = ((np.nanstd(logR, ddof=1) / np.sqrt(len(param["ct"]))) **2)
    param["betaLambda"] = np.repeat(betaLambdaVal, len(param["ct"]))

    param["alphaLambda"] = np.repeat(param["alphaLambda"], K)

    S = param["numberSamples"]

    logR_var = 1 / ((np.nanstd(logR, ddof=1, axis=0) / np.sqrt(len(param["ct"]))) **2)

    if logR.ndim == 2 and logR.shape[1] > 1:
        param["lambda"] = np.tile(logR_var, (K, S))
    else:
        param["lambda"] = np.repeat(logR_var, len(param["ct"]))
        param["lambda"][np.in1d(param["ct"], np.array([2]))] = logR_var 
        param["lambda"][np.in1d(param["ct"], np.array([1,3]))] = logR_var 
        param["lambda"][param["ct"] >= 4] = logR_var / 5
        param["lambda"][param["ct"] == np.max(param["ct"])] = logR_var / 15
        param["lambda"][param["ct_sc_status"]] = logR_var / 10

    # Define joint copy number states 
    param["jointCNstates"] = pd.DataFrame(np.repeat(param["ct"], S).astype(int), columns=["Sample_1"])
    param["jointSCstatus"] = pd.DataFrame(np.repeat(param["ct_sc_status"], S), columns=["Sample_1"])

    # Initialize transition matrix to the prior
    txn = getTransitionMatrix(K**S, e, strength)
    cnStateDiff = param["jointCNstates"].apply(lambda x: (np.abs(np.max(x) - np.min(x))), axis=1).values

    if e_sameState > 0 and S > 1:
        txn["A"][:, cnStateDiff==0] = txn["A"][:, cnStateDiff==0] * e_sameState * K 
        txn["A"][:, cnStateDiff>=3] = txn["A"][:, cnStateDiff>=3] / e_sameState / K

    for i in range(txn["A"].shape[0]):
        for j in range(txn["A"].shape[1]):
            if i == j:
                txn["A"][i, j] = e

    txn["A"] = normalize(txn["A"])
    param["A"] = txn["A"]
    param["dirPrior"] = txn["A"] * strength

    param["A"][:, param["ct_sc_status"]] = param["A"][:, param["ct_sc_status"]] / 10
    param["A"] = normalize(param["A"])
    param["dirPrior"][:, param["ct_sc_status"]] = param["dirPrior"][:, param["ct_sc_status"]] / 10

    if includeHOMD:
        K = len(param["ct"])
        param["A"][0, 1:K] = param["A"][0, 1:K] * 1e-5
        param["A"][1:K, 0] = param["A"][1:K, 0] * 1e-5
        param["A"][0, 0] = param["A"][0, 0] * 1e-5
        param["A"] = normalize(param["A"])
        param["dirPrior"] = param["A"] * param["strength"]

    param["kappa"] = np.repeat(75, K**S)
    param["kappa"][cnStateDiff == 0] = param["kappa"][cnStateDiff == 0] + 125
    param["kappa"][cnStateDiff >=3] = param["kappa"][cnStateDiff >=3] - 50
    param["kappa"][np.sum(param["jointCNstates"].values==2, axis=1) == S] = 800

    return param


def runViterbi(convergedParams, chroms, verbose=False):
    """


    Parameters
    ----------

    
    Returns
    -------


    """

    if verbose: print("runViterbi: Segmenting and classifying")
    
    indexes = np.unique(chroms, return_index=True)[1]
    chrs = np.array([chroms[index] for index in sorted(indexes)])
    chrsI = []

    # initialise the chromosome index and the init state distributions
    for i in range(len(chrs)):
        chrsI.append(np.where(chroms == chrs[i])[0])

    segs = []
    py = convergedParams["py"]
    N = py.shape[1]
    Z = np.zeros(N, dtype=int)
    convergeIter = convergedParams["iter"]
    piG = convergedParams["pi"][:, convergeIter]
    A = convergedParams["A"]


    for c in range(len(chrsI)):
        I = chrsI[c]
        output = viterbi(np.log(piG), np.log(A), np.log(py[:, I]))
        Z[I] = output["path"]
        segs.append(output["seg"])

    return {"segs":segs, "states":Z}
        

def merge_Trues(x):
    """


    Parameters
    ----------
        x : numpy.ndarray
    
    Returns
    -------
        Generator of tuples

    """
    
    # Merge nearby True values
    x1 = np.hstack([ [False], x, [False] ])  # padding
    d = np.diff(x1.astype(int))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]

    # Determine indices
    for segment in zip(starts,ends):
        yield segment


def segment_data(x, states, convergedParams):
    """


    Parameters
    ----------
        x : IntervalFrame
    
    Returns
    -------
        total_segList : IntervalFrame

    """

    if np.sum(convergedParams["param"]["ct"] == 0) == 0:
        includeHOMD = False
    else:
        includeHOMD = True
    
    if ~includeHOMD:
        names = ["HETD","NEUT","GAIN","AMP","HLAMP"] + ["HLAMP"+str(i) for i in range(2,25)]
    else:
        names = ["HOMD","HETD","NEUT","GAIN","AMP","HLAMP"] + ["HLAMP"+str(i) for i in range(2,25)]
    
    jointStates = convergedParams["param"]["jointCNstates"].values

    #indexes = np.sort(np.unique(x.labels(), return_index=True)[1])
    chroms = x.index.unique_labels
    sample_segList = []
    sample_intervals = {}

    sample_intervals = LabeledIntervalArray()
    for i in range(len(chroms)):
        chrom = chroms[i]
        #sample_intervals[chrom] = LabeledIntervalArray()
        chrom_selected = x.loc[chrom,:]
        chrom_positions = chrom_selected.starts()
        chrom_values = chrom_selected.df["ratios"].values
        chrom_states = states[x.index.get_locs([chrom])]
        chrom_segList = []
        for state in np.unique(chrom_states):
            name = names[int(state)]
            for start, end in merge_Trues(chrom_states==state):
                end = min(end, len(chrom_positions)-1)
                median = np.nanmedian(chrom_values[start:end])
                #chrom_segList.append([chrom, chrom_positions[start], chrom_positions[end], int(state), name, median, jointStates[int(state),j]])
                sample_intervals.add(chrom_positions[start], chrom_positions[end], str(chrom))
                chrom_segList.append([int(state), name, median, jointStates[int(state)]])

        #chrom_segList.sort(key = lambda x: x[1])
        sample_segList += chrom_segList

    #total_segList.append(pd.DataFrame(sample_segList, columns=["chrom", "start", "end", "state", "event", "median", "copy_number"]))
    total_segList = IntervalFrame(sample_intervals,
                                  pd.DataFrame(sample_segList,
                                               columns=["state", "event", "median", "copy_number"]))

    return total_segList
    

def HMMsegment(x, validInd=None, dataType="ratios", param=None, 
               chrTrain=np.arange(22).astype(str), maxiter=50, estimateNormal=True, estimatePloidy=True, 
               estimatePrecision=True, estimateSubclone=True, estimateTransition=True,
               estimateInitDist=True, logTransform=False, verbose=False):
    """


    Parameters
    ----------
        x : IntervalFrame

        validInd : array-like

        dataType : str

        param : dict

        chrTrain : array-like

        maxiter : int

        estimateNormal : bool

        estimate Ploidy : bool

        estimatePrecision : bool

        estimateSubclone : bool

        estimateTransition : bool

        estimateInitDist : bool

        logTransform : bool

        verbose : bool
    
    Returns
    -------
        dict

    """
    
    # Find chromosomes present
    chroms = x.index.extract_labels()

    # Setup columns
    dataMat = x.df.loc[:,dataType].to_frame()

    # Normalize by median and log data
    if logTransform:
        dataMat = dataMat.apply(lambda x: np.log(x / np.nanmedian(x)), axis=1)
    else:
        dataMat = np.log(2**dataMat)

    # Update variable x with loge instead of log2
    x.df.loc[:, dataType] = dataMat.iloc[:, 0]

    chrInd = np.in1d(chroms, chrTrain)
    if validInd is not None:
        chrInd = np.logical_and(chrInd, validInd)
    if param is None:
        param = getDefaultParameters(dataMat.loc[chrInd,:])

    ####### RUN EM ##########

    convergedParams = em_utilities.runEM(dataMat, chroms, chrInd, param, maxiter, 
                                        verbose, estimateNormal=estimateNormal, estimatePloidy=estimatePloidy, 
                                        estimateSubclone=estimateSubclone, estimatePrecision=estimatePrecision, 
                                        estimateTransition=estimateTransition, estimateInitDist=estimateInitDist)
    
    viterbiResults = runViterbi(convergedParams, chroms, verbose=verbose)

    # Segment data based on called states
    segs = segment_data(x, viterbiResults["states"], convergedParams)
    
    names = np.array(["HOMD","HETD","NEUT","GAIN","AMP","HLAMP"] + ["HLAMP"+str(i) for i in range(2,25)])

    #cnaList = {}
    #pid = list(x.keys())[i]
    copyNumber = param["jointCNstates"].values[viterbiResults["states"]]
    subclone_status = param["jointSCstatus"].values[viterbiResults["states"]]

    chroms = x.index.extract_labels()

    #cnaList = pd.DataFrame([chroms], index=["chr"]).T
    #cnaList.loc[:,"start"] = x.starts()
    #cnaList.loc[:,"end"] = x.ends()
    cnaList = IntervalFrame.from_array(x.starts(), x.ends(), labels=chroms)
    cnaList.loc[:,"copy_number"] = copyNumber
    cnaList.loc[:,"event"] = names[copyNumber]
    cnaList.loc[:,"logR"] = np.log2(np.exp(dataMat.values))
    cnaList.loc[:,"subclone_status"] = subclone_status

    ## order by chromosome ##
    #indexes = np.sort(np.unique(x[pid].loc[:,"seqnames"].values, return_index=True)[1])
    #chrOrder_chroms = x.index.unique_labels
    #chrOrder = np.zeros(len(chroms), dtype=int)
    #shift = 0
    #for c in chrOrder_chroms:
    #    selected = np.where(chroms == c)[0]
    #    chrOrder[shift:shift+len(selected)] = selected
    #    shift += len(selected)

    #cnaList = cnaList.loc[chrOrder_chroms,:]

    ## segment mean loge -> log2
    segs.df["median"] = np.log2(np.exp(segs.df["median"].values))
    ## add subclone status
    segs.df["subclone_status"] = param["jointSCstatus"].values[segs.df["state"].values]
    
    convergedParams["segs"] = segs

    return {"cna":cnaList, "results":convergedParams, "viterbiResults":viterbiResults}


def logRbasedCN(x, purity, ploidyT, cellPrev=np.nan, cn = 2):
    """
    compute copy number using corrected log ratio

    Parameters
    ----------

    
    Returns
    -------


    """

    if len(cellPrev) == 1 and np.isnan(cellPrev):
        cellPrev = 1
    else: #if cellPrev is a vector
        cellPrev[np.isnan(cellPrev)] = 1
    
    ct = (2**x 
            * (cn * (1 - purity) + purity * ploidyT * (cn / 2)) 
            - (cn * (1 - purity)) 
            - (cn * purity * (1 - cellPrev)))
    ct = ct / (purity * cellPrev)
    ct = np.maximum(ct, 1/2**6)

    # Replace inf with 0
    ct[ct == np.inf] = 0
    
    return ct



def correctIntegerCN(cn, segs, purity, ploidy, cellPrev, callColName = "event", maxCNtoCorrect_autosomes = None, 
		             maxCNtoCorrect_X = None, correctHOMD = True, minPurityToCorrect = 0.2, gender = "male",
                     chrs = np.append(np.arange(1,23).astype(str), np.array(["X"]))):
    """
    Recompute integer CN for high-level amplifications
    compute logR-corrected copy number

    Parameters
    ----------

    
    Returns
    -------


    """

    names = np.array(["HOMD","HETD","NEUT","GAIN","AMP","HLAMP"] + ["HLAMP"+str(i) for i in range(2,3000)] + ["NA"])

    ## set up chromosome style
    autosomeStr = np.array([c for c in chrs if "X" not in c and "Y" not in c])
    chrXStr = [c for c in chrs if "X" in c][0]

    if maxCNtoCorrect_autosomes is None:
        maxCNtoCorrect_autosomes = np.nanmax(segs.loc[np.in1d(segs.loc[:,"chrom"].values, autosomeStr), "copy_number"].values)
    
    if maxCNtoCorrect_X is None and gender == "female" and len(chrXStr) > 0:
        maxCNtoCorrect_X = np.nanmax(segs.loc[segs.loc[:,"chrom"].values == chrXStr, "copy_number"])
    
    ## correct log ratio and compute corrected CN
    cellPrev_seg = np.ones(segs.shape[0])
    cellPrev_seg[segs.df["subclone_status"]] = cellPrev
    segs.loc[:,"logR_Copy_Number"] = logRbasedCN(segs.loc[:,"median"].values, purity, ploidy, cellPrev_seg, cn=2)
    if gender == "male" and len(chrXStr) > 0: ## analyze chrX separately
        ind_cnChrX = np.where(segs.loc[:,"chrom"].values == chrXStr)[0]
        segs.loc[ind_cnChrX,"logR_Copy_Number"] = logRbasedCN(segs.loc[:,"median"].values[ind_cnChrX], purity, ploidy, cellPrev_seg[ind_cnChrX], cn=1)
    
    ## assign copy number to use - Corrected_Copy_Number
    # same ichorCNA calls for autosomes - no change in copy number
    segs.loc[:,"Corrected_Copy_Number"] = segs.loc[:,"copy_number"].values.astype(int)
    segs.loc[:,"Corrected_Call"] = segs.loc[:,callColName].values

    ind_change = np.array([])
    if purity >= minPurityToCorrect:
        # ichorCNA calls adjusted for >= copies - HLAMP
        # perform on all chromosomes
        ind_cn = np.where(np.logical_or(segs.loc[:,"copy_number"].values >= maxCNtoCorrect_autosomes,
                                        segs.loc[:,"logR_Copy_Number"].values >= maxCNtoCorrect_autosomes * 1.2))[0]
        segs.df.loc[ind_cn,"Corrected_Copy_Number"] = np.round(segs.df.loc[:,"logR_Copy_Number"].values[ind_cn]).astype(int)
        segs.df.loc[ind_cn,"Corrected_Call"] = names[segs.df.loc[:,"Corrected_Copy_Number"].values[ind_cn] + 1]
        ind_change = np.append(ind_change, ind_cn)

        # ichorCNA calls adjust for HOMD
        if correctHOMD:
            ind_cn = np.where(np.logical_and(np.in1d(segs.index.extract_labels(), chrs),
                                             np.logical_or(segs.loc[:,"copy_number"].values == 0,
                                                           segs.loc[:,"logR_Copy_Number"].values == 1/2**6)))[0]
            segs.df.loc[ind_cn,"Corrected_Copy_Number"] = np.round(segs.df.loc[:,"logR_Copy_Number"].values[ind_cn])
            segs.df.loc[ind_cn,"Corrected_Call"] = names[segs.df.loc[:,"Corrected_Copy_Number"].vaues[ind_cn] + 1]
            ind_change = np.append(ind_change, ind_cn)
        
        # Re-adjust chrX copy number for males (females already handled above)
        if gender == "male" and len(chrXStr) > 0:
            ind_cn = np.where(np.logical_and(segs.index.extract_labels() == chrXStr,
                                            np.logical_or(segs.df.loc[:,"copy_number"].values >= maxCNtoCorrect_X,
                                                          segs.df.loc[:,"logR_Copy_Number"].values >= maxCNtoCorrect_X * 1.2)))[0]
            segs.df.loc[ind_cn,"Corrected_Copy_Number"] = np.round(segs.df.loc[:,"logR_Copy_Number"].values[ind_cn]).astype(int)
            segs.df.loc[ind_cn,"Corrected_Call"] = names[segs.df.loc[:,"Corrected_Copy_Number"].values[ind_cn] + 2]
            ind_change = np.append(ind_change, ind_cn)
        
    ## adjust the bin level data ##
    # 1) assign the original calls
    cn.loc[:,"Corrected_Copy_Number"] = cn.loc[:,"copy_number"].values
    cn.loc[:,"Corrected_Call"] = cn.loc[:,callColName].values
    cellPrev_cn =  np.ones(cn.shape[0])
    cellPrev_cn[cn.loc[:,"subclone_status"].values] = cellPrev
    cn.loc[:,"logR_Copy_Number"] = logRbasedCN(cn.loc[:,"logR"].values, purity, ploidy, cellPrev_cn, cn=2)
    if gender == "male" and len(chrXStr) > 0: ## analyze chrX separately
        ind_cnChrX = np.where(cn.loc[:,"chr"].values == chrXStr)[0]
        cn.loc[ind_cnChrX,"logR_Copy_Number"] = logRbasedCN(cn.loc[:,"logR"].values[ind_cnChrX], purity, ploidy, cellPrev_cn[ind_cnChrX], cn=1)
    
    # 2) find which segs changed/adjusted
    ind_change = np.unique(ind_change).astype(int)

    # 3) correct bins overlapping adjusted segs
    hits = np.array([])
    for i in ind_change:
        fields = segs.iloc[i,:]

        overlaps = cn.index.has_hit(fields.index[0].start, fields.index[0].end, fields.index[0].label)
        #overlaps = np.logical_and(cn.loc[:,"chr"].values == fields.loc["chrom"],
        #                          np.logical_and(cn.loc[:,"start"].values <= fields.loc["end"],
        #                          cn.loc[:,"end"].values >= fields.loc["start"]))
        cn.df.loc[overlaps,"Corrected_Copy_Number"] = fields.df.loc[:,"Corrected_Copy_Number"]
        cn.df.loc[overlaps,"Corrected_Call"] = fields.df.loc[:,"Corrected_Call"]
        hits = np.append(hits, np.where(overlaps)[0])

    # 4) correct bins that are missed as high level amplifications
    ind_cn = np.where(np.logical_or(cn.df.loc[:,"copy_number"].values >= maxCNtoCorrect_autosomes,
                                    cn.df.loc[:,"logR_Copy_Number"].values >= maxCNtoCorrect_autosomes * 1.2))[0]
    ind_cn = ind_cn[~np.in1d(ind_cn, hits)]
    cn.df.loc[ind_cn,"Corrected_Copy_Number"] = np.round(cn.df.loc[:,"logR_Copy_Number"].values[ind_cn])
    corr_copy = np.round(cn.df.loc[:,"logR_Copy_Number"].values[ind_cn])
    corr_copy[pd.isnull(corr_copy)] = -2
    corr_copy[corr_copy >= len(names)] = len(names) - 2 # scale to max label
    corr_copy = corr_copy.astype(int)
    cn.df.loc[ind_cn,"Corrected_Call"] = names[corr_copy + 1]
    
    return {"cn":cn, "segs":segs}
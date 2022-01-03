import numpy as np
import pandas as pd
from math import lgamma
from scipy.special import betaln, gammaln
from . import hmm_utilities
from scipy.special import gamma
from .hmm import forward_backward
from scipy.optimize import minimize, Bounds
import time


def get2and3ComponentMixture(ct, ct_sc_status, n, sp, phi):
    """


    Parameters
    ----------

    
    Returns
    -------


    """

    #S = len(n)
    cn = 2
    mu = np.tile(np.nan, ct.shape)
    #for s in range(S):
        #subclonal 3 component
    mu[ct_sc_status == True] = (((1 - n) * (1 - sp) * ct[ct_sc_status == True]) + 
                ((1 - n) * sp * cn) + (n * cn)) / ((1 - n) * phi + n * cn)
    #clonal 2 component
    mu[ct_sc_status == False] = ((1 - n) * ct[ct_sc_status == False] + n * cn) / ((1 - n) * phi + n * cn)
    
    return np.log(mu)


def tdistPDF(x, mu, lambda_p, nu):
    """


    Parameters
    ----------
        x : np.ndarrau
    
    Returns
    -------


    """

    S = x.shape[1]
    if S is not None:
        p = None
        for s in range(S):
            tpdf = (gamma(nu / 2 + 0.5) / gamma(nu / 2)) * ((lambda_p[s] / (np.pi * nu)) **(0.5)) * (1 + (lambda_p[s] * (x[:, s] - mu[s]) **2) / nu) **(-0.5 * nu - 0.5)
            p = tpdf if p is None else np.array([p, tpdf])
    else:
        nu = np.repeat(nu, len(mu))
        p = (gamma(nu / 2 + 0.5) / gamma(nu / 2)) * ((lambda_p / (np.pi * nu)) **(0.5)) * (1 + (lambda_p * (x - mu) **2) / nu) **(-0.5 * nu - 0.5)
    
    p[np.isnan(p)] = 1

    #nu = np.repeat(nu, len(mu))
    #p = (gamma(nu / 2 + 0.5) / gamma(nu / 2)) * ((lambda_p / (np.pi * nu)) **(0.5)) * (1 + (lambda_p * (x - mu) **2) / nu) **(-0.5 * nu - 0.5)
    #p[np.isnan(p)] = 1
    
    return p


def dirichletpdf(x, alpha, verbose=False):
    """


    Parameters
    ----------

    
    Returns
    -------


    """

    if (x < 0).any():
        return 0
    
    if np.abs(np.sum(x) - 1) > 1e-3:
        if verobose: print("Dirichlet PDF: probabilities do not sum to 1")
        return 0
    
    p = np.exp(gammaln(np.sum(alpha)) - np.sum(gammaln(alpha))) * np.prod(x **(alpha - 1))
    return p


def dirichletpdflog(x, k):
    """


    Parameters
    ----------

    
    Returns
    -------


    """

    c = gammaln(np.nansum(k)) - np.nansum(gammaln(k))  #normalizing constant
    l = np.nansum((k - 1) * np.log(x))  #likelihood
    y = c + l

    return y


def gammapdflog(x, a, b): #rate and scale parameterization
    """


    Parameters
    ----------

    
    Returns
    -------


    """

    c = a * np.log(b) - gammaln(a)  # normalizing constant  
    l = (a - 1) * np.log(x) + (-b * x)  #likelihood  
    y = c + l

    return y


def betapdflog(x, a, b):
    """


    Parameters
    ----------

    
    Returns
    -------


    """

    y = -betaln(a, b) + (a - 1) * np.log(x) + (b - 1) * np.log(1 - x)

    return y


def estimateMixWeightsParamMap(rho, kappa):
    """


    Parameters
    ----------

    
    Returns
    -------


    """

    K = rho.shape[0]
    pi = (np.nansum(rho, axis=1) + kappa - 1) / (np.sum(np.nansum(rho, axis=1)) + np.sum(kappa) - K)

    return pi


def stLikelihood(n, sp, phi, lambda_p, params, D, rho):
    """
    Student's t likelihood function

    Parameters
    ----------

    
    Returns
    -------


    """

    KS = rho.shape[0]
    lik = 0
    
    # Recalculate the likelihood
    lambdaKS = lambda_p.reshape(-1,1)
    mus = get2and3ComponentMixture(params["jointCNstates"].values, params["jointSCstatus"].values, n, sp, phi)
    for ks in range(KS):
        probs = np.log(tdistPDF(D.values, mus[ks,:], lambdaKS[ks,:], params["nu"]))    
        # multiply across samples for each data point to get joint likelihood.
        l = np.dot(rho[ks,:], probs)
        lik = lik + l
    
    return lik


def completeLikelihoodFun(x, pType, n, sp, phi, lambda_p, piG, A, params, D, rho, Zcounts,
                                  estimateNormal = True, estimatePloidy = True,
                                  estimatePrecision = True, estimateTransition = False,
                                  estimateInitDist = True, estimateSubclone = True):
    """


    Parameters
    ----------

    
    Returns
    -------


    """
    
    KS = rho.shape[0]
    N = rho.shape[1]
    S = params["numberSamples"]
    K = len(params["ct"])

    lambda_p = np.tile(lambda_p, S)

    if estimatePrecision and np.sum(pType == "lambda") > 0:
        lambda_p = np.tile(x[pType == "lambda"], S)

    if estimateNormal and np.sum(pType == "n") > 0:
        n = x[pType == "n"]

    if estimateSubclone and np.sum(pType == "sp") > 0:
        sp = x[pType == "sp"]
    
    if estimatePloidy and np.sum(pType == "phi") > 0:
        phi = x[pType == "phi"]

    ## prior probabilities ##
    prior = priorProbs(n, sp, phi, lambda_p, piG, A, params, 
                        estimateNormal = estimateNormal, estimatePloidy = estimatePloidy,
                        estimatePrecision = estimatePrecision, estimateTransition = estimateTransition,
                        estimateInitDist = estimateInitDist, estimateSubclone = estimateSubclone)
    ## likelihood terms ##
    likObs = stLikelihood(n, sp, phi, lambda_p, params, D, rho)
    likInit = np.dot(rho[:, 1], np.log(piG))
    likTxn = 0
    for ks in range(KS):
        likTxn = likTxn + np.dot(Zcounts[ks,:], np.log(A[ks,:]))

    ## sum together ##
    lik = likObs + likInit + likTxn
    prior = prior["prior"]
    f = lik + prior
    
    # Multiply by -1, because we want to maximize, not minimize
    return f * -1


def priorProbs(n, sp, phi, lambda_p, piG, A, params, 
                estimateNormal = True, estimatePloidy = True,
                estimatePrecision = True, estimateTransition = True,
                estimateInitDist = True, estimateSubclone = True):
    """


    Parameters
    ----------

    
    Returns
    -------


    """

    S = params["numberSamples"]
    K = len(params["ct"])
    KS = K **S
    
    ## prior terms ##
    priorA = 0
    if estimateTransition:
        for ks in range(KS):
            priorA = priorA + dirichletpdflog(A[ks,:], params["dirPrior"][ks,:])

    priorLambda = 0
    if estimatePrecision:
        if S > 1:
            for s in range(S):
                for k in range(K):
                    priorLambda = priorLambda + gammapdflog(lambda_p[k, s], params["alphaLambda"][k], params["betaLambda"][k, s])
        else:
            for k in range(K):
                priorLambda = priorLambda + gammapdflog(lambda_p[k], params["alphaLambda"][k], params["betaLambda"][k])

    priorLambda = priorLambda
    priorN = 0
    if estimateNormal:
        priorN = np.sum(betapdflog(n, params["alphaN"], params["betaN"]))

    priorSP = 0
    if estimateNormal:
        priorSP = np.sum(betapdflog(sp, params["alphaSp"], params["betaSp"]))
    
    priorPhi = 0
    if estimatePloidy:
        priorPhi = np.sum(gammapdflog(phi, params["alphaPhi"], params["betaPhi"]))
    
    priorPi = 0
    if estimateInitDist:
        priorPi = dirichletpdflog(piG, params["kappa"])
    
    prior = priorA + priorLambda + priorN + priorSP + priorPhi + priorPi

    return {"prior":prior, "priorA":priorA, "priorLambda":priorLambda, 
            "priorN":priorN, "priorSP":priorSP, "priorPhi":priorPhi, "priorPi":priorPi}


def estimateParamsMap(D, n_prev, sp_prev, phi_prev, lambda_prev, pi_prev, A_prev, 
                              params, rho, Zcounts, 
                              estimateNormal = True, estimatePloidy = True,
                              estimatePrecision = True, estimateInitDist = True, 
                              estimateTransition = True, estimateSubclone = True,
                              verbose = True):
    """


    Parameters
    ----------

    
    Returns
    -------


    """

    KS = rho.shape[0]
    K = len(params["ct"])
    T = rho.shape[1]
    S = len(n_prev)

    intervalNormal = (1e-6, 1 - 1e-6)
    intervalSubclone = (1e-6, 1 - 1e-6)
    intervalPhi = (np.finfo(np.double).eps, 10)
    intervalLambda = (1e-5, 1e4)

    # initialize params to be estimated
    n_hat = n_prev
    sp_hat = sp_prev
    phi_hat = phi_prev
    lambda_hat = lambda_prev
    pi_hat = pi_prev
    A_hat = A_prev

    # Update transition matrix A
    if estimateTransition:
        for k in range(KS):
            A_hat[k,:] = Zcounts[k,:] + params["dirPrior"][k,:]
            A_hat[k,:] = hmm_utilities.normalize(A_hat[k,:])

    # map estimate for pi (initial state dist)
    if estimateInitDist:
        pi_hat = estimateMixWeightsParamMap(rho, params["kappa"])

    # map estimate for normal 
    if estimateNormal:
        estNorm = minimize(completeLikelihoodFun, n_prev, args=(np.repeat("n", S), n_prev, sp_prev, phi_prev, 
                                                        lambda_prev, pi_hat, A_hat, params, D, rho, Zcounts, 
                                                        estimateNormal, estimatePloidy, estimatePrecision, estimateInitDist,
                                                        estimateTransition, estimateSubclone), method='L-BFGS-B',
                                                        bounds=Bounds(intervalNormal[0],intervalNormal[1]),
                                                        options={'eps':1e-8})
        n_hat = estNorm.x

    if estimateSubclone:
        estSubclone = minimize(completeLikelihoodFun, sp_prev, args=(np.repeat("sp", S), n_hat, sp_prev, phi_prev, 
                                                        lambda_prev, pi_hat, A_hat, params, D, rho, Zcounts, 
                                                        estimateNormal, estimatePloidy, estimatePrecision, estimateInitDist,
                                                        estimateTransition, estimateSubclone), method='L-BFGS-B',
                                                        bounds=Bounds(intervalNormal[0],intervalNormal[1]),
                                                        options={'eps':1e-8})
        sp_hat = estSubclone.x
    
    if estimatePloidy:
        estPhi = minimize(completeLikelihoodFun, phi_prev, args=(np.repeat("phi", len(phi_prev)), n_hat, sp_hat, phi_prev, 
                                                        lambda_prev, pi_hat, A_hat, params, D, rho, Zcounts, 
                                                        estimateNormal, estimatePloidy, estimatePrecision, estimateInitDist,
                                                        estimateTransition, estimateSubclone), method='L-BFGS-B',
                                                        bounds=Bounds(intervalPhi[0],intervalPhi[1]),
                                                        options={'eps':1e-8})
        phi_hat = estPhi.x

    if estimatePrecision:
        estLambda = minimize(completeLikelihoodFun, lambda_prev, args=(np.repeat("lambda", K*S), n_hat, sp_hat, phi_hat, 
                                                        lambda_prev, pi_hat, A_hat, params, D, rho, Zcounts, 
                                                        estimateNormal, estimatePloidy, estimatePrecision, estimateInitDist,
                                                        estimateTransition, estimateSubclone), method='L-BFGS-B',
                                                        bounds=Bounds(intervalLambda[0],np.inf),
                                                        options={'eps':1e-8})
        lambda_hat = estLambda.x

    return {"n":n_hat, "sp":sp_hat, "phi":phi_hat, "lambda":lambda_hat, "piG":pi_hat, "A":A_hat, "F":estLambda.fun*-1}



def runEM(copy, chroms, chrTrain, param, maxiter, verbose=False, 
            estimateNormal=True, estimatePloidy=True, estimatePrecision=True,
            estimateTransition=True, estimateInitDist=True, estimateSubclone=True,
            likChangeConvergence=1e-3):
    """


    Parameters
    ----------
        copy : pandas.DataFrame
    
    Returns
    -------


    """

    if copy.shape[0] != len(chroms) or copy.shape[0] != len(chrTrain):
        raise IndexError("runEM: Length of inputs do not match for one of: copy, chr, chrTrain")

    if param["ct"] is None or param["lambda"] is None or param["nu"] is None or param["kappa"] is None:
        raise IndexError("runEM: Parameter missing, ensure all parameters exist as columns in data frame: ct, lambda, nu, kappa")

    S = param["numberSamples"]
    K = len(param["ct"])
    Z = np.sum(param["ct_sc_status"]) #number of subclonal states (# repeated copy number states)
    KS = K**S
    N = copy.shape[0]
    rho = np.zeros((KS, N))
    py = np.zeros((KS, N), order="F")            # Local evidence
    mus = [np.zeros((KS, S)) for i in range(maxiter)]     # State means
    lambdas = [np.zeros((K, S)) for i in range(maxiter)]  # State Variances
    phi = np.tile(np.nan, (len(param["phi_0"]), maxiter))					 # Tumour ploidy
    n = np.tile(np.nan, (S, maxiter))						 # Normal contamination
    sp = np.tile(np.nan, (S, maxiter))     # cellular prevalence (tumor does not contain event)
    piG = np.zeros((KS, maxiter), order="F")     # Initial state distribution
    converged = False               # Flag for convergence
    Zcounts = np.zeros((KS,KS))
    loglik = np.zeros(maxiter)

    ptmTotal = time.perf_counter() # start total timer

    # SET UP
    # Set up the chromosome indices and make cell array of chromosome indicies
    indexes = np.unique(chroms, return_index=True)[1]
    chrs = np.array([chroms[index] for index in sorted(indexes)])
    chrsI = []

    # initialise the chromosome index and the init state distributions
    for i in range(len(chrs)):
        chrsI.append(np.where(chroms == chrs[i])[0])
    
    # INITIALIZATION
    if verbose: print("runEM: Initialization")
    i = 0
    piG[:, i] = hmm_utilities.normalize(param["kappa"])
    n[:, i] = param["n_0"]
    sp[:, i] = param["sp_0"]
    phi[:, i] = param["phi_0"]
    lambdas[i] = param["lambda"]
    lambdasKS = pd.DataFrame(lambdas[0])
    mus[i] = get2and3ComponentMixture(param["jointCNstates"].values, param["jointSCstatus"].values, n[:, i], sp[:, i], phi[:, i])

    # Likelihood #
    for ks in range(KS):
        probs = tdistPDF(copy.values, mus[i][ks,:], lambdasKS.iloc[ks,:], param["nu"])
        if probs.ndim > 1:
            py[ks,:] = probs.apply(prod, axis=0) # multiply across samples for each data point to get joint likelihood.
        else:
            py[ks,:] = probs

    # initialize transition prior #
    A = hmm_utilities.normalize(param["A"])
    A_prior = A
    dirPrior = param["dirPrior"]

    loglik[i] = -np.inf

    while converged == False and i < maxiter-1:
        ptm = time.perf_counter()

        i = i + 1

        ################ E-step ####################
        if verbose: print("runEM iter", i-1 , ": Expectation")

        Zcounts = np.zeros((KS, KS))
        for j in range(len(chrsI)):
            I = chrsI[j][np.in1d(chrsI[j], np.where(chrTrain)[0])]
            if len(I) > 0:
                #output = forward_backward.forward_backward(piG[:, i - 1], A, py[:, I])
                output = forward_backward(piG[:, i - 1], A, py[:, I])
                rho[:, I] = output["rho"]
                loglik[i] = loglik[i] + output["loglik"]
                Zcounts = Zcounts + output["xi"].sum(axis=2)
            
        ################ M-step ####################
        if verbose: print("runEM iter", i-1 , ": Maximization")

        output = estimateParamsMap(copy.loc[chrTrain, :], n[:, i - 1], sp[:, i - 1], phi[:, i - 1], 
                                lambdas[i - 1], piG[:, i - 1], A, param, rho[:, chrTrain], Zcounts, 
                                estimateNormal = estimateNormal, estimatePloidy = estimatePloidy,
                                estimatePrecision = estimatePrecision, estimateTransition = estimateTransition,
                                estimateInitDist = estimateInitDist, estimateSubclone = estimateSubclone)

        if verbose :
            for s in range(S):
                print("Sample", s, " n=", output["n"][s], ", sp=", output["sp"][s], 
                ", phi=", output["phi"][s], 
                ", lambda=", output["lambda"][s],
                ", F=", output["F"])
        
        n[:, i] = output["n"]
        sp[:, i] = output["sp"]
        phi[:, i] = output["phi"]
        lambdas[i] = output["lambda"]
        piG[:, i] = output["piG"]
        A = output["A"]
        estF = output["F"]

        # Recalculate the likelihood
        lambdasKS = pd.DataFrame(lambdas[i])
        mus[i] = get2and3ComponentMixture(param["jointCNstates"].values, param["jointSCstatus"].values, n[:, i], sp[:, i], phi[:, i])
        for ks in range(KS):
            probs = tdistPDF(copy.values, mus[i][ks,:], lambdasKS.iloc[ks,:], param["nu"])
            if probs.ndim > 1:
                py[ks,:] = probs.apply(prod, axis=0) # multiply across samples for each data point to get joint likelihood.
            else:
                py[ks,:] = probs

        prior = priorProbs(n[:, i], sp[:, i], phi[:, i], lambdas[i], piG[:, i], A, param, 
                        estimateNormal = estimateNormal, estimatePloidy = estimatePloidy,
                        estimatePrecision = estimatePrecision, estimateTransition = estimateTransition,
                        estimateInitDist = estimateInitDist, estimateSubclone = estimateSubclone)

        # check converence 
        loglik[i] = loglik[i] + prior["prior"]
        elapsedTime = time.perf_counter() - ptm
        if verbose:
            print("runEM iter", i-1, " Log likelihood:", loglik[i]) 
            print("runEM iter", i-1, " Time: ", elapsedTime / 60, " min.")

        if (np.abs(loglik[i] - loglik[i - 1]) / np.abs(loglik[i])) < likChangeConvergence:
            converged = True

        if loglik[i] < loglik[i - 1]:
            i = i - 1
            converged = True
        
    if converged:
        # Perform one last round of E-step to get latest responsibilities
        # E-step
        if verbose: print("runEM iter", i-1 ,": Re-calculating responsibilties from converged parameters.")
        for j in range(len(chrsI)):
            I = chrsI[j]
            output = forward_backward(piG[:, i], A, py[:, I])
            rho[:, I] = output["rho"]
    
    if verbose:
        totalTime = time.perf_counter() - ptmTotal
        print("runEM: Using optimal parameters from iter", i-1)
        print("runEM: Total elapsed time: ", totalTime / 60, "min.")
    
    #### Return parameters ####
    n = n[:, :i+1]
    sp = sp[:, :i+1]
    phi = phi[:, :i+1]
    mus = mus[:i+1]
    lambdas = lambdas[:i+1]
    piG = piG[:, :i+1]
    loglik = loglik[:i+1]

    output = {}
    output["n"] = n
    output["sp"] = sp
    output["phi"] = phi
    output["mus"] = mus
    output["lambdas"] = lambdas
    output["pi"] = piG
    output["A"] = A
    output["loglik"] = loglik
    output["rho"] = rho
    output["param"] = param
    output["py"] = py
    output["iter"] = i

    return output
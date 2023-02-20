import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics as stat
import argparse
import os
import colorsys
import scipy.optimize

import models

plt.rcParams.update({'font.size': 24})


model_lookup = {
    "athey": models.generate_AtheyModel,
    "confounder": models.generate_ConfounderModel,
    "instrument": models.generate_InstrumentModel,
    "confounderpoly": models.generate_ConfounderModelPoly,
    "instrumentpoly": models.generate_InstrumentModelPoly
}

model_params = {
    "athey": 8,
    "confounder": 5,
    "instrument": 6,
    "confounderpoly": 5,
    "instrumentpoly": 6,
}

def fmt(x):
    s = f"{x:.2f}"
    if s.endswith("0"):
        s = f"{x:.0f}"
    return rf"{s}" if plt.rcParams["text.usetex"] else f"{s}"

def get_color_old(red_to_green):
    assert 0 <= red_to_green <= 1
    # in HSV, red is 0 deg and green is 120 deg (out of 360);
    # divide red_to_green with 3 to map [0, 1] to [0, 1./3.]
    hue = red_to_green / 3.0
    hue = 1 - red_to_green / 1.5
    r, g, b = colorsys.hsv_to_rgb(hue, 1, 1)
    return map(lambda x: int(255 * x), (r, g, b))
    
def get_color(red_to_green):
    assert 0 <= red_to_green <= 1
    # in HSV, red is 0 deg and green is 120 deg (out of 360);
    # divide red_to_green with 3 to map [0, 1] to [0, 1./3.]
    r = 1 - 2*red_to_green
    g = 2*red_to_green - 1
    b = 1-2*np.abs(red_to_green - 0.5)
    print(r,g,b)
    if r<0:
        r=0
    if g<0:
        g=0
    return map(lambda x: int(255 * x), (r, g, b))
    
def get_color_bg(blue_to_green):
    assert 0 <= blue_to_green <= 1
    # in HSV, red is 0 deg and green is 120 deg (out of 360);
    # divide red_to_green with 3 to map [0, 1] to [0, 1./3.]
    r = 0
    g = blue_to_green
    b = 1-blue_to_green
    print(r,g,b)
    return map(lambda x: int(255 * x), (r, g, b))

def estimatorTrueFDC(sampledicts,coeffdict):

    numruns = len(sampledicts)
    
    Wsamples = [sampledicts[run]["W"] for run in range(numruns)]
    Xsamples = [sampledicts[run]["X"] for run in range(numruns)]
    Msamples = [sampledicts[run]["M"] for run in range(numruns)]
    Ysamples = [sampledicts[run]["Y"] for run in range(numruns)]
    
    # Recall true values
    a = coeffdict['a']
    c = coeffdict['c']

    cEstimators = [np.dot(Xsamples[run],Msamples[run])/np.dot(Xsamples[run],Xsamples[run]) for run in range(numruns)]
    
    aEstimators = [(np.dot(Msamples[run],Ysamples[run]) * np.dot(Xsamples[run],Xsamples[run])- np.dot(Msamples[run],Xsamples[run]) * np.dot(Xsamples[run],Ysamples[run])) / (np.dot(Msamples[run],Msamples[run]) * np.dot(Xsamples[run],Xsamples[run])- np.dot(Msamples[run],Xsamples[run]) * np.dot(Xsamples[run],Msamples[run])) for run in range(numruns)]
    
    cMean = stat.mean(cEstimators)
    cBias = cMean - c
    cVar = stat.variance(cEstimators)

    #print("Theoretical mean(c) = 0")
    #THcVar = 1./((numsamp-2)*(d**2 + 1))
    #print("Theoretical var(c) = ","{:.6f}".format(THcVar))

    aMean = stat.mean(aEstimators)
    aBias = aMean - a
    aVar = stat.variance(aEstimators)

    #print("Theoretical bias(a) = 0")
    #THaVar = (b**2 + (d**2 + 1))/((numsamp-3)*(d**2 + 1))
    #print("Theoretical var(a) = ","{:.4f}".format(THaVar))

    acEstimators = [aEstimators[i] * cEstimators[i] for i in range(numruns) ]
    acMean = stat.mean(acEstimators)
    acBias = acMean - a*c
    acVar = stat.variance(acEstimators)

    #THacBias = 
    #print("Theoretical bias(ac) = ","{:.4f}".format(THacBias)
    #THacVar = 
    #print("Theoretical var(ac) = ","{:.4f}".format(THacVar))
    
    return {"c":  [cBias, cVar], 
            "a":  [aBias, aVar],
            "ac": [acBias, acVar]
            }
            
def estimatorApproxFDC(sampledicts,coeffdict):            
            
    numruns = len(sampledicts)
    
    Wsamples = [sampledicts[run]["W"] for run in range(numruns)]
    Xsamples = [sampledicts[run]["X"] for run in range(numruns)]
    Msamples = [sampledicts[run]["M"] for run in range(numruns)]
    Ysamples = [sampledicts[run]["Y"] for run in range(numruns)]
    
    # Recall true values
    a = coeffdict['a']
    c = coeffdict['c']

    cEstimators = [np.dot(Xsamples[run],Msamples[run])/np.dot(Xsamples[run],Xsamples[run]) for run in range(numruns)]
    cMean = stat.mean(cEstimators)
    cBias = cMean - c
    cVar = stat.variance(cEstimators)

    #print("Theoretical mean(c) = 0")
    #THcVar = 1./((numsamp-2)*(d**2 + 1))
    #print("Theoretical var(c) = ","{:.6f}".format(THcVar))

    aApproxEstimators = [np.dot(Msamples[run],Ysamples[run]) / np.dot(Msamples[run],Msamples[run]) for run in range(numruns)]
    aApproxMean = stat.mean(aApproxEstimators)
    aBias = aApproxMean - a
    aVar = stat.variance(aApproxEstimators)

    #THaBias = b*c*d / (1 + c**2 * (d**2 + 1))
    #print("Theoretical bias(a) = ","{:.4f}".format(THaBias))

    acApproxEstimators = [aApproxEstimators[i] * cEstimators[i] for i in range(numruns) ]
    acApproxMean = stat.mean(acApproxEstimators)
    acBias = acApproxMean - a*c
    acVar = stat.variance(acApproxEstimators)
           
    return {"c":  [cBias, cVar], 
            "a":  [aBias, aVar],
            "ac": [acBias, acVar]
            }
            
            
def estimatorInstFDC(sampledicts,coeffdict):         

    numruns = len(sampledicts)   
            
    Wsamples = [sampledicts[run]["W"] for run in range(numruns)]
    Xsamples = [sampledicts[run]["X"] for run in range(numruns)]
    Msamples = [sampledicts[run]["M"] for run in range(numruns)]
    Ysamples = [sampledicts[run]["Y"] for run in range(numruns)]
    
    numsamp = len(Wsamples[0])
    
    # Recall true values
    print(coeffdict)
    a = coeffdict['a']
    c = coeffdict['c']

    # Calculate the regression slope from x -> m (c)
    cEstimators = [np.dot(Xsamples[run],Msamples[run])/np.dot(Xsamples[run],Xsamples[run]) for run in range(numruns)]
    
    # Calculate the residuals um for each datapoint {x,m}
    uMsamples = [[Msamples[run][samp] - cEstimators[run] * Xsamples[run][samp] for samp in range(numsamp)] for run in range(numruns)]
    
    ## Intermediate check: calculate all correlations of residuals um
    # Estimate the means of r (per run)
            
    rmeans = [np.mean(uMsamples[run]) for run in range(numruns)]
        
    # Estimate the variances of r (per run)
    
    rvars = [np.var(uMsamples[run]) for run in range(numruns)]
    
    # Estimate the covariances of X with r (per run)
    
    rcovsX = [np.cov(Xsamples[run],uMsamples[run],bias=True)[1][0] for run in range(numruns)]
    
    # Estimate the covariances of M with r (per run)
    
    rcovsM = [np.cov(Msamples[run],uMsamples[run],bias=True)[1][0] for run in range(numruns)]
    
    # Estimate the covariances of W with r (per run)
    
    rcovsW = [np.cov(Wsamples[run],uMsamples[run],bias=True)[1][0] for run in range(numruns)]
    
    meanrmean = stat.mean(rmeans)
    meanrvar = stat.mean(rvars)
    meanrcovX = stat.mean(rcovsX)
    meanrcovM = stat.mean(rcovsM)
    meanrcovW = stat.mean(rcovsW)
    
    # Use um as an instrumental variable to estimate m -> y (a):
    # A) regress y on um
    
    YuMslope = [np.dot(Ysamples[run],uMsamples[run])/np.dot(uMsamples[run],uMsamples[run]) for run in range(numruns)]
    
    # B) regress m on um
    
    MuMslope = [np.dot(Msamples[run],uMsamples[run])/np.dot(uMsamples[run],uMsamples[run]) for run in range(numruns)]
    
    # C) divide these coefficients
    
    aInstEstimators = [YuMslope[run]/MuMslope[run] for run in range(numruns)]
    #aInstEstimators = MuMslope     # to run this test, switch aBias to aInstMean - 1
    
    cMean = stat.mean(cEstimators)
    cBias = cMean - c
    cVar = stat.variance(cEstimators)

    #print("Theoretical mean(c) = 0")
    #THcVar = 1./((numsamp-2)*(d**2 + 1))
    #print("Theoretical var(c) = ","{:.6f}".format(THcVar))

    aInstMean = stat.mean(aInstEstimators)
    aBias = aInstMean - a
    aVar = stat.variance(aInstEstimators)

    #THaBias = b*c*d / (1 + c**2 * (d**2 + 1))
    #print("Theoretical bias(a) = ","{:.4f}".format(THaBias))

    acInstEstimators = [aInstEstimators[i] * cEstimators[i] for i in range(numruns) ]
    acInstMean = stat.mean(acInstEstimators)
    acBias = acInstMean - a*c
    acVar = stat.variance(acInstEstimators)
                   
    return {"c":  [round(cBias,4), round(cVar,4)], 
            "a":  [round(aBias,4), round(aVar,4)],
            "ac": [round(acBias,4), round(acVar,4)],
            "rmean": round(meanrmean,4),
            "rvar": round(meanrvar,4),
            "rcovX": round(meanrcovX,4),
            "rcovM": round(meanrcovM,4),
            "rcovW": round(meanrcovW,4),
            "adata": aInstEstimators,
            "acdata": acInstEstimators
            }
            
def estimatorEDRemainderFDC(sampledicts,coeffdict): 

    numruns = len(sampledicts)
    
    Wsamples = [sampledicts[run]["W"] for run in range(numruns)]
    Xsamples = [sampledicts[run]["X"] for run in range(numruns)]
    Msamples = [sampledicts[run]["M"] for run in range(numruns)]
    Ysamples = [sampledicts[run]["Y"] for run in range(numruns)]
    
    numsamp = len(Wsamples[0])
    
    # Recall true values
    a = coeffdict['a']
    c = coeffdict['c']
    e = coeffdict['e']
    d = coeffdict['d']

    # Use um as an instrumental variable to estimate x -> m (c):
    
    cEstimators = [np.dot(Xsamples[run],Msamples[run])/np.dot(Xsamples[run],Xsamples[run]) for run in range(numruns)]

    # Assume we already know some perfect e/d estimator
    edivd = e/d
    
    # Calculate the residuals eW + um for each datapoint {x,m}, and their means
    eWplusuMsamples = [[Msamples[run][samp] - c * Xsamples[run][samp] for samp in range(numsamp)] for run in range(numruns)]
    
    # Calculate the remainder for each datapoint {x,m} given perfect knowledge of 
    #uMsamples = [[Msamples[run][samp] - c * Xsamples[run][samp] for samp in range(numsamp)] for run in range(numruns)]
    uMsamples = [[Msamples[run][samp] - c * Xsamples[run][samp] - edivd * Xsamples[run][samp] for samp in range(numsamp)] for run in range(numruns)]
    
    ## Intermediate check: calculate all correlations of remainders um
    # Estimate the means of r (per run)
            
    rmeans = [np.mean(uMsamples[run]) for run in range(numruns)]
        
    # Estimate the variances of r (per run)
    
    rvars = [np.var(uMsamples[run]) for run in range(numruns)]
    
    # Estimate the covariances of X with r (per run)
    
    rcovsX = [np.cov(Xsamples[run],uMsamples[run],bias=True)[1][0] for run in range(numruns)]
    
    # Estimate the covariances of M with r (per run)
    
    rcovsM = [np.cov(Msamples[run],uMsamples[run],bias=True)[1][0] for run in range(numruns)]
    
    # Estimate the covariances of W with r (per run)
    
    rcovsW = [np.cov(Wsamples[run],uMsamples[run],bias=True)[1][0] for run in range(numruns)]
    
    meanrmean = stat.mean(rmeans)
    meanrvar = stat.mean(rvars)
    meanrcovX = stat.mean(rcovsX)
    meanrcovM = stat.mean(rcovsM)
    meanrcovW = stat.mean(rcovsW)
    
    # Use um as an instrumental variable to estimate m -> y (a):
    # A) regress y on um
    
    YuMslope = [np.dot(Ysamples[run],uMsamples[run])/np.dot(uMsamples[run],uMsamples[run]) for run in range(numruns)]
    
    # B) regress m on um
    
    MuMslope = [np.dot(Msamples[run],uMsamples[run])/np.dot(uMsamples[run],uMsamples[run]) for run in range(numruns)]
    
    print("Remainder instrumental coefficients")
    print("YR slope: ", np.mean(YuMslope))
    print("MR slope: ", np.mean(MuMslope))
    
    
    
    # C) divide these coefficients
    
    
    
    aInstEstimators = [YuMslope[run]/MuMslope[run] for run in range(numruns)]
    #aInstEstimators = MuMslope     # to run this test, switch aBias to aInstMean - 1
    
    cBias = 0
    cVar = 0

    #print("Theoretical mean(c) = 0")
    #THcVar = 1./((numsamp-2)*(d**2 + 1))
    #print("Theoretical var(c) = ","{:.6f}".format(THcVar))

    aInstMean = stat.mean(aInstEstimators)
    aBias = aInstMean - a
    aVar = stat.variance(aInstEstimators)

    #THaBias = b*c*d / (1 + c**2 * (d**2 + 1))
    #print("Theoretical bias(a) = ","{:.4f}".format(THaBias))

    acInstEstimators = [aInstEstimators[i] * c for i in range(numruns) ]
    acInstMean = stat.mean(acInstEstimators)
    acBias = acInstMean - a*c
    acVar = stat.variance(acInstEstimators)
                   
    return {"c":  [round(cBias,4), round(cVar,4)], 
            "a":  [round(aBias,4), round(aVar,4)],
            "ac": [round(acBias,4), round(acVar,4)],
            "rmean": round(meanrmean,4),
            "rvar": round(meanrvar,4),
            "rcovX": round(meanrcovX,4),
            "rcovM": round(meanrcovM,4),
            "rcovW": round(meanrcovW,4),
            "adata": aInstEstimators,
            "acdata": acInstEstimators
            }    
            
def estimatorEDRatioRemainderFDC(sampledicts,coeffdict): 

    numruns = len(sampledicts)
    
    Wsamples = [sampledicts[run]["W"] for run in range(numruns)]
    Xsamples = [sampledicts[run]["X"] for run in range(numruns)]
    Msamples = [sampledicts[run]["M"] for run in range(numruns)]
    Ysamples = [sampledicts[run]["Y"] for run in range(numruns)]
    
    numsamp = len(Wsamples[0])
    
    # Recall true values
    a = coeffdict['a']
    c = coeffdict['c']
    e = coeffdict['e']
    d = coeffdict['d']

    # Use um as an instrumental variable to estimate x -> m (c):
    
    cEstimators = [np.dot(Xsamples[run],Msamples[run])/np.dot(Xsamples[run],Xsamples[run]) for run in range(numruns)]

    # Assume we already know some perfect e/d estimator
    edivd = e/d
    
    # Calculate the residuals eW + um for each datapoint {x,m}, and their means
    eWplusuMsamples = [[Msamples[run][samp] - c * Xsamples[run][samp] for samp in range(numsamp)] for run in range(numruns)]
    
    eWmeans = [np.mean(eWplusuMsamples[run]) for run in range(numruns)]
    
    dWmeans = [np.mean(Xsamples[run]) for run in range(numruns)]

    print("meane: ", np.mean(eWmeans))
    print("meand: ", np.mean(dWmeans))
    # Construct the ratio e/d estimator
    edivdEst2 = [eWmeans[run] / dWmeans[run] for run in range(numruns)]
    print("e/d estimate: ", edivdEst2)
    
    # Calculate the remainder for each datapoint {x,m} given perfect knowledge of 
    #uMsamples = [[Msamples[run][samp] - c * Xsamples[run][samp] for samp in range(numsamp)] for run in range(numruns)]
    uMsamples = [[Msamples[run][samp] - c * Xsamples[run][samp] - edivdEst2[run] * Xsamples[run][samp] for samp in range(numsamp)] for run in range(numruns)]
    
    ## Intermediate check: calculate all correlations of remainders um
    # Estimate the means of r (per run)
            
    rmeans = [np.mean(uMsamples[run]) for run in range(numruns)]
        
    # Estimate the variances of r (per run)
    
    rvars = [np.var(uMsamples[run]) for run in range(numruns)]
    
    # Estimate the covariances of X with r (per run)
    
    rcovsX = [np.cov(Xsamples[run],uMsamples[run],bias=True)[1][0] for run in range(numruns)]
    
    # Estimate the covariances of M with r (per run)
    
    rcovsM = [np.cov(Msamples[run],uMsamples[run],bias=True)[1][0] for run in range(numruns)]
    
    # Estimate the covariances of W with r (per run)
    
    rcovsW = [np.cov(Wsamples[run],uMsamples[run],bias=True)[1][0] for run in range(numruns)]
    
    meanrmean = stat.mean(rmeans)
    meanrvar = stat.mean(rvars)
    meanrcovX = stat.mean(rcovsX)
    meanrcovM = stat.mean(rcovsM)
    meanrcovW = stat.mean(rcovsW)
    
    # Use um as an instrumental variable to estimate m -> y (a):
    # A) regress y on um
    
    YuMslope = [np.dot(Ysamples[run],uMsamples[run])/np.dot(uMsamples[run],uMsamples[run]) for run in range(numruns)]
    
    # B) regress m on um
    
    MuMslope = [np.dot(Msamples[run],uMsamples[run])/np.dot(uMsamples[run],uMsamples[run]) for run in range(numruns)]
    
    #print("Remainder instrumental coefficients")
    #print("YR slope: ", np.mean(YuMslope))
    #print("MR slope: ", np.mean(MuMslope))
    
    
    
    # C) divide these coefficients
    
    
    
    aInstEstimators = [YuMslope[run]/MuMslope[run] for run in range(numruns)]
    #aInstEstimators = MuMslope     # to run this test, switch aBias to aInstMean - 1
    
    cBias = 0
    cVar = 0

    #print("Theoretical mean(c) = 0")
    #THcVar = 1./((numsamp-2)*(d**2 + 1))
    #print("Theoretical var(c) = ","{:.6f}".format(THcVar))

    aInstMean = stat.mean(aInstEstimators)
    aBias = aInstMean - a
    aVar = stat.variance(aInstEstimators)

    #THaBias = b*c*d / (1 + c**2 * (d**2 + 1))
    #print("Theoretical bias(a) = ","{:.4f}".format(THaBias))

    acInstEstimators = [aInstEstimators[i] * c for i in range(numruns) ]
    acInstMean = stat.mean(acInstEstimators)
    acBias = acInstMean - a*c
    acVar = stat.variance(acInstEstimators)
                   
    return {"c":  [round(cBias,4), round(cVar,4)], 
            "a":  [round(aBias,4), round(aVar,4)],
            "ac": [round(acBias,4), round(acVar,4)],
            "rmean": round(meanrmean,4),
            "rvar": round(meanrvar,4),
            "rcovX": round(meanrcovX,4),
            "rcovM": round(meanrcovM,4),
            "rcovW": round(meanrcovW,4),
            "adata": aInstEstimators,
            "acdata": acInstEstimators
            }                        

def estimatorInstFDCPoly(sampledicts,coeffdict): 

    numruns = len(sampledicts)
    
    Wsamples = [sampledicts[run]["W"] for run in range(numruns)]
    Xsamples = [sampledicts[run]["X"] for run in range(numruns)]
    Msamples = [sampledicts[run]["M"] for run in range(numruns)]
    Ysamples = [sampledicts[run]["Y"] for run in range(numruns)]
    
    numsamp = len(Wsamples[0])
    
    # Recall true values
    a = coeffdict['a'][1]
    c = coeffdict['c'][1]
    e = coeffdict['e'][1] #linear
    d = coeffdict['d'][1] #linear

    # Use um as an instrumental variable to estimate x -> m (c):
    
    cEstimators = [np.dot(Xsamples[run],Msamples[run])/np.dot(Xsamples[run],Xsamples[run]) for run in range(numruns)]
    
    # Calculate the residuals r = um for each datapoint {x,m}
    uMsamples = [[Msamples[run][samp] - cEstimators[run] * Xsamples[run][samp] for samp in range(numsamp)] for run in range(numruns)]
    
    ## Intermediate check: calculate all correlations of residuals um
    # Estimate the means of r (per run)
            
    rmeans = [np.mean(uMsamples[run]) for run in range(numruns)]
        
    # Estimate the variances of r (per run)
    
    rvars = [np.var(uMsamples[run]) for run in range(numruns)]
    
    # Estimate the covariances of X with r (per run)
    
    rcovsX = [np.cov(Xsamples[run],uMsamples[run],bias=True)[1][0] for run in range(numruns)]
    
    # Estimate the covariances of M with r (per run)
    
    rcovsM = [np.cov(Msamples[run],uMsamples[run],bias=True)[1][0] for run in range(numruns)]
    
    # Estimate the covariances of W with r (per run)
    
    rcovsW = [np.cov(Wsamples[run],uMsamples[run],bias=True)[1][0] for run in range(numruns)]
    
    meanrmean = stat.mean(rmeans)
    meanrvar = stat.mean(rvars)
    meanrcovX = stat.mean(rcovsX)
    meanrcovM = stat.mean(rcovsM)
    meanrcovW = stat.mean(rcovsW)
    
    ## Use um as an instrumental variable to estimate m -> y (a):
    # A) regress y on um
    
    YuMslope = [np.dot(Ysamples[run],uMsamples[run])/np.dot(uMsamples[run],uMsamples[run]) for run in range(numruns)]
    
    # B) regress m on um
    
    MuMslope = [np.dot(Msamples[run],uMsamples[run])/np.dot(uMsamples[run],uMsamples[run]) for run in range(numruns)]
    
    # C) divide these coefficients
    
    aInstEstimators = [YuMslope[run]/MuMslope[run] for run in range(numruns)]
    #aInstEstimators = MuMslope     # to run this test, switch aBias to aInstMean - 1
    
    cMean = stat.mean(cEstimators)
    cBias = cMean - c
    cVar = stat.variance(cEstimators)

    #print("Theoretical mean(c) = 0")
    #THcVar = 1./((numsamp-2)*(d**2 + 1))
    #print("Theoretical var(c) = ","{:.6f}".format(THcVar))

    aInstMean = stat.mean(aInstEstimators)
    aBias = aInstMean - a
    aVar = stat.variance(aInstEstimators)

    #THaBias = b*c*d / (1 + c**2 * (d**2 + 1))
    #print("Theoretical bias(a) = ","{:.4f}".format(THaBias))

    acInstEstimators = [aInstEstimators[i] * cEstimators[i] for i in range(numruns) ]
    acInstMean = stat.mean(acInstEstimators)
    acBias = acInstMean - a*c
    acVar = stat.variance(acInstEstimators)
                   
    return {"c":  [round(cBias,4), round(cVar,4)], 
            "a":  [round(aBias,4), round(aVar,4)],
            "ac": [round(acBias,4), round(acVar,4)],
            "rmean": round(meanrmean,4),
            "rvar": round(meanrvar,4),
            "rcovX": round(meanrcovX,4),
            "rcovM": round(meanrcovM,4),
            "rcovW": round(meanrcovW,4)
            }            
            
            
def estimatorEDRemainderFDCPoly(sampledicts,coeffdict): 

    numruns = len(sampledicts)
    
    Wsamples = [sampledicts[run]["W"] for run in range(numruns)]
    Xsamples = [sampledicts[run]["X"] for run in range(numruns)]
    Msamples = [sampledicts[run]["M"] for run in range(numruns)]
    Ysamples = [sampledicts[run]["Y"] for run in range(numruns)]
    
    numsamp = len(Wsamples[0])
    
    # Recall true values
    a = coeffdict['a'][1]
    c = coeffdict['c'][1]
    elist = coeffdict['e']
    dlist = coeffdict['d']

    # Use um as an instrumental variable to estimate x -> m (c):
    
    cEstimators = [np.dot(Xsamples[run],Msamples[run])/np.dot(Xsamples[run],Xsamples[run]) for run in range(numruns)]

    # Assume we already know some perfect e/d estimator
    edivd = lambda x: divpolys(elist,dlist,x)
    
    # Calculate the remainder for each datapoint {x,m} given perfect knowledge of 
    #uMsamples = [[Msamples[run][samp] - c * Xsamples[run][samp] for samp in range(numsamp)] for run in range(numruns)]
    uMsamples = [[Msamples[run][samp] - c * Xsamples[run][samp] - edivd(Xsamples[run][samp]) for samp in range(numsamp)] for run in range(numruns)]
    
    ## Intermediate check: calculate all correlations of remainders um
    # Estimate the means of r (per run)
            
    rmeans = [np.mean(uMsamples[run]) for run in range(numruns)]
        
    # Estimate the variances of r (per run)
    
    rvars = [np.var(uMsamples[run]) for run in range(numruns)]
    
    # Estimate the covariances of X with r (per run)
    
    rcovsX = [np.cov(Xsamples[run],uMsamples[run],bias=True)[1][0] for run in range(numruns)]
    
    # Estimate the covariances of M with r (per run)
    
    rcovsM = [np.cov(Msamples[run],uMsamples[run],bias=True)[1][0] for run in range(numruns)]
    
    # Estimate the covariances of W with r (per run)
    
    rcovsW = [np.cov(Wsamples[run],uMsamples[run],bias=True)[1][0] for run in range(numruns)]
    
    meanrmean = stat.mean(rmeans)
    meanrvar = stat.mean(rvars)
    meanrcovX = stat.mean(rcovsX)
    meanrcovM = stat.mean(rcovsM)
    meanrcovW = stat.mean(rcovsW)
    
    # Use um as an instrumental variable to estimate m -> y (a):
    # A) regress y on um
    
    YuMslope = [np.dot(Ysamples[run],uMsamples[run])/np.dot(uMsamples[run],uMsamples[run]) for run in range(numruns)]
    
    # B) regress m on um
    
    MuMslope = [np.dot(Msamples[run],uMsamples[run])/np.dot(uMsamples[run],uMsamples[run]) for run in range(numruns)]
    
    print("Remainder instrumental coefficients")
    print("YR slope: ", np.mean(YuMslope))
    print("MR slope: ", np.mean(MuMslope))
    
    
    
    # C) divide these coefficients
    
    
    
    aInstEstimators = [YuMslope[run]/MuMslope[run] for run in range(numruns)]
    #aInstEstimators = MuMslope     # to run this test, switch aBias to aInstMean - 1
    
    cBias = 0
    cVar = 0

    #print("Theoretical mean(c) = 0")
    #THcVar = 1./((numsamp-2)*(d**2 + 1))
    #print("Theoretical var(c) = ","{:.6f}".format(THcVar))

    aInstMean = stat.mean(aInstEstimators)
    aBias = aInstMean - a
    aVar = stat.variance(aInstEstimators)

    #THaBias = b*c*d / (1 + c**2 * (d**2 + 1))
    #print("Theoretical bias(a) = ","{:.4f}".format(THaBias))

    acInstEstimators = [aInstEstimators[i] * c for i in range(numruns) ]
    acInstMean = stat.mean(acInstEstimators)
    acBias = acInstMean - a*c
    acVar = stat.variance(acInstEstimators)
                   
    return {"c":  [round(cBias,4), round(cVar,4)], 
            "a":  [round(aBias,4), round(aVar,4)],
            "ac": [round(acBias,4), round(acVar,4)],
            "rmean": round(meanrmean,4),
            "rvar": round(meanrvar,4),
            "rcovX": round(meanrcovX,4),
            "rcovM": round(meanrcovM,4),
            "rcovW": round(meanrcovW,4)
            }                       

def estimatorPriorInstFDC(sampledicts,coeffdict): 

    numruns = len(sampledicts)
    
    Vsamples = [sampledicts[run]["V"] for run in range(numruns)]
    Wsamples = [sampledicts[run]["W"] for run in range(numruns)]
    Xsamples = [sampledicts[run]["X"] for run in range(numruns)]
    Msamples = [sampledicts[run]["M"] for run in range(numruns)]
    Ysamples = [sampledicts[run]["Y"] for run in range(numruns)]
    
    numsamp = len(Wsamples[0])
    
    # Recall true values
    a = coeffdict['a']
    c = coeffdict['c']
    e = coeffdict['e']
    d = coeffdict['d']

    # Use um as an instrumental variable to estimate x -> m (c):
    # A) regress m on v
    
    MVslope = [np.dot(Msamples[run],Vsamples[run])/np.dot(Vsamples[run],Vsamples[run]) for run in range(numruns)]
    
    # B) regress x on v  (note that this is an unbiased estimator for g)
    
    XVslope = [np.dot(Xsamples[run],Vsamples[run])/np.dot(Vsamples[run],Vsamples[run]) for run in range(numruns)]
    
    # C) divide these coefficients
    
    cInstEstimators = [MVslope[run]/XVslope[run] for run in range(numruns)]
    
    # Calculate the residuals r = um for each datapoint {x,m}
    uMsamples = [[Msamples[run][samp] - cInstEstimators[run] * Xsamples[run][samp] for samp in range(numsamp)] for run in range(numruns)]
    
    ## Intermediate check: calculate all correlations of residuals um
    # Estimate the means of r (per run)
            
    rmeans = [np.mean(uMsamples[run]) for run in range(numruns)]
        
    # Estimate the variances of r (per run)
    
    rvars = [np.var(uMsamples[run]) for run in range(numruns)]
    
    # Estimate the covariances of X with r (per run)
    
    rcovsX = [np.cov(Xsamples[run],uMsamples[run],bias=True)[1][0] for run in range(numruns)]
    
    # Estimate the covariances of M with r (per run)
    
    rcovsM = [np.cov(Msamples[run],uMsamples[run],bias=True)[1][0] for run in range(numruns)]
    
    # Estimate the covariances of W with r (per run)
    
    rcovsW = [np.cov(Wsamples[run],uMsamples[run],bias=True)[1][0] for run in range(numruns)]
    
    meanrmean = stat.mean(rmeans)
    meanrvar = stat.mean(rvars)
    meanrcovX = stat.mean(rcovsX)
    meanrcovM = stat.mean(rcovsM)
    meanrcovW = stat.mean(rcovsW)
    
    ## Use um as an instrumental variable to estimate m -> y (a):
    # A) regress y on um
    
    YuMslope = [np.dot(Ysamples[run],uMsamples[run])/np.dot(uMsamples[run],uMsamples[run]) for run in range(numruns)]
    
    # B) regress m on um
    
    MuMslope = [np.dot(Msamples[run],uMsamples[run])/np.dot(uMsamples[run],uMsamples[run]) for run in range(numruns)]
    
    # C) divide these coefficients
    
    aInstEstimators = [YuMslope[run]/MuMslope[run] for run in range(numruns)]
    #aInstEstimators = MuMslope     # to run this test, switch aBias to aInstMean - 1
    
    cMean = stat.mean(cInstEstimators)
    cBias = cMean - c
    cVar = stat.variance(cInstEstimators)

    #print("Theoretical mean(c) = 0")
    #THcVar = 1./((numsamp-2)*(d**2 + 1))
    #print("Theoretical var(c) = ","{:.6f}".format(THcVar))

    aInstMean = stat.mean(aInstEstimators)
    aBias = aInstMean - a
    aVar = stat.variance(aInstEstimators)

    #THaBias = b*c*d / (1 + c**2 * (d**2 + 1))
    #print("Theoretical bias(a) = ","{:.4f}".format(THaBias))

    acInstEstimators = [aInstEstimators[i] * cInstEstimators[i] for i in range(numruns) ]
    acInstMean = stat.mean(acInstEstimators)
    acBias = acInstMean - a*c
    acVar = stat.variance(acInstEstimators)
                   
    return {"c":  [round(cBias,4), round(cVar,4)], 
            "a":  [round(aBias,4), round(aVar,4)],
            "ac": [round(acBias,4), round(acVar,4)],
            "rmean": round(meanrmean,4),
            "rvar": round(meanrvar,4),
            "rcovX": round(meanrcovX,4),
            "rcovM": round(meanrcovM,4),
            "rcovW": round(meanrcovW,4)
            }
           
def estimatorPriorEDResidualFDC(sampledicts,coeffdict): 

    numruns = len(sampledicts)
    
    Vsamples = [sampledicts[run]["V"] for run in range(numruns)]
    Wsamples = [sampledicts[run]["W"] for run in range(numruns)]
    Xsamples = [sampledicts[run]["X"] for run in range(numruns)]
    Msamples = [sampledicts[run]["M"] for run in range(numruns)]
    Ysamples = [sampledicts[run]["Y"] for run in range(numruns)]
    
    numsamp = len(Wsamples[0])
    
    # Recall true values
    a = coeffdict['a']
    c = coeffdict['c']
    e = coeffdict['e']
    d = coeffdict['d']

    # Use um as an instrumental variable to estimate x -> m (c):
    # A) regress m on v
    
    MVslope = [np.dot(Msamples[run],Vsamples[run])/np.dot(Vsamples[run],Vsamples[run]) for run in range(numruns)]
    
    # B) regress x on v  (note that this is an unbiased estimator for g)
    
    XVslope = [np.dot(Xsamples[run],Vsamples[run])/np.dot(Vsamples[run],Vsamples[run]) for run in range(numruns)]
    
    # C) divide these coefficients
    
    cInstEstimators = [MVslope[run]/XVslope[run] for run in range(numruns)]

            
    # Calculate the residuals eW + um for each datapoint {x,m}
    eWplusuMsamples = [[Msamples[run][samp] - cInstEstimators[run] * Xsamples[run][samp] for samp in range(numsamp)] for run in range(numruns)]
    
    # Calculate the residuals dW + ux for each datapoint {v,x}
    dWplusuXsamples = [[Xsamples[run][samp] - XVslope[run] * Vsamples[run][samp] for samp in range(numsamp)] for run in range(numruns)]

    # Assume we already know some perfect e/d estimator
    edivd = e/d
    
    # Calculate the residuals um for each datapoint {x,m}
    uMsamples = [[eWplusuMsamples[run][samp] - edivd * dWplusuXsamples[run][samp] for samp in range(numsamp)] for run in range(numruns)]
    
    ## Intermediate check: calculate all correlations of residuals um
    # Estimate the means of r (per run)
            
    rmeans = [np.mean(uMsamples[run]) for run in range(numruns)]
        
    # Estimate the variances of r (per run)
    
    rvars = [np.var(uMsamples[run]) for run in range(numruns)]
    
    # Estimate the covariances of X with r (per run)
    
    rcovsX = [np.cov(Xsamples[run],uMsamples[run],bias=True)[1][0] for run in range(numruns)]
    
    # Estimate the covariances of M with r (per run)
    
    rcovsM = [np.cov(Msamples[run],uMsamples[run],bias=True)[1][0] for run in range(numruns)]
    
    # Estimate the covariances of W with r (per run)
    
    rcovsW = [np.cov(Wsamples[run],uMsamples[run],bias=True)[1][0] for run in range(numruns)]
    
    meanrmean = stat.mean(rmeans)
    meanrvar = stat.mean(rvars)
    meanrcovX = stat.mean(rcovsX)
    meanrcovM = stat.mean(rcovsM)
    meanrcovW = stat.mean(rcovsW)
    
    # Use um as an instrumental variable to estimate m -> y (a):
    # A) regress y on um
    
    YuMslope = [np.dot(Ysamples[run],uMsamples[run])/np.dot(uMsamples[run],uMsamples[run]) for run in range(numruns)]
    
    # B) regress m on um
    
    MuMslope = [np.dot(Msamples[run],uMsamples[run])/np.dot(uMsamples[run],uMsamples[run]) for run in range(numruns)]
    
    # C) divide these coefficients
    
    print("Residual instrumental coefficients")
    print("YR slope: ", np.mean(YuMslope))
    print("MR slope: ", np.mean(MuMslope))
    
    
    
    aInstEstimators = [YuMslope[run]/MuMslope[run] for run in range(numruns)]
    #aInstEstimators = MuMslope     # to run this test, switch aBias to aInstMean - 1
    
    cMean = stat.mean(cInstEstimators)
    cBias = cMean - c
    cVar = stat.variance(cInstEstimators)

    #print("Theoretical mean(c) = 0")
    #THcVar = 1./((numsamp-2)*(d**2 + 1))
    #print("Theoretical var(c) = ","{:.6f}".format(THcVar))

    aInstMean = stat.mean(aInstEstimators)
    aBias = aInstMean - a
    aVar = stat.variance(aInstEstimators)

    #THaBias = b*c*d / (1 + c**2 * (d**2 + 1))
    #print("Theoretical bias(a) = ","{:.4f}".format(THaBias))

    acInstEstimators = [aInstEstimators[i] * cInstEstimators[i] for i in range(numruns) ]
    acInstMean = stat.mean(acInstEstimators)
    acBias = acInstMean - a*c
    acVar = stat.variance(acInstEstimators)
                   
    return {"c":  [round(cBias,4), round(cVar,4)], 
            "a":  [round(aBias,4), round(aVar,4)],
            "ac": [round(acBias,4), round(acVar,4)],
            "rmean": round(meanrmean,4),
            "rvar": round(meanrvar,4),
            "rcovX": round(meanrcovX,4),
            "rcovM": round(meanrcovM,4),
            "rcovW": round(meanrcovW,4)
            } 
            
def estimatorPriorEDRemainderFDC(sampledicts,coeffdict): 

    numruns = len(sampledicts)
    
    Vsamples = [sampledicts[run]["V"] for run in range(numruns)]
    Wsamples = [sampledicts[run]["W"] for run in range(numruns)]
    Xsamples = [sampledicts[run]["X"] for run in range(numruns)]
    Msamples = [sampledicts[run]["M"] for run in range(numruns)]
    Ysamples = [sampledicts[run]["Y"] for run in range(numruns)]
    
    numsamp = len(Wsamples[0])
    
    # Recall true values
    a = coeffdict['a']
    c = coeffdict['c']
    e = coeffdict['e']
    d = coeffdict['d']

    # Use um as an instrumental variable to estimate x -> m (c):
    # A) regress m on v
    
    MVslope = [np.dot(Msamples[run],Vsamples[run])/np.dot(Vsamples[run],Vsamples[run]) for run in range(numruns)]
    
    # B) regress x on v  (note that this is an unbiased estimator for g)
    
    XVslope = [np.dot(Xsamples[run],Vsamples[run])/np.dot(Vsamples[run],Vsamples[run]) for run in range(numruns)]
    
    # C) divide these coefficients
    
    cInstEstimators = [MVslope[run]/XVslope[run] for run in range(numruns)]

    # Assume we already know some perfect e/d estimator
    edivd = e/d
    
    # Calculate the remainder for each datapoint {x,m} given perfect knowledge of 
    #uMsamples = [[Msamples[run][samp] - c * Xsamples[run][samp] for samp in range(numsamp)] for run in range(numruns)]
    uMsamples = [[Msamples[run][samp] - (c + edivd) * Xsamples[run][samp] for samp in range(numsamp)] for run in range(numruns)]
    
    ## Intermediate check: calculate all correlations of remainders um
    # Estimate the means of r (per run)
            
    rmeans = [np.mean(uMsamples[run]) for run in range(numruns)]
        
    # Estimate the variances of r (per run)
    
    rvars = [np.var(uMsamples[run]) for run in range(numruns)]
    
    # Estimate the covariances of X with r (per run)
    
    rcovsX = [np.cov(Xsamples[run],uMsamples[run],bias=True)[1][0] for run in range(numruns)]
    
    # Estimate the covariances of M with r (per run)
    
    rcovsM = [np.cov(Msamples[run],uMsamples[run],bias=True)[1][0] for run in range(numruns)]
    
    # Estimate the covariances of W with r (per run)
    
    rcovsW = [np.cov(Wsamples[run],uMsamples[run],bias=True)[1][0] for run in range(numruns)]
    
    meanrmean = stat.mean(rmeans)
    meanrvar = stat.mean(rvars)
    meanrcovX = stat.mean(rcovsX)
    meanrcovM = stat.mean(rcovsM)
    meanrcovW = stat.mean(rcovsW)
    
    # Use um as an instrumental variable to estimate m -> y (a):
    # A) regress y on um
    
    YuMslope = [np.dot(Ysamples[run],uMsamples[run])/np.dot(uMsamples[run],uMsamples[run]) for run in range(numruns)]
    
    # B) regress m on um
    
    MuMslope = [np.dot(Msamples[run],uMsamples[run])/np.dot(uMsamples[run],uMsamples[run]) for run in range(numruns)]
    
    print("Remainder instrumental coefficients")
    print("YR slope: ", np.mean(YuMslope))
    print("MR slope: ", np.mean(MuMslope))
    
    
    
    # C) divide these coefficients
    
    
    
    aInstEstimators = [YuMslope[run]/MuMslope[run] for run in range(numruns)]
    #aInstEstimators = MuMslope     # to run this test, switch aBias to aInstMean - 1
    
    cBias = 0
    cVar = 0

    #print("Theoretical mean(c) = 0")
    #THcVar = 1./((numsamp-2)*(d**2 + 1))
    #print("Theoretical var(c) = ","{:.6f}".format(THcVar))

    aInstMean = stat.mean(aInstEstimators)
    aBias = aInstMean - a
    aVar = stat.variance(aInstEstimators)

    #THaBias = b*c*d / (1 + c**2 * (d**2 + 1))
    #print("Theoretical bias(a) = ","{:.4f}".format(THaBias))

    acInstEstimators = [aInstEstimators[i] * c for i in range(numruns) ]
    acInstMean = stat.mean(acInstEstimators)
    acBias = acInstMean - a*c
    acVar = stat.variance(acInstEstimators)
                   
    return {"c":  [round(cBias,4), round(cVar,4)], 
            "a":  [round(aBias,4), round(aVar,4)],
            "ac": [round(acBias,4), round(acVar,4)],
            "rmean": round(meanrmean,4),
            "rvar": round(meanrvar,4),
            "rcovX": round(meanrcovX,4),
            "rcovM": round(meanrcovM,4),
            "rcovW": round(meanrcovW,4)
            }                  
 
def estimatorPriorSubEDResidualFDC(sampledicts,coeffdict): 

    numruns = len(sampledicts)
    
    Vsamples = [sampledicts[run]["V"] for run in range(numruns)]
    Wsamples = [sampledicts[run]["W"] for run in range(numruns)]
    Xsamples = [sampledicts[run]["X"] for run in range(numruns)]
    Msamples = [sampledicts[run]["M"] for run in range(numruns)]
    Ysamples = [sampledicts[run]["Y"] for run in range(numruns)]
    
    numsamp = len(Wsamples[0])
    
    # Recall true values
    a = coeffdict['a']
    c = coeffdict['c']
    e = coeffdict['e']
    d = coeffdict['d']
    
    # Calculate the regression slope from x -> m (c + e/d)
    cEstimators = [np.dot(Xsamples[run],Msamples[run])/np.dot(Xsamples[run],Xsamples[run]) for run in range(numruns)]

    # Use um as an instrumental variable to estimate x -> m (c):
    # A) regress m on v
    
    MVslope = [np.dot(Msamples[run],Vsamples[run])/np.dot(Vsamples[run],Vsamples[run]) for run in range(numruns)]
    
    # B) regress x on v  (note that this is an unbiased estimator for g)
    
    XVslope = [np.dot(Xsamples[run],Vsamples[run])/np.dot(Vsamples[run],Vsamples[run]) for run in range(numruns)]
    
    # C) divide these coefficients
    
    
    cInstEstimators = [MVslope[run]/XVslope[run] for run in range(numruns)]

            
    # Calculate the residuals eW + um for each datapoint {x,m}
    eWplusuMsamples = [[Msamples[run][samp] - cInstEstimators[run] * Xsamples[run][samp] for samp in range(numsamp)] for run in range(numruns)]
    
    # Calculate the residuals dW + ux for each datapoint {v,x}
    dWplusuXsamples = [[Xsamples[run][samp] - XVslope[run] * Vsamples[run][samp] for samp in range(numsamp)] for run in range(numruns)]

    # Construct the subtracted e/d estimator
    edivdEst2 = [cEstimators[run] - cInstEstimators[run] for run in range(numruns)]

    # Calculate the residuals um for each datapoint {x,m}
    uMsamples = [[eWplusuMsamples[run][samp] - edivdEst2[run] * dWplusuXsamples[run][samp] for samp in range(numsamp)] for run in range(numruns)]
    
    ## Intermediate check: calculate all correlations of residuals um
    # Estimate the means of r (per run)
            
    rmeans = [np.mean(uMsamples[run]) for run in range(numruns)]
        
    # Estimate the variances of r (per run)
    
    rvars = [np.var(uMsamples[run]) for run in range(numruns)]
    
    # Estimate the covariances of X with r (per run)
    
    rcovsX = [np.cov(Xsamples[run],uMsamples[run],bias=True)[1][0] for run in range(numruns)]
    
    # Estimate the covariances of M with r (per run)
    
    rcovsM = [np.cov(Msamples[run],uMsamples[run],bias=True)[1][0] for run in range(numruns)]
    
    # Estimate the covariances of W with r (per run)
    
    rcovsW = [np.cov(Wsamples[run],uMsamples[run],bias=True)[1][0] for run in range(numruns)]
    
    meanrmean = stat.mean(rmeans)
    meanrvar = stat.mean(rvars)
    meanrcovX = stat.mean(rcovsX)
    meanrcovM = stat.mean(rcovsM)
    meanrcovW = stat.mean(rcovsW)
    
    # Use um as an instrumental variable to estimate m -> y (a):
    # A) regress y on um
    
    YuMslope = [np.dot(Ysamples[run],uMsamples[run])/np.dot(uMsamples[run],uMsamples[run]) for run in range(numruns)]
    
    # B) regress m on um
    
    MuMslope = [np.dot(Msamples[run],uMsamples[run])/np.dot(uMsamples[run],uMsamples[run]) for run in range(numruns)]
    
    # C) divide these coefficients
    
    
    
    aInstEstimators = [YuMslope[run]/MuMslope[run] for run in range(numruns)]
    #aInstEstimators = MuMslope     # to run this test, switch aBias to aInstMean - 1
    
    cMean = stat.mean(cInstEstimators)
    cBias = cMean - c
    cVar = stat.variance(cInstEstimators)

    #print("Theoretical mean(c) = 0")
    #THcVar = 1./((numsamp-2)*(d**2 + 1))
    #print("Theoretical var(c) = ","{:.6f}".format(THcVar))

    aInstMean = stat.mean(aInstEstimators)
    aBias = aInstMean - a
    aVar = stat.variance(aInstEstimators)

    #THaBias = b*c*d / (1 + c**2 * (d**2 + 1))
    #print("Theoretical bias(a) = ","{:.4f}".format(THaBias))

    acInstEstimators = [aInstEstimators[i] * cInstEstimators[i] for i in range(numruns) ]
    acInstMean = stat.mean(acInstEstimators)
    acBias = acInstMean - a*c
    acVar = stat.variance(acInstEstimators)
                   
    return {"c":  [round(cBias,4), round(cVar,4)], 
            "a":  [round(aBias,4), round(aVar,4)],
            "ac": [round(acBias,4), round(acVar,4)],
            "rmean": round(meanrmean,4),
            "rvar": round(meanrvar,4),
            "rcovX": round(meanrcovX,4),
            "rcovM": round(meanrcovM,4),
            "rcovW": round(meanrcovW,4)
            } 

def estimatorPriorSubEDRemainderFDC(sampledicts,coeffdict): 

    numruns = len(sampledicts)
    
    Vsamples = [sampledicts[run]["V"] for run in range(numruns)]
    Wsamples = [sampledicts[run]["W"] for run in range(numruns)]
    Xsamples = [sampledicts[run]["X"] for run in range(numruns)]
    Msamples = [sampledicts[run]["M"] for run in range(numruns)]
    Ysamples = [sampledicts[run]["Y"] for run in range(numruns)]
    
    numsamp = len(Wsamples[0])
    
    # Recall true values
    a = coeffdict['a']
    c = coeffdict['c']
    e = coeffdict['e']
    d = coeffdict['d']
    
    # Calculate the regression slope from x -> m (c + e/d)
    cEstimators = [np.dot(Xsamples[run],Msamples[run])/np.dot(Xsamples[run],Xsamples[run]) for run in range(numruns)]

    # Use um as an instrumental variable to estimate x -> m (c):
    # A) regress m on v
    
    MVslope = [np.dot(Msamples[run],Vsamples[run])/np.dot(Vsamples[run],Vsamples[run]) for run in range(numruns)]
    
    # B) regress x on v  (note that this is an unbiased estimator for g)
    
    XVslope = [np.dot(Xsamples[run],Vsamples[run])/np.dot(Vsamples[run],Vsamples[run]) for run in range(numruns)]
    
    # C) divide these coefficients
    
    cInstEstimators = [MVslope[run]/XVslope[run] for run in range(numruns)]

    
    # Construct the subtracted e/d estimator
    print("Mean cEstimator: ",np.mean(cEstimators))
    print("Mean cInstEstimator:",np.mean(cInstEstimators))
    edivdEst2 = [3*(cEstimators[run] - c) for run in range(numruns)]
    #print(edivdEst2)

    # Calculate the remainder for each datapoint {x,m} given perfect knowledge of 
    #uMsamples = [[Msamples[run][samp] - c * Xsamples[run][samp] for samp in range(numsamp)] for run in range(numruns)]
    uMsamples = [[Msamples[run][samp] - (c + edivdEst2[run]) * Xsamples[run][samp] for samp in range(numsamp)] for run in range(numruns)]
    
    ## Intermediate check: calculate all correlations of remainders um
    # Estimate the means of r (per run)
            
    rmeans = [np.mean(uMsamples[run]) for run in range(numruns)]
        
    # Estimate the variances of r (per run)
    
    rvars = [np.var(uMsamples[run]) for run in range(numruns)]
    
    # Estimate the covariances of X with r (per run)
    
    rcovsX = [np.cov(Xsamples[run],uMsamples[run],bias=True)[1][0] for run in range(numruns)]
    
    # Estimate the covariances of M with r (per run)
    
    rcovsM = [np.cov(Msamples[run],uMsamples[run],bias=True)[1][0] for run in range(numruns)]
    
    # Estimate the covariances of W with r (per run)
    
    rcovsW = [np.cov(Wsamples[run],uMsamples[run],bias=True)[1][0] for run in range(numruns)]
    
    meanrmean = stat.mean(rmeans)
    meanrvar = stat.mean(rvars)
    meanrcovX = stat.mean(rcovsX)
    meanrcovM = stat.mean(rcovsM)
    meanrcovW = stat.mean(rcovsW)
    
    # Use um as an instrumental variable to estimate m -> y (a):
    # A) regress y on um
    
    YuMslope = [np.dot(Ysamples[run],uMsamples[run])/np.dot(uMsamples[run],uMsamples[run]) for run in range(numruns)]
    
    # B) regress m on um
    
    MuMslope = [np.dot(Msamples[run],uMsamples[run])/np.dot(uMsamples[run],uMsamples[run]) for run in range(numruns)]
    
    # C) divide these coefficients
    
    
    
    aInstEstimators = [YuMslope[run]/MuMslope[run] for run in range(numruns)]
    #aInstEstimators = MuMslope     # to run this test, switch aBias to aInstMean - 1
    
    cBias = 0
    cVar = 0

    #print("Theoretical mean(c) = 0")
    #THcVar = 1./((numsamp-2)*(d**2 + 1))
    #print("Theoretical var(c) = ","{:.6f}".format(THcVar))

    aInstMean = stat.mean(aInstEstimators)
    aBias = aInstMean - a
    aVar = stat.variance(aInstEstimators)

    #THaBias = b*c*d / (1 + c**2 * (d**2 + 1))
    #print("Theoretical bias(a) = ","{:.4f}".format(THaBias))

    acInstEstimators = [aInstEstimators[i] * c for i in range(numruns) ]
    acInstMean = stat.mean(acInstEstimators)
    acBias = acInstMean - a*c
    acVar = stat.variance(acInstEstimators)
                   
    return {"c":  [round(cBias,4), round(cVar,4)], 
            "a":  [round(aBias,4), round(aVar,4)],
            "ac": [round(acBias,4), round(acVar,4)],
            "rmean": round(meanrmean,4),
            "rvar": round(meanrvar,4),
            "rcovX": round(meanrcovX,4),
            "rcovM": round(meanrcovM,4),
            "rcovW": round(meanrcovW,4)
            }                    

def estimatorPriorRatioEDResidualFDC(sampledicts,coeffdict): 

    numruns = len(sampledicts)
    
    Vsamples = [sampledicts[run]["V"] for run in range(numruns)]
    Wsamples = [sampledicts[run]["W"] for run in range(numruns)]
    Xsamples = [sampledicts[run]["X"] for run in range(numruns)]
    Msamples = [sampledicts[run]["M"] for run in range(numruns)]
    Ysamples = [sampledicts[run]["Y"] for run in range(numruns)]
    
    numsamp = len(Wsamples[0])
    
    # Recall true values
    a = coeffdict['a']
    c = coeffdict['c']
    e = coeffdict['e']
    d = coeffdict['d']
    
    # Calculate the regression slope from x -> m (c + e/d)
    cEstimators = [np.dot(Xsamples[run],Msamples[run])/np.dot(Xsamples[run],Xsamples[run]) for run in range(numruns)]

    # Use um as an instrumental variable to estimate x -> m (c):
    # A) regress m on v
    
    MVslope = [np.dot(Msamples[run],Vsamples[run])/np.dot(Vsamples[run],Vsamples[run]) for run in range(numruns)]
    
    # B) regress x on v  (note that this is an unbiased estimator for g)
    
    XVslope = [np.dot(Xsamples[run],Vsamples[run])/np.dot(Vsamples[run],Vsamples[run]) for run in range(numruns)]
    
    # C) divide these coefficients
    
    
    cInstEstimators = [MVslope[run]/XVslope[run] for run in range(numruns)]

            
    # Calculate the residuals eW + um for each datapoint {x,m}, and their means
    eWplusuMsamples = [[Msamples[run][samp] - cInstEstimators[run] * Xsamples[run][samp] for samp in range(numsamp)] for run in range(numruns)]
    
    eWmeans = [np.mean(eWplusuMsamples[run]) for run in range(numruns)]
    #print(eWmeans)
    
    # Calculate the residuals dW + ux for each datapoint {v,x}, and their means
    dWplusuXsamples = [[Xsamples[run][samp] - XVslope[run] * Vsamples[run][samp] for samp in range(numsamp)] for run in range(numruns)]
    
    dWmeans = [np.mean(dWplusuXsamples[run]) for run in range(numruns)]
    #print(dWmeans)

    # Construct the ratio e/d estimator
    edivdEst2 = [eWmeans[run] / dWmeans[run] for run in range(numruns)]

    # Calculate the residuals um for each datapoint {x,m}
    uMsamples = [[eWplusuMsamples[run][samp] - edivdEst2[run] * dWplusuXsamples[run][samp] for samp in range(numsamp)] for run in range(numruns)]
    
    ## Intermediate check: calculate all correlations of residuals um
    # Estimate the means of r (per run)
            
    rmeans = [np.mean(uMsamples[run]) for run in range(numruns)]
        
    # Estimate the variances of r (per run)
    
    rvars = [np.var(uMsamples[run]) for run in range(numruns)]
    
    # Estimate the covariances of X with r (per run)
    
    rcovsX = [np.cov(Xsamples[run],uMsamples[run],bias=True)[1][0] for run in range(numruns)]
    
    # Estimate the covariances of M with r (per run)
    
    rcovsM = [np.cov(Msamples[run],uMsamples[run],bias=True)[1][0] for run in range(numruns)]
    
    # Estimate the covariances of W with r (per run)
    
    rcovsW = [np.cov(Wsamples[run],uMsamples[run],bias=True)[1][0] for run in range(numruns)]
    
    meanrmean = stat.mean(rmeans)
    meanrvar = stat.mean(rvars)
    meanrcovX = stat.mean(rcovsX)
    meanrcovM = stat.mean(rcovsM)
    meanrcovW = stat.mean(rcovsW)
    
    # Use um as an instrumental variable to estimate m -> y (a):
    # A) regress y on um
    
    YuMslope = [np.dot(Ysamples[run],uMsamples[run])/np.dot(uMsamples[run],uMsamples[run]) for run in range(numruns)]
    
    # B) regress m on um
    
    MuMslope = [np.dot(Msamples[run],uMsamples[run])/np.dot(uMsamples[run],uMsamples[run]) for run in range(numruns)]
    
    # C) divide these coefficients
    
    
    
    aInstEstimators = [YuMslope[run]/MuMslope[run] for run in range(numruns)]
    #aInstEstimators = MuMslope     # to run this test, switch aBias to aInstMean - 1
    
    cMean = stat.mean(cInstEstimators)
    cBias = cMean - c
    cVar = stat.variance(cInstEstimators)

    #print("Theoretical mean(c) = 0")
    #THcVar = 1./((numsamp-2)*(d**2 + 1))
    #print("Theoretical var(c) = ","{:.6f}".format(THcVar))

    aInstMean = stat.mean(aInstEstimators)
    aBias = aInstMean - a
    aVar = stat.variance(aInstEstimators)

    #THaBias = b*c*d / (1 + c**2 * (d**2 + 1))
    #print("Theoretical bias(a) = ","{:.4f}".format(THaBias))

    acInstEstimators = [aInstEstimators[i] * cInstEstimators[i] for i in range(numruns) ]
    acInstMean = stat.mean(acInstEstimators)
    acBias = acInstMean - a*c
    acVar = stat.variance(acInstEstimators)
                   
    return {"c":  [round(cBias,4), round(cVar,4)], 
            "a":  [round(aBias,4), round(aVar,4)],
            "ac": [round(acBias,4), round(acVar,4)],
            "rmean": round(meanrmean,4),
            "rvar": round(meanrvar,4),
            "rcovX": round(meanrcovX,4),
            "rcovM": round(meanrcovM,4),
            "rcovW": round(meanrcovW,4)
            } 

def estimatorPriorRatioEDRemainderFDC(sampledicts,coeffdict): 

    numruns = len(sampledicts)
    
    Vsamples = [sampledicts[run]["V"] for run in range(numruns)]
    Wsamples = [sampledicts[run]["W"] for run in range(numruns)]
    Xsamples = [sampledicts[run]["X"] for run in range(numruns)]
    Msamples = [sampledicts[run]["M"] for run in range(numruns)]
    Ysamples = [sampledicts[run]["Y"] for run in range(numruns)]
    
    numsamp = len(Wsamples[0])
    
    # Recall true values
    a = coeffdict['a']
    c = coeffdict['c']
    e = coeffdict['e']
    d = coeffdict['d']
    
    # Calculate the regression slope from x -> m (c + e/d)
    cEstimators = [np.dot(Xsamples[run],Msamples[run])/np.dot(Xsamples[run],Xsamples[run]) for run in range(numruns)]

    # Use um as an instrumental variable to estimate x -> m (c):
    # A) regress m on v
    
    MVslope = [np.dot(Msamples[run],Vsamples[run])/np.dot(Vsamples[run],Vsamples[run]) for run in range(numruns)]
    
    # B) regress x on v  (note that this is an unbiased estimator for g)
    
    XVslope = [np.dot(Xsamples[run],Vsamples[run])/np.dot(Vsamples[run],Vsamples[run]) for run in range(numruns)]
    
    # C) divide these coefficients
    
    cInstEstimators = [MVslope[run]/XVslope[run] for run in range(numruns)]

    
    # Calculate the residuals eW + um for each datapoint {x,m}, and their means
    eWplusuMsamples = [[Msamples[run][samp] - cInstEstimators[run] * Xsamples[run][samp] for samp in range(numsamp)] for run in range(numruns)]
    
    eWmeans = [np.mean(eWplusuMsamples[run]) for run in range(numruns)]
    
    # Calculate the residuals dW + ux for each datapoint {v,x}, and their means
    dWplusuXsamples = [[Xsamples[run][samp] - XVslope[run] * Vsamples[run][samp] for samp in range(numsamp)] for run in range(numruns)]
    
    dWmeans = [np.mean(dWplusuXsamples[run]) for run in range(numruns)]

    print("meane: ", np.mean(eWmeans))
    print("meand: ", np.mean(dWmeans))
    # Construct the ratio e/d estimator
    edivdEst2 = [eWmeans[run] / dWmeans[run] for run in range(numruns)]
    print(edivdEst2)

    # Calculate the remainder for each datapoint {x,m} given perfect knowledge of 
    #uMsamples = [[Msamples[run][samp] - c * Xsamples[run][samp] for samp in range(numsamp)] for run in range(numruns)]
    uMsamples = [[Msamples[run][samp] - (c + edivdEst2[run]) * Xsamples[run][samp] for samp in range(numsamp)] for run in range(numruns)]
    
    ## Intermediate check: calculate all correlations of remainders um
    # Estimate the means of r (per run)
            
    rmeans = [np.mean(uMsamples[run]) for run in range(numruns)]
        
    # Estimate the variances of r (per run)
    
    rvars = [np.var(uMsamples[run]) for run in range(numruns)]
    
    # Estimate the covariances of X with r (per run)
    
    rcovsX = [np.cov(Xsamples[run],uMsamples[run],bias=True)[1][0] for run in range(numruns)]
    
    # Estimate the covariances of M with r (per run)
    
    rcovsM = [np.cov(Msamples[run],uMsamples[run],bias=True)[1][0] for run in range(numruns)]
    
    # Estimate the covariances of W with r (per run)
    
    rcovsW = [np.cov(Wsamples[run],uMsamples[run],bias=True)[1][0] for run in range(numruns)]
    
    meanrmean = stat.mean(rmeans)
    meanrvar = stat.mean(rvars)
    meanrcovX = stat.mean(rcovsX)
    meanrcovM = stat.mean(rcovsM)
    meanrcovW = stat.mean(rcovsW)
    
    # Use um as an instrumental variable to estimate m -> y (a):
    # A) regress y on um
    
    YuMslope = [np.dot(Ysamples[run],uMsamples[run])/np.dot(uMsamples[run],uMsamples[run]) for run in range(numruns)]
    print(np.mean(YuMslope))
    
    # B) regress m on um
    
    MuMslope = [np.dot(Msamples[run],uMsamples[run])/np.dot(uMsamples[run],uMsamples[run]) for run in range(numruns)]
    print(np.mean(MuMslope))
    
    # C) divide these coefficients
    
    
    
    aInstEstimators = [YuMslope[run]/MuMslope[run] for run in range(numruns)]
    #aInstEstimators = MuMslope     # to run this test, switch aBias to aInstMean - 1
    
    cBias = 0
    cVar = 0

    #print("Theoretical mean(c) = 0")
    #THcVar = 1./((numsamp-2)*(d**2 + 1))
    #print("Theoretical var(c) = ","{:.6f}".format(THcVar))

    aInstMean = stat.mean(aInstEstimators)
    aBias = aInstMean - a
    aVar = stat.variance(aInstEstimators)

    #THaBias = b*c*d / (1 + c**2 * (d**2 + 1))
    #print("Theoretical bias(a) = ","{:.4f}".format(THaBias))

    acInstEstimators = [aInstEstimators[i] * c for i in range(numruns) ]
    acInstMean = stat.mean(acInstEstimators)
    acBias = acInstMean - a*c
    acVar = stat.variance(acInstEstimators)
                   
    return {"c":  [round(cBias,4), round(cVar,4)], 
            "a":  [round(aBias,4), round(aVar,4)],
            "ac": [round(acBias,4), round(acVar,4)],
            "rmean": round(meanrmean,4),
            "rvar": round(meanrvar,4),
            "rcovX": round(meanrcovX,4),
            "rcovM": round(meanrcovM,4),
            "rcovW": round(meanrcovW,4)
            }        

def estimatorPriorInstFDCwithX(sampledicts,coeffdict): 

    numruns = len(sampledicts)
    
    Vsamples = [sampledicts[run]["V"] for run in range(numruns)]
    Wsamples = [sampledicts[run]["W"] for run in range(numruns)]
    Xsamples = [sampledicts[run]["X"] for run in range(numruns)]
    Msamples = [sampledicts[run]["M"] for run in range(numruns)]
    Ysamples = [sampledicts[run]["Y"] for run in range(numruns)]
    
    numsamp = len(Wsamples[0])
    
    # Recall true values
    a = coeffdict['a']
    c = coeffdict['c']
    e = coeffdict['e']
    d = coeffdict['d']

    # Use um as an instrumental variable to estimate x -> m (c):
    # A) regress m on v
    
    MVslope = [np.dot(Msamples[run],Vsamples[run])/np.dot(Vsamples[run],Vsamples[run]) for run in range(numruns)]
    
    # B) regress x on v  (note that this is an unbiased estimator for g)
    
    XVslope = [np.dot(Xsamples[run],Vsamples[run])/np.dot(Vsamples[run],Vsamples[run]) for run in range(numruns)]
    
    # C) divide these coefficients
    
    cInstEstimators = [MVslope[run]/XVslope[run] for run in range(numruns)]
    
    # Calculate the residuals r = um for each datapoint {x,m}
    uMsamples = [[Msamples[run][samp] - cInstEstimators[run] * Xsamples[run][samp] for samp in range(numsamp)] for run in range(numruns)]
    
    ## Intermediate check: calculate all correlations of residuals um
    # Estimate the means of r (per run)
            
    rmeans = [np.mean(uMsamples[run]) for run in range(numruns)]
        
    # Estimate the variances of r (per run)
    
    rvars = [np.var(uMsamples[run]) for run in range(numruns)]
    
    # Estimate the covariances of X with r (per run)
    
    rcovsX = [np.cov(Xsamples[run],uMsamples[run],bias=True)[1][0] for run in range(numruns)]
    
    # Estimate the covariances of M with r (per run)
    
    rcovsM = [np.cov(Msamples[run],uMsamples[run],bias=True)[1][0] for run in range(numruns)]
    
    # Estimate the covariances of W with r (per run)
    
    rcovsW = [np.cov(Wsamples[run],uMsamples[run],bias=True)[1][0] for run in range(numruns)]
    
    meanrmean = stat.mean(rmeans)
    meanrvar = stat.mean(rvars)
    meanrcovX = stat.mean(rcovsX)
    meanrcovM = stat.mean(rcovsM)
    meanrcovW = stat.mean(rcovsW)
    
    ## Use um as an instrumental variable to estimate m -> y (a):
    # A) regress y on um AND x
    
    YuMslope = [ (np.dot(uMsamples[run],Ysamples[run]) * np.dot(Xsamples[run],Xsamples[run]) - np.dot(uMsamples[run],Xsamples[run]) * np.dot(Xsamples[run],Ysamples[run]) ) / (np.dot(uMsamples[run],uMsamples[run]) * np.dot(Xsamples[run],Xsamples[run]) - np.dot(uMsamples[run],Xsamples[run])**2) for run in range(numruns)]
    
    # B) regress m on um
    
    #MuMslope = [np.dot(Msamples[run],uMsamples[run])/np.dot(uMsamples[run],uMsamples[run]) for run in range(numruns)]
    MuMslope = [ (np.dot(uMsamples[run],Msamples[run]) * np.dot(Xsamples[run],Xsamples[run]) - np.dot(uMsamples[run],Xsamples[run]) * np.dot(Xsamples[run],Msamples[run]) ) / (np.dot(uMsamples[run],uMsamples[run]) * np.dot(Xsamples[run],Xsamples[run]) - np.dot(uMsamples[run],Xsamples[run])**2) for run in range(numruns)]
    
    # C) divide these coefficients
    
    aInstEstimators = [YuMslope[run]/MuMslope[run] for run in range(numruns)]
    #aInstEstimators = MuMslope     # to run this test, switch aBias to aInstMean - 1
    
    cMean = stat.mean(cInstEstimators)
    cBias = cMean - c
    cVar = stat.variance(cInstEstimators)

    #print("Theoretical mean(c) = 0")
    #THcVar = 1./((numsamp-2)*(d**2 + 1))
    #print("Theoretical var(c) = ","{:.6f}".format(THcVar))

    aInstMean = stat.mean(aInstEstimators)
    aBias = aInstMean - a
    aVar = stat.variance(aInstEstimators)

    #THaBias = b*c*d / (1 + c**2 * (d**2 + 1))
    #print("Theoretical bias(a) = ","{:.4f}".format(THaBias))

    acInstEstimators = [aInstEstimators[i] * cInstEstimators[i] for i in range(numruns) ]
    acInstMean = stat.mean(acInstEstimators)
    acBias = acInstMean - a*c
    acVar = stat.variance(acInstEstimators)
                   
    return {"c":  [round(cBias,4), round(cVar,4)], 
            "a":  [round(aBias,4), round(aVar,4)],
            "ac": [round(acBias,4), round(acVar,4)],
            "rmean": round(meanrmean,4),
            "rvar": round(meanrvar,4),
            "rcovX": round(meanrcovX,4),
            "rcovM": round(meanrcovM,4),
            "rcovW": round(meanrcovW,4)
            }
           
def estimatorPriorEDResidualFDCwithX(sampledicts,coeffdict): 

    numruns = len(sampledicts)
    
    Vsamples = [sampledicts[run]["V"] for run in range(numruns)]
    Wsamples = [sampledicts[run]["W"] for run in range(numruns)]
    Xsamples = [sampledicts[run]["X"] for run in range(numruns)]
    Msamples = [sampledicts[run]["M"] for run in range(numruns)]
    Ysamples = [sampledicts[run]["Y"] for run in range(numruns)]
    
    numsamp = len(Wsamples[0])
    
    # Recall true values
    a = coeffdict['a']
    c = coeffdict['c']
    e = coeffdict['e']
    d = coeffdict['d']

    # Use um as an instrumental variable to estimate x -> m (c):
    # A) regress m on v
    
    MVslope = [np.dot(Msamples[run],Vsamples[run])/np.dot(Vsamples[run],Vsamples[run]) for run in range(numruns)]
    
    # B) regress x on v  (note that this is an unbiased estimator for g)
    
    XVslope = [np.dot(Xsamples[run],Vsamples[run])/np.dot(Vsamples[run],Vsamples[run]) for run in range(numruns)]
    
    # C) divide these coefficients
    
    cInstEstimators = [MVslope[run]/XVslope[run] for run in range(numruns)]

            
    # Calculate the residuals eW + um for each datapoint {x,m}
    eWplusuMsamples = [[Msamples[run][samp] - cInstEstimators[run] * Xsamples[run][samp] for samp in range(numsamp)] for run in range(numruns)]
    
    # Calculate the residuals dW + ux for each datapoint {v,x}
    dWplusuXsamples = [[Xsamples[run][samp] - XVslope[run] * Vsamples[run][samp] for samp in range(numsamp)] for run in range(numruns)]

    # Assume we already know some perfect e/d estimator
    edivd = e/d
    
    # Calculate the residuals um for each datapoint {x,m}
    uMsamples = [[eWplusuMsamples[run][samp] - edivd * dWplusuXsamples[run][samp] for samp in range(numsamp)] for run in range(numruns)]
    
    ## Intermediate check: calculate all correlations of residuals um
    # Estimate the means of r (per run)
            
    rmeans = [np.mean(uMsamples[run]) for run in range(numruns)]
        
    # Estimate the variances of r (per run)
    
    rvars = [np.var(uMsamples[run]) for run in range(numruns)]
    
    # Estimate the covariances of X with r (per run)
    
    rcovsX = [np.cov(Xsamples[run],uMsamples[run],bias=True)[1][0] for run in range(numruns)]
    
    # Estimate the covariances of M with r (per run)
    
    rcovsM = [np.cov(Msamples[run],uMsamples[run],bias=True)[1][0] for run in range(numruns)]
    
    # Estimate the covariances of W with r (per run)
    
    rcovsW = [np.cov(Wsamples[run],uMsamples[run],bias=True)[1][0] for run in range(numruns)]
    
    meanrmean = stat.mean(rmeans)
    meanrvar = stat.mean(rvars)
    meanrcovX = stat.mean(rcovsX)
    meanrcovM = stat.mean(rcovsM)
    meanrcovW = stat.mean(rcovsW)
    
    # Use um as an instrumental variable to estimate m -> y (a):
    # A) regress y on um AND x
    
    YuMslope = [ (np.dot(uMsamples[run],Ysamples[run]) * np.dot(Xsamples[run],Xsamples[run]) - np.dot(uMsamples[run],Xsamples[run]) * np.dot(Xsamples[run],Ysamples[run]) ) / (np.dot(uMsamples[run],uMsamples[run]) * np.dot(Xsamples[run],Xsamples[run]) - np.dot(uMsamples[run],Xsamples[run])**2) for run in range(numruns)]
    
    # B) regress m on um
    
    #MuMslope = [np.dot(Msamples[run],uMsamples[run])/np.dot(uMsamples[run],uMsamples[run]) for run in range(numruns)]
    MuMslope = [ (np.dot(uMsamples[run],Msamples[run]) * np.dot(Xsamples[run],Xsamples[run]) - np.dot(uMsamples[run],Xsamples[run]) * np.dot(Xsamples[run],Msamples[run]) ) / (np.dot(uMsamples[run],uMsamples[run]) * np.dot(Xsamples[run],Xsamples[run]) - np.dot(uMsamples[run],Xsamples[run])**2) for run in range(numruns)]
    
    # C) divide these coefficients
    
    print("Residual instrumental coefficients, joint with X")
    print("YR slope: ", np.mean(YuMslope))
    print("MR slope: ", np.mean(MuMslope))
    
    
    
    aInstEstimators = [YuMslope[run]/MuMslope[run] for run in range(numruns)]
    #aInstEstimators = MuMslope     # to run this test, switch aBias to aInstMean - 1
    
    cMean = stat.mean(cInstEstimators)
    cBias = cMean - c
    cVar = stat.variance(cInstEstimators)

    #print("Theoretical mean(c) = 0")
    #THcVar = 1./((numsamp-2)*(d**2 + 1))
    #print("Theoretical var(c) = ","{:.6f}".format(THcVar))

    aInstMean = stat.mean(aInstEstimators)
    aBias = aInstMean - a
    aVar = stat.variance(aInstEstimators)

    #THaBias = b*c*d / (1 + c**2 * (d**2 + 1))
    #print("Theoretical bias(a) = ","{:.4f}".format(THaBias))

    acInstEstimators = [aInstEstimators[i] * cInstEstimators[i] for i in range(numruns) ]
    acInstMean = stat.mean(acInstEstimators)
    acBias = acInstMean - a*c
    acVar = stat.variance(acInstEstimators)
                   
    return {"c":  [round(cBias,4), round(cVar,4)], 
            "a":  [round(aBias,4), round(aVar,4)],
            "ac": [round(acBias,4), round(acVar,4)],
            "rmean": round(meanrmean,4),
            "rvar": round(meanrvar,4),
            "rcovX": round(meanrcovX,4),
            "rcovM": round(meanrcovM,4),
            "rcovW": round(meanrcovW,4)
            } 
            
def estimatorPriorEDRemainderFDCwithX(sampledicts,coeffdict): 

    numruns = len(sampledicts)
    
    Vsamples = [sampledicts[run]["V"] for run in range(numruns)]
    Wsamples = [sampledicts[run]["W"] for run in range(numruns)]
    Xsamples = [sampledicts[run]["X"] for run in range(numruns)]
    Msamples = [sampledicts[run]["M"] for run in range(numruns)]
    Ysamples = [sampledicts[run]["Y"] for run in range(numruns)]
    
    numsamp = len(Wsamples[0])
    
    # Recall true values
    a = coeffdict['a']
    c = coeffdict['c']
    e = coeffdict['e']
    d = coeffdict['d']

    # Use um as an instrumental variable to estimate x -> m (c):
    # A) regress m on v
    
    MVslope = [np.dot(Msamples[run],Vsamples[run])/np.dot(Vsamples[run],Vsamples[run]) for run in range(numruns)]
    
    # B) regress x on v  (note that this is an unbiased estimator for g)
    
    XVslope = [np.dot(Xsamples[run],Vsamples[run])/np.dot(Vsamples[run],Vsamples[run]) for run in range(numruns)]
    
    # C) divide these coefficients
    
    cInstEstimators = [MVslope[run]/XVslope[run] for run in range(numruns)]

    # Assume we already know some perfect e/d estimator
    edivd = e/d
    
    # Calculate the remainder for each datapoint {x,m} given perfect knowledge of 
    #uMsamples = [[Msamples[run][samp] - c * Xsamples[run][samp] for samp in range(numsamp)] for run in range(numruns)]
    uMsamples = [[Msamples[run][samp] - (c + edivd) * Xsamples[run][samp] for samp in range(numsamp)] for run in range(numruns)]
    
    ## Intermediate check: calculate all correlations of remainders um
    # Estimate the means of r (per run)
            
    rmeans = [np.mean(uMsamples[run]) for run in range(numruns)]
        
    # Estimate the variances of r (per run)
    
    rvars = [np.var(uMsamples[run]) for run in range(numruns)]
    
    # Estimate the covariances of X with r (per run)
    
    rcovsX = [np.cov(Xsamples[run],uMsamples[run],bias=True)[1][0] for run in range(numruns)]
    
    # Estimate the covariances of M with r (per run)
    
    rcovsM = [np.cov(Msamples[run],uMsamples[run],bias=True)[1][0] for run in range(numruns)]
    
    # Estimate the covariances of W with r (per run)
    
    rcovsW = [np.cov(Wsamples[run],uMsamples[run],bias=True)[1][0] for run in range(numruns)]
    
    meanrmean = stat.mean(rmeans)
    meanrvar = stat.mean(rvars)
    meanrcovX = stat.mean(rcovsX)
    meanrcovM = stat.mean(rcovsM)
    meanrcovW = stat.mean(rcovsW)
    
    # Use um as an instrumental variable to estimate m -> y (a):
    # A) regress y on um AND x
    
    YuMslope = [ (np.dot(uMsamples[run],Ysamples[run]) * np.dot(Xsamples[run],Xsamples[run]) - np.dot(uMsamples[run],Xsamples[run]) * np.dot(Xsamples[run],Ysamples[run]) ) / (np.dot(uMsamples[run],uMsamples[run]) * np.dot(Xsamples[run],Xsamples[run]) - np.dot(uMsamples[run],Xsamples[run])**2) for run in range(numruns)]
    
    # B) regress m on um
    
    #MuMslope = [np.dot(Msamples[run],uMsamples[run])/np.dot(uMsamples[run],uMsamples[run]) for run in range(numruns)]
    MuMslope = [ (np.dot(uMsamples[run],Msamples[run]) * np.dot(Xsamples[run],Xsamples[run]) - np.dot(uMsamples[run],Xsamples[run]) * np.dot(Xsamples[run],Msamples[run]) ) / (np.dot(uMsamples[run],uMsamples[run]) * np.dot(Xsamples[run],Xsamples[run]) - np.dot(uMsamples[run],Xsamples[run])**2) for run in range(numruns)]
    
    print("Remainder instrumental coefficients, joint with X")
    print("YR slope: ", np.mean(YuMslope))
    print("MR slope: ", np.mean(MuMslope))
    
    
    
    # C) divide these coefficients
    
    
    
    aInstEstimators = [YuMslope[run]/MuMslope[run] for run in range(numruns)]
    #aInstEstimators = MuMslope     # to run this test, switch aBias to aInstMean - 1
    
    cBias = 0
    cVar = 0

    #print("Theoretical mean(c) = 0")
    #THcVar = 1./((numsamp-2)*(d**2 + 1))
    #print("Theoretical var(c) = ","{:.6f}".format(THcVar))

    aInstMean = stat.mean(aInstEstimators)
    aBias = aInstMean - a
    aVar = stat.variance(aInstEstimators)

    #THaBias = b*c*d / (1 + c**2 * (d**2 + 1))
    #print("Theoretical bias(a) = ","{:.4f}".format(THaBias))

    acInstEstimators = [aInstEstimators[i] * c for i in range(numruns) ]
    acInstMean = stat.mean(acInstEstimators)
    acBias = acInstMean - a*c
    acVar = stat.variance(acInstEstimators)
                   
    return {"c":  [round(cBias,4), round(cVar,4)], 
            "a":  [round(aBias,4), round(aVar,4)],
            "ac": [round(acBias,4), round(acVar,4)],
            "rmean": round(meanrmean,4),
            "rvar": round(meanrvar,4),
            "rcovX": round(meanrcovX,4),
            "rcovM": round(meanrcovM,4),
            "rcovW": round(meanrcovW,4)
            }                  
 
def estimatorPriorSubEDResidualFDCwithX(sampledicts,coeffdict): 

    numruns = len(sampledicts)
    
    Vsamples = [sampledicts[run]["V"] for run in range(numruns)]
    Wsamples = [sampledicts[run]["W"] for run in range(numruns)]
    Xsamples = [sampledicts[run]["X"] for run in range(numruns)]
    Msamples = [sampledicts[run]["M"] for run in range(numruns)]
    Ysamples = [sampledicts[run]["Y"] for run in range(numruns)]
    
    numsamp = len(Wsamples[0])
    
    # Recall true values
    a = coeffdict['a']
    c = coeffdict['c']
    e = coeffdict['e']
    d = coeffdict['d']
    
    # Calculate the regression slope from x -> m (c + e/d)
    cEstimators = [np.dot(Xsamples[run],Msamples[run])/np.dot(Xsamples[run],Xsamples[run]) for run in range(numruns)]

    # Use um as an instrumental variable to estimate x -> m (c):
    # A) regress m on v
    
    MVslope = [np.dot(Msamples[run],Vsamples[run])/np.dot(Vsamples[run],Vsamples[run]) for run in range(numruns)]
    
    # B) regress x on v  (note that this is an unbiased estimator for g)
    
    XVslope = [np.dot(Xsamples[run],Vsamples[run])/np.dot(Vsamples[run],Vsamples[run]) for run in range(numruns)]
    
    # C) divide these coefficients
    
    
    cInstEstimators = [MVslope[run]/XVslope[run] for run in range(numruns)]

            
    # Calculate the residuals eW + um for each datapoint {x,m}
    eWplusuMsamples = [[Msamples[run][samp] - cInstEstimators[run] * Xsamples[run][samp] for samp in range(numsamp)] for run in range(numruns)]
    
    # Calculate the residuals dW + ux for each datapoint {v,x}
    dWplusuXsamples = [[Xsamples[run][samp] - XVslope[run] * Vsamples[run][samp] for samp in range(numsamp)] for run in range(numruns)]

    # Construct the subtracted e/d estimator
    edivdEst2 = [cEstimators[run] - cInstEstimators[run] for run in range(numruns)]

    # Calculate the residuals um for each datapoint {x,m}
    uMsamples = [[eWplusuMsamples[run][samp] - edivdEst2[run] * dWplusuXsamples[run][samp] for samp in range(numsamp)] for run in range(numruns)]
    
    ## Intermediate check: calculate all correlations of residuals um
    # Estimate the means of r (per run)
            
    rmeans = [np.mean(uMsamples[run]) for run in range(numruns)]
        
    # Estimate the variances of r (per run)
    
    rvars = [np.var(uMsamples[run]) for run in range(numruns)]
    
    # Estimate the covariances of X with r (per run)
    
    rcovsX = [np.cov(Xsamples[run],uMsamples[run],bias=True)[1][0] for run in range(numruns)]
    
    # Estimate the covariances of M with r (per run)
    
    rcovsM = [np.cov(Msamples[run],uMsamples[run],bias=True)[1][0] for run in range(numruns)]
    
    # Estimate the covariances of W with r (per run)
    
    rcovsW = [np.cov(Wsamples[run],uMsamples[run],bias=True)[1][0] for run in range(numruns)]
    
    meanrmean = stat.mean(rmeans)
    meanrvar = stat.mean(rvars)
    meanrcovX = stat.mean(rcovsX)
    meanrcovM = stat.mean(rcovsM)
    meanrcovW = stat.mean(rcovsW)
    
    # Use um as an instrumental variable to estimate m -> y (a):
    # A) regress y on um AND x
    
    YuMslope = [ (np.dot(uMsamples[run],Ysamples[run]) * np.dot(Xsamples[run],Xsamples[run]) - np.dot(uMsamples[run],Xsamples[run]) * np.dot(Xsamples[run],Ysamples[run]) ) / (np.dot(uMsamples[run],uMsamples[run]) * np.dot(Xsamples[run],Xsamples[run]) - np.dot(uMsamples[run],Xsamples[run])**2) for run in range(numruns)]
    
    # B) regress m on um
    
    #MuMslope = [np.dot(Msamples[run],uMsamples[run])/np.dot(uMsamples[run],uMsamples[run]) for run in range(numruns)]
    MuMslope = [ (np.dot(uMsamples[run],Msamples[run]) * np.dot(Xsamples[run],Xsamples[run]) - np.dot(uMsamples[run],Xsamples[run]) * np.dot(Xsamples[run],Msamples[run]) ) / (np.dot(uMsamples[run],uMsamples[run]) * np.dot(Xsamples[run],Xsamples[run]) - np.dot(uMsamples[run],Xsamples[run])**2) for run in range(numruns)]
    
    # C) divide these coefficients
    
    
    
    aInstEstimators = [YuMslope[run]/MuMslope[run] for run in range(numruns)]
    #aInstEstimators = MuMslope     # to run this test, switch aBias to aInstMean - 1
    
    cMean = stat.mean(cInstEstimators)
    cBias = cMean - c
    cVar = stat.variance(cInstEstimators)

    #print("Theoretical mean(c) = 0")
    #THcVar = 1./((numsamp-2)*(d**2 + 1))
    #print("Theoretical var(c) = ","{:.6f}".format(THcVar))

    aInstMean = stat.mean(aInstEstimators)
    aBias = aInstMean - a
    aVar = stat.variance(aInstEstimators)

    #THaBias = b*c*d / (1 + c**2 * (d**2 + 1))
    #print("Theoretical bias(a) = ","{:.4f}".format(THaBias))

    acInstEstimators = [aInstEstimators[i] * cInstEstimators[i] for i in range(numruns) ]
    acInstMean = stat.mean(acInstEstimators)
    acBias = acInstMean - a*c
    acVar = stat.variance(acInstEstimators)
                   
    return {"c":  [round(cBias,4), round(cVar,4)], 
            "a":  [round(aBias,4), round(aVar,4)],
            "ac": [round(acBias,4), round(acVar,4)],
            "rmean": round(meanrmean,4),
            "rvar": round(meanrvar,4),
            "rcovX": round(meanrcovX,4),
            "rcovM": round(meanrcovM,4),
            "rcovW": round(meanrcovW,4)
            } 

def estimatorPriorSubEDRemainderFDCwithX(sampledicts,coeffdict): 

    numruns = len(sampledicts)
    
    Vsamples = [sampledicts[run]["V"] for run in range(numruns)]
    Wsamples = [sampledicts[run]["W"] for run in range(numruns)]
    Xsamples = [sampledicts[run]["X"] for run in range(numruns)]
    Msamples = [sampledicts[run]["M"] for run in range(numruns)]
    Ysamples = [sampledicts[run]["Y"] for run in range(numruns)]
    
    numsamp = len(Wsamples[0])
    
    # Recall true values
    a = coeffdict['a']
    c = coeffdict['c']
    e = coeffdict['e']
    d = coeffdict['d']
    
    # Calculate the regression slope from x -> m (c + e/d)
    cEstimators = [np.dot(Xsamples[run],Msamples[run])/np.dot(Xsamples[run],Xsamples[run]) for run in range(numruns)]

    # Use um as an instrumental variable to estimate x -> m (c):
    # A) regress m on v
    
    MVslope = [np.dot(Msamples[run],Vsamples[run])/np.dot(Vsamples[run],Vsamples[run]) for run in range(numruns)]
    
    # B) regress x on v  (note that this is an unbiased estimator for g)
    
    XVslope = [np.dot(Xsamples[run],Vsamples[run])/np.dot(Vsamples[run],Vsamples[run]) for run in range(numruns)]
    
    # C) divide these coefficients
    
    cInstEstimators = [MVslope[run]/XVslope[run] for run in range(numruns)]

    
    # Construct the subtracted e/d estimator
    #print("Mean cEstimator: ",np.mean(cEstimators))
    #print("Mean cInstEstimator:",np.mean(cInstEstimators))
    edivdEst2 = [3*(cEstimators[run] - c) for run in range(numruns)]
    #print(edivdEst2)

    # Calculate the remainder for each datapoint {x,m} given perfect knowledge of 
    #uMsamples = [[Msamples[run][samp] - c * Xsamples[run][samp] for samp in range(numsamp)] for run in range(numruns)]
    uMsamples = [[Msamples[run][samp] - (c + edivdEst2[run]) * Xsamples[run][samp] for samp in range(numsamp)] for run in range(numruns)]
    
    ## Intermediate check: calculate all correlations of remainders um
    # Estimate the means of r (per run)
            
    rmeans = [np.mean(uMsamples[run]) for run in range(numruns)]
        
    # Estimate the variances of r (per run)
    
    rvars = [np.var(uMsamples[run]) for run in range(numruns)]
    
    # Estimate the covariances of X with r (per run)
    
    rcovsX = [np.cov(Xsamples[run],uMsamples[run],bias=True)[1][0] for run in range(numruns)]
    
    # Estimate the covariances of M with r (per run)
    
    rcovsM = [np.cov(Msamples[run],uMsamples[run],bias=True)[1][0] for run in range(numruns)]
    
    # Estimate the covariances of W with r (per run)
    
    rcovsW = [np.cov(Wsamples[run],uMsamples[run],bias=True)[1][0] for run in range(numruns)]
    
    meanrmean = stat.mean(rmeans)
    meanrvar = stat.mean(rvars)
    meanrcovX = stat.mean(rcovsX)
    meanrcovM = stat.mean(rcovsM)
    meanrcovW = stat.mean(rcovsW)
    
    # Use um as an instrumental variable to estimate m -> y (a):
    # A) regress y on um AND x
    
    YuMslope = [ (np.dot(uMsamples[run],Ysamples[run]) * np.dot(Xsamples[run],Xsamples[run]) - np.dot(uMsamples[run],Xsamples[run]) * np.dot(Xsamples[run],Ysamples[run]) ) / (np.dot(uMsamples[run],uMsamples[run]) * np.dot(Xsamples[run],Xsamples[run]) - np.dot(uMsamples[run],Xsamples[run])**2) for run in range(numruns)]
    
    # B) regress m on um
    
    #MuMslope = [np.dot(Msamples[run],uMsamples[run])/np.dot(uMsamples[run],uMsamples[run]) for run in range(numruns)]
    MuMslope = [ (np.dot(uMsamples[run],Msamples[run]) * np.dot(Xsamples[run],Xsamples[run]) - np.dot(uMsamples[run],Xsamples[run]) * np.dot(Xsamples[run],Msamples[run]) ) / (np.dot(uMsamples[run],uMsamples[run]) * np.dot(Xsamples[run],Xsamples[run]) - np.dot(uMsamples[run],Xsamples[run])**2) for run in range(numruns)]
    
    # C) divide these coefficients

    aInstEstimators = [YuMslope[run]/MuMslope[run] for run in range(numruns)]
    #aInstEstimators = MuMslope     # to run this test, switch aBias to aInstMean - 1
    
    cBias = 0
    cVar = 0

    #print("Theoretical mean(c) = 0")
    #THcVar = 1./((numsamp-2)*(d**2 + 1))
    #print("Theoretical var(c) = ","{:.6f}".format(THcVar))

    aInstMean = stat.mean(aInstEstimators)
    aBias = aInstMean - a
    aVar = stat.variance(aInstEstimators)

    #THaBias = b*c*d / (1 + c**2 * (d**2 + 1))
    #print("Theoretical bias(a) = ","{:.4f}".format(THaBias))

    acInstEstimators = [aInstEstimators[i] * c for i in range(numruns) ]
    acInstMean = stat.mean(acInstEstimators)
    acBias = acInstMean - a*c
    acVar = stat.variance(acInstEstimators)
                   
    return {"c":  [round(cBias,4), round(cVar,4)], 
            "a":  [round(aBias,4), round(aVar,4)],
            "ac": [round(acBias,4), round(acVar,4)],
            "rmean": round(meanrmean,4),
            "rvar": round(meanrvar,4),
            "rcovX": round(meanrcovX,4),
            "rcovM": round(meanrcovM,4),
            "rcovW": round(meanrcovW,4)
            }
            
def estimatorPriorInstFDCPoly(sampledicts,coeffdict): 

    numruns = len(sampledicts)
    
    Vsamples = [sampledicts[run]["V"] for run in range(numruns)]
    Wsamples = [sampledicts[run]["W"] for run in range(numruns)]
    Xsamples = [sampledicts[run]["X"] for run in range(numruns)]
    Msamples = [sampledicts[run]["M"] for run in range(numruns)]
    Ysamples = [sampledicts[run]["Y"] for run in range(numruns)]
    
    numsamp = len(Wsamples[0])
    
    # Recall true values
    a = coeffdict['a'][1]
    c = coeffdict['c'][1]
    e = coeffdict['e'][1] #linear
    d = coeffdict['d'][1] #linear

    # Use um as an instrumental variable to estimate x -> m (c):
    # A) regress m on v
    
    MVslope = [np.dot(Msamples[run],Vsamples[run])/np.dot(Vsamples[run],Vsamples[run]) for run in range(numruns)]
    
    # B) regress x on v  (note that this is an unbiased estimator for g)
    
    XVslope = [np.dot(Xsamples[run],Vsamples[run])/np.dot(Vsamples[run],Vsamples[run]) for run in range(numruns)]
    
    # C) divide these coefficients
    
    cInstEstimators = [MVslope[run]/XVslope[run] for run in range(numruns)]
    
    # Calculate the residuals r = um for each datapoint {x,m}
    uMsamples = [[Msamples[run][samp] - cInstEstimators[run] * Xsamples[run][samp] for samp in range(numsamp)] for run in range(numruns)]
    
    ## Intermediate check: calculate all correlations of residuals um
    # Estimate the means of r (per run)
            
    rmeans = [np.mean(uMsamples[run]) for run in range(numruns)]
        
    # Estimate the variances of r (per run)
    
    rvars = [np.var(uMsamples[run]) for run in range(numruns)]
    
    # Estimate the covariances of X with r (per run)
    
    rcovsX = [np.cov(Xsamples[run],uMsamples[run],bias=True)[1][0] for run in range(numruns)]
    
    # Estimate the covariances of M with r (per run)
    
    rcovsM = [np.cov(Msamples[run],uMsamples[run],bias=True)[1][0] for run in range(numruns)]
    
    # Estimate the covariances of W with r (per run)
    
    rcovsW = [np.cov(Wsamples[run],uMsamples[run],bias=True)[1][0] for run in range(numruns)]
    
    meanrmean = stat.mean(rmeans)
    meanrvar = stat.mean(rvars)
    meanrcovX = stat.mean(rcovsX)
    meanrcovM = stat.mean(rcovsM)
    meanrcovW = stat.mean(rcovsW)
    
    ## Use um as an instrumental variable to estimate m -> y (a):
    # A) regress y on um
    
    YuMslope = [np.dot(Ysamples[run],uMsamples[run])/np.dot(uMsamples[run],uMsamples[run]) for run in range(numruns)]
    
    # B) regress m on um
    
    MuMslope = [np.dot(Msamples[run],uMsamples[run])/np.dot(uMsamples[run],uMsamples[run]) for run in range(numruns)]
    
    # C) divide these coefficients
    
    aInstEstimators = [YuMslope[run]/MuMslope[run] for run in range(numruns)]
    #aInstEstimators = MuMslope     # to run this test, switch aBias to aInstMean - 1
    
    cMean = stat.mean(cInstEstimators)
    cBias = cMean - c
    cVar = stat.variance(cInstEstimators)

    #print("Theoretical mean(c) = 0")
    #THcVar = 1./((numsamp-2)*(d**2 + 1))
    #print("Theoretical var(c) = ","{:.6f}".format(THcVar))

    aInstMean = stat.mean(aInstEstimators)
    aBias = aInstMean - a
    aVar = stat.variance(aInstEstimators)

    #THaBias = b*c*d / (1 + c**2 * (d**2 + 1))
    #print("Theoretical bias(a) = ","{:.4f}".format(THaBias))

    acInstEstimators = [aInstEstimators[i] * cInstEstimators[i] for i in range(numruns) ]
    acInstMean = stat.mean(acInstEstimators)
    acBias = acInstMean - a*c
    acVar = stat.variance(acInstEstimators)
                   
    return {"c":  [round(cBias,4), round(cVar,4)], 
            "a":  [round(aBias,4), round(aVar,4)],
            "ac": [round(acBias,4), round(acVar,4)],
            "rmean": round(meanrmean,4),
            "rvar": round(meanrvar,4),
            "rcovX": round(meanrcovX,4),
            "rcovM": round(meanrcovM,4),
            "rcovW": round(meanrcovW,4)
            }
            
def divpolys(es,ds,x):
    dfunc = lambda y: np.sum([d*(y**n) for n,d in enumerate(ds)]) - x
    #print(x)
    #print(dfunc(0)+x)
    #print(dfunc(1)+x)
    #print(dfunc(2)+x)
    dinvx = scipy.optimize.fsolve(dfunc,0.,xtol=0.0001)
    return np.sum([e*(dinvx**m) for m,e in enumerate(es)])
    
            
def estimatorPriorEDRemainderFDCPoly(sampledicts,coeffdict): 

    numruns = len(sampledicts)
    
    Vsamples = [sampledicts[run]["V"] for run in range(numruns)]
    Wsamples = [sampledicts[run]["W"] for run in range(numruns)]
    Xsamples = [sampledicts[run]["X"] for run in range(numruns)]
    Msamples = [sampledicts[run]["M"] for run in range(numruns)]
    Ysamples = [sampledicts[run]["Y"] for run in range(numruns)]
    
    numsamp = len(Wsamples[0])
    
    # Recall true values
    a = coeffdict['a'][1]
    c = coeffdict['c'][1]
    elist = coeffdict['e']
    dlist = coeffdict['d']

    # Use um as an instrumental variable to estimate x -> m (c):
    # A) regress m on v
    
    MVslope = [np.dot(Msamples[run],Vsamples[run])/np.dot(Vsamples[run],Vsamples[run]) for run in range(numruns)]
    
    # B) regress x on v  (note that this is an unbiased estimator for g)
    
    XVslope = [np.dot(Xsamples[run],Vsamples[run])/np.dot(Vsamples[run],Vsamples[run]) for run in range(numruns)]
    
    # C) divide these coefficients
    
    cInstEstimators = [MVslope[run]/XVslope[run] for run in range(numruns)]

    # Assume we already know some perfect e/d estimator
    edivd = lambda x: divpolys(elist,dlist,x)
    
    # Calculate the remainder for each datapoint {x,m} given perfect knowledge of 
    #uMsamples = [[Msamples[run][samp] - c * Xsamples[run][samp] for samp in range(numsamp)] for run in range(numruns)]
    uMsamples = [[Msamples[run][samp] - c * Xsamples[run][samp] - edivd(Xsamples[run][samp]) for samp in range(numsamp)] for run in range(numruns)]
    
    ## Intermediate check: calculate all correlations of remainders um
    # Estimate the means of r (per run)
            
    rmeans = [np.mean(uMsamples[run]) for run in range(numruns)]
        
    # Estimate the variances of r (per run)
    
    rvars = [np.var(uMsamples[run]) for run in range(numruns)]
    
    # Estimate the covariances of X with r (per run)
    
    rcovsX = [np.cov(Xsamples[run],uMsamples[run],bias=True)[1][0] for run in range(numruns)]
    
    # Estimate the covariances of M with r (per run)
    
    rcovsM = [np.cov(Msamples[run],uMsamples[run],bias=True)[1][0] for run in range(numruns)]
    
    # Estimate the covariances of W with r (per run)
    
    rcovsW = [np.cov(Wsamples[run],uMsamples[run],bias=True)[1][0] for run in range(numruns)]
    
    meanrmean = stat.mean(rmeans)
    meanrvar = stat.mean(rvars)
    meanrcovX = stat.mean(rcovsX)
    meanrcovM = stat.mean(rcovsM)
    meanrcovW = stat.mean(rcovsW)
    
    # Use um as an instrumental variable to estimate m -> y (a):
    # A) regress y on um
    
    YuMslope = [np.dot(Ysamples[run],uMsamples[run])/np.dot(uMsamples[run],uMsamples[run]) for run in range(numruns)]
    
    # B) regress m on um
    
    MuMslope = [np.dot(Msamples[run],uMsamples[run])/np.dot(uMsamples[run],uMsamples[run]) for run in range(numruns)]
    
    print("Remainder instrumental coefficients")
    print("YR slope: ", np.mean(YuMslope))
    print("MR slope: ", np.mean(MuMslope))
    
    
    
    # C) divide these coefficients
    
    
    
    aInstEstimators = [YuMslope[run]/MuMslope[run] for run in range(numruns)]
    #aInstEstimators = MuMslope     # to run this test, switch aBias to aInstMean - 1
    
    cBias = 0
    cVar = 0

    #print("Theoretical mean(c) = 0")
    #THcVar = 1./((numsamp-2)*(d**2 + 1))
    #print("Theoretical var(c) = ","{:.6f}".format(THcVar))

    aInstMean = stat.mean(aInstEstimators)
    aBias = aInstMean - a
    aVar = stat.variance(aInstEstimators)

    #THaBias = b*c*d / (1 + c**2 * (d**2 + 1))
    #print("Theoretical bias(a) = ","{:.4f}".format(THaBias))

    acInstEstimators = [aInstEstimators[i] * c for i in range(numruns) ]
    acInstMean = stat.mean(acInstEstimators)
    acBias = acInstMean - a*c
    acVar = stat.variance(acInstEstimators)
                   
    return {"c":  [round(cBias,4), round(cVar,4)], 
            "a":  [round(aBias,4), round(aVar,4)],
            "ac": [round(acBias,4), round(acVar,4)],
            "rmean": round(meanrmean,4),
            "rvar": round(meanrvar,4),
            "rcovX": round(meanrcovX,4),
            "rcovM": round(meanrcovM,4),
            "rcovW": round(meanrcovW,4)
            }           
           
estimator_lookup = {
    "truefdc": estimatorTrueFDC,
    "approxfdc": estimatorApproxFDC,
    "instfdc": estimatorInstFDC,
    "edremfdc": estimatorEDRemainderFDC,
    "ratioedremfdc": estimatorEDRatioRemainderFDC,
    "instfdc-poly": estimatorInstFDCPoly,
    "edremfdc-poly": estimatorEDRemainderFDCPoly,
    "pi-instfdc": estimatorPriorInstFDC,
    "pi-edresfdc": estimatorPriorEDResidualFDC,
    "pi-edremfdc": estimatorPriorEDRemainderFDC,
    "pi-subedresfdc": estimatorPriorSubEDResidualFDC,
    "pi-subedremfdc": estimatorPriorSubEDRemainderFDC,
    "pi-ratioedresfdc": estimatorPriorRatioEDResidualFDC,
    "pi-ratioedremfdc": estimatorPriorRatioEDRemainderFDC,
    "pi-instfdc-x": estimatorPriorInstFDCwithX,
    "pi-edresfdc-x": estimatorPriorEDResidualFDCwithX,
    "pi-edremfdc-x": estimatorPriorEDRemainderFDCwithX,
    "pi-subedresfdc-x": estimatorPriorSubEDResidualFDCwithX,
    "pi-subedremfdc-x": estimatorPriorSubEDRemainderFDCwithX,
    "pi-instfdc-poly": estimatorPriorInstFDCPoly,
    "pi-edremfdc-poly": estimatorPriorEDRemainderFDCPoly
    
}

estimator_name = {
    "truefdc": "True FDC",
    "approxfdc": "Approximate FDC",
    "instfdc": "Instrumental FDC",
    "edremfdc": "e/d-improved Remainder FDC",
    "ratioedremfdc": "ratio-e/d-improved Remainder FDC",
    "instfdc-poly": "Instrumental FDC, Polynomial Couplings",
    "edremfdc-poly": "e/d-improved Remainder FDC, Polynomial Couplings",
    "pi-instfdc": "Instrumental FDC with Prior Instrument",
    "pi-edresfdc": "e/d-improved Residual FDC with Prior Instrument",
    "pi-edremfdc": "e/d-improved Remainder FDC with Prior Instrument",
    "pi-subedresfdc": "sub-e/d-improved Residual FDC with Prior Instrument",
    "pi-subedremfdc": "sub-e/d-improved Remainder FDC with Prior Instrument",
    "pi-ratioedresfdc": "ratio-e/d-improved Residual FDC with Prior Instrument",
    "pi-ratioedremfdc": "ratio-e/d-improved Remainder FDC with Prior Instrument",
    "pi-instfdc-x": "Instrumental FDC+X with Prior Instrument",
    "pi-edresfdc-x": "e/d-improved Residual FDC+X with Prior Instrument",
    "pi-edremfdc-x": "e/d-improved Remainder FDC+X with Prior Instrument",
    "pi-subedresfdc-x": "sub-e/d-improved Residual FDC+X with Prior Instrument",
    "pi-subedremfdc-x": "sub-e/d-improved Remainder FDC+X with Prior Instrument",
    "pi-instfdc-poly": "Instrumental FDC with Prior Instrument, Polynomial Couplings",
    "pi-edremfdc-poly": "e/d-improved Remainder FDC with Prior Instrument, Polynomial Couplings"
}

estimator_shortname = {
    "truefdc": "True FDC",
    "approxfdc": "Approximate FDC",
    "instfdc": "Instrumental FDC",
    "edremfdc": "e/d-rem",
    "ratioedremfdc": "ratio-e/d-rem",
    "instfdc-poly": "poly-ifdc",
    "edremfdc-poly": "poly-edremfdc",
    "pi-instfdc": "stand-res",
    "pi-edresfdc": "e/d-res",
    "pi-edremfdc": "e/d-rem",
    "pi-subedresfdc": "sub-e/d-res",
    "pi-subedremfdc": "sub-e/d-rem",
    "pi-ratioedresfdc": "ratio-e/d-res",
    "pi-ratioedremfdc": "ratio-e/d-rem",
    "pi-instfdc-x": "Instrumental FDC+X with Prior Instrument",
    "pi-edresfdc-x": "e/d-improved Residual FDC+X with Prior Instrument",
    "pi-edremfdc-x": "e/d-improved Remainder FDC+X with Prior Instrument",
    "pi-subedresfdc-x": "sub-e/d-improved Residual FDC+X with Prior Instrument",
    "pi-subedremfdc-x": "sub-e/d-improved Remainder FDC+X with Prior Instrument",
    "pi-instfdc-poly": "poly-ifdc",
    "pi-edremfdc-poly": "poly-remfdc"
}
    # Note both params and scanparams are dictionaries i.e. {var: val} or {var: [min,max,step]}
#def estimator_scan(model,scanparams,params,noise,estimators,numsamp=1e4,numrun=1e2):



def estimator_draw(model,params,noise,estimators,numsamp=1e4,numrun=1e2,shift=0):

    # Input cleaning
    numsamp = int(numsamp)
    numrun = int(numrun)
    if model not in list(model_lookup.keys()):
        raise ValueError('Model not defined in model dictionary.')
        
    if len(params) > model_params[model]:
        raise ValueError('Too many parameters for model, model will exit.')
        
    print(params)
    print(len(params))
    if len(params) < model_params[model]:
        raise Warning('Not all model parameters specified, model will resort to defaults.')

    # First the graph object for the chosen model is generated
    modelgen = model_lookup[model]
    confmodel = modelgen(params,noise=noise,shift=shift)

    coeffdict = dict(zip(confmodel.edgevars,confmodel.coeffs))

    runvalues = []
        
    for run in range(numrun):
        
        os.system('cls||clear')
        #print("Params:" + paramstr)
        print(params)
        print("Drawing samples from " + confmodel.name)
        #print("a = ", a, " in range [", arange[0], ", ", arange[1], "]")
        #print("c = ", c, " in range [", crange[0], ", ", crange[1], "]")
        print("Run: ",run," of ",numrun)
        confsamples = confmodel.sample(numsamp)
        
        values = [[x for x in samp.values()] for samp in confsamples]
        valuesT = np.array(values).T.tolist()
        runvalues.append(valuesT)
       
    # Retrieve samples from all model nodes, in order of confmodel.nodes
    nodedict = dict([(node,confmodel.nodes.index(node)) for node in confmodel.nodes])
    sampledicts = [dict([(node,runvalues[run][nodedict[node]]) for node in confmodel.nodes]) for run in range(numrun)]
    
    # The chosen estimators are now generated and dictionaried
    biasvars = {}
    for estimator in estimators:
        estimatorfunc = estimator_lookup[estimator]
        biasvars[estimator] = estimatorfunc(sampledicts,coeffdict)
            
    return biasvars
    
def estimator_scan(model,scanvar,scanrange, scannum,fixparams,noise,estimators,numsamp=1e4,numrun=1e2,shift=0):

    biasvarscan = []
    scanvals = np.linspace(scanrange[0],scanrange[1],scannum+1)
    for e in scanvals:
        params = fixparams + [e]
        biasvarscan.append(estimator_draw(model,params, noise,estimators,numsamp=numsamp,numrun=numrun,shift=shift))
    return biasvarscan
    
def estimator_filedata(fname,paramdict,coeffdict,estimators):

    # Input cleaning
    df = pd.read_csv(filename, encoding="iso-8859-1")
    sampledict = {} # Plural later in case of multiple runs
    coeffdict = {} 
    for param in paramdict:
        sampledict[param] = df[paramdict[param]].values
        
    numsamp = len(sampledict[param])
    numrun = 1
   
    
    # The chosen estimators are now generated from data and dictionaried
    biasvars = {}
    for estimator in estimators:
        estimatorfunc = estimator_lookup[estimator]
        biasvars[estimator] = estimatorfunc([sampledict],coeffdict)
            
    return biasvars
    
def estimator_dfdata(dfs,coeffdict,estimators):

    # Input cleaning
    sampledicts = [] # Plural later in case of multiple runs
    for df in dfs:
        sampledict = {}
        for name,values in df.iteritems():
            sampledict[name] = values
            numsamp = len(sampledict[name])
        sampledicts.append(sampledict)

    numrun = len(dfs)
   
    
    # The chosen estimators are now generated from data and dictionaried
    biasvars = {}
    for estimator in estimators:
        estimatorfunc = estimator_lookup[estimator]
        biasvars[estimator] = estimatorfunc(sampledicts,coeffdict)
            
    return biasvars


def test():
    #biasvars = estimator_draw("confounder",[1,1,1,1,1],1,["truefdc","instfdc"])
    #estlist = ["pi-instfdc","pi-edresfdc","pi-edremfdc","pi-subedresfdc","pi-subedremfdc"]
    estlist = ["pi-instfdc","pi-edresfdc","pi-edremfdc","pi-instfdc-x","pi-edresfdc-x","pi-edremfdc-x"]
    # d-c-b-a-g-e
    biasvars = estimator_draw("instrument",[1,1,1,1,1,1],1,estlist,numsamp=1e4,numrun=2e2)
    
    for est in estlist:
        print(estimator_name[est])
        print(biasvars[est])
        print("\n")
        
def test_shifted(shift=1):
    #biasvars = estimator_draw("confounder",[1,1,1,1,1],1,["truefdc","instfdc"])
    estlist = ["pi-instfdc","pi-edresfdc","pi-edremfdc","pi-subedresfdc","pi-subedremfdc","pi-ratioedresfdc","pi-ratioedremfdc"]
    #estlist = ["pi-instfdc","pi-edresfdc","pi-edremfdc","pi-instfdc-x","pi-edresfdc-x","pi-edremfdc-x"]
    # d-c-b-a-g-e
    biasvars = estimator_draw("instrument",[1,1,1,1,1,100],1,estlist,numsamp=1e4,numrun=2e2,shift=1)
    
    for est in estlist:
        print(estimator_name[est])
        print(biasvars[est])
        print("\n")
    
def test_escan_shifted(shift=1,ofile="test"):
    #biasvars = estimator_draw("confounder",[1,1,1,1,1],1,["truefdc","instfdc"])
    estlist = ["pi-instfdc","pi-edresfdc","pi-edremfdc","pi-subedresfdc","pi-subedremfdc","pi-ratioedresfdc","pi-ratioedremfdc"]
    estlist = ["pi-instfdc","pi-edresfdc","pi-edremfdc","pi-ratioedresfdc","pi-ratioedremfdc"]
    numests = len(estlist)
    #colests = [np.array(list(get_color(x)))/255.0 for x in np.linspace(0,1,numests)]
    colests = ['r','g','b','g','b']
    linestyles = ['solid','solid','solid','dashed','dashed']
    scanrange = [0,3]
    scannum = 30
    #estlist = ["pi-instfdc","pi-edresfdc","pi-edremfdc","pi-instfdc-x","pi-edresfdc-x","pi-edremfdc-x"]
    # d-c-b-a-g-e
    biasvarscan = estimator_scan("instrument",'e',scanrange,scannum,[1,1,1,1,1],1,estlist,numsamp=5e4,numrun=1e2,shift=1)
    
    scanvals = np.linspace(scanrange[0],scanrange[1],scannum+1)
    fig, ax = plt.subplots(1,1)
    
    for i,est in enumerate(estlist):
        print(colests[i])
        biasvarsesta = [biasvars[est]['ac'][0] for biasvars in biasvarscan]
        biasvarsestac = [biasvars[est]['ac'][1] for biasvars in biasvarscan]
        ax.plot(scanvals,biasvarsesta, color=colests[i], linestyle = linestyles[i], label=estimator_shortname[est])
        #ax[1].plot(scanvals,biasvarsestac, color=colests[i], label=estimator_shortname[est])
    ax.legend()
    #ax[1].legend()
    ax.set_xlabel('e')
    #ax[1].set_xlabel('e')
    ax.set_ylabel('Bias[ac]')
    ax.set_ylim([-0.2,0.6])
    #ax[1].set_ylabel('Var[ac]')
    
    fig.set_size_inches((10,10), forward=False)
    fig.tight_layout()
    fig.savefig(ofile + '.pdf', format='pdf', dpi=1200)
    
def test_scanenoise_shifted(shift=1,ofile="test"):
    #biasvars = estimator_draw("confounder",[1,1,1,1,1],1,["truefdc","instfdc"])
    estlist = ["instfdc","ratioedremfdc"]
    numests = len(estlist)
    numnoises = 3
    noiserange = [0.1,1.0]
    noises = np.linspace(noiserange[0],noiserange[1],numnoises)
    noises = [0.1,0.5,1.0]
    print(noises)
    colests = [np.array(list(get_color_bg(x)))/255.0 for x in np.linspace(0,1,numnoises)]
    linestyles = ['dashed','solid']
    scanrange = [0,2]
    scannum = 30
    #estlist = ["pi-instfdc","pi-edresfdc","pi-edremfdc","pi-instfdc-x","pi-edresfdc-x","pi-edremfdc-x"]
    # d-c-b-a-g-e
    biasvarscan = []
    for i,noise in enumerate(noises):
        thisbiasvarscan = estimator_scan("confounder",'e',scanrange,scannum,[1,1,1,1],noise,estlist,numsamp=1e4,numrun=1e2,shift=1)
        biasvarscan.append(thisbiasvarscan)
    
    scanvals = np.linspace(scanrange[0],scanrange[1],scannum+1)
    fig, ax = plt.subplots(2,2)
    
    for i,noise in enumerate(noises):
        print(colests[i])
        ifdc_biasesta = [biasvars[estlist[0]]['a'][0] for biasvars in biasvarscan[i]]
        ifdc_varesta = [biasvars[estlist[0]]['a'][1] for biasvars in biasvarscan[i]]
        rremifdc_biasesta = [biasvars[estlist[1]]['a'][0] for biasvars in biasvarscan[i]]
        rremifdc_varesta = [biasvars[estlist[1]]['a'][1] for biasvars in biasvarscan[i]]
        
        thislabel = "$\sigma^2$ = " + str(noise)
        
        ax[0][0].plot(scanvals,ifdc_biasesta, color=colests[i], linestyle = 'solid', label=thislabel)
        ax[1][0].plot(scanvals,ifdc_varesta, color=colests[i], linestyle = 'solid', label=thislabel)
        ax[0][1].plot(scanvals,rremifdc_biasesta, color=colests[i], linestyle = 'solid', label=thislabel)
        ax[1][1].plot(scanvals,rremifdc_varesta, color=colests[i], linestyle = 'solid', label=thislabel)
        #ax[1].plot(scanvals,biasvarsestac, color=colests[i], label=estimator_shortname[est])
    ax[0][0].legend(prop={'size': 18})
    ax[0][1].legend(prop={'size': 18})
    ax[1][0].legend(prop={'size': 18})
    ax[1][1].legend(prop={'size': 18})
    ax[1][0].set_xlabel('e')
    ax[1][1].set_xlabel('e')
    ax[0][0].set_ylabel('Bias[a]')
    ax[1][0].set_ylabel('Var[a]')
    ax[0][0].set_ylim([-0.1,0.6])
    ax[0][1].set_ylim([-0.1,0.6])
    ax[1][0].set_ylim([0,0.001])
    ax[1][1].set_ylim([0,0.1])
    ax[0][0].set_title('IFDC')
    ax[0][1].set_title('e/d-improved IFDC')
    
    fig.set_size_inches((16,12), forward=False)
    fig.tight_layout()
    fig.savefig(ofile + '.pdf', format='pdf', dpi=1200)

def test_poly(ofile="test"):
    #biasvars = estimator_draw("confounder",[1,1,1,1,1],1,["truefdc","instfdc"])
    estlist = ["pi-instfdc-poly","pi-edremfdc-poly"]
    numests = len(estlist)
    colests = [np.array(list(get_color(x)))/255.0 for x in np.linspace(0,1,numests)]
    #colests = [(r,g,b) for r,g,b in colests]
    scanrange = [0,2]
    scannum = 40
    #estlist = ["pi-instfdc","pi-edresfdc","pi-edremfdc","pi-instfdc-x","pi-edresfdc-x","pi-edremfdc-x"]
    # d-c-b-a-g-e
    #coeffd = [0,1,0,0.2]
    coeffd = [0,1,-0.75,0.2]                             # maximal quadratic
    coeffd = [0,1,0,-1/3.,0,2/15.,0,-17/315.,0,62/2835.] #tanh
    coeffc = [0,1]
    coeffb = [0,1]
    coeffa = [0,1]
    coeffg = [0,1]
    coeffs = [coeffd,coeffc,coeffb,coeffa,coeffg]
    coeffe = [0,1]
    #coeffe = [0,1,0,-1/3.,0,2/15.,0,-17/315.,0,62/2835.] #tanh
    
    biasvars = estimator_draw("instrumentpoly",coeffs + [coeffe],1,estlist,numsamp=1e3,numrun=1e2)
    
    scanvals = np.linspace(scanrange[0],scanrange[1],scannum+1)
    fig, ax = plt.subplots(1,2)
    
    for i,est in enumerate(estlist):
        print(colests[i])
        biasvarsesta = biasvars[est]['ac'][0]
        biasvarsestac = biasvars[est]['ac'][1]

        print(estimator_name[est])
        print(biasvars[est])
        print("\n")

def test_dscan_poly(ofile="test"):
    #biasvars = estimator_draw("confounder",[1,1,1,1,1],1,["truefdc","instfdc"])
    estlist = ["pi-instfdc-poly","pi-edremfdc-poly"]
    numests = len(estlist)
    colests = [np.array(list(get_color(x)))/255.0 for x in np.linspace(0,1,numests)]
    #colests = [(r,g,b) for r,g,b in colests]
    scanrange = [-0.75,0.75] # sqrt(0.6) = +-0.7746 is the invertibility bound
    scannum = 10
    #estlist = ["pi-instfdc","pi-edresfdc","pi-edremfdc","pi-instfdc-x","pi-edresfdc-x","pi-edremfdc-x"]
    # d-c-b-a-g-e
    coeffd = [0,1,0,0.2]
    #coeffd = [0,1,-0.75,0.2]                             # maximal quadratic
    #coeffd = [0,1,0,-1/3.,0,2/15.,0,-17/315.,0,62/2835.] #tanh
    coeffc = [0,1]
    coeffb = [0,1]
    coeffa = [0,1]
    coeffg = [0,1]
    coeffs = [coeffd,coeffc,coeffb,coeffa,coeffg]
    #coeffe = [0,2]
    coeffe = [0,1,0,-1/3.,0,2/15.,0,-17/315.,0,62/2835.] #tanh
    
    biasvarscan = []
    scanvals = np.linspace(scanrange[0],scanrange[1],scannum+1)
    
    for d2 in scanvals:
        coeffd[2] = d2
        coeffs = [coeffd,coeffc,coeffb,coeffa,coeffg]
        biasvarscan.append(estimator_draw("instrumentpoly",coeffs + [coeffe],1,estlist,numsamp=1e3,numrun=1e3))
    
    fig, ax = plt.subplots(1,2)
    
    for i,est in enumerate(estlist):
        print(colests[i])
        biasvarsesta = [biasvars[est]['ac'][0] for biasvars in biasvarscan]
        biasvarsestac = [biasvars[est]['ac'][1] for biasvars in biasvarscan]
        print(biasvarsesta)
        ax[0].plot(scanvals,biasvarsesta, color=colests[i], label=estimator_shortname[est])
        ax[1].plot(scanvals,biasvarsestac, color=colests[i], label=estimator_shortname[est])
    ax[0].legend()
    ax[1].legend()
    ax[0].set_xlabel('d[2]')
    ax[1].set_xlabel('d[2]')
    ax[0].set_ylabel('Bias[ac]')
    ax[1].set_ylabel('Var[ac]')
    
    fig.set_size_inches((20,10), forward=False)
    fig.tight_layout()
    fig.savefig(ofile + '.pdf', format='pdf', dpi=1200)

def test_e3scan_poly(ofile="test"):
    #biasvars = estimator_draw("confounder",[1,1,1,1,1],1,["truefdc","instfdc"])
    estlist = ["instfdc-poly","edremfdc-poly"]
    numests = len(estlist)
    colests = [np.array(list(get_color(x)))/255.0 for x in np.linspace(0,1,numests)]
    #colests = [(r,g,b) for r,g,b in colests]
    e2scanrange = [-1,1]
    e2scannum = 5
    e3scanrange = [-0.4,0.4] # sqrt(0.6) = +-0.7746 is the invertibility bound
    e3scannum = 4
    cole3s = [np.array(list(get_color(x)))/255.0 for x in np.linspace(0,1,e3scannum+1)]
    #estlist = ["pi-instfdc","pi-edresfdc","pi-edremfdc","pi-instfdc-x","pi-edresfdc-x","pi-edremfdc-x"]
    # d-c-b-a-g-e
    #coeffd = [0,1,0,0.2]
    coeffd = [0,1]                             # maximal quadratic
    #coeffd = [0,1,0,-1/3.,0,2/15.,0,-17/315.,0,62/2835.] #tanh
    coeffc = [0,1]
    coeffb = [0,1]
    coeffa = [0,1]
    #coeffg = [0,1]
    #coeffs = [coeffd,coeffc,coeffb,coeffa,coeffg]
    coeffs = [coeffd,coeffc,coeffb,coeffa]
    coeffe = [0,2.0,0,0]
    #coeffe = [0,1,0,-1/3.,0,2/15.,0,-17/315.,0,62/2835.] #tanh
    
    biasvarscan = []
    e3scanvals = np.linspace(e3scanrange[0],e3scanrange[1],e3scannum+1)
    e2scanvals = np.linspace(e2scanrange[0],e2scanrange[1],e2scannum+1)
    
    for e3 in e3scanvals:
        coeffe[3] = e3
        
        thisbiasvarscan = []
        for e2 in e2scanvals:
            coeffe[2] = e2
            #coeffs = [coeffd,coeffc,coeffb,coeffa,coeffg]
            coeffs = [coeffd,coeffc,coeffb,coeffa]
            thisbiasvarscan.append(estimator_draw("confounderpoly",coeffs + [coeffe],0.3,estlist,numsamp=1e4,numrun=1e2))
        biasvarscan.append(thisbiasvarscan)
    
    fig, ax = plt.subplots(1,2)
    
    for i,e3 in enumerate(e3scanvals):
        print(e3scanvals[i])
        print(e2scanvals)
        #print(biasvarscan[0][estlist[0]])
        biasvarsestifdc = [biasvars[estlist[0]]['a'][0] for biasvars in biasvarscan[i]]
        biasvarsestrem = [biasvars[estlist[1]]['a'][0] for biasvars in biasvarscan[i]]
        print(e2scanvals)
        print(biasvarsestifdc)
        #lab0 = estimator_shortname[estlist[0]] + " e3=" + '{0:3.1f}'.format(e3)
        #lab1 = estimator_shortname[estlist[1]] + " e3=" + '{0:3.1f}'.format(e3)
        lab0 = "e3=" + '{0:3.1f}'.format(e3)
        lab1 = "e3=" + '{0:3.1f}'.format(e3)
        ax[0].plot(e2scanvals,biasvarsestifdc, color=cole3s[i], label=lab0)
        ax[1].plot(e2scanvals,biasvarsestrem, color=cole3s[i], label=lab1)
    ax[0].legend()
    ax[1].legend()
    ax[0].set_xlabel('e[2]')
    ax[1].set_xlabel('e[2]')
    ax[0].set_ylabel('Bias[a]')
    ax[1].set_ylabel('Bias[a]')
    ax[0].set_ylim([0.25,0.35])
    ax[1].set_ylim([-0.6,0.2])
    ax[0].set_title('IFDC')
    ax[1].set_title('e/d-improved IFDC')
    
    fig.set_size_inches((20,10), forward=False)
    fig.tight_layout()
    fig.savefig(ofile + '.pdf', format='pdf', dpi=1200)

def test():
    #biasvars = estimator_draw("confounder",[1,1,1,1,1],1,["truefdc","instfdc"])
    #estlist = ["pi-instfdc","pi-edresfdc","pi-edremfdc","pi-subedresfdc","pi-subedremfdc"]
    estlist = ["pi-instfdc","pi-edresfdc","pi-edremfdc","pi-instfdc-x","pi-edresfdc-x","pi-edremfdc-x"]
    # d-c-b-a-g-e
    biasvars = estimator_draw("instrument",[1,1,1,1,1,1],1,estlist,numsamp=1e4,numrun=2e2)
    
    for est in estlist:
        print(estimator_name[est])
        print(biasvars[est])
        print("\n")
        
    scanvals = np.linspace(scanrange[0],scanrange[1],scannum+1)
    fig, ax = plt.subplots(1,2)
    
    for i,est in enumerate(estlist):
        print(colests[i])
        biasvarsesta = biasvars[est]['ac'][0]
        biasvarsestac = biasvars[est]['ac'][1]

        print(estimator_name[est])
        print(biasvars[est])
        print("\n")
        
def test_d3scan_poly(ofile="test"):
    #biasvars = estimator_draw("confounder",[1,1,1,1,1],1,["truefdc","instfdc"])
    estlist = ["instfdc-poly","edremfdc-poly"]
    numests = len(estlist)
    colests = [np.array(list(get_color_bg(x)))/255.0 for x in np.linspace(0,1,numests)]
    #colests = [(r,g,b) for r,g,b in colests]
    d2scannum = 5
    d3scanrange = [0.1,0.5] # sqrt(0.6) = +-0.7746 is the invertibility bound
    d3scannum = 4
    cold3s = [np.array(list(get_color_bg(x)))/255.0 for x in np.linspace(0,1,d3scannum+1)]
    #estlist = ["pi-instfdc","pi-edresfdc","pi-edremfdc","pi-instfdc-x","pi-edresfdc-x","pi-edremfdc-x"]
    # d-c-b-a-g-e
    #coeffd = [0,1,0,0.2]
    coeffd = [0,1,-0.75,0.2]                             # maximal quadratic
    #coeffd = [0,1,0,-1/3.,0,2/15.,0,-17/315.,0,62/2835.] #tanh
    coeffc = [0,1]
    coeffb = [0,1]
    coeffa = [0,1]
    #coeffg = [0,1]
    #coeffs = [coeffd,coeffc,coeffb,coeffa,coeffg]
    coeffs = [coeffd,coeffc,coeffb,coeffa]
    coeffe = [0,2.0]
    #coeffe = [0,1,0,-1/3.,0,2/15.,0,-17/315.,0,62/2835.] #tanh
    
    biasvarscan = []
    d3scanvals = np.linspace(d3scanrange[0],d3scanrange[1],d3scannum+1)
    d2scanvals = []
    
    for d3 in d3scanvals:
        coeffd[3] = d3
        d2scanbound = np.sqrt(3*d3)
        if d2scanbound > 0.5:
            d2scanbound = 0.5
        thisd2scanrange = [-1.*d2scanbound,d2scanbound]
        thisd2scanvals = np.linspace(thisd2scanrange[0],thisd2scanrange[1],d2scannum+1)
        d2scanvals.append(thisd2scanvals)
        
        thisbiasvarscan = []
        for d2 in thisd2scanvals:
            coeffd[2] = d2
            #coeffs = [coeffd,coeffc,coeffb,coeffa,coeffg]
            coeffs = [coeffd,coeffc,coeffb,coeffa]
            thisbiasvarscan.append(estimator_draw("confounderpoly",coeffs + [coeffe],0.2,estlist,numsamp=1e3,numrun=1e2))
        biasvarscan.append(thisbiasvarscan)
    
    fig, ax = plt.subplots(1,2)
    
    for i,d3 in enumerate(d3scanvals):
        print(d3scanvals[i])
        print(d2scanvals[i])
        #print(biasvarscan[0][estlist[0]])
        biasvarsestifdc = [biasvars[estlist[0]]['a'][0] for biasvars in biasvarscan[i]]
        biasvarsestrem = [biasvars[estlist[1]]['a'][0] for biasvars in biasvarscan[i]]
        print(d2scanvals[i])
        print(biasvarsestifdc)
        #lab0 = estimator_shortname[estlist[0]] + " d3=" + '{0:3.1f}'.format(d3)
        #lab1 = estimator_shortname[estlist[1]] + " d3=" + '{0:3.1f}'.format(d3)
        lab0 = "d3=" + '{0:3.1f}'.format(d3)
        lab1 = "d3=" + '{0:3.1f}'.format(d3)
        ax[0].plot(d2scanvals[i],biasvarsestifdc, color=cold3s[i], label=lab0)
        ax[1].plot(d2scanvals[i],biasvarsestrem, color=cold3s[i], label=lab1)
    ax[0].legend()
    ax[1].legend()
    ax[0].set_xlabel('d[2]')
    ax[1].set_xlabel('d[2]')
    ax[0].set_ylabel('Bias[a]')
    ax[1].set_ylabel('Bias[a]')
    ax[0].set_ylim([0.35,0.45])
    ax[1].set_ylim([-0.1,0.1])
    ax[0].set_title('IFDC')
    ax[1].set_title('e/d-improved IFDC')
    
    fig.set_size_inches((20,10), forward=False)
    fig.tight_layout()
    fig.savefig(ofile + '.pdf', format='pdf', dpi=1200)
    
def test_ISTdata(ofile="test"):
    
    estlist = ["instfdc","edremfdc"]
    
    # Input Setup
    inpath = "../IST_data/IST_corrected.csv"
    fields = ['RSBP','AGE']
    df = pd.read_csv(inpath, encoding="iso-8859-1", usecols=fields)
    
    RSBP = df['RSBP'].values # Blood pressure, as treatment X
    maxRSBP = float(max(RSBP))
    print(maxRSBP)
    df['RSBP'] = df['RSBP']/maxRSBP
    RSBP = RSBP/maxRSBP
    AGE = df['AGE'].values # Age, as confounder W
    maxAGE = float(max(AGE))
    print(maxAGE)
    df['AGE'] = df['AGE']/maxAGE
    AGE = AGE/maxAGE
    
    # Note: Age should influence blood pressure, so arrow d: W -> X already nonzero
    numsamp = len(RSBP)
    numruns = 1000
    
    d = 0.5  # d unknown
    c = 1.  # chosen value for simulation
    e = 0.2 # chosen value for simulation
    a = 1.  # chosen value for simulation
    b = 0.5  # chosen value for simulation
    uYvar = uMvar = 0.1 # chosen value for noise variance
    
    rundfs = []
    df.rename({'RSBP': 'X', 'AGE': 'W'}, axis=1, inplace=True)
    for run in range(numruns):
        uM = np.random.normal(0,np.sqrt(uMvar),numsamp)
        uY = np.random.normal(0,np.sqrt(uYvar),numsamp)
        
        M = c*RSBP + e*AGE + uM
        Y = a*M + b*AGE + uY
        
        rundf = df.copy()
        rundf['M'] = M
        rundf['Y'] = Y
        rundfs.append(rundf)

    
    # Store true causation values
    coeffdict = {}
    coeffdict['a'] = a
    coeffdict['c'] = c
    coeffdict['d'] = d
    coeffdict['e'] = e
    
    biasvarscan = estimator_dfdata(rundfs,coeffdict,estlist)
    adata = {}
    acdata = {}
    
    fig, ax = plt.subplots(2,2)
    
    for i,est in enumerate(estlist):
        print(estimator_name[est],"\n")
        print("Bias on a: ", biasvarscan[est]['a'][0])
        print("Variance on a: ", biasvarscan[est]['a'][1])
        print("Bias on ac: ", biasvarscan[est]['ac'][0])
        print("Variance on ac: ", biasvarscan[est]['ac'][1])
        print("\n")
        
        adata[est] = np.array(biasvarscan[est]['adata'])
        acdata[est] = np.array(biasvarscan[est]['acdata'])
        ax[i][0].hist(adata[est]-a, bins=20)
        ax[i][1].hist(acdata[est]-(a*c),bins=20)
        ax[i][0].set_xlabel('Bias[a]')
        ax[i][1].set_xlabel('Bias[ac]')
        ax[i][0].set_ylabel('Count')
    
    
    plt.show()      
    
    
    fig.set_size_inches((20,20), forward=False)
    fig.tight_layout()
    fig.savefig(ofile + '.pdf', format='pdf', dpi=1200)

def test_scanenoise_ISTdata(ofile="test"):
    
    estlist = ["instfdc","ratioedremfdc"]
    
    # Input Setup
    inpath = "../IST_data/IST_corrected.csv"
    fields = ['RSBP','AGE']
    df = pd.read_csv(inpath, encoding="iso-8859-1", usecols=fields)
    
    RSBP = df['RSBP'].values # Blood pressure, as treatment X
    maxRSBP = float(max(RSBP))
    print(maxRSBP)
    df['RSBP'] = df['RSBP']/maxRSBP
    RSBP = RSBP/maxRSBP
    AGE = df['AGE'].values # Age, as confounder W
    maxAGE = float(max(AGE))
    print(maxAGE)
    df['AGE'] = df['AGE']/maxAGE
    AGE = AGE/maxAGE
    
    # Note: Age should influence blood pressure, so arrow d: W -> X already nonzero
    numsamp = len(RSBP)
    numruns = 200
    
    edom = np.linspace(0.0, 3.0, 20)
    noises = [0.1, 0.5, 1.0]
    print(noises)
    numnoises = len(noises)
    colests = [np.array(list(get_color_bg(x)))/255.0 for x in np.linspace(0,1,numnoises)]
    
    d = 1.  # d unknown
    c = 1.  # chosen value for simulation
    a = 1.  # chosen value for simulation
    b = 1.  # chosen value for simulation
    
    biasvarscans = []
    
    for noise in noises:
        print("Noise: ", noise)
        biasvarscan = []
        for e in edom:
            print("e: ", e)
            rundfs = []
            df.rename({'RSBP': 'X', 'AGE': 'W'}, axis=1, inplace=True)
            for run in range(numruns):
                uM = np.random.normal(0,np.sqrt(noise),numsamp)
                uY = np.random.normal(0,np.sqrt(noise),numsamp)
                
                M = c*RSBP + e*AGE + uM
                Y = a*M + b*AGE + uY
                
                rundf = df.copy()
                rundf['M'] = M
                rundf['Y'] = Y
                rundfs.append(rundf)

    
            # Store true causation values
            coeffdict = {}
            coeffdict['a'] = a
            coeffdict['c'] = c
            coeffdict['d'] = d
            coeffdict['e'] = e
                
            biasvarscan.append(estimator_dfdata(rundfs,coeffdict,estlist))
        biasvarscans.append(biasvarscan)
    
    fig, ax = plt.subplots(1,1)
    
    for i,noise in enumerate(noises):
        ifdc_abiasscan = [biasvars[estlist[0]]['a'][0] for biasvars in biasvarscans[i]]
        #ifdc_avarscan = [biasvars[estlist[0]]['a'][1] for biasvars in biasvarscans[i]]
        rremifdc_abiasscan = [biasvars[estlist[1]]['a'][0] for biasvars in biasvarscans[i]]
        #rremifdc_avarscan = [biasvars[estlist[1]]['a'][1] for biasvars in biasvarscans[i]]
        
        #maxabias = np.max(np.array(abiasscan).flatten())
        #maxacbias = np.max(np.array(acbiasscan).flatten())
        #minabias = np.min(np.array(abiasscan).flatten())
        #minacbias = np.min(np.array(acbiasscan).flatten())
        #maxbias = np.max([maxabias,maxacbias])
        #minbias = np.min([minabias,minacbias])
        #print(maxbias)
        #print(minbias)
        #biaslevels = np.linspace(minbias, maxbias, 31)
        #biaslevels = [-10,-8,-6,-4,-2,-1,-0.8,-0.6,-0.4,-0.2,-0.1,0,0.1,0.2,0.4,0.6,0.8,1]
        thislabel = "$\sigma^2$ = " + str(noise)
        
        ax.plot(edom,ifdc_abiasscan, color=colests[i], linestyle = 'dashed')
        #ax.plot(edom,ifdc_avarscan, color=colests[i], linestyle = 'dashed', label=thislabel)
        ax.plot(edom,rremifdc_abiasscan, color=colests[i], linestyle = 'solid', label=thislabel)
        #ax.plot(edom,rremifdc_avarscan, color=colests[i], linestyle = 'solid', label=thislabel)
        
    ax.legend(prop={'size': 18})
    ax.set_xlabel('e')
    ax.set_ylabel('Bias[a]')
    ax.set_ylim([-0.05,0.3])
    
    fig.set_size_inches((18,10), forward=False)
    fig.tight_layout()
    fig.savefig(ofile + '.pdf', format='pdf', dpi=1200)
    
def test_ISTdata_ASPHEP(ofile="test"):
    
    estlist = ["instfdc","edremfdc"]
    
    yntobin = {'Y': 1., 'N': 0.}
    mlntobin = {'H': 1., 'M': 1., 'L': 0.5, 'N': 0.}
    
    # Input Setup
    inpath = "../IST_data/IST_corrected.csv"
    fields = ['RXASP','RXHEP','AGE']
    df = pd.read_csv(inpath, encoding="iso-8859-1", usecols=fields)
    
    RXASP = np.array([yntobin[val] for val in df['RXASP'].values]) # Aspirin treatment, Y/N
    RXHEP = np.array([mlntobin[val] for val in df['RXHEP'].values]) # Heparin treatment, M/L/N
    X = (RXASP + RXHEP)/2.0
    df['X'] = X

    AGE = df['AGE'].values # Age, as confounder W
    maxAGE = float(max(AGE))
    print(maxAGE)
    df['AGE'] = df['AGE']/maxAGE
    AGE = AGE/maxAGE
    
    # Note: Age should influence blood pressure, so arrow d: W -> X already nonzero
    numsamp = len(RXASP)
    numruns = 100
    
    d = 0.5  # d unknown
    c = 1.  # chosen value for simulation
    e = 0.2 # chosen value for simulation
    a = 1.  # chosen value for simulation
    b = 0.5  # chosen value for simulation
    uYvar = uMvar = 0.1 # chosen value for noise variance
    
    rundfs = []
    df.rename({'AGE': 'W'}, axis=1, inplace=True)
    for run in range(numruns):
        uM = np.random.normal(0,np.sqrt(uMvar),numsamp)
        uY = np.random.normal(0,np.sqrt(uYvar),numsamp)
        
        M = c*X + e*AGE + uM
        Y = a*M + b*AGE + uY
        
        rundf = df.copy()
        rundf['M'] = M
        rundf['Y'] = Y
        rundfs.append(rundf)

    
    # Store true causation values
    coeffdict = {}
    coeffdict['a'] = a
    coeffdict['c'] = c
    coeffdict['d'] = d
    coeffdict['e'] = e
    
    biasvarscan = estimator_dfdata(rundfs,coeffdict,estlist)
    adata = {}
    acdata = {}
    
    fig, ax = plt.subplots(2,2)
    
    for i,est in enumerate(estlist):
        print(estimator_name[est],"\n")
        print("Bias on a: ", biasvarscan[est]['a'][0])
        print("Variance on a: ", biasvarscan[est]['a'][1])
        print("Bias on ac: ", biasvarscan[est]['ac'][0])
        print("Variance on ac: ", biasvarscan[est]['ac'][1])
        print("\n")
        
        adata[est] = np.array(biasvarscan[est]['adata'])
        acdata[est] = np.array(biasvarscan[est]['acdata'])
        ax[i][0].hist(adata[est]-a, bins=20)
        ax[i][1].hist(acdata[est]-(a*c),bins=20)
        ax[i][0].set_xlabel('Bias[a]')
        ax[i][1].set_xlabel('Bias[ac]')
        ax[i][0].set_ylabel('Count')
    
    
    plt.show()      
    
    
    fig.set_size_inches((20,20), forward=False)
    fig.tight_layout()
    fig.savefig(ofile + '.pdf', format='pdf', dpi=1200)

def scanenoise_ISTdata_ASPHEP(ofile="test"):
    
    estlist = ["instfdc","edremfdc"]
    
    yntobin = {'Y': 1., 'N': 0.}
    mlntobin = {'H': 1., 'M': 1., 'L': 0.5, 'N': 0.}
    
    # Input Setup
    inpath = "../IST_data/IST_corrected.csv"
    fields = ['RXASP','RXHEP','AGE']
    df = pd.read_csv(inpath, encoding="iso-8859-1", usecols=fields)
    
    RXASP = np.array([yntobin[val] for val in df['RXASP'].values]) # Aspirin treatment, Y/N
    RXHEP = np.array([mlntobin[val] for val in df['RXHEP'].values]) # Heparin treatment, M/L/N
    X = (RXASP + RXHEP)/2.0
    df['X'] = X

    AGE = df['AGE'].values # Age, as confounder W
    maxAGE = float(max(AGE))
    print(maxAGE)
    df['AGE'] = df['AGE']/maxAGE
    AGE = AGE/maxAGE
    
    # Note: Age should influence blood pressure, so arrow d: W -> X already nonzero
    numsamp = len(RXASP)
    numruns = 20
    
    edom = np.linspace(0.1, 1.0, 5)
    noisedom = np.linspace(0.1, 1.0, 5)
    
    d = 0.5  # d unknown
    c = 1.  # chosen value for simulation
    a = 1.  # chosen value for simulation
    b = 0.5  # chosen value for simulation
    
    biasvarscans = []
    
    for e in edom:
        biasvarscan = []
        for noise in noisedom:
            rundfs = []
            df.rename({'AGE': 'W'}, axis=1, inplace=True)
            for run in range(numruns):
                uM = np.random.normal(0,np.sqrt(noise),numsamp)
                uY = np.random.normal(0,np.sqrt(noise),numsamp)
                
                M = c*X + e*AGE + uM
                Y = a*M + b*AGE + uY
                
                rundf = df.copy()
                rundf['M'] = M
                rundf['Y'] = Y
                rundfs.append(rundf)

    
            # Store true causation values
            coeffdict = {}
            coeffdict['a'] = a
            coeffdict['c'] = c
            coeffdict['d'] = d
            coeffdict['e'] = e
                
            biasvarscan.append(estimator_dfdata(rundfs,coeffdict,estlist))
        biasvarscans.append(biasvarscan)
    
    fig, ax = plt.subplots(2,2)
    
    for i,est in enumerate(estlist):
        abiasscan = [[biasvars[est]['a'][0] for biasvars in biasvarscan] for biasvarscan in biasvarscans]
        acbiasscan = [[biasvars[est]['ac'][0] for biasvars in biasvarscan] for biasvarscan in biasvarscans]
        
        maxabias = np.max(np.array(abiasscan).flatten())
        maxacbias = np.max(np.array(acbiasscan).flatten())
        minabias = np.min(np.array(abiasscan).flatten())
        minacbias = np.min(np.array(acbiasscan).flatten())
        maxbias = np.max([maxabias,maxacbias])
        minbias = np.min([minabias,minacbias])
        print(maxbias)
        print(minbias)
        
        biaslevels = np.linspace(minbias, maxbias, 31)
        #biaslevels = [-10,-8,-6,-4,-2,-1,-0.8,-0.6,-0.4,-0.2,-0.1,0,0.1,0.2,0.4,0.6,0.8,1]
        
        cf00 = ax[i][0].contour(edom, noisedom, abiasscan, biaslevels, linestyles="solid")
        cf01 = ax[i][1].contour(edom, noisedom, acbiasscan, biaslevels, linestyles="solid")
        
        ax[i][0].clabel(cf00, cf00.levels, inline=True, fmt=fmt, fontsize=10)
        ax[i][1].clabel(cf01, cf01.levels, inline=True, fmt=fmt, fontsize=10)
        
    ax[0][0].set_ylabel('noise')
    ax[1][0].set_ylabel('noise')
    ax[1][0].set_xlabel('e')
    ax[1][1].set_xlabel('e')
  
    plt.show()      
    
    
    fig.set_size_inches((20,20), forward=False)
    fig.tight_layout()
    fig.savefig(ofile + '.pdf', format='pdf', dpi=1200)

    
# Categorical to Numerical Input Conversion
def yntobin(word):
    if word!=word:
        return 0.
    yntobindict = {'Y': 1., 'N': 0., 'U': 0., 'u': 0.}
    return yntobindict[word]
    
def mlntobin(word):
    if word!=word:
        return 0.
    mlntobindict = {'H': 1., 'M': 1., 'L': 0.5, 'N': 0.}
    return mlntobindict[word]

def identity(word):
    if word!=word:
        return 0.
    return word

def test_ISTdata_ASPHEP_14day(ofile="test"):
    
    estlist = ["instfdc","edremfdc"]
    
    # Input Setup
    inpath = "../IST_data/IST_corrected.csv"
    fields = ['RXASP','RXHEP','DDIAGISC','DDIAGHA','DDIAGUN','DRSISC','DRSH','DRSUNK','DPE','DDEAD','FDEAD','AGE']
    fielddict = {'RXASP':       yntobin,
                 'RXHEP':       mlntobin,
                 'DDIAGISC':    yntobin,
                 'DDIAGHA':     yntobin,
                 'DDIAGUN':     yntobin,
                 'DRSISC':      yntobin,
                 'DRSH':        yntobin,
                 'DRSUNK':      yntobin,
                 'DPE':         yntobin,
                 'DDEAD':       yntobin,
                 'FDEAD':       yntobin,
                 'AGE':         identity}
                 
    df = pd.read_csv(inpath, encoding="iso-8859-1", usecols=fields)
    #print(df[1:30])
    
    #df['DDIAGISC'].value_counts().plot(kind='bar')
    listvals = list(df['DDIAGISC'].values)
    nanvals = [(1 if (val!=val) else 0) for val in listvals]
    print(sum(nanvals))
    plt.show()
    
    valdf = {}
    for field in fields:
        print(field)
        valdf[field] = np.array([fielddict[field](val) for val in df[field].values])
    
    X = (valdf['RXASP'] + valdf['RXHEP']) / 2.0
    df['X'] = X
    
    M = (valdf['DDIAGISC'] + valdf['DDIAGHA'] + valdf['DDIAGUN'] + valdf['DRSISC'] + valdf['DRSH'] + valdf['DRSUNK'] + valdf['DPE'] + valdf['DDEAD']) / 3.0
    df['M'] = M
    
    #df.hist(column='X',bins=100)
    #plt.show()
    #df.hist(column='M',bins=100)
    #plt.show()

    AGE = df['AGE'].values # Age, as confounder W
    maxAGE = float(max(AGE))
    print(maxAGE)
    df['AGE'] = df['AGE']/maxAGE
    AGE = AGE/maxAGE
    
    # Note: Age should influence blood pressure, so arrow d: W -> X already nonzero
    numsamp = len(valdf['RXASP'])
    numruns = 1000
    
    d = 0.5  # d unknown
    c = 1.  # chosen value for simulation
    e = 0.2 # chosen value for simulation
    a = 1.  # chosen value for simulation
    b = 0.5  # chosen value for simulation
    uYvar = uMvar = 0.1 # chosen value for noise variance
    
    rundfs = []
    df.rename({'AGE': 'W'}, axis=1, inplace=True)
    for run in range(numruns):
        #uM = np.random.normal(0,np.sqrt(uMvar),numsamp)
        uY = np.random.normal(0,np.sqrt(uYvar),numsamp)
        
        #M = c*X + e*AGE + uM
        Y = a*M + b*AGE + uY
        
        rundf = df.copy()
        #rundf['M'] = M
        rundf['Y'] = Y
        rundfs.append(rundf)

    
    # Store true causation values
    coeffdict = {}
    coeffdict['a'] = a
    coeffdict['c'] = c
    coeffdict['d'] = d
    coeffdict['e'] = e
    
    biasvarscan = estimator_dfdata(rundfs,coeffdict,estlist)
    adata = {}
    acdata = {}
    
    fig, ax = plt.subplots(2,2)
    
    for i,est in enumerate(estlist):
        print(estimator_name[est],"\n")
        print("Bias on a: ", biasvarscan[est]['a'][0])
        print("Variance on a: ", biasvarscan[est]['a'][1])
        print("Bias on ac: ", biasvarscan[est]['ac'][0])
        print("Variance on ac: ", biasvarscan[est]['ac'][1])
        print("\n")
        
        adata[est] = np.array(biasvarscan[est]['adata'])
        acdata[est] = np.array(biasvarscan[est]['acdata'])
        ax[i][0].hist(adata[est]-a, bins=20)
        ax[i][1].hist(acdata[est]-(a*c),bins=20)
        ax[i][0].set_xlabel('Bias[a]')
        ax[i][1].set_xlabel('Bias[ac]')
        ax[i][0].set_ylabel('Count')
    
    
    plt.show()      
    
    
    fig.set_size_inches((20,20), forward=False)
    fig.tight_layout()
    fig.savefig(ofile + '.pdf', format='pdf', dpi=1200)

def test():
    #biasvars = estimator_draw("confounder",[1,1,1,1,1],1,["truefdc","instfdc"])
    #estlist = ["pi-instfdc","pi-edresfdc","pi-edremfdc","pi-subedresfdc","pi-subedremfdc"]
    estlist = ["pi-instfdc","pi-edresfdc","pi-edremfdc","pi-instfdc-x","pi-edresfdc-x","pi-edremfdc-x"]
    # d-c-b-a-g-e
    biasvars = estimator_draw("instrument",[1,1,1,1,1,1],1,estlist,numsamp=1e4,numrun=2e2)
    
    for est in estlist:
        print(estimator_name[est])
        print(biasvars[est])
        print("\n")
        
    scanvals = np.linspace(scanrange[0],scanrange[1],scannum+1)
    fig, ax = plt.subplots(1,2)
    
    for i,est in enumerate(estlist):
        print(colests[i])
        biasvarsesta = biasvars[est]['ac'][0]
        biasvarsestac = biasvars[est]['ac'][1]

        print(estimator_name[est])
        print(biasvars[est])
        print("\n")

parser = argparse.ArgumentParser()

parser.add_argument("--outfile", "-o", help="output file name", default="marg_out")

# Read arguments from the command line
args = parser.parse_args()
ofile = args.outfile

# List of colormap options: https://matplotlib.org/3.1.1/gallery/color/colormap_reference.html
                  
#test()
#test_shifted()
#test_escan_shifted(shift=1,ofile=ofile)
test_scanenoise_shifted(shift=1,ofile=ofile)
#test_poly(ofile=ofile)
#test_dscan_poly(ofile=ofile)
#test_d3scan_poly(ofile=ofile)
#test_e3scan_poly(ofile=ofile)
#test_ISTdata(ofile=ofile)
#test_ISTdata_ASPHEP(ofile=ofile)
#test_ISTdata_ASPHEP_14day(ofile=ofile)

#test_scanenoise_ISTdata(ofile=ofile)

#scanenoise_ISTdata_ASPHEP(ofile=ofile)

#get_color(0.5)

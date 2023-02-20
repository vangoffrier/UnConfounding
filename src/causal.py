import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics as stat
import argparse
import os
import matplotlib.colors as clr
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import models
import estimators

np.random.seed(4812)

plt.rcParams.update({'font.size': 24})

def ModelCompare():

    a,b,c,d,e = 1,1,1,1,4

    confmodel = models.generate_ConfounderModel([d,c,a,b])
    confmodelimperf = models.generate_ConfounderModel([d,c,a,b],imperfect=e)

    runvalues = []
    runvaluesimperfect = []

    numsamp = int(1e4)
    numruns = int(100)
    for run in range(numruns):
        print("Run: ",run," of ",numruns)
        confsamples = confmodel.sample(numsamp)
        confsamplesimperf = confmodelimperf.sample(numsamp)
        #for samp in confsamples[-3:-1]:
        #    print(samp)
        
        values = [[x for x in samp.values()] for samp in confsamples]
        valuesimperfect = [[x for x in samp.values()] for samp in confsamplesimperf]

        valuesT = np.array(values).T.tolist()
        valuesimperfectT = np.array(valuesimperfect).T.tolist()
        
        runvalues.append(valuesT)
        runvaluesimperfect.append(valuesimperfectT)

    nodedict = dict([(node,confmodel.nodes.index(node)) for node in confmodel.nodes])
    sampledicts = [dict([(node,runvalues[run][nodedict[node]]) for node in confmodel.nodes]) for run in range(numruns)]

    Wsamples = [sampledicts[run]["W"] for run in range(numruns)]
    Xsamples = [sampledicts[run]["X"] for run in range(numruns)]
    Msamples = [sampledicts[run]["M"] for run in range(numruns)]
    Ysamples = [sampledicts[run]["Y"] for run in range(numruns)]

    print("\n\nTRUE FDC")
    cEstimators = [np.dot(Xsamples[run],Msamples[run])/np.dot(Xsamples[run],Xsamples[run]) for run in range(numruns)]
    cMean = stat.mean(cEstimators)
    cBias = cMean - 1
    cVar = stat.variance(cEstimators)
    print("mean(c) = ","{:.6f}".format(cMean))
    print("bias(c) = ","{:.6f}".format(cBias))
    print("var(c) = ","{:.6f}".format(cVar))

    print("Theoretical mean(c) = 0")
    THcVar = 1./((numsamp-2)*(d**2 + 1))
    print("Theoretical var(c) = ","{:.6f}".format(THcVar))


    aEstimators = [(np.dot(Msamples[run],Ysamples[run]) * np.dot(Xsamples[run],Xsamples[run])- np.dot(Msamples[run],Xsamples[run]) * np.dot(Xsamples[run],Ysamples[run])) / (np.dot(Msamples[run],Msamples[run]) * np.dot(Xsamples[run],Xsamples[run])- np.dot(Msamples[run],Xsamples[run]) * np.dot(Xsamples[run],Msamples[run])) for run in range(numruns)]
    aMean = stat.mean(aEstimators)
    aBias = aMean - 1
    aVar = stat.variance(aEstimators)
    print("mean(a) = ","{:.6f}".format(aMean))
    print("bias(a) = ","{:.6f}".format(aBias))
    print("var(a) = ","{:.6f}".format(aVar))

    print("Theoretical bias(a) = 0")
    THaVar = (b**2 + (d**2 + 1))/((numsamp-3)*(d**2 + 1))
    print("Theoretical var(a) = ","{:.4f}".format(THaVar))

    acEstimators = [aEstimators[i] * cEstimators[i] for i in range(numruns) ]
    acMean = stat.mean(acEstimators)
    acBias = acMean - 1
    acVar = stat.variance(acEstimators)
    print("mean(ac) = ","{:.6f}".format(acMean))
    print("bias(ac) = ","{:.6f}".format(acBias))
    print("var(ac) = ","{:.6f}".format(acVar))

    print("Theoretical bias(ac) = 0")
    #THacVar = (b**2 + (d**2 + 1))/((numsamp-3)*(d**2 + 1))
    #print("Theoretical var(a) = ","{:.4f}".format(THaVar))

    print("\n\nAPPROXIMATE FDC")
    aApproxEstimators = [np.dot(Msamples[run],Ysamples[run]) / np.dot(Msamples[run],Msamples[run]) for run in range(numruns)]
    aApproxMean = stat.mean(aApproxEstimators)
    aApproxBias = aApproxMean - 1
    aApproxVar = stat.variance(aApproxEstimators)
    print("mean(a) = ","{:.6f}".format(aApproxMean))
    print("bias(a) = ","{:.6f}".format(aApproxBias))
    print("var(a) = ","{:.6f}".format(aApproxVar))

    THaBias = b*c*d / (1 + c**2 * (d**2 + 1))
    print("Theoretical bias(a) = ","{:.4f}".format(THaBias))

    acApproxEstimators = [aApproxEstimators[i] * cEstimators[i] for i in range(numruns) ]
    acApproxMean = stat.mean(acApproxEstimators)
    acApproxBias = acApproxMean - 1
    acApproxVar = stat.variance(acApproxEstimators)
    print("mean(ac) = ","{:.6f}".format(acApproxMean))
    print("bias(ac) = ","{:.6f}".format(acApproxBias))
    print("var(ac) = ","{:.6f}".format(acApproxVar))

    print("\n\nAPPROXIMATE APPROXIMATE FDC (Direct Estimator)")
    ac2ApproxEstimators = [np.dot(Xsamples[run],Ysamples[run]) / np.dot(Xsamples[run],Xsamples[run]) for run in range(numruns)]
    ac2ApproxMean = stat.mean(ac2ApproxEstimators)
    ac2ApproxBias = ac2ApproxMean - 1
    ac2ApproxVar = stat.variance(ac2ApproxEstimators)
    print("mean(ac) = ","{:.6f}".format(ac2ApproxMean))
    print("bias(ac) = ","{:.6f}".format(ac2ApproxBias))
    print("var(ac) = ","{:.6f}".format(ac2ApproxVar))

    THacBias = b*d / (d**2 + 1)
    print("Theoretical bias(ac) = ","{:.4f}".format(THacBias))




    print("\n\nCONFOUNDED MEDIATOR")
    nodedictimperf = dict([(node,confmodelimperf.nodes.index(node)) for node in confmodelimperf.nodes])
    sampledictsimperf = [dict([(node,runvaluesimperfect[run][nodedictimperf[node]]) for node in confmodelimperf.nodes]) for run in range(numruns)]

    Wsamples = [sampledictsimperf[run]["W"] for run in range(numruns)]
    Xsamples = [sampledictsimperf[run]["X"] for run in range(numruns)]
    Msamples = [sampledictsimperf[run]["M"] for run in range(numruns)]
    Ysamples = [sampledictsimperf[run]["Y"] for run in range(numruns)]

    cEstimators = [np.dot(Xsamples[run],Msamples[run])/np.dot(Xsamples[run],Xsamples[run]) for run in range(numruns)]
    cMean = stat.mean(cEstimators)
    cBias = cMean - 1
    cVar = stat.variance(cEstimators)
    print("mean(c) = ","{:.6f}".format(cMean))
    print("bias(c) = ","{:.6f}".format(cBias))
    print("var(c) = ","{:.6f}".format(cVar))

    THcMean = d*e / (d**2 + 1)
    print("Theoretical bias(c) = ","{:.6f}".format(THcMean))

    print("Theoretical var(c) = ?")


    aEstimators = [(np.dot(Msamples[run],Ysamples[run]) * np.dot(Xsamples[run],Xsamples[run])- np.dot(Msamples[run],Xsamples[run]) * np.dot(Xsamples[run],Ysamples[run])) / (np.dot(Msamples[run],Msamples[run]) * np.dot(Xsamples[run],Xsamples[run])- np.dot(Msamples[run],Xsamples[run]) * np.dot(Xsamples[run],Msamples[run])) for run in range(numruns)]
    aMean = stat.mean(aEstimators)
    aBias = aMean - 1
    aVar = stat.variance(aEstimators)
    print("mean(a) = ","{:.6f}".format(aMean))
    print("bias(a) = ","{:.6f}".format(aBias))
    print("var(a) = ","{:.6f}".format(aVar))

    THaMean = b*e / (e**2 + d**2 + 1)
    print("Theoretical bias(a) = ","{:.6f}".format(THaMean))

    print("Theoretical var(a) = ?")


    bins = np.linspace(-10,10,100)



    fig,ax = plt.subplots(1,2)

    for i in range(confmodel.numnodes):
        ax[0].hist(valuesT[i],bins,alpha=0.5,label=confmodel.nodes[i])
        
    for i in range(confmodel.numnodes):
        ax[1].hist(valuesimperfectT[i],bins,alpha=0.5,label=confmodel.nodes[i])
        
    ax[0].legend(loc='upper right')
    ax[1].legend(loc='upper right')
    plt.show()

    #print(model.coeffdict)
    #print(model.get_children("U"))

    print(confmodel.get_node_distribution("Y"))
    print(confmodelimperf.get_node_distribution("Y"))


    print(confmodel.get_node_distribution("Y"))
    print(confmodelimperf.get_node_distribution("Y"))
    
    #atheymodel.plot()
    #atheymodelimperfect.plot()

def acScan_trueFDC(arange,crange,anum,cnum,numsamp=1e4,numruns=1e2,e=0,b=1,d=1,noise=1):

    avals = np.linspace(arange[0],arange[1],anum)
    cvals = np.linspace(crange[0],crange[1],cnum)
    
    cBias = np.empty([anum,cnum])
    cVar = np.empty([anum,cnum])
    aBias = np.empty([anum,cnum])
    aVar = np.empty([anum,cnum])
    acBias = np.empty([anum,cnum])
    acVar = np.empty([anum,cnum])
    
    for ai,a in enumerate(avals):
        for ci,c in enumerate(cvals):

            confmodel = models.generate_ConfounderModel([d,c,a,b,e],noise=noise)

            runvalues = []

            for run in range(numruns):
            
                #BEGIN (numruns) THREADS, VARS runvalues, ?samples
                
                os.system('cls||clear')
                print("Running True FDC Estimator")
                print("a = ", a, " in range [", arange[0], ", ", arange[1], "]")
                print("c = ", c, " in range [", crange[0], ", ", crange[1], "]")
                print("Run: ",run," of ",numruns)
                confsamples = confmodel.sample(numsamp)
                #for samp in confsamples[-3:-1]:
                #    print(samp)
                
                values = [[x for x in samp.values()] for samp in confsamples]
                valuesT = np.array(values).T.tolist()
                runvalues.append(valuesT)
                
                #CONT THREADS

            nodedict = dict([(node,confmodel.nodes.index(node)) for node in confmodel.nodes])
            
            #CONT THREADS
            
            sampledicts = [dict([(node,runvalues[run][nodedict[node]]) for node in confmodel.nodes]) for run in range(numruns)]

            Wsamples = [sampledicts[run]["W"] for run in range(numruns)]
            Xsamples = [sampledicts[run]["X"] for run in range(numruns)]
            Msamples = [sampledicts[run]["M"] for run in range(numruns)]
            Ysamples = [sampledicts[run]["Y"] for run in range(numruns)]

            cEstimators = [np.dot(Xsamples[run],Msamples[run])/np.dot(Xsamples[run],Xsamples[run]) for run in range(numruns)]
            
            aEstimators = [(np.dot(Msamples[run],Ysamples[run]) * np.dot(Xsamples[run],Xsamples[run])- np.dot(Msamples[run],Xsamples[run]) * np.dot(Xsamples[run],Ysamples[run])) / (np.dot(Msamples[run],Msamples[run]) * np.dot(Xsamples[run],Xsamples[run])- np.dot(Msamples[run],Xsamples[run]) * np.dot(Xsamples[run],Msamples[run])) for run in range(numruns)]
            
            #END THREADS, MERGE (numruns) cEstimators, aEstimators
            
            cMean = stat.mean(cEstimators)
            cBias[ai,ci] = cMean - c
            cVar[ai,ci] = stat.variance(cEstimators)

            print("Theoretical mean(c) = 0")
            THcVar = 1./((numsamp-2)*(d**2 + 1))
            print("Theoretical var(c) = ","{:.6f}".format(THcVar))


            aMean = stat.mean(aEstimators)
            aBias[ai,ci] = aMean - a
            aVar[ai,ci] = stat.variance(aEstimators)
      
            print("Theoretical bias(a) = 0")
            THaVar = (b**2 + (d**2 + 1))/((numsamp-3)*(d**2 + 1))
            print("Theoretical var(a) = ","{:.4f}".format(THaVar))

            acEstimators = [aEstimators[i] * cEstimators[i] for i in range(numruns) ]
            acMean = stat.mean(acEstimators)
            acBias[ai,ci] = acMean - a*c
            acVar[ai,ci] = stat.variance(acEstimators)

            print("Theoretical bias(ac) = 0")
            THacVar = (b**2 + (d**2 + 1))/((numsamp-3)*(d**2 + 1))
            print("Theoretical var(a) = ","{:.4f}".format(THaVar))
            
    return cBias, cVar, aBias, aVar, acBias, acVar
    
def acScan_approxFDC(arange,crange,anum,cnum,numsamp=1e4,numruns=1e2,e=0,b=1,d=1,noise=1):

    avals = np.linspace(arange[0],arange[1],anum)
    cvals = np.linspace(crange[0],crange[1],cnum)
    
    cBias = np.empty([anum,cnum])
    cVar = np.empty([anum,cnum])
    aBias = np.empty([anum,cnum])
    aVar = np.empty([anum,cnum])
    acBias = np.empty([anum,cnum])
    acVar = np.empty([anum,cnum])
    
    for ai,a in enumerate(avals):
        for ci,c in enumerate(cvals):

            confmodel = models.generate_ConfounderModel([d,c,a,b,e],noise=noise)

            runvalues = []

            for run in range(numruns):
                os.system('cls||clear')
                print("Running Approximate FDC Estimator")
                print("a = ", a, " in range [", arange[0], ", ", arange[1], "]")
                print("c = ", c, " in range [", crange[0], ", ", crange[1], "]")
                print("Run: ",run," of ",numruns)
                confsamples = confmodel.sample(numsamp)
                #for samp in confsamples[-3:-1]:
                #    print(samp)
                
                values = [[x for x in samp.values()] for samp in confsamples]
                valuesT = np.array(values).T.tolist()
                runvalues.append(valuesT)

            nodedict = dict([(node,confmodel.nodes.index(node)) for node in confmodel.nodes])
            sampledicts = [dict([(node,runvalues[run][nodedict[node]]) for node in confmodel.nodes]) for run in range(numruns)]

            Wsamples = [sampledicts[run]["W"] for run in range(numruns)]
            Xsamples = [sampledicts[run]["X"] for run in range(numruns)]
            Msamples = [sampledicts[run]["M"] for run in range(numruns)]
            Ysamples = [sampledicts[run]["Y"] for run in range(numruns)]

            cEstimators = [np.dot(Xsamples[run],Msamples[run])/np.dot(Xsamples[run],Xsamples[run]) for run in range(numruns)]
            cMean = stat.mean(cEstimators)
            cBias[ai,ci] = cMean - c
            cVar[ai,ci] = stat.variance(cEstimators)

            #print("Theoretical mean(c) = 0")
            THcVar = 1./((numsamp-2)*(d**2 + 1))
            #print("Theoretical var(c) = ","{:.6f}".format(THcVar))

            aApproxEstimators = [np.dot(Msamples[run],Ysamples[run]) / np.dot(Msamples[run],Msamples[run]) for run in range(numruns)]
            aApproxMean = stat.mean(aApproxEstimators)
            aBias[ai,ci] = aApproxMean - a
            aVar[ai,ci] = stat.variance(aApproxEstimators)

            THaBias = b*c*d / (1 + c**2 * (d**2 + 1))
            #print("Theoretical bias(a) = ","{:.4f}".format(THaBias))

            acApproxEstimators = [aApproxEstimators[i] * cEstimators[i] for i in range(numruns) ]
            acApproxMean = stat.mean(acApproxEstimators)
            acBias[ai,ci] = acApproxMean - a*c
            acVar[ai,ci] = stat.variance(acApproxEstimators)
                   
    return cBias, cVar, aBias, aVar, acBias, acVar
    
    
def acScan_instrumentalFDC(arange,crange,anum,cnum,numsamp=1e4,numruns=1e2,e=0,b=1,d=1,noise=1):

    avals = np.linspace(arange[0],arange[1],anum)
    cvals = np.linspace(crange[0],crange[1],cnum)
    
    b = 1
    d = 1
    
    cBias = np.empty([anum,cnum])
    cVar = np.empty([anum,cnum])
    aBias = np.empty([anum,cnum])
    aVar = np.empty([anum,cnum])
    acBias = np.empty([anum,cnum])
    acVar = np.empty([anum,cnum])
    
    for ai,a in enumerate(avals):
        for ci,c in enumerate(cvals):

            confmodel = models.generate_ConfounderModel([d,c,a,b,e],noise=noise)

            runvalues = []

            for run in range(numruns):
                os.system('cls||clear')
                print("Running Instrumental FDC Estimator")
                print("a = ", a, " in range [", arange[0], ", ", arange[1], "]")
                print("c = ", c, " in range [", crange[0], ", ", crange[1], "]")
                print("Run: ",run," of ",numruns)
                confsamples = confmodel.sample(numsamp)
                #for samp in confsamples[-3:-1]:
                #    print(samp)
                
                values = [[x for x in samp.values()] for samp in confsamples]
                valuesT = np.array(values).T.tolist()
                runvalues.append(valuesT)

            nodedict = dict([(node,confmodel.nodes.index(node)) for node in confmodel.nodes])
            sampledicts = [dict([(node,runvalues[run][nodedict[node]]) for node in confmodel.nodes]) for run in range(numruns)]

            Wsamples = [sampledicts[run]["W"] for run in range(numruns)]
            Xsamples = [sampledicts[run]["X"] for run in range(numruns)]
            Msamples = [sampledicts[run]["M"] for run in range(numruns)]
            Ysamples = [sampledicts[run]["Y"] for run in range(numruns)]

            # Calculate the regression slope from x -> m (c)
            cEstimators = [np.dot(Xsamples[run],Msamples[run])/np.dot(Xsamples[run],Xsamples[run]) for run in range(numruns)]
            
            # Calculate the residuals um for each datapoint {x,m}
            uMsamples = [[Msamples[run][samp] - cEstimators[run] * Xsamples[run][samp] for samp in range(numsamp)] for run in range(numruns)]
            
            # Use um as an instrumental variable to estimate m -> y (a):
            # A) regress y on um
            
            YuMslope = [np.dot(Ysamples[run],uMsamples[run])/np.dot(uMsamples[run],uMsamples[run]) for run in range(numruns)]
            
            # B) regress m on um
            
            MuMslope = [np.dot(Msamples[run],uMsamples[run])/np.dot(uMsamples[run],uMsamples[run]) for run in range(numruns)]
            
            # C) divide these coefficients
            
            aInstEstimators = [YuMslope[run]/MuMslope[run] for run in range(numruns)]
            #aInstEstimators = MuMslope     # to run this test, switch aBias to aInstMean - 1
            
            cMean = stat.mean(cEstimators)
            cBias[ai,ci] = cMean - c
            cVar[ai,ci] = stat.variance(cEstimators)

            #print("Theoretical mean(c) = 0")
            THcVar = 1./((numsamp-2)*(d**2 + 1))
            #print("Theoretical var(c) = ","{:.6f}".format(THcVar))

            aInstMean = stat.mean(aInstEstimators)
            aBias[ai,ci] = aInstMean - a
            aVar[ai,ci] = stat.variance(aInstEstimators)

            THaBias = b*c*d / (1 + c**2 * (d**2 + 1))
            #print("Theoretical bias(a) = ","{:.4f}".format(THaBias))

            acInstEstimators = [aInstEstimators[i] * cEstimators[i] for i in range(numruns) ]
            acInstMean = stat.mean(acInstEstimators)
            acBias[ai,ci] = acInstMean - a*c
            acVar[ai,ci] = stat.variance(acInstEstimators)
                   
    return cBias, cVar, aBias, aVar, acBias, acVar
 
def acScan_instrumentalFDCwithX(arange,crange,anum,cnum,numsamp=1e4,numruns=1e2,e=0,b=1,d=1,noise=1):
 
    avals = np.linspace(arange[0],arange[1],anum)
    cvals = np.linspace(crange[0],crange[1],cnum)
    
    b = 1
    d = 1
    
    cBias = np.empty([anum,cnum])
    cVar = np.empty([anum,cnum])
    aBias = np.empty([anum,cnum])
    aVar = np.empty([anum,cnum])
    acBias = np.empty([anum,cnum])
    acVar = np.empty([anum,cnum])
    
    for ai,a in enumerate(avals):
        for ci,c in enumerate(cvals):

            confmodel = models.generate_ConfounderModel([d,c,a,b,e],noise=noise)

            runvalues = []

            for run in range(numruns):
                os.system('cls||clear')
                print("Running Instrumental FDC Estimator")
                print("a = ", a, " in range [", arange[0], ", ", arange[1], "]")
                print("c = ", c, " in range [", crange[0], ", ", crange[1], "]")
                print("Run: ",run," of ",numruns)
                confsamples = confmodel.sample(numsamp)
                #for samp in confsamples[-3:-1]:
                #    print(samp)
                
                values = [[x for x in samp.values()] for samp in confsamples]
                valuesT = np.array(values).T.tolist()
                runvalues.append(valuesT)

            nodedict = dict([(node,confmodel.nodes.index(node)) for node in confmodel.nodes])
            sampledicts = [dict([(node,runvalues[run][nodedict[node]]) for node in confmodel.nodes]) for run in range(numruns)]

            Wsamples = [sampledicts[run]["W"] for run in range(numruns)]
            Xsamples = [sampledicts[run]["X"] for run in range(numruns)]
            Msamples = [sampledicts[run]["M"] for run in range(numruns)]
            Ysamples = [sampledicts[run]["Y"] for run in range(numruns)]

            # Calculate the regression slope from x -> m (c)
            cEstimators = [np.dot(Xsamples[run],Msamples[run])/np.dot(Xsamples[run],Xsamples[run]) for run in range(numruns)]
            
            # Calculate the residuals um for each datapoint {x,m}
            uMsamples = [[Msamples[run][samp] - cEstimators[run] * Xsamples[run][samp] for samp in range(numsamp)] for run in range(numruns)]
            
            # Use um as an instrumental variable to estimate m -> y (a):
            # A) regress y on um
            
            YuMslope = [ (np.dot(uMsamples[run],Ysamples[run]) * np.dot(Xsamples[run],Xsamples[run]) - np.dot(uMsamples[run],Xsamples[run]) * np.dot(Xsamples[run],Ysamples[run]) ) / (np.dot(uMsamples[run],uMsamples[run]) * np.dot(Xsamples[run],Xsamples[run]) - np.dot(uMsamples[run],Xsamples[run])**2) for run in range(numruns)]
            
            # B) regress m on um
            
            MuMslope = [ (np.dot(uMsamples[run],Msamples[run]) * np.dot(Xsamples[run],Xsamples[run]) - np.dot(uMsamples[run],Xsamples[run]) * np.dot(Xsamples[run],Msamples[run]) ) / (np.dot(uMsamples[run],uMsamples[run]) * np.dot(Xsamples[run],Xsamples[run]) - np.dot(uMsamples[run],Xsamples[run])**2) for run in range(numruns)]
            
            # C) divide these coefficients
            
            aInstEstimators = [YuMslope[run]/MuMslope[run] for run in range(numruns)]
            #aInstEstimators = MuMslope     # to run this test, switch aBias to aInstMean - 1
            
            cMean = stat.mean(cEstimators)
            cBias[ai,ci] = cMean - c
            cVar[ai,ci] = stat.variance(cEstimators)

            #print("Theoretical mean(c) = 0")
            THcVar = 1./((numsamp-2)*(d**2 + 1))
            #print("Theoretical var(c) = ","{:.6f}".format(THcVar))

            aInstMean = stat.mean(aInstEstimators)
            aBias[ai,ci] = aInstMean - a
            aVar[ai,ci] = stat.variance(aInstEstimators)

            THaBias = b*c*d / (1 + c**2 * (d**2 + 1))
            #print("Theoretical bias(a) = ","{:.4f}".format(THaBias))

            acInstEstimators = [aInstEstimators[i] * cEstimators[i] for i in range(numruns) ]
            acInstMean = stat.mean(acInstEstimators)
            acBias[ai,ci] = acInstMean - a*c
            acVar[ai,ci] = stat.variance(acInstEstimators)
                   
    return cBias, cVar, aBias, aVar, acBias, acVar   

def acScan_instrumentalFDCedivd_CM(arange,crange,anum,cnum,tfdc,numsamp=1e4,numruns=1e2,e=1,b=1,d=1,noise=1):

    avals = np.linspace(arange[0],arange[1],anum)
    cvals = np.linspace(crange[0],crange[1],cnum)
    
    #b = 1
    #d = 1
    
    cBias = np.empty([anum,cnum])
    cVar = np.empty([anum,cnum])
    aBias = np.empty([anum,cnum])
    aVar = np.empty([anum,cnum])
    acBias = np.empty([anum,cnum])
    acVar = np.empty([anum,cnum])
    
    for ai,a in enumerate(avals):
        for ci,c in enumerate(cvals):

            # graph edges: [("V","X"),("W","X"),("X","M"),("M","Y"),("W","Y"),("W","M")]
            confmodel = models.generate_ConfounderModel([d,c,a,b,e],noise=noise)

            runvalues = []

            for run in range(numruns):
            
                #BEGIN (numruns) THREADS, VARS runvalues, ?samples
                
                os.system('cls||clear')
                print("Running Instrumental ac Estimators")
                print("a = ", a, " in range [", arange[0], ", ", arange[1], "]")
                print("c = ", c, " in range [", crange[0], ", ", crange[1], "]")
                print("Run: ",run," of ",numruns)
                
                #if run>1:
                #    print("Mean g estimates:")
                #    print( np.mean(XVslope[run-1]))
                #    
                #    print("Mean c estimates:")
                #    print( np.mean(cInstEstimators[run-1]) )
                    
                
                confsamples = confmodel.sample(numsamp)
                #for samp in confsamples[-3:-1]:
                #    print(samp)
                
                values = [[x for x in samp.values()] for samp in confsamples]
                valuesT = np.array(values).T.tolist()
                runvalues.append(valuesT)
                
                #CONT THREADS

            nodedict = dict([(node,confmodel.nodes.index(node)) for node in confmodel.nodes])
            
            #CONT THREADS
            
            sampledicts = [dict([(node,runvalues[run][nodedict[node]]) for node in confmodel.nodes]) for run in range(numruns)]


            Wsamples = [sampledicts[run]["W"] for run in range(numruns)]
            Xsamples = [sampledicts[run]["X"] for run in range(numruns)]
            Msamples = [sampledicts[run]["M"] for run in range(numruns)]
            Ysamples = [sampledicts[run]["Y"] for run in range(numruns)]
            
            # Calculate the regression slope from x -> m (c + e/d)
            cEstimators = [np.dot(Xsamples[run],Msamples[run])/np.dot(Xsamples[run],Xsamples[run]) for run in range(numruns)]
            
            # Calculate the residuals eW + um for each datapoint {x,m}
            eWplusuMsamples = [[Msamples[run][samp] - cEstimators[run] * Xsamples[run][samp] for samp in range(numsamp)] for run in range(numruns)]


            if tfdc:  # This is the standard FDC for this graph, nothing changes
                
                uMsamples = eWplusuMsamples

            if not tfdc:   # We compare with our improved e/d approach
        
                # Assume we already know some perfect e/d estimator
                edivd = e/d
                
                # Calculate the residuals um for each datapoint {x,m}
                uMsamples = [[eWplusuMsamples[run][samp] - edivd * Xsamples[run][samp] for samp in range(numsamp)] for run in range(numruns)]
            
                     
            # Use um as an instrumental variable to estimate m -> y (a):
            # A) regress y on um
            
            YuMslope = [ (np.dot(uMsamples[run],Ysamples[run]) * np.dot(Xsamples[run],Xsamples[run]) - np.dot(uMsamples[run],Xsamples[run]) * np.dot(Xsamples[run],Ysamples[run]) ) / (np.dot(uMsamples[run],uMsamples[run]) * np.dot(Xsamples[run],Xsamples[run]) - np.dot(uMsamples[run],Xsamples[run])**2) for run in range(numruns)]
            
            # B) regress m on um
            
            MuMslope = [ (np.dot(uMsamples[run],Msamples[run]) * np.dot(Xsamples[run],Xsamples[run]) - np.dot(uMsamples[run],Xsamples[run]) * np.dot(Xsamples[run],Msamples[run]) ) / (np.dot(uMsamples[run],uMsamples[run]) * np.dot(Xsamples[run],Xsamples[run]) - np.dot(uMsamples[run],Xsamples[run])**2) for run in range(numruns)]
            
            # C) divide these coefficients
            
            aInstEstimators = [YuMslope[run]/MuMslope[run] for run in range(numruns)]
            #aInstEstimators = MuMslope     # to run this test, switch aBias to aInstMean - 1
            
            cMean = stat.mean(cEstimators)
            cBias[ai,ci] = cMean - c
            cVar[ai,ci] = stat.variance(cEstimators)

            #print("Theoretical mean(c) = 0")
            THcVar = 1./((numsamp-2)*(d**2 + 1))
            #print("Theoretical var(c) = ","{:.6f}".format(THcVar))

            aInstMean = stat.mean(aInstEstimators)
            aBias[ai,ci] = aInstMean - a
            aVar[ai,ci] = stat.variance(aInstEstimators)

            THaBias = b*c*d / (1 + c**2 * (d**2 + 1))
            #print("Theoretical bias(a) = ","{:.4f}".format(THaBias))

            acInstEstimators = [aInstEstimators[i] * cEstimators[i] for i in range(numruns) ]
            acInstMean = stat.mean(acInstEstimators)
            acBias[ai,ci] = acInstMean - a*c
            acVar[ai,ci] = stat.variance(acInstEstimators)
                   
    return cBias, cVar, aBias, aVar, acBias, acVar
 
def acScan_instrumentalFDCedivd_ICM(arange,crange,anum,cnum,tfdc,tperfedivd,numsamp=1e4,numruns=1e2,e=1,b=1,d=1,g=1,noise=1):

    avals = np.linspace(arange[0],arange[1],anum)
    cvals = np.linspace(crange[0],crange[1],cnum)
    
    #b = 1
    #d = 1
    
    cBias = np.empty([anum,cnum])
    cVar = np.empty([anum,cnum])
    aBias = np.empty([anum,cnum])
    aVar = np.empty([anum,cnum])
    acBias = np.empty([anum,cnum])
    acVar = np.empty([anum,cnum])
    
    for ai,a in enumerate(avals):
        for ci,c in enumerate(cvals):

            # graph edges: [("V","X"),("W","X"),("X","M"),("M","Y"),("W","Y"),("W","M")]
            confmodel = models.generate_InstrumentModel([d,c,a,b,g,e],noise=noise)

            runvalues = []

            for run in range(numruns):
            
                #BEGIN (numruns) THREADS, VARS runvalues, ?samples
                
                os.system('cls||clear')
                print("Running Instrumental ac Estimators")
                print("a = ", a, " in range [", arange[0], ", ", arange[1], "]")
                print("c = ", c, " in range [", crange[0], ", ", crange[1], "]")
                print("Run: ",run," of ",numruns)
                
                #if run>1:
                #    print("Mean g estimates:")
                #    print( np.mean(XVslope[run-1]))
                #    
                #    print("Mean c estimates:")
                #    print( np.mean(cInstEstimators[run-1]) )
                    
                
                confsamples = confmodel.sample(numsamp)
                #for samp in confsamples[-3:-1]:
                #    print(samp)
                
                values = [[x for x in samp.values()] for samp in confsamples]
                valuesT = np.array(values).T.tolist()
                runvalues.append(valuesT)
                
                #CONT THREADS

            nodedict = dict([(node,confmodel.nodes.index(node)) for node in confmodel.nodes])
            
            #CONT THREADS
            
            sampledicts = [dict([(node,runvalues[run][nodedict[node]]) for node in confmodel.nodes]) for run in range(numruns)]

            Vsamples = [sampledicts[run]["V"] for run in range(numruns)]
            Wsamples = [sampledicts[run]["W"] for run in range(numruns)]
            Xsamples = [sampledicts[run]["X"] for run in range(numruns)]
            Msamples = [sampledicts[run]["M"] for run in range(numruns)]
            Ysamples = [sampledicts[run]["Y"] for run in range(numruns)]
            
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

            if tfdc:  # This is the standard FDC for this graph
            
                # Calculate the regression slope from x -> m (c + e/d)
                cEstimators = [np.dot(Xsamples[run],Msamples[run])/np.dot(Xsamples[run],Xsamples[run]) for run in range(numruns)]
                
                # Calculate the residuals ResMX = um - e/d ux - ge/d uV for each datapoint {x,m}
                uMsamples = [[Msamples[run][samp] - cEstimators[run] * Xsamples[run][samp] for samp in range(numsamp)] for run in range(numruns)]

            else:   # We compare with our improved e/d approach
            
                if tperfedivd:
                    # Assume we already know some perfect e/d estimator
                    edivd = e/d
                    
                    # Calculate the residuals um for each datapoint {x,m}
                    uMsamples = [[eWplusuMsamples[run][samp] - edivd * dWplusuXsamples[run][samp] for samp in range(numsamp)] for run in range(numruns)]
                
                else:
                    # Calculate the regression slope from x -> m (c + e/d)
                    cEstimators = [np.dot(Xsamples[run],Msamples[run])/np.dot(Xsamples[run],Xsamples[run]) for run in range(numruns)]
                    
                    edivdEst = [np.linalg.norm(eWplusuMsamples[run]) / np.linalg.norm(dWplusuXsamples[run]) for run in range(numruns)]
                    
                    edivdEst2 = [cEstimators[run] - cInstEstimators[run] for run in range(numruns)]

                    # Calculate the residuals um for each datapoint {x,m}
                    uMsamples = [[eWplusuMsamples[run][samp] - edivdEst2[run] * dWplusuXsamples[run][samp] for samp in range(numsamp)] for run in range(numruns)]
                
            # Use um as an instrumental variable to estimate m -> y (a):
            # A) regress y on um
            
            YuMslope = [ (np.dot(uMsamples[run],Ysamples[run]) * np.dot(Xsamples[run],Xsamples[run]) - np.dot(uMsamples[run],Xsamples[run]) * np.dot(Xsamples[run],Ysamples[run]) ) / (np.dot(uMsamples[run],uMsamples[run]) * np.dot(Xsamples[run],Xsamples[run]) - np.dot(uMsamples[run],Xsamples[run])**2) for run in range(numruns)]
            
            # B) regress m on um
            
            MuMslope = [ (np.dot(uMsamples[run],Msamples[run]) * np.dot(Xsamples[run],Xsamples[run]) - np.dot(uMsamples[run],Xsamples[run]) * np.dot(Xsamples[run],Msamples[run]) ) / (np.dot(uMsamples[run],uMsamples[run]) * np.dot(Xsamples[run],Xsamples[run]) - np.dot(uMsamples[run],Xsamples[run])**2) for run in range(numruns)]
            
            # C) divide these coefficients
            
            aInstEstimators = [YuMslope[run]/MuMslope[run] for run in range(numruns)]
            #aInstEstimators = MuMslope     # to run this test, switch aBias to aInstMean - 1
            
            cMean = stat.mean(cInstEstimators)
            cBias[ai,ci] = cMean - c
            cVar[ai,ci] = stat.variance(cInstEstimators)

            #print("Theoretical mean(c) = 0")
            THcVar = 1./((numsamp-2)*(d**2 + 1))
            #print("Theoretical var(c) = ","{:.6f}".format(THcVar))

            aInstMean = stat.mean(aInstEstimators)
            aBias[ai,ci] = aInstMean - a
            aVar[ai,ci] = stat.variance(aInstEstimators)

            THaBias = b*c*d / (1 + c**2 * (d**2 + 1))
            #print("Theoretical bias(a) = ","{:.4f}".format(THaBias))

            acInstEstimators = [aInstEstimators[i] * cInstEstimators[i] for i in range(numruns) ]
            acInstMean = stat.mean(acInstEstimators)
            acBias[ai,ci] = acInstMean - a*c
            acVar[ai,ci] = stat.variance(acInstEstimators)
                   
    return cBias, cVar, aBias, aVar, acBias, acVar
    
def acScan_instrumentalFDCedivd_ICM_withalpha(arange,crange,anum,cnum,tfdc,tperfedivd,numsamp=1e4,numruns=1e2,e=1,b=1,d=1,g=1,noise=1):

    avals = np.linspace(arange[0],arange[1],anum)
    cvals = np.linspace(crange[0],crange[1],cnum)
    
    #b = 1
    #d = 1
    
    cBias = np.empty([anum,cnum])
    cVar = np.empty([anum,cnum])
    aBias = np.empty([anum,cnum])
    aVar = np.empty([anum,cnum])
    acBias = np.empty([anum,cnum])
    acVar = np.empty([anum,cnum])
    
    for ai,a in enumerate(avals):
        for ci,c in enumerate(cvals):

            # graph edges: [("V","X"),("W","X"),("X","M"),("M","Y"),("W","Y"),("W","M")]
            confmodel = models.generate_InstrumentModel([d,c,a,b,g,e],noise=noise)

            runvalues = []

            for run in range(numruns):
            
                #BEGIN (numruns) THREADS, VARS runvalues, ?samples
                
                os.system('cls||clear')
                print("Running Instrumental ac Estimators")
                print("a = ", a, " in range [", arange[0], ", ", arange[1], "]")
                print("c = ", c, " in range [", crange[0], ", ", crange[1], "]")
                print("Run: ",run," of ",numruns)
                
                #if run>1:
                #    print("Mean g estimates:")
                #    print( np.mean(XVslope[run-1]))
                #    
                #    print("Mean c estimates:")
                #    print( np.mean(cInstEstimators[run-1]) )
                    
                
                confsamples = confmodel.sample(numsamp)
                #for samp in confsamples[-3:-1]:
                #    print(samp)
                
                values = [[x for x in samp.values()] for samp in confsamples]
                valuesT = np.array(values).T.tolist()
                runvalues.append(valuesT)
                
                #CONT THREADS

            nodedict = dict([(node,confmodel.nodes.index(node)) for node in confmodel.nodes])
            
            #CONT THREADS
            
            sampledicts = [dict([(node,runvalues[run][nodedict[node]]) for node in confmodel.nodes]) for run in range(numruns)]

            Vsamples = [sampledicts[run]["V"] for run in range(numruns)]
            Wsamples = [sampledicts[run]["W"] for run in range(numruns)]
            Xsamples = [sampledicts[run]["X"] for run in range(numruns)]
            Msamples = [sampledicts[run]["M"] for run in range(numruns)]
            Ysamples = [sampledicts[run]["Y"] for run in range(numruns)]
            
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

            if tfdc:  # This is the standard FDC for this graph
            
                # Calculate the regression slope from x -> m (c + e/d)
                cEstimators = [np.dot(Xsamples[run],Msamples[run])/np.dot(Xsamples[run],Xsamples[run]) for run in range(numruns)]
                
                # Calculate the residuals ResMX = um - e/d ux - ge/d uV for each datapoint {x,m}
                uMsamples = [[Msamples[run][samp] - cEstimators[run] * Xsamples[run][samp] for samp in range(numsamp)] for run in range(numruns)]

            else:   # We compare with our improved e/d approach
            
                if tperfedivd:
                    # Assume we already know some perfect e/d estimator
                    edivd = e/d
                    
                    # Calculate the residuals um for each datapoint {x,m}
                    uMsamples = [[eWplusuMsamples[run][samp] - edivd * dWplusuXsamples[run][samp] for samp in range(numsamp)] for run in range(numruns)]
                
                else:
                    # Calculate the regression slope from x -> m (c + e/d)
                    cEstimators = [np.dot(Xsamples[run],Msamples[run])/np.dot(Xsamples[run],Xsamples[run]) for run in range(numruns)]
                    
                    edivdEst = [np.linalg.norm(eWplusuMsamples[run]) / np.linalg.norm(dWplusuXsamples[run]) for run in range(numruns)]
                    
                    edivdEst2 = [cEstimators[run] - cInstEstimators[run] for run in range(numruns)]

                    # Calculate the residuals um for each datapoint {x,m}
                    uMsamples = [[eWplusuMsamples[run][samp] - edivdEst2[run] * dWplusuXsamples[run][samp] for samp in range(numsamp)] for run in range(numruns)]
                
          
            
            # Estimate the covariances of X with res1 and res2 (per run)
            
            uMcovsX = [np.cov(Xsamples[run],uMsamples[run],bias=True)[1][0] for run in range(numruns)]
            
            # Estimate the variance of X
            
            Xvariance = [np.var(Xsamples[run]) for run in range(numruns)]
            
            
            # Generate adjustment samples scaled from X
            singalpha = 0.56 # corresponds to full adjustment in range [0,1]
            adjXsamples = [[Xsamples[run][samp] * singalpha * uMcovsX[run] / np.sqrt(Xvariance[run]) for samp in range(numsamp)] for run in range(numruns)]
            adjuMsamples = [[uMsamples[run][samp] - adjXsamples[run][samp] for samp in range(numsamp)] for run in range(numruns)]
            
            
            # Use um as an instrumental variable to estimate m -> y (a):
            # A) regress y on um
            
            YuMslope = [ (np.dot(adjuMsamples[run],Ysamples[run]) * np.dot(Xsamples[run],Xsamples[run]) - np.dot(adjuMsamples[run],Xsamples[run]) * np.dot(Xsamples[run],Ysamples[run]) ) / (np.dot(adjuMsamples[run],adjuMsamples[run]) * np.dot(Xsamples[run],Xsamples[run]) - np.dot(adjuMsamples[run],Xsamples[run])**2) for run in range(numruns)]
            
            # B) regress m on um
            
            MuMslope = [ (np.dot(adjuMsamples[run],Msamples[run]) * np.dot(Xsamples[run],Xsamples[run]) - np.dot(adjuMsamples[run],Xsamples[run]) * np.dot(Xsamples[run],Msamples[run]) ) / (np.dot(adjuMsamples[run],adjuMsamples[run]) * np.dot(Xsamples[run],Xsamples[run]) - np.dot(adjuMsamples[run],Xsamples[run])**2) for run in range(numruns)]
            
            # C) divide these coefficients
            
            aInstEstimators = [YuMslope[run]/MuMslope[run] for run in range(numruns)]
            #aInstEstimators = MuMslope     # to run this test, switch aBias to aInstMean - 1
            
            cMean = stat.mean(cInstEstimators)
            cBias[ai,ci] = cMean - c
            cVar[ai,ci] = stat.variance(cInstEstimators)

            #print("Theoretical mean(c) = 0")
            THcVar = 1./((numsamp-2)*(d**2 + 1))
            #print("Theoretical var(c) = ","{:.6f}".format(THcVar))

            aInstMean = stat.mean(aInstEstimators)
            aBias[ai,ci] = aInstMean - a
            aVar[ai,ci] = stat.variance(aInstEstimators)

            THaBias = b*c*d / (1 + c**2 * (d**2 + 1))
            #print("Theoretical bias(a) = ","{:.4f}".format(THaBias))

            acInstEstimators = [aInstEstimators[i] * cInstEstimators[i] for i in range(numruns) ]
            acInstMean = stat.mean(acInstEstimators)
            acBias[ai,ci] = acInstMean - a*c
            acVar[ai,ci] = stat.variance(acInstEstimators)
                   
    return cBias, cVar, aBias, aVar, acBias, acVar

def acScan_IFDCres_ICM(arange,crange,anum,cnum,tperfedivd,numsamp=1e4,numruns=1e2,e=1,b=1,d=1,g=1,noise=1):

    avals = np.linspace(arange[0],arange[1],anum)
    cvals = np.linspace(crange[0],crange[1],cnum)
    
    #b = 1
    #d = 1
    
    meanres1var = np.empty([anum,cnum])
    meanres2var = np.empty([anum,cnum])
    meanres1covX = np.empty([anum,cnum])
    meanres2covX = np.empty([anum,cnum])
    meanres1covM = np.empty([anum,cnum])
    meanres2covM = np.empty([anum,cnum])
    meanres1covW = np.empty([anum,cnum])
    meanres2covW = np.empty([anum,cnum])
    
    meanadjres2var = np.empty([anum,cnum])
    meanadjres2covX = np.empty([anum,cnum])
    meanadjres2covM = np.empty([anum,cnum])
    meanadjres2covW = np.empty([anum,cnum])\
    
    numalpha = 10
    meanadjres2varscan = np.empty([anum,cnum,numalpha])
    meanadjres2covXscan = np.empty([anum,cnum,numalpha])
    meanadjres2covMscan = np.empty([anum,cnum,numalpha])
    meanadjres2covWscan = np.empty([anum,cnum,numalpha])
    
    
    for ai,a in enumerate(avals):
        for ci,c in enumerate(cvals):

            # graph edges: [("V","X"),("W","X"),("X","M"),("M","Y"),("W","Y"),("W","M")]
            confmodel = models.generate_InstrumentModel([d,c,a,b,g,e],noise=noise)

            runvalues = []

            for run in range(numruns):
            
                #BEGIN (numruns) THREADS, VARS runvalues, ?samples
                
                os.system('cls||clear')
                print("Running Instrumental ac Estimators")
                print("a = ", a, " in range [", arange[0], ", ", arange[1], "]")
                print("c = ", c, " in range [", crange[0], ", ", crange[1], "]")
                print("Run: ",run," of ",numruns)
                
                #if run>1:
                #    print("Mean g estimates:")
                #    print( np.mean(XVslope[run-1]))
                #    
                #    print("Mean c estimates:")
                #    print( np.mean(cInstEstimators[run-1]) )
                    
                
                confsamples = confmodel.sample(numsamp)
                #for samp in confsamples[-3:-1]:
                #    print(samp)
                
                values = [[x for x in samp.values()] for samp in confsamples]
                valuesT = np.array(values).T.tolist()
                runvalues.append(valuesT)
                
                #CONT THREADS

            nodedict = dict([(node,confmodel.nodes.index(node)) for node in confmodel.nodes])
            
            #CONT THREADS
            
            sampledicts = [dict([(node,runvalues[run][nodedict[node]]) for node in confmodel.nodes]) for run in range(numruns)]

            Vsamples = [sampledicts[run]["V"] for run in range(numruns)]
            Wsamples = [sampledicts[run]["W"] for run in range(numruns)]
            Xsamples = [sampledicts[run]["X"] for run in range(numruns)]
            Msamples = [sampledicts[run]["M"] for run in range(numruns)]
            Ysamples = [sampledicts[run]["Y"] for run in range(numruns)]
            
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

            # This is the standard FDC for this graph
            
            # Calculate the regression slope from x -> m (c + e/d)
            cEstimators = [np.dot(Xsamples[run],Msamples[run])/np.dot(Xsamples[run],Xsamples[run]) for run in range(numruns)]
            
            # Calculate the residuals ResMX = um - e/d ux - ge/d uV for each datapoint {x,m}
            res1samples = [[Msamples[run][samp] - cEstimators[run] * Xsamples[run][samp] for samp in range(numsamp)] for run in range(numruns)]

            # We compare with our improved e/d approach
            
            if tperfedivd:
                # Assume we already know some perfect e/d estimator
                edivd = e/d
                
                # Calculate the residuals um for each datapoint {x,m}
                res2samples = [[eWplusuMsamples[run][samp] - edivd * dWplusuXsamples[run][samp] for samp in range(numsamp)] for run in range(numruns)]
                
            else:
                # Calculate the regression slope from x -> m (c + e/d)
                cEstimators = [np.dot(Xsamples[run],Msamples[run])/np.dot(Xsamples[run],Xsamples[run]) for run in range(numruns)]
                
                edivdEst = [np.linalg.norm(eWplusuMsamples[run]) / np.linalg.norm(dWplusuXsamples[run]) for run in range(numruns)]
                
                edivdEst2 = [cEstimators[run] - cInstEstimators[run] for run in range(numruns)]

                # Calculate the residuals um for each datapoint {x,m}
                res2samples = [[eWplusuMsamples[run][samp] - edivdEst2[run] * dWplusuXsamples[run][samp] for samp in range(numsamp)] for run in range(numruns)]
                 
                
            # Estimate the means of res1 and res2 (per run)
            
            res1means = [np.mean(res1samples[run]) for run in range(numruns)]
            res2means = [np.mean(res2samples[run]) for run in range(numruns)]
                
            # Estimate the variances of res1 and res2 (per run)
            
            res1vars = [np.var(res1samples[run]) for run in range(numruns)]
            res2vars = [np.var(res2samples[run]) for run in range(numruns)]
            
            # Estimate the covariances of X with res1 and res2 (per run)
            
            res1covsX = [np.cov(Xsamples[run],res1samples[run],bias=True)[1][0] for run in range(numruns)]
            res2covsX = [np.cov(Xsamples[run],res2samples[run],bias=True)[1][0] for run in range(numruns)]
            
            # Estimate the covariances of M with res1 and res2 (per run)
            
            print(np.cov(Msamples[run],res1samples[run],bias=True))
            res1covsM = [np.cov(Msamples[run],res1samples[run],bias=True)[1][0] for run in range(numruns)]
            res2covsM = [np.cov(Msamples[run],res2samples[run],bias=True)[1][0] for run in range(numruns)]
            
            # Estimate the covariances of W with res1 and res2 (per run)
            
            res1covsW = [np.cov(Wsamples[run],res1samples[run],bias=True)[1][0] for run in range(numruns)]
            res2covsW = [np.cov(Wsamples[run],res2samples[run],bias=True)[1][0] for run in range(numruns)]
            
            # Estimate variance of X
            Xvariance = [np.var(Xsamples[run]) for run in range(numruns)]
            
            # Generate adjustment samples scaled from X
            singalpha = 1 # corresponds to full adjustment in range [0,1]
            singalpha = 0.56 # corresponds to full adjustment in range [0,1]
            adjXsamples = [[Xsamples[run][samp] * singalpha * res2covsX[run] / np.sqrt(Xvariance[run]) for samp in range(numsamp)] for run in range(numruns)]
            adjres2samples = [[res2samples[run][samp] - adjXsamples[run][samp] for samp in range(numsamp)] for run in range(numruns)]
            
            alphas = np.linspace(0,1,numalpha)
            adjXsamplesscan = []
            adjres2samplesscan = []
            for alpha in alphas:
                tempadjXsamples = [[Xsamples[run][samp] * alpha * res2covsX[run] / np.sqrt(Xvariance[run]) for samp in range(numsamp)] for run in range(numruns)]
                adjXsamplesscan = adjXsamplesscan + [tempadjXsamples]
                adjres2samplesscan = adjres2samplesscan + [[[res2samples[run][samp] - tempadjXsamples[run][samp] for samp in range(numsamp)] for run in range(numruns)]]
                
            
            
            # Estimate statistics for adjusted res2
            adjres2vars = [np.var(adjres2samples[run]) for run in range(numruns)]
            adjres2covsX = [np.cov(Xsamples[run],adjres2samples[run],bias=True)[1][0] for run in range(numruns)]
            adjres2covsM = [np.cov(Msamples[run],adjres2samples[run],bias=True)[1][0] for run in range(numruns)]
            adjres2covsW = [np.cov(Wsamples[run],adjres2samples[run],bias=True)[1][0] for run in range(numruns)]
            
            adjres2varsscan = [[np.var(adjres2samplesscan[ia][run]) for run in range(numruns)] for ia in range(numalpha)] 
            adjres2covsXscan = [[np.cov(Xsamples[run],adjres2samplesscan[ia][run],bias=True)[1][0] for run in range(numruns)] for ia in range(numalpha)]
            adjres2covsMscan = [[np.cov(Msamples[run],adjres2samplesscan[ia][run],bias=True)[1][0] for run in range(numruns)] for ia in range(numalpha)]
            adjres2covsWscan = [[np.cov(Wsamples[run],adjres2samplesscan[ia][run],bias=True)[1][0] for run in range(numruns)] for ia in range(numalpha)]
            
            print(adjres2covsXscan)
            
            
            
            meanres1var[ai,ci] = stat.mean(res1vars)
            #varres1var = stat.variance(res1vars)
            
            meanres2var[ai,ci] = stat.mean(res2vars)
            #varres2var = stat.variance(res2vars)
            
            meanres1covX[ai,ci] = stat.mean(res1covsX)
            meanres2covX[ai,ci] = stat.mean(res2covsX)
            meanres1covM[ai,ci] = stat.mean(res1covsM)
            meanres2covM[ai,ci] = stat.mean(res2covsM)
            meanres1covW[ai,ci] = stat.mean(res1covsW)
            meanres2covW[ai,ci] = stat.mean(res2covsW)
            
            meanadjres2var[ai,ci] = stat.mean(adjres2vars)
            meanadjres2covX[ai,ci] = stat.mean(adjres2covsX)
            meanadjres2covM[ai,ci] = stat.mean(adjres2covsM)
            meanadjres2covW[ai,ci] = stat.mean(adjres2covsW)
            
            for ia in range(numalpha):
              
                meanadjres2varscan[ai,ci,ia] = stat.mean(adjres2varsscan[ia])
                meanadjres2covXscan[ai,ci,ia] = stat.mean(adjres2covsXscan[ia])
                meanadjres2covMscan[ai,ci,ia] = stat.mean(adjres2covsMscan[ia])
                meanadjres2covWscan[ai,ci,ia] = stat.mean(adjres2covsWscan[ia])
                print(meanadjres2varscan[ai,ci,ia])
                   
    return meanres1var, meanres2var, meanres1covX, meanres2covX, meanres1covM, meanres2covM, meanres1covW, meanres2covW, meanadjres2var, meanadjres2covX, meanadjres2covM, meanadjres2covW, meanadjres2varscan, meanadjres2covXscan, meanadjres2covMscan, meanadjres2covWscan

def edScan_instrumentalFDCedivd(erange,drange,enum,dnum,tfdc,tperfedivd,numsamp=1e4,numruns=1e2,a=1,b=1,c=1,g=1,noise=1):

    evals = np.linspace(erange[0],erange[1],enum)
    dvals = np.linspace(drange[0],drange[1],dnum)
    
    #b = 1
    #d = 1
    
    cBias = np.empty([enum,dnum])
    cVar = np.empty([enum,dnum])
    aBias = np.empty([enum,dnum])
    aVar = np.empty([enum,dnum])
    acBias = np.empty([enum,dnum])
    acVar = np.empty([enum,dnum])
    
    for ei,e in enumerate(evals):
        for di,d in enumerate(dvals):

            # graph edges: [("V","X"),("W","X"),("X","M"),("M","Y"),("W","Y"),("W","M")]
            confmodel = models.generate_InstrumentModel([d,c,a,b,g,e],noise=noise)

            runvalues = []

            for run in range(numruns):
            
                #BEGIN (numruns) THREADS, VARS runvalues, ?samples
                
                os.system('cls||clear')
                print("Running Instrumental ac Estimators")
                print("e = ", e, " in range [", erange[0], ", ", erange[1], "]")
                print("d = ", d, " in range [", drange[0], ", ", drange[1], "]")
                print("Run: ",run," of ",numruns)
                
                #if run>1:
                #    print("Mean g estimates:")
                #    print( np.mean(XVslope[run-1]))
                #    
                #    print("Mean c estimates:")
                #    print( np.mean(cInstEstimators[run-1]) )
                    
                
                confsamples = confmodel.sample(numsamp)
                #for samp in confsamples[-3:-1]:
                #    print(samp)
                
                values = [[x for x in samp.values()] for samp in confsamples]
                valuesT = np.array(values).T.tolist()
                runvalues.append(valuesT)
                
                #CONT THREADS

            nodedict = dict([(node,confmodel.nodes.index(node)) for node in confmodel.nodes])
            
            #CONT THREADS
            
            sampledicts = [dict([(node,runvalues[run][nodedict[node]]) for node in confmodel.nodes]) for run in range(numruns)]

            Vsamples = [sampledicts[run]["V"] for run in range(numruns)]
            Wsamples = [sampledicts[run]["W"] for run in range(numruns)]
            Xsamples = [sampledicts[run]["X"] for run in range(numruns)]
            Msamples = [sampledicts[run]["M"] for run in range(numruns)]
            Ysamples = [sampledicts[run]["Y"] for run in range(numruns)]
            
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

            if tfdc:  # This is the standard FDC for this graph
            
                # Calculate the regression slope from x -> m (c + e/d)
                cEstimators = [np.dot(Xsamples[run],Msamples[run])/np.dot(Xsamples[run],Xsamples[run]) for run in range(numruns)]
                
                # Calculate the residuals ResMX = um - e/d ux - ge/d uV for each datapoint {x,m}
                uMsamples = [[Msamples[run][samp] - cEstimators[run] * Xsamples[run][samp] for samp in range(numsamp)] for run in range(numruns)]

            else:   # We compare with our improved e/d approach
            
                if tperfedivd:
                    # Assume we already know some perfect e/d estimator
                    edivd = e/d
                    
                    # Calculate the residuals um for each datapoint {x,m}
                    uMsamples = [[eWplusuMsamples[run][samp] - edivd * dWplusuXsamples[run][samp] for samp in range(numsamp)] for run in range(numruns)]
                
                else:   
                    # Calculate the regression slope from x -> m (c + e/d)
                    cEstimators = [np.dot(Xsamples[run],Msamples[run])/np.dot(Xsamples[run],Xsamples[run]) for run in range(numruns)]
                    
                    edivdEst = [np.linalg.norm(eWplusuMsamples[run]) / np.linalg.norm(dWplusuXsamples[run]) for run in range(numruns)]
                    
                    edivdEst2 = [cEstimators[run] - cInstEstimators[run] for run in range(numruns)]

                    # Calculate the residuals um for each datapoint {x,m}
                    uMsamples = [[eWplusuMsamples[run][samp] - edivdEst2[run] * dWplusuXsamples[run][samp] for samp in range(numsamp)] for run in range(numruns)]
            
            # Use um as an instrumental variable to estimate m -> y (a):
            # A) regress y on um
            
            YuMslope = [ (np.dot(uMsamples[run],Ysamples[run]) * np.dot(Xsamples[run],Xsamples[run]) - np.dot(uMsamples[run],Xsamples[run]) * np.dot(Xsamples[run],Ysamples[run]) ) / (np.dot(uMsamples[run],uMsamples[run]) * np.dot(Xsamples[run],Xsamples[run]) - np.dot(uMsamples[run],Xsamples[run])**2) for run in range(numruns)]
            
            # B) regress m on um
            
            MuMslope = [ (np.dot(uMsamples[run],Msamples[run]) * np.dot(Xsamples[run],Xsamples[run]) - np.dot(uMsamples[run],Xsamples[run]) * np.dot(Xsamples[run],Msamples[run]) ) / (np.dot(uMsamples[run],uMsamples[run]) * np.dot(Xsamples[run],Xsamples[run]) - np.dot(uMsamples[run],Xsamples[run])**2) for run in range(numruns)]
            
            # C) divide these coefficients
            
            aInstEstimators = [YuMslope[run]/MuMslope[run] for run in range(numruns)]
            #aInstEstimators = MuMslope     # to run this test, switch aBias to aInstMean - 1
            
            cMean = stat.mean(cInstEstimators)
            cBias[ei,di] = cMean - c
            cVar[ei,di] = stat.variance(cInstEstimators)

            #print("Theoretical mean(c) = 0")
            THcVar = 1./((numsamp-2)*(d**2 + 1))
            #print("Theoretical var(c) = ","{:.6f}".format(THcVar))

            aInstMean = stat.mean(aInstEstimators)
            aBias[ei,di] = aInstMean - a
            aVar[ei,di] = stat.variance(aInstEstimators)

            THaBias = b*c*d / (1 + c**2 * (d**2 + 1))
            #print("Theoretical bias(a) = ","{:.4f}".format(THaBias))

            acInstEstimators = [aInstEstimators[i] * cInstEstimators[i] for i in range(numruns) ]
            acInstMean = stat.mean(acInstEstimators)
            acBias[ei,di] = acInstMean - a*c
            acVar[ei,di] = stat.variance(acInstEstimators)
                   
    return cBias, cVar, aBias, aVar, acBias, acVar 

def acScan_fdc_ifdc(ofile,plottype):

    
    arange = [-2,2]
    crange = [-2,2]
    anum = cnum = 11
    numsamp = int(1e3)
    numruns = int(1e2)
    e = 1
    b=1
    d=1
    g=1
    noise = 1
    
    
    
    save=True
    savename="acdata_fdc-ifdc_1ksamp_100run_11div_e1"
    #savename="acdatainst_2ksamp_500run_10div"
    load=False
    
    if load:
        npzfile = np.load(savename + ".npz")
        
        cBiasTrue =  npzfile['cBiasTrue']
        cVarTrue =   npzfile['cVarTrue']
        aBiasTrue =  npzfile['aBiasTrue']
        aVarTrue =   npzfile['aVarTrue']
        acBiasTrue = npzfile['acBiasTrue']
        acVarTrue =  npzfile['acVarTrue']
        
        cBiasApprox =  npzfile['cBiasApprox']
        cVarApprox =   npzfile['cVarApprox']
        aBiasApprox =  npzfile['aBiasApprox']
        aVarApprox =   npzfile['aVarApprox']
        acBiasApprox = npzfile['acBiasApprox']
        acVarApprox =  npzfile['acVarApprox']
        
        cBiasInst =  npzfile['cBiasInst']
        cVarInst =   npzfile['cVarInst']
        aBiasInst =  npzfile['aBiasInst']
        aVarInst =   npzfile['aVarInst']
        acBiasInst = npzfile['acBiasInst']
        acVarInst =  npzfile['acVarInst']
        
    
    else:
        #cBiasTrue, cVarTrue, aBiasTrue, aVarTrue, acBiasTrue, acVarTrue = acScan_trueFDC(arange,crange,anum,cnum,numsamp,numruns,e)
        cBiasTrue, cVarTrue, aBiasTrue, aVarTrue, acBiasTrue, acVarTrue = acScan_trueFDC(arange,crange,anum,cnum,numsamp,numruns,e=e,b=b,d=d,noise=noise)
        #cBiasApprox, cVarApprox, aBiasApprox, aVarApprox, acBiasApprox, acVarApprox = acScan_instrumentalFDCedivd(arange,crange,anum,cnum,False,False,numsamp,numruns,e=e,b=b,d=d,g=g,noise=noise)
        #cBiasInst, cVarInst, aBiasInst, aVarInst, acBiasInst, acVarInst = acScan_instrumentalFDC(arange,crange,anum,cnum,numsamp,numruns,e)
        cBiasInst, cVarInst, aBiasInst, aVarInst, acBiasInst, acVarInst = acScan_instrumentalFDC(arange,crange,anum,cnum,numsamp,numruns,e=e,b=b,d=d,noise=noise)
        
        if save:
            np.savez(savename + ".npz", cBiasTrue=cBiasTrue, cVarTrue=cVarTrue, aBiasTrue=aBiasTrue, aVarTrue=aVarTrue, acBiasTrue=acBiasTrue, acVarTrue=acVarTrue,  cBiasInst=cBiasInst, cVarInst=cVarInst, aBiasInst=aBiasInst, aVarInst=aVarInst, acBiasInst=acBiasInst, acVarInst=acVarInst)
            #np.savez(savename + ".npz", cBiasTrue=cBiasTrue, cVarTrue=cVarTrue, aBiasTrue=aBiasTrue, aVarTrue=aVarTrue, acBiasTrue=acBiasTrue, acVarTrue=acVarTrue, cBiasInst=cBiasInst, cVarInst=cVarInst, aBiasInst=aBiasInst, aVarInst=aVarInst, acBiasInst=acBiasInst, acVarInst=acVarInst)
    
        
    
    fig, ax = plt.subplots(2,2)
    
    avals = np.linspace(arange[0],arange[1],anum)
    cvals = np.linspace(crange[0],crange[1],cnum)
    
    if plottype=="ac":
        print("TRUE")
        
        print(max(aBiasTrue.flatten()))
        print(min(aBiasTrue.flatten()))
        print(np.mean(aBiasTrue.flatten()))
        
        print(max(cBiasTrue.flatten()))
        print(min(cBiasTrue.flatten()))
        print(np.mean(cBiasTrue.flatten()))
        
        acScoreTrue = acBiasTrue + np.sqrt(acVarTrue)
        #acScoreApprox = acBiasApprox + np.sqrt(acVarApprox)
        acScoreInst = acBiasInst + np.sqrt(acVarInst)
        
        #acScoreTrue = acBiasTrue * np.sqrt(acVarTrue)
        #acScoreApprox = acBiasApprox * np.sqrt(acVarApprox)
        #acScoreInst = acBiasInst * np.sqrt(acVarInst)
        
    
        acMaxTrue = np.max(np.abs(acBiasTrue))
        #acMaxApprox = np.max(np.abs(acBiasApprox))
        #acMaxApprox = 0
        acMaxInst = np.max(np.abs(acBiasInst))
        acMax = np.max([acMaxTrue,acMaxInst])
        acLevels = np.linspace(-1 * acMax, acMax, 21)
        
        acMaxVarTrue = np.max(np.abs(acVarTrue))
        #acMaxVarApprox = np.max(np.abs(acVarApprox))
        #acMaxVarApprox = 0
        acMaxVarInst = np.max(np.abs(acVarInst))
        acMaxVar = np.max([acMaxVarTrue,acMaxVarInst])
        acVarLevels = np.linspace(0, acMaxVar, 21)
        
        
        
        cf00 = ax[0][0].contourf(avals, cvals, acBiasTrue, acLevels, linestyles="solid")
        #cf01 = ax[0][1].contourf(avals, cvals, acBiasApprox, acLevels, linestyles="solid")
        cf02 = ax[0][1].contourf(avals, cvals, acBiasInst, acLevels, linestyles="solid")
        cf10 = ax[1][0].contourf(avals, cvals, acVarTrue, acVarLevels, linestyles="solid")
        #cf11 = ax[1][1].contourf(avals, cvals, acVarApprox, acVarLevels, linestyles="solid")
        cf12 = ax[1][1].contourf(avals, cvals, acVarInst, acVarLevels, linestyles="solid")
        
        print(acBiasTrue)
        #print(acBiasApprox)
        print(acBiasInst)
        
        
    if plottype=="a":
    
        aMaxTrue = np.max(np.abs(aBiasTrue))
        aMaxApprox = np.max(np.abs(aBiasApprox))
        aMaxInst = np.max(np.abs(aBiasInst))
        aMax = np.max([aMaxTrue,aMaxApprox,aMaxInst])
        aLevels = np.linspace(-1 * aMax, aMax, 21)
        
        aMaxVarTrue = np.max(np.abs(aVarTrue))
        aMaxVarApprox = np.max(np.abs(aVarApprox))
        aMaxVarInst = np.max(np.abs(aVarInst))
        aMaxVar = np.max([aMaxVarTrue,aMaxVarApprox,aMaxVarInst])
        aVarLevels = np.linspace(0, aMaxVar, 21)
        
        print("Mean bias(a), True FDC ",np.mean(aBiasTrue))
        print("Mean bias(a), Instrumental FDC with e/d estimator ",np.mean(aBiasApprox))
        print("Mean bias(a), Instrumental FDC with perfect e/d ",np.mean(aBiasInst))
        
        
        cf00 = ax[0][0].contourf(avals, cvals, aBiasTrue, aLevels, linestyles="solid")
        #cf01 = ax[0][1].contourf(avals, cvals, aBiasApprox, aLevels, linestyles="solid")
        cf02 = ax[0][1].contourf(avals, cvals, aBiasInst, aLevels, linestyles="solid")
        cf10 = ax[1][0].contourf(avals, cvals, aVarTrue, aVarLevels, linestyles="solid")
        #cf11 = ax[1][1].contourf(avals, cvals, aVarApprox, aVarLevels, linestyles="solid")
        cf12 = ax[1][1].contourf(avals, cvals, aVarInst, aVarLevels, linestyles="solid")
        
        print(aBiasTrue)
        print(aBiasApprox)
        print(aBiasInst)

    if plottype=="c":
    
        cMaxTrue = np.max(np.abs(cBiasTrue))
        cMaxApprox = np.max(np.abs(cBiasApprox))
        cMaxInst = np.max(np.abs(cBiasInst))
        cMax = np.max([cMaxTrue,cMaxApprox,cMaxInst])
        cLevels = np.linspace(-1 * cMax, cMax, 21)
        
        cMaxVarTrue = np.max(np.abs(cVarTrue))
        cMaxVarApprox = np.max(np.abs(cVarApprox))
        cMaxVarInst = np.max(np.abs(cVarInst))
        cMaxVar = np.max([cMaxVarTrue,cMaxVarApprox,cMaxVarInst])
        cVarLevels = np.linspace(0, cMaxVar, 21)
       
        print("Mean bias(c), True FDC ",np.mean(cBiasTrue))
        print("Mean bias(c), Instrumental FDC with e/d estimator ",np.mean(cBiasApprox))
        print("Mean bias(c), Instrumental FDC with perfect e/d ",np.mean(cBiasInst))
        
        cf00 = ax[0][0].contourf(avals, cvals, cBiasTrue, cLevels, linestyles="solid")
        #cf01 = ax[0][1].contourf(avals, cvals, cBiasApprox, cLevels, linestyles="solid")
        cf02 = ax[0][1].contourf(avals, cvals, cBiasInst, cLevels, linestyles="solid")
        cf10 = ax[1][0].contourf(avals, cvals, cVarTrue, cVarLevels, linestyles="solid")
        #cf11 = ax[1][1].contourf(avals, cvals, cVarApprox, cVarLevels, linestyles="solid")
        cf12 = ax[1][1].contourf(avals, cvals, cVarInst, cVarLevels, linestyles="solid")
        
        print(cBiasTrue)
        print(cBiasApprox)
        print(cBiasInst)
        
    if plottype=="debug":
    
        cMaxTrue = np.max(np.abs(cBiasTrue))
        cMaxApprox = np.max(np.abs(cBiasApprox))
        cMax = np.max([cMaxTrue,cMaxApprox])
        cLevels = np.linspace(-1 * cMax, cMax, 21)
        
        cMaxVarTrue = np.max(np.abs(cVarTrue))
        cMaxVarApprox = np.max(np.abs(cVarApprox))
        cMaxVar = np.max([cMaxVarTrue,cMaxVarApprox])
        cVarLevels = np.linspace(0, cMaxVar, 21)
        
        cf00 = ax[0][0].contourf(avals, cvals, aBiasTrue, cLevels, linestyles="solid")
        cf01 = ax[0][1].contourf(avals, cvals, cBiasTrue, cLevels, linestyles="solid")
        cf10 = ax[1][0].contourf(avals, cvals, acBiasTrue, cVarLevels, linestyles="solid")
        #cf11 = ax[1][1].contourf(avals, cvals, cVarApprox, cVarLevels, linestyles="solid")
    
        
    #colorlabel = 'Probability Density'
    ax[1][0].set_xlabel("a")
    ax[1][1].set_xlabel("a")
    #ax[1][2].set_xlabel("a")
    ax[0][0].set_ylabel("c")
    ax[1][0].set_ylabel("c")
    #ax[2][0].set_ylabel("c")
    
    ax[0][0].set_title('True FDC', fontstyle='italic')
    ax[0][1].set_title('Instrumental FDC', fontstyle='italic')
    #ax[0][2].set_title('\" with perfect e/d', fontstyle='italic')
    
    axins0 = inset_axes(ax[0][1],
           width="5%", # width = 10% of parent_bbox width
           height="100%", # height : 50%
           loc=6,
           bbox_to_anchor=(1.05, 0., 1, 1),
           bbox_transform=ax[0][1].transAxes,
           borderpad=0,
       )
    axins1 = inset_axes(ax[1][1],
           width="5%", # width = 10% of parent_bbox width
           height="100%", # height : 50%
           loc=6,
           bbox_to_anchor=(1.05, 0., 1, 1),
           bbox_transform=ax[1][1].transAxes,
           borderpad=0,
       )

    cbar0 = plt.colorbar(cf02, cax=axins0)
    cbar0.set_label("Bias")
    
    cbar1 = fig.colorbar(cf12, cax=axins1)
    cbar1.set_label("Variance")
    

    #ax[0][1].collections[0].colorbar.set_label("Bias")
    #ax[1][1].collections[0].colorbar.set_label("Variance")
    
    
    fig.set_size_inches((16,16), forward=False)
    #fig.tight_layout()
    fig.savefig(ofile + '.pdf', format='pdf', dpi=1200)


def acScan_ifdc_jointx(ofile,plottype):

    
    arange = crange = [-2,2]
    anum = cnum = 4
    numsamp = int(1e3)
    numruns = int(1e2)
    e = 1
    b=1
    d=1
    g=1
    noise = 1
    
    
    
    save=True
    savename="acdata_ifdc-jointx_1ksamp_100run_4div_e1"
    #savename="acdatainst_2ksamp_500run_10div"
    load=False
    
    if load:
        npzfile = np.load(savename + ".npz")
        
        cBiasTrue =  npzfile['cBiasTrue']
        cVarTrue =   npzfile['cVarTrue']
        aBiasTrue =  npzfile['aBiasTrue']
        aVarTrue =   npzfile['aVarTrue']
        acBiasTrue = npzfile['acBiasTrue']
        acVarTrue =  npzfile['acVarTrue']
        
        cBiasApprox =  npzfile['cBiasApprox']
        cVarApprox =   npzfile['cVarApprox']
        aBiasApprox =  npzfile['aBiasApprox']
        aVarApprox =   npzfile['aVarApprox']
        acBiasApprox = npzfile['acBiasApprox']
        acVarApprox =  npzfile['acVarApprox']
        
        cBiasInst =  npzfile['cBiasInst']
        cVarInst =   npzfile['cVarInst']
        aBiasInst =  npzfile['aBiasInst']
        aVarInst =   npzfile['aVarInst']
        acBiasInst = npzfile['acBiasInst']
        acVarInst =  npzfile['acVarInst']
        
    
    else:
        #cBiasTrue, cVarTrue, aBiasTrue, aVarTrue, acBiasTrue, acVarTrue = acScan_trueFDC(arange,crange,anum,cnum,numsamp,numruns,e)
        cBiasTrue, cVarTrue, aBiasTrue, aVarTrue, acBiasTrue, acVarTrue = acScan_instrumentalFDC(arange,crange,anum,cnum,numsamp,numruns,e=e,b=b,d=d,noise=noise)
        #cBiasApprox, cVarApprox, aBiasApprox, aVarApprox, acBiasApprox, acVarApprox = acScan_instrumentalFDCedivd(arange,crange,anum,cnum,False,False,numsamp,numruns,e=e,b=b,d=d,g=g,noise=noise)
        #cBiasInst, cVarInst, aBiasInst, aVarInst, acBiasInst, acVarInst = acScan_instrumentalFDC(arange,crange,anum,cnum,numsamp,numruns,e)
        cBiasInst, cVarInst, aBiasInst, aVarInst, acBiasInst, acVarInst = acScan_instrumentalFDCwithX(arange,crange,anum,cnum,numsamp,numruns,e=e,b=b,d=d,noise=noise)
        
        if save:
            np.savez(savename + ".npz", cBiasTrue=cBiasTrue, cVarTrue=cVarTrue, aBiasTrue=aBiasTrue, aVarTrue=aVarTrue, acBiasTrue=acBiasTrue, acVarTrue=acVarTrue,  cBiasInst=cBiasInst, cVarInst=cVarInst, aBiasInst=aBiasInst, aVarInst=aVarInst, acBiasInst=acBiasInst, acVarInst=acVarInst)
            #np.savez(savename + ".npz", cBiasTrue=cBiasTrue, cVarTrue=cVarTrue, aBiasTrue=aBiasTrue, aVarTrue=aVarTrue, acBiasTrue=acBiasTrue, acVarTrue=acVarTrue, cBiasInst=cBiasInst, cVarInst=cVarInst, aBiasInst=aBiasInst, aVarInst=aVarInst, acBiasInst=acBiasInst, acVarInst=acVarInst)
    
        
    
    fig, ax = plt.subplots(2,2)
    
    avals = np.linspace(arange[0],arange[1],anum)
    cvals = np.linspace(crange[0],crange[1],cnum)
    
    if plottype=="ac":
        print("TRUE")
        
        print(max(aBiasTrue.flatten()))
        print(min(aBiasTrue.flatten()))
        print(np.mean(aBiasTrue.flatten()))
        
        print(max(cBiasTrue.flatten()))
        print(min(cBiasTrue.flatten()))
        print(np.mean(cBiasTrue.flatten()))
        
        acScoreTrue = acBiasTrue + np.sqrt(acVarTrue)
        #acScoreApprox = acBiasApprox + np.sqrt(acVarApprox)
        acScoreInst = acBiasInst + np.sqrt(acVarInst)
        
        #acScoreTrue = acBiasTrue * np.sqrt(acVarTrue)
        #acScoreApprox = acBiasApprox * np.sqrt(acVarApprox)
        #acScoreInst = acBiasInst * np.sqrt(acVarInst)
        
    
        acMaxTrue = np.max(np.abs(acBiasTrue))
        #acMaxApprox = np.max(np.abs(acBiasApprox))
        #acMaxApprox = 0
        acMaxInst = np.max(np.abs(acBiasInst))
        acMax = np.max([acMaxTrue,acMaxInst])
        acLevels = np.linspace(-1 * acMax, acMax, 21)
        
        acMaxVarTrue = np.max(np.abs(acVarTrue))
        #acMaxVarApprox = np.max(np.abs(acVarApprox))
        #acMaxVarApprox = 0
        acMaxVarInst = np.max(np.abs(acVarInst))
        acMaxVar = np.max([acMaxVarTrue,acMaxVarInst])
        acVarLevels = np.linspace(0, acMaxVar, 21)
        
        
        
        cf00 = ax[0][0].contourf(avals, cvals, acBiasTrue, acLevels, linestyles="solid")
        #cf01 = ax[0][1].contourf(avals, cvals, acBiasApprox, acLevels, linestyles="solid")
        cf02 = ax[0][1].contourf(avals, cvals, acBiasInst, acLevels, linestyles="solid")
        cf10 = ax[1][0].contourf(avals, cvals, acVarTrue, acVarLevels, linestyles="solid")
        #cf11 = ax[1][1].contourf(avals, cvals, acVarApprox, acVarLevels, linestyles="solid")
        cf12 = ax[1][1].contourf(avals, cvals, acVarInst, acVarLevels, linestyles="solid")
        
        print(acBiasTrue)
        #print(acBiasApprox)
        print(acBiasInst)
        
        
    if plottype=="a":
    
        aMaxTrue = np.max(np.abs(aBiasTrue))
        aMaxApprox = np.max(np.abs(aBiasApprox))
        aMaxInst = np.max(np.abs(aBiasInst))
        aMax = np.max([aMaxTrue,aMaxApprox,aMaxInst])
        aLevels = np.linspace(-1 * aMax, aMax, 21)
        
        aMaxVarTrue = np.max(np.abs(aVarTrue))
        aMaxVarApprox = np.max(np.abs(aVarApprox))
        aMaxVarInst = np.max(np.abs(aVarInst))
        aMaxVar = np.max([aMaxVarTrue,aMaxVarApprox,aMaxVarInst])
        aVarLevels = np.linspace(0, aMaxVar, 21)
        
        print("Mean bias(a), True FDC ",np.mean(aBiasTrue))
        print("Mean bias(a), Instrumental FDC with e/d estimator ",np.mean(aBiasApprox))
        print("Mean bias(a), Instrumental FDC with perfect e/d ",np.mean(aBiasInst))
        
        
        cf00 = ax[0][0].contourf(avals, cvals, aBiasTrue, aLevels, linestyles="solid")
        #cf01 = ax[0][1].contourf(avals, cvals, aBiasApprox, aLevels, linestyles="solid")
        cf02 = ax[0][1].contourf(avals, cvals, aBiasInst, aLevels, linestyles="solid")
        cf10 = ax[1][0].contourf(avals, cvals, aVarTrue, aVarLevels, linestyles="solid")
        #cf11 = ax[1][1].contourf(avals, cvals, aVarApprox, aVarLevels, linestyles="solid")
        cf12 = ax[1][1].contourf(avals, cvals, aVarInst, aVarLevels, linestyles="solid")
        
        print(aBiasTrue)
        print(aBiasApprox)
        print(aBiasInst)

    if plottype=="c":
    
        cMaxTrue = np.max(np.abs(cBiasTrue))
        cMaxApprox = np.max(np.abs(cBiasApprox))
        cMaxInst = np.max(np.abs(cBiasInst))
        cMax = np.max([cMaxTrue,cMaxApprox,cMaxInst])
        cLevels = np.linspace(-1 * cMax, cMax, 21)
        
        cMaxVarTrue = np.max(np.abs(cVarTrue))
        cMaxVarApprox = np.max(np.abs(cVarApprox))
        cMaxVarInst = np.max(np.abs(cVarInst))
        cMaxVar = np.max([cMaxVarTrue,cMaxVarApprox,cMaxVarInst])
        cVarLevels = np.linspace(0, cMaxVar, 21)
       
        print("Mean bias(c), True FDC ",np.mean(cBiasTrue))
        print("Mean bias(c), Instrumental FDC with e/d estimator ",np.mean(cBiasApprox))
        print("Mean bias(c), Instrumental FDC with perfect e/d ",np.mean(cBiasInst))
        
        cf00 = ax[0][0].contourf(avals, cvals, cBiasTrue, cLevels, linestyles="solid")
        #cf01 = ax[0][1].contourf(avals, cvals, cBiasApprox, cLevels, linestyles="solid")
        cf02 = ax[0][1].contourf(avals, cvals, cBiasInst, cLevels, linestyles="solid")
        cf10 = ax[1][0].contourf(avals, cvals, cVarTrue, cVarLevels, linestyles="solid")
        #cf11 = ax[1][1].contourf(avals, cvals, cVarApprox, cVarLevels, linestyles="solid")
        cf12 = ax[1][1].contourf(avals, cvals, cVarInst, cVarLevels, linestyles="solid")
        
        print(cBiasTrue)
        print(cBiasApprox)
        print(cBiasInst)
        
    if plottype=="debug":
    
        cMaxTrue = np.max(np.abs(cBiasTrue))
        cMaxApprox = np.max(np.abs(cBiasApprox))
        cMax = np.max([cMaxTrue,cMaxApprox])
        cLevels = np.linspace(-1 * cMax, cMax, 21)
        
        cMaxVarTrue = np.max(np.abs(cVarTrue))
        cMaxVarApprox = np.max(np.abs(cVarApprox))
        cMaxVar = np.max([cMaxVarTrue,cMaxVarApprox])
        cVarLevels = np.linspace(0, cMaxVar, 21)
        
        cf00 = ax[0][0].contourf(avals, cvals, aBiasTrue, cLevels, linestyles="solid")
        cf01 = ax[0][1].contourf(avals, cvals, cBiasTrue, cLevels, linestyles="solid")
        cf10 = ax[1][0].contourf(avals, cvals, acBiasTrue, cVarLevels, linestyles="solid")
        #cf11 = ax[1][1].contourf(avals, cvals, cVarApprox, cVarLevels, linestyles="solid")
    
        
    #colorlabel = 'Probability Density'
    ax[1][0].set_xlabel("a")
    ax[1][1].set_xlabel("a")
    #ax[1][2].set_xlabel("a")
    ax[0][0].set_ylabel("c")
    ax[1][0].set_ylabel("c")
    #ax[2][0].set_ylabel("c")
    
    ax[0][0].set_title('True FDC', fontstyle='italic')
    ax[0][1].set_title('Instrumental FDC', fontstyle='italic')
    #ax[0][2].set_title('\" with perfect e/d', fontstyle='italic')
    
    axins0 = inset_axes(ax[0][1],
           width="5%", # width = 10% of parent_bbox width
           height="100%", # height : 50%
           loc=6,
           bbox_to_anchor=(1.05, 0., 1, 1),
           bbox_transform=ax[0][1].transAxes,
           borderpad=0,
       )
    axins1 = inset_axes(ax[1][1],
           width="5%", # width = 10% of parent_bbox width
           height="100%", # height : 50%
           loc=6,
           bbox_to_anchor=(1.05, 0., 1, 1),
           bbox_transform=ax[1][1].transAxes,
           borderpad=0,
       )

    cbar0 = plt.colorbar(cf02, cax=axins0)
    cbar0.set_label("Bias")
    
    cbar1 = fig.colorbar(cf12, cax=axins1)
    cbar1.set_label("Variance")
    

    #ax[0][1].collections[0].colorbar.set_label("Bias")
    #ax[1][1].collections[0].colorbar.set_label("Variance")
    
    
    fig.set_size_inches((16,16), forward=False)
    #fig.tight_layout()
    fig.savefig(ofile + '.pdf', format='pdf', dpi=1200)

def acScan_edifdc_cm(ofile,plottype):

    
    arange = crange = [-2,2]
    anum = cnum = 11
    numsamp = int(1e3)
    numruns = int(1e2)
    e = 1
    b=1
    d=1
    g=1
    noise = 1
    
    
    
    save=True
    savename="acdata_edifdc_cm_1ksamp_100run_11div_e1"
    #savename="acdatainst_2ksamp_500run_10div"
    load= not save
    
    if load:
        npzfile = np.load(savename + ".npz")
        
        cBiasTrue =  npzfile['cBiasTrue']
        cVarTrue =   npzfile['cVarTrue']
        aBiasTrue =  npzfile['aBiasTrue']
        aVarTrue =   npzfile['aVarTrue']
        acBiasTrue = npzfile['acBiasTrue']
        acVarTrue =  npzfile['acVarTrue']
        
        cBiasApprox =  npzfile['cBiasApprox']
        cVarApprox =   npzfile['cVarApprox']
        aBiasApprox =  npzfile['aBiasApprox']
        aVarApprox =   npzfile['aVarApprox']
        acBiasApprox = npzfile['acBiasApprox']
        acVarApprox =  npzfile['acVarApprox']
        
        cBiasInst =  npzfile['cBiasInst']
        cVarInst =   npzfile['cVarInst']
        aBiasInst =  npzfile['aBiasInst']
        aVarInst =   npzfile['aVarInst']
        acBiasInst = npzfile['acBiasInst']
        acVarInst =  npzfile['acVarInst']
        
    
    else:
        #cBiasTrue, cVarTrue, aBiasTrue, aVarTrue, acBiasTrue, acVarTrue = acScan_trueFDC(arange,crange,anum,cnum,numsamp,numruns,e)
        cBiasTrue, cVarTrue, aBiasTrue, aVarTrue, acBiasTrue, acVarTrue = acScan_instrumentalFDCedivd_CM(arange,crange,anum,cnum,True,numsamp,numruns,e=e,b=b,d=d,noise=noise)
        #cBiasApprox, cVarApprox, aBiasApprox, aVarApprox, acBiasApprox, acVarApprox = acScan_instrumentalFDCedivd(arange,crange,anum,cnum,False,numsamp,numruns,e=e,b=b,d=d,noise=noise)
        #cBiasInst, cVarInst, aBiasInst, aVarInst, acBiasInst, acVarInst = acScan_instrumentalFDC(arange,crange,anum,cnum,numsamp,numruns,e)
        cBiasInst, cVarInst, aBiasInst, aVarInst, acBiasInst, acVarInst = acScan_instrumentalFDCedivd_CM(arange,crange,anum,cnum,False,numsamp,numruns,e=e,b=b,d=d,noise=noise)
        
        if save:
            np.savez(savename + ".npz", cBiasTrue=cBiasTrue, cVarTrue=cVarTrue, aBiasTrue=aBiasTrue, aVarTrue=aVarTrue, acBiasTrue=acBiasTrue, acVarTrue=acVarTrue, cBiasInst=cBiasInst, cVarInst=cVarInst, aBiasInst=aBiasInst, aVarInst=aVarInst, acBiasInst=acBiasInst, acVarInst=acVarInst)
            #np.savez(savename + ".npz", cBiasTrue=cBiasTrue, cVarTrue=cVarTrue, aBiasTrue=aBiasTrue, aVarTrue=aVarTrue, acBiasTrue=acBiasTrue, acVarTrue=acVarTrue, cBiasInst=cBiasInst, cVarInst=cVarInst, aBiasInst=aBiasInst, aVarInst=aVarInst, acBiasInst=acBiasInst, acVarInst=acVarInst)
    
        
    
    fig, ax = plt.subplots(2,2)
    
    avals = np.linspace(arange[0],arange[1],anum)
    cvals = np.linspace(crange[0],crange[1],cnum)
    
    if plottype=="ac":
        print("TRUE")
        
        print(max(aBiasTrue.flatten()))
        print(min(aBiasTrue.flatten()))
        print(np.mean(aBiasTrue.flatten()))
        
        print(max(cBiasTrue.flatten()))
        print(min(cBiasTrue.flatten()))
        print(np.mean(cBiasTrue.flatten()))
        
        acScoreTrue = acBiasTrue + np.sqrt(acVarTrue)
        #acScoreApprox = acBiasApprox + np.sqrt(acVarApprox)
        acScoreInst = acBiasInst + np.sqrt(acVarInst)
        
        #acScoreTrue = acBiasTrue * np.sqrt(acVarTrue)
        #acScoreApprox = acBiasApprox * np.sqrt(acVarApprox)
        #acScoreInst = acBiasInst * np.sqrt(acVarInst)
        
    
        acMaxTrue = np.max(np.abs(acBiasTrue))
        #acMaxApprox = np.max(np.abs(acBiasApprox))
        #acMaxApprox = 0
        acMaxInst = np.max(np.abs(acBiasInst))
        acMax = np.max([acMaxTrue,acMaxInst])
        acLevels = np.linspace(-1 * acMax, acMax, 21)
        
        acMaxVarTrue = np.max(np.abs(acVarTrue))
        #acMaxVarApprox = np.max(np.abs(acVarApprox))
        #acMaxVarApprox = 0
        acMaxVarInst = np.max(np.abs(acVarInst))
        acMaxVar = np.max([acMaxVarTrue,acMaxVarInst])
        acVarLevels = np.linspace(0, acMaxVar, 21)
       
        
        cf00 = ax[0][0].contourf(avals, cvals, acBiasTrue, acLevels, linestyles="solid")
        #cf01 = ax[0][1].contourf(avals, cvals, acBiasApprox, acLevels, linestyles="solid")
        cf02 = ax[0][1].contourf(avals, cvals, acBiasInst, acLevels, linestyles="solid")
        cf10 = ax[1][0].contourf(avals, cvals, acVarTrue, acVarLevels, linestyles="solid")
        #cf11 = ax[1][1].contourf(avals, cvals, acVarApprox, acVarLevels, linestyles="solid")
        cf12 = ax[1][1].contourf(avals, cvals, acVarInst, acVarLevels, linestyles="solid")
         
        print(acBiasTrue)
        #print(acBiasApprox)
        print(acBiasInst)
        
        
    if plottype=="a":
    
        aMaxTrue = np.max(np.abs(aBiasTrue))
        aMaxApprox = np.max(np.abs(aBiasApprox))
        aMaxInst = np.max(np.abs(aBiasInst))
        aMax = np.max([aMaxTrue,aMaxApprox,aMaxInst])
        aLevels = np.linspace(-1 * aMax, aMax, 21)
        
        aMaxVarTrue = np.max(np.abs(aVarTrue))
        aMaxVarApprox = np.max(np.abs(aVarApprox))
        aMaxVarInst = np.max(np.abs(aVarInst))
        aMaxVar = np.max([aMaxVarTrue,aMaxVarApprox,aMaxVarInst])
        aVarLevels = np.linspace(0, aMaxVar, 21)
        
        print("Mean bias(a), True FDC ",np.mean(aBiasTrue))
        print("Mean bias(a), IFDC with e/d estimator ",np.mean(aBiasApprox))
        print("Mean bias(a), IFDC with e/d oracle ",np.mean(aBiasInst))
        
        
        cf00 = ax[0][0].contourf(avals, cvals, aBiasTrue, aLevels, linestyles="solid")
        cf01 = ax[0][1].contourf(avals, cvals, aBiasApprox, aLevels, linestyles="solid")
        cf02 = ax[0][2].contourf(avals, cvals, aBiasInst, aLevels, linestyles="solid")
        cf10 = ax[1][0].contourf(avals, cvals, aVarTrue, aVarLevels, linestyles="solid")
        cf11 = ax[1][1].contourf(avals, cvals, aVarApprox, aVarLevels, linestyles="solid")
        cf12 = ax[1][2].contourf(avals, cvals, aVarInst, aVarLevels, linestyles="solid")
        
        print(aBiasTrue)
        print(aBiasApprox)
        print(aBiasInst)

    if plottype=="c":
    
        cMaxTrue = np.max(np.abs(cBiasTrue))
        cMaxApprox = np.max(np.abs(cBiasApprox))
        cMaxInst = np.max(np.abs(cBiasInst))
        cMax = np.max([cMaxTrue,cMaxApprox,cMaxInst])
        cLevels = np.linspace(-1 * cMax, cMax, 21)
        
        cMaxVarTrue = np.max(np.abs(cVarTrue))
        cMaxVarApprox = np.max(np.abs(cVarApprox))
        cMaxVarInst = np.max(np.abs(cVarInst))
        cMaxVar = np.max([cMaxVarTrue,cMaxVarApprox,cMaxVarInst])
        cVarLevels = np.linspace(0, cMaxVar, 21)
       
        print("Mean bias(c), True FDC ",np.mean(cBiasTrue))
        print("Mean bias(c), Instrumental FDC with e/d estimator ",np.mean(cBiasApprox))
        print("Mean bias(c), Instrumental FDC with perfect e/d ",np.mean(cBiasInst))
        
        cf00 = ax[0][0].contourf(avals, cvals, cBiasTrue, cLevels, linestyles="solid")
        cf01 = ax[0][1].contourf(avals, cvals, cBiasApprox, cLevels, linestyles="solid")
        cf02 = ax[0][2].contourf(avals, cvals, cBiasInst, cLevels, linestyles="solid")
        cf10 = ax[1][0].contourf(avals, cvals, cVarTrue, cVarLevels, linestyles="solid")
        cf11 = ax[1][1].contourf(avals, cvals, cVarApprox, cVarLevels, linestyles="solid")
        cf12 = ax[1][2].contourf(avals, cvals, cVarInst, cVarLevels, linestyles="solid")
        
        print(cBiasTrue)
        print(cBiasApprox)
        print(cBiasInst)
        
    if plottype=="debug":
    
        cMaxTrue = np.max(np.abs(cBiasTrue))
        cMaxApprox = np.max(np.abs(cBiasApprox))
        cMax = np.max([cMaxTrue,cMaxApprox])
        cLevels = np.linspace(-1 * cMax, cMax, 21)
        
        cMaxVarTrue = np.max(np.abs(cVarTrue))
        cMaxVarApprox = np.max(np.abs(cVarApprox))
        cMaxVar = np.max([cMaxVarTrue,cMaxVarApprox])
        cVarLevels = np.linspace(0, cMaxVar, 21)
        
        cf00 = ax[0][0].contourf(avals, cvals, aBiasTrue, cLevels, linestyles="solid")
        cf01 = ax[0][1].contourf(avals, cvals, cBiasTrue, cLevels, linestyles="solid")
        cf10 = ax[1][0].contourf(avals, cvals, acBiasTrue, cVarLevels, linestyles="solid")
        #cf11 = ax[1][1].contourf(avals, cvals, cVarApprox, cVarLevels, linestyles="solid")
    
        
    #colorlabel = 'Probability Density'
    ax[1][0].set_xlabel("a")
    ax[1][1].set_xlabel("a")
    #ax[1][2].set_xlabel("a")
    ax[0][0].set_ylabel("c")
    ax[1][0].set_ylabel("c")
    #ax[2][0].set_ylabel("c")
    
    ax[0][0].set_title('True FDC', fontstyle='italic')
    #ax[0][1].set_title('IFDC with e/d estimator', fontstyle='italic')
    ax[0][1].set_title('IFDC with e/d oracle', fontstyle='italic')
    
    axins0 = inset_axes(ax[0][1],
           width="5%", # width = 10% of parent_bbox width
           height="100%", # height : 50%
           loc=6,
           bbox_to_anchor=(1.05, 0., 1, 1),
           bbox_transform=ax[0][1].transAxes,
           borderpad=0,
       )
    axins1 = inset_axes(ax[1][1],
           width="5%", # width = 10% of parent_bbox width
           height="100%", # height : 50%
           loc=6,
           bbox_to_anchor=(1.05, 0., 1, 1),
           bbox_transform=ax[1][1].transAxes,
           borderpad=0,
       )
    

    cbar0 = plt.colorbar(cf02, cax=axins0)
    cbar0.set_label("Bias")
    
    cbar1 = fig.colorbar(cf12, cax=axins1)
    cbar1.set_label("Variance")
    

    #ax[0][1].collections[0].colorbar.set_label("Bias")
    #ax[1][1].collections[0].colorbar.set_label("Variance")
    
    
    fig.set_size_inches((16,16), forward=False)
    #fig.tight_layout()
    fig.savefig(ofile + '.pdf', format='pdf', dpi=1200)
    
def acScan_edifdc_icm(ofile,plottype):

    
    arange = crange = [-2,2]
    anum = cnum = 11
    numsamp = int(1e3)
    numruns = int(1e2)
    e = 1
    b=1
    d=1
    g=1
    noise = 1
    
    
    
    save=True
    savename="acdatainstedivd_icm_1ksamp_100run_10div_e1"
    #savename="acdatainst_2ksamp_500run_10div"
    load= not save
    
    if load:
        npzfile = np.load(savename + ".npz")
        
        cBiasTrue =  npzfile['cBiasTrue']
        cVarTrue =   npzfile['cVarTrue']
        aBiasTrue =  npzfile['aBiasTrue']
        aVarTrue =   npzfile['aVarTrue']
        acBiasTrue = npzfile['acBiasTrue']
        acVarTrue =  npzfile['acVarTrue']
        
        cBiasApprox =  npzfile['cBiasApprox']
        cVarApprox =   npzfile['cVarApprox']
        aBiasApprox =  npzfile['aBiasApprox']
        aVarApprox =   npzfile['aVarApprox']
        acBiasApprox = npzfile['acBiasApprox']
        acVarApprox =  npzfile['acVarApprox']
        
        cBiasInst =  npzfile['cBiasInst']
        cVarInst =   npzfile['cVarInst']
        aBiasInst =  npzfile['aBiasInst']
        aVarInst =   npzfile['aVarInst']
        acBiasInst = npzfile['acBiasInst']
        acVarInst =  npzfile['acVarInst']
        
    
    else:
        #cBiasTrue, cVarTrue, aBiasTrue, aVarTrue, acBiasTrue, acVarTrue = acScan_trueFDC(arange,crange,anum,cnum,numsamp,numruns,e)
        cBiasTrue, cVarTrue, aBiasTrue, aVarTrue, acBiasTrue, acVarTrue = acScan_instrumentalFDCedivd_ICM(arange,crange,anum,cnum,True,True,numsamp,numruns,e=e,b=b,d=d,g=g,noise=noise)
        cBiasApprox, cVarApprox, aBiasApprox, aVarApprox, acBiasApprox, acVarApprox = acScan_instrumentalFDCedivd_ICM(arange,crange,anum,cnum,False,False,numsamp,numruns,e=e,b=b,d=d,g=g,noise=noise)
        #cBiasInst, cVarInst, aBiasInst, aVarInst, acBiasInst, acVarInst = acScan_instrumentalFDC(arange,crange,anum,cnum,numsamp,numruns,e)
        cBiasInst, cVarInst, aBiasInst, aVarInst, acBiasInst, acVarInst = acScan_instrumentalFDCedivd_ICM(arange,crange,anum,cnum,False,True,numsamp,numruns,e=e,b=b,d=d,g=g,noise=noise)
        
        if save:
            np.savez(savename + ".npz", cBiasTrue=cBiasTrue, cVarTrue=cVarTrue, aBiasTrue=aBiasTrue, aVarTrue=aVarTrue, acBiasTrue=acBiasTrue, acVarTrue=acVarTrue, cBiasApprox=cBiasApprox, cVarApprox=cVarApprox, aBiasApprox=aBiasApprox, aVarApprox=aVarApprox, acBiasApprox=acBiasApprox, acVarApprox=acVarApprox, cBiasInst=cBiasInst, cVarInst=cVarInst, aBiasInst=aBiasInst, aVarInst=aVarInst, acBiasInst=acBiasInst, acVarInst=acVarInst)
            #np.savez(savename + ".npz", cBiasTrue=cBiasTrue, cVarTrue=cVarTrue, aBiasTrue=aBiasTrue, aVarTrue=aVarTrue, acBiasTrue=acBiasTrue, acVarTrue=acVarTrue, cBiasInst=cBiasInst, cVarInst=cVarInst, aBiasInst=aBiasInst, aVarInst=aVarInst, acBiasInst=acBiasInst, acVarInst=acVarInst)
    
        
    
    fig, ax = plt.subplots(2,3)
    
    avals = np.linspace(arange[0],arange[1],anum)
    cvals = np.linspace(crange[0],crange[1],cnum)
    
    if plottype=="ac":
        print("TRUE")
        
        print(max(aBiasTrue.flatten()))
        print(min(aBiasTrue.flatten()))
        print(np.mean(aBiasTrue.flatten()))
        
        print(max(cBiasTrue.flatten()))
        print(min(cBiasTrue.flatten()))
        print(np.mean(cBiasTrue.flatten()))
        
        acScoreTrue = acBiasTrue + np.sqrt(acVarTrue)
        acScoreApprox = acBiasApprox + np.sqrt(acVarApprox)
        acScoreInst = acBiasInst + np.sqrt(acVarInst)
        
        #acScoreTrue = acBiasTrue * np.sqrt(acVarTrue)
        #acScoreApprox = acBiasApprox * np.sqrt(acVarApprox)
        #acScoreInst = acBiasInst * np.sqrt(acVarInst)
        
    
        acMaxTrue = np.max(np.abs(acBiasTrue))
        acMaxApprox = np.max(np.abs(acBiasApprox))
        #acMaxApprox = 0
        acMaxInst = np.max(np.abs(acBiasInst))
        acMax = np.max([acMaxTrue,acMaxApprox,acMaxInst])
        acLevels = np.linspace(-1 * acMax, acMax, 21)
        
        acMaxVarTrue = np.max(np.abs(acVarTrue))
        acMaxVarApprox = np.max(np.abs(acVarApprox))
        #acMaxVarApprox = 0
        acMaxVarInst = np.max(np.abs(acVarInst))
        acMaxVar = np.max([acMaxVarTrue,acMaxVarApprox,acMaxVarInst])
        acVarLevels = np.linspace(0, acMaxVar, 21)
       
        
        cf00 = ax[0][0].contourf(avals, cvals, acBiasTrue, acLevels, linestyles="solid")
        cf01 = ax[0][1].contourf(avals, cvals, acBiasApprox, acLevels, linestyles="solid")
        cf02 = ax[0][2].contourf(avals, cvals, acBiasInst, acLevels, linestyles="solid")
        cf10 = ax[1][0].contourf(avals, cvals, acVarTrue, acVarLevels, linestyles="solid")
        cf11 = ax[1][1].contourf(avals, cvals, acVarApprox, acVarLevels, linestyles="solid")
        cf12 = ax[1][2].contourf(avals, cvals, acVarInst, acVarLevels, linestyles="solid")
         
        print(acBiasTrue)
        #print(acBiasApprox)
        print(acBiasInst)
        
        
    if plottype=="a":
    
        aMaxTrue = np.max(np.abs(aBiasTrue))
        aMaxApprox = np.max(np.abs(aBiasApprox))
        aMaxInst = np.max(np.abs(aBiasInst))
        aMax = np.max([aMaxTrue,aMaxApprox,aMaxInst])
        aLevels = np.linspace(-1 * aMax, aMax, 21)
        
        aMaxVarTrue = np.max(np.abs(aVarTrue))
        aMaxVarApprox = np.max(np.abs(aVarApprox))
        aMaxVarInst = np.max(np.abs(aVarInst))
        aMaxVar = np.max([aMaxVarTrue,aMaxVarApprox,aMaxVarInst])
        aVarLevels = np.linspace(0, aMaxVar, 21)
        
        print("Mean bias(a), True FDC ",np.mean(aBiasTrue))
        print("Mean bias(a), IFDC with e/d estimator ",np.mean(aBiasApprox))
        print("Mean bias(a), IFDC with e/d oracle ",np.mean(aBiasInst))
        
        
        cf00 = ax[0][0].contourf(avals, cvals, aBiasTrue, aLevels, linestyles="solid")
        cf01 = ax[0][1].contourf(avals, cvals, aBiasApprox, aLevels, linestyles="solid")
        cf02 = ax[0][2].contourf(avals, cvals, aBiasInst, aLevels, linestyles="solid")
        cf10 = ax[1][0].contourf(avals, cvals, aVarTrue, aVarLevels, linestyles="solid")
        cf11 = ax[1][1].contourf(avals, cvals, aVarApprox, aVarLevels, linestyles="solid")
        cf12 = ax[1][2].contourf(avals, cvals, aVarInst, aVarLevels, linestyles="solid")
        
        print(aBiasTrue)
        print(aBiasApprox)
        print(aBiasInst)

    if plottype=="c":
    
        cMaxTrue = np.max(np.abs(cBiasTrue))
        cMaxApprox = np.max(np.abs(cBiasApprox))
        cMaxInst = np.max(np.abs(cBiasInst))
        cMax = np.max([cMaxTrue,cMaxApprox,cMaxInst])
        cLevels = np.linspace(-1 * cMax, cMax, 21)
        
        cMaxVarTrue = np.max(np.abs(cVarTrue))
        cMaxVarApprox = np.max(np.abs(cVarApprox))
        cMaxVarInst = np.max(np.abs(cVarInst))
        cMaxVar = np.max([cMaxVarTrue,cMaxVarApprox,cMaxVarInst])
        cVarLevels = np.linspace(0, cMaxVar, 21)
       
        print("Mean bias(c), True FDC ",np.mean(cBiasTrue))
        print("Mean bias(c), Instrumental FDC with e/d estimator ",np.mean(cBiasApprox))
        print("Mean bias(c), Instrumental FDC with perfect e/d ",np.mean(cBiasInst))
        
        cf00 = ax[0][0].contourf(avals, cvals, cBiasTrue, cLevels, linestyles="solid")
        cf01 = ax[0][1].contourf(avals, cvals, cBiasApprox, cLevels, linestyles="solid")
        cf02 = ax[0][2].contourf(avals, cvals, cBiasInst, cLevels, linestyles="solid")
        cf10 = ax[1][0].contourf(avals, cvals, cVarTrue, cVarLevels, linestyles="solid")
        cf11 = ax[1][1].contourf(avals, cvals, cVarApprox, cVarLevels, linestyles="solid")
        cf12 = ax[1][2].contourf(avals, cvals, cVarInst, cVarLevels, linestyles="solid")
        
        print(cBiasTrue)
        print(cBiasApprox)
        print(cBiasInst)
        
    if plottype=="debug":
    
        cMaxTrue = np.max(np.abs(cBiasTrue))
        cMaxApprox = np.max(np.abs(cBiasApprox))
        cMax = np.max([cMaxTrue,cMaxApprox])
        cLevels = np.linspace(-1 * cMax, cMax, 21)
        
        cMaxVarTrue = np.max(np.abs(cVarTrue))
        cMaxVarApprox = np.max(np.abs(cVarApprox))
        cMaxVar = np.max([cMaxVarTrue,cMaxVarApprox])
        cVarLevels = np.linspace(0, cMaxVar, 21)
        
        cf00 = ax[0][0].contourf(avals, cvals, aBiasTrue, cLevels, linestyles="solid")
        cf01 = ax[0][1].contourf(avals, cvals, cBiasTrue, cLevels, linestyles="solid")
        cf10 = ax[1][0].contourf(avals, cvals, acBiasTrue, cVarLevels, linestyles="solid")
        #cf11 = ax[1][1].contourf(avals, cvals, cVarApprox, cVarLevels, linestyles="solid")
    
        
    #colorlabel = 'Probability Density'
    ax[1][0].set_xlabel("a")
    ax[1][1].set_xlabel("a")
    ax[1][2].set_xlabel("a")
    ax[0][0].set_ylabel("c")
    ax[1][0].set_ylabel("c")
    #ax[2][0].set_ylabel("c")
    
    ax[0][0].set_title('True FDC', fontstyle='italic')
    ax[0][1].set_title('IFDC with e/d estimator', fontstyle='italic')
    ax[0][2].set_title('IFDC with e/d oracle', fontstyle='italic')
    
    axins0 = inset_axes(ax[0][2],
           width="5%", # width = 10% of parent_bbox width
           height="100%", # height : 50%
           loc=6,
           bbox_to_anchor=(1.05, 0., 1, 1),
           bbox_transform=ax[0][2].transAxes,
           borderpad=0,
       )
    axins1 = inset_axes(ax[1][2],
           width="5%", # width = 10% of parent_bbox width
           height="100%", # height : 50%
           loc=6,
           bbox_to_anchor=(1.05, 0., 1, 1),
           bbox_transform=ax[1][2].transAxes,
           borderpad=0,
       )
    

    cbar0 = plt.colorbar(cf02, cax=axins0)
    cbar0.set_label("Bias")
    
    cbar1 = fig.colorbar(cf12, cax=axins1)
    cbar1.set_label("Variance")
    

    #ax[0][1].collections[0].colorbar.set_label("Bias")
    #ax[1][1].collections[0].colorbar.set_label("Variance")
    
    
    fig.set_size_inches((22,16), forward=False)
    #fig.tight_layout()
    fig.savefig(ofile + '.pdf', format='pdf', dpi=1200)

def acScan_edifdc_icm_withalpha(ofile,plottype):

    
    arange = crange = [-2,2]
    anum = cnum = 11
    numsamp = int(1e3)
    numruns = int(1e2)
    e = 1
    b=1
    d=1
    g=1
    noise = 1
    
    
    
    save=False
    savename="acdatainstedivd_alpha58_icm_1ksamp_100run_10div_e1_subedivd"
    #savename="acdatainst_2ksamp_500run_10div"
    load= not save
    
    tperfedivd=False
    
    if load:
        npzfile = np.load(savename + ".npz")
        
        cBiasTrue =  npzfile['cBiasTrue']
        cVarTrue =   npzfile['cVarTrue']
        aBiasTrue =  npzfile['aBiasTrue']
        aVarTrue =   npzfile['aVarTrue']
        acBiasTrue = npzfile['acBiasTrue']
        acVarTrue =  npzfile['acVarTrue']
        
        cBiasApprox =  npzfile['cBiasApprox']
        cVarApprox =   npzfile['cVarApprox']
        aBiasApprox =  npzfile['aBiasApprox']
        aVarApprox =   npzfile['aVarApprox']
        acBiasApprox = npzfile['acBiasApprox']
        acVarApprox =  npzfile['acVarApprox']
        
        cBiasInst =  npzfile['cBiasInst']
        cVarInst =   npzfile['cVarInst']
        aBiasInst =  npzfile['aBiasInst']
        aVarInst =   npzfile['aVarInst']
        acBiasInst = npzfile['acBiasInst']
        acVarInst =  npzfile['acVarInst']
        
    
    else:
        #cBiasTrue, cVarTrue, aBiasTrue, aVarTrue, acBiasTrue, acVarTrue = acScan_trueFDC(arange,crange,anum,cnum,numsamp,numruns,e)
        cBiasTrue, cVarTrue, aBiasTrue, aVarTrue, acBiasTrue, acVarTrue = acScan_instrumentalFDCedivd_ICM(arange,crange,anum,cnum,True,True,numsamp,numruns,e=e,b=b,d=d,g=g,noise=noise)
        cBiasApprox, cVarApprox, aBiasApprox, aVarApprox, acBiasApprox, acVarApprox = acScan_instrumentalFDCedivd_ICM(arange,crange,anum,cnum,False,tperfedivd,numsamp,numruns,e=e,b=b,d=d,g=g,noise=noise)
        #cBiasInst, cVarInst, aBiasInst, aVarInst, acBiasInst, acVarInst = acScan_instrumentalFDC(arange,crange,anum,cnum,numsamp,numruns,e)
        cBiasInst, cVarInst, aBiasInst, aVarInst, acBiasInst, acVarInst = acScan_instrumentalFDCedivd_ICM_withalpha(arange,crange,anum,cnum,False,tperfedivd,numsamp,numruns,e=e,b=b,d=d,g=g,noise=noise)
        
        if save:
            np.savez(savename + ".npz", cBiasTrue=cBiasTrue, cVarTrue=cVarTrue, aBiasTrue=aBiasTrue, aVarTrue=aVarTrue, acBiasTrue=acBiasTrue, acVarTrue=acVarTrue, cBiasApprox=cBiasApprox, cVarApprox=cVarApprox, aBiasApprox=aBiasApprox, aVarApprox=aVarApprox, acBiasApprox=acBiasApprox, acVarApprox=acVarApprox, cBiasInst=cBiasInst, cVarInst=cVarInst, aBiasInst=aBiasInst, aVarInst=aVarInst, acBiasInst=acBiasInst, acVarInst=acVarInst)
            #np.savez(savename + ".npz", cBiasTrue=cBiasTrue, cVarTrue=cVarTrue, aBiasTrue=aBiasTrue, aVarTrue=aVarTrue, acBiasTrue=acBiasTrue, acVarTrue=acVarTrue, cBiasInst=cBiasInst, cVarInst=cVarInst, aBiasInst=aBiasInst, aVarInst=aVarInst, acBiasInst=acBiasInst, acVarInst=acVarInst)
    
        
    
    fig, ax = plt.subplots(2,3)
    
    avals = np.linspace(arange[0],arange[1],anum)
    cvals = np.linspace(crange[0],crange[1],cnum)
    
    if plottype=="ac":
        print("TRUE")
        
        print(max(aBiasTrue.flatten()))
        print(min(aBiasTrue.flatten()))
        print(np.mean(aBiasTrue.flatten()))
        
        print(max(cBiasTrue.flatten()))
        print(min(cBiasTrue.flatten()))
        print(np.mean(cBiasTrue.flatten()))
        
        acScoreTrue = acBiasTrue + np.sqrt(acVarTrue)
        acScoreApprox = acBiasApprox + np.sqrt(acVarApprox)
        acScoreInst = acBiasInst + np.sqrt(acVarInst)
        
        #acScoreTrue = acBiasTrue * np.sqrt(acVarTrue)
        #acScoreApprox = acBiasApprox * np.sqrt(acVarApprox)
        #acScoreInst = acBiasInst * np.sqrt(acVarInst)
        
    
        acMaxTrue = np.max(np.abs(acBiasTrue))
        acMaxApprox = np.max(np.abs(acBiasApprox))
        #acMaxApprox = 0
        acMaxInst = np.max(np.abs(acBiasInst))
        acMax = np.max([acMaxTrue,acMaxApprox,acMaxInst])
        acLevels = np.linspace(-1 * acMax, acMax, 21)
        
        acMaxVarTrue = np.max(np.abs(acVarTrue))
        acMaxVarApprox = np.max(np.abs(acVarApprox))
        #acMaxVarApprox = 0
        acMaxVarInst = np.max(np.abs(acVarInst))
        acMaxVar = np.max([acMaxVarTrue,acMaxVarApprox,acMaxVarInst])
        acVarLevels = np.linspace(0, acMaxVar, 21)
       
        
        cf00 = ax[0][0].contourf(avals, cvals, acBiasTrue, acLevels, linestyles="solid")
        cf01 = ax[0][1].contourf(avals, cvals, acBiasApprox, acLevels, linestyles="solid")
        cf02 = ax[0][2].contourf(avals, cvals, acBiasInst, acLevels, linestyles="solid")
        cf10 = ax[1][0].contourf(avals, cvals, acVarTrue, acVarLevels, linestyles="solid")
        cf11 = ax[1][1].contourf(avals, cvals, acVarApprox, acVarLevels, linestyles="solid")
        cf12 = ax[1][2].contourf(avals, cvals, acVarInst, acVarLevels, linestyles="solid")
         
        print(acBiasTrue)
        #print(acBiasApprox)
        print(acBiasInst)
        
        
    if plottype=="a":
    
        aMaxTrue = np.max(np.abs(aBiasTrue))
        aMaxApprox = np.max(np.abs(aBiasApprox))
        aMaxInst = np.max(np.abs(aBiasInst))
        aMax = np.max([aMaxTrue,aMaxApprox,aMaxInst])
        aLevels = np.linspace(-1 * aMax, aMax, 21)
        
        aMaxVarTrue = np.max(np.abs(aVarTrue))
        aMaxVarApprox = np.max(np.abs(aVarApprox))
        aMaxVarInst = np.max(np.abs(aVarInst))
        aMaxVar = np.max([aMaxVarTrue,aMaxVarApprox,aMaxVarInst])
        aVarLevels = np.linspace(0, aMaxVar, 21)
        
        print("Mean bias(a), True FDC ",np.mean(aBiasTrue))
        print("Mean bias(a), IFDC with e/d oracle ",np.mean(aBiasApprox))
        print("Mean bias(a), IFDC - alpha=0.58 ",np.mean(aBiasInst))
        
        
        cf00 = ax[0][0].contourf(avals, cvals, aBiasTrue, aLevels, linestyles="solid")
        cf01 = ax[0][1].contourf(avals, cvals, aBiasApprox, aLevels, linestyles="solid")
        cf02 = ax[0][2].contourf(avals, cvals, aBiasInst, aLevels, linestyles="solid")
        cf10 = ax[1][0].contourf(avals, cvals, aVarTrue, aVarLevels, linestyles="solid")
        cf11 = ax[1][1].contourf(avals, cvals, aVarApprox, aVarLevels, linestyles="solid")
        cf12 = ax[1][2].contourf(avals, cvals, aVarInst, aVarLevels, linestyles="solid")
        
        print(aBiasTrue)
        print(aBiasApprox)
        print(aBiasInst)

    if plottype=="c":
    
        cMaxTrue = np.max(np.abs(cBiasTrue))
        cMaxApprox = np.max(np.abs(cBiasApprox))
        cMaxInst = np.max(np.abs(cBiasInst))
        cMax = np.max([cMaxTrue,cMaxApprox,cMaxInst])
        cLevels = np.linspace(-1 * cMax, cMax, 21)
        
        cMaxVarTrue = np.max(np.abs(cVarTrue))
        cMaxVarApprox = np.max(np.abs(cVarApprox))
        cMaxVarInst = np.max(np.abs(cVarInst))
        cMaxVar = np.max([cMaxVarTrue,cMaxVarApprox,cMaxVarInst])
        cVarLevels = np.linspace(0, cMaxVar, 21)
       
        print("Mean bias(a), True FDC ",np.mean(aBiasTrue))
        print("Mean bias(a), IFDC with e/d oracle ",np.mean(aBiasApprox))
        print("Mean bias(a), IFDC - alpha=0.58 ",np.mean(aBiasInst))
        
        cf00 = ax[0][0].contourf(avals, cvals, cBiasTrue, cLevels, linestyles="solid")
        cf01 = ax[0][1].contourf(avals, cvals, cBiasApprox, cLevels, linestyles="solid")
        cf02 = ax[0][2].contourf(avals, cvals, cBiasInst, cLevels, linestyles="solid")
        cf10 = ax[1][0].contourf(avals, cvals, cVarTrue, cVarLevels, linestyles="solid")
        cf11 = ax[1][1].contourf(avals, cvals, cVarApprox, cVarLevels, linestyles="solid")
        cf12 = ax[1][2].contourf(avals, cvals, cVarInst, cVarLevels, linestyles="solid")
        
        print(cBiasTrue)
        print(cBiasApprox)
        print(cBiasInst)
        
    if plottype=="debug":
    
        cMaxTrue = np.max(np.abs(cBiasTrue))
        cMaxApprox = np.max(np.abs(cBiasApprox))
        cMax = np.max([cMaxTrue,cMaxApprox])
        cLevels = np.linspace(-1 * cMax, cMax, 21)
        
        cMaxVarTrue = np.max(np.abs(cVarTrue))
        cMaxVarApprox = np.max(np.abs(cVarApprox))
        cMaxVar = np.max([cMaxVarTrue,cMaxVarApprox])
        cVarLevels = np.linspace(0, cMaxVar, 21)
        
        cf00 = ax[0][0].contourf(avals, cvals, aBiasTrue, cLevels, linestyles="solid")
        cf01 = ax[0][1].contourf(avals, cvals, cBiasTrue, cLevels, linestyles="solid")
        cf10 = ax[1][0].contourf(avals, cvals, acBiasTrue, cVarLevels, linestyles="solid")
        #cf11 = ax[1][1].contourf(avals, cvals, cVarApprox, cVarLevels, linestyles="solid")
    
        
    #colorlabel = 'Probability Density'
    ax[1][0].set_xlabel("a")
    ax[1][1].set_xlabel("a")
    ax[1][2].set_xlabel("a")
    ax[0][0].set_ylabel("c")
    ax[1][0].set_ylabel("c")
    #ax[2][0].set_ylabel("c")
    
    ax[0][0].set_title('True FDC', fontstyle='italic')
    ax[0][1].set_title('IFDC with e/d oracle', fontstyle='italic')
    ax[0][2].set_title('IFDC - alpha=0.58', fontstyle='italic')
    
    axins0 = inset_axes(ax[0][2],
           width="5%", # width = 10% of parent_bbox width
           height="100%", # height : 50%
           loc=6,
           bbox_to_anchor=(1.05, 0., 1, 1),
           bbox_transform=ax[0][2].transAxes,
           borderpad=0,
       )
    axins1 = inset_axes(ax[1][2],
           width="5%", # width = 10% of parent_bbox width
           height="100%", # height : 50%
           loc=6,
           bbox_to_anchor=(1.05, 0., 1, 1),
           bbox_transform=ax[1][2].transAxes,
           borderpad=0,
       )
    

    cbar0 = plt.colorbar(cf02, cax=axins0)
    cbar0.set_label("Bias")
    
    cbar1 = fig.colorbar(cf12, cax=axins1)
    cbar1.set_label("Variance")
    

    #ax[0][1].collections[0].colorbar.set_label("Bias")
    #ax[1][1].collections[0].colorbar.set_label("Variance")
    
    
    fig.set_size_inches((22,16), forward=False)
    #fig.tight_layout()
    fig.savefig(ofile + '.pdf', format='pdf', dpi=1200)


def edScan_plot(ofile,plottype):

    
    erange = drange = [-2,2]
    enum = dnum = 10
    numsamp = int(1e3)
    numruns = int(1e2)
    b=1
    g=1
    noise = 1
    a=1
    c=1
    
    
    save=True
    savename="eddatainstedivd_1ksamp_100run_10div_e01"
    #savename="acdatainst_2ksamp_500run_10div"
    load=False
    
    if load:
        npzfile = np.load(savename + ".npz")
        
        cBiasTrue =  npzfile['cBiasTrue']
        cVarTrue =   npzfile['cVarTrue']
        aBiasTrue =  npzfile['aBiasTrue']
        aVarTrue =   npzfile['aVarTrue']
        acBiasTrue = npzfile['acBiasTrue']
        acVarTrue =  npzfile['acVarTrue']
        
        cBiasApprox =  npzfile['cBiasApprox']
        cVarApprox =   npzfile['cVarApprox']
        aBiasApprox =  npzfile['aBiasApprox']
        aVarApprox =   npzfile['aVarApprox']
        acBiasApprox = npzfile['acBiasApprox']
        acVarApprox =  npzfile['acVarApprox']
        
        cBiasInst =  npzfile['cBiasInst']
        cVarInst =   npzfile['cVarInst']
        aBiasInst =  npzfile['aBiasInst']
        aVarInst =   npzfile['aVarInst']
        acBiasInst = npzfile['acBiasInst']
        acVarInst =  npzfile['acVarInst']
        
    
    else:
        #cBiasTrue, cVarTrue, aBiasTrue, aVarTrue, acBiasTrue, acVarTrue = acScan_trueFDC(arange,crange,anum,cnum,numsamp,numruns,e)
        cBiasTrue, cVarTrue, aBiasTrue, aVarTrue, acBiasTrue, acVarTrue = edScan_instrumentalFDCedivd(erange,drange,enum,dnum,True,True,numsamp,numruns,a=a,b=b,c=c,g=g,noise=noise)
        cBiasApprox, cVarApprox, aBiasApprox, aVarApprox, acBiasApprox, acVarApprox = edScan_instrumentalFDCedivd(erange,drange,enum,dnum,False,False,numsamp,numruns,a=a,b=b,c=c,g=g,noise=noise)
        #cBiasInst, cVarInst, aBiasInst, aVarInst, acBiasInst, acVarInst = acScan_instrumentalFDC(arange,crange,anum,cnum,numsamp,numruns,e)
        cBiasInst, cVarInst, aBiasInst, aVarInst, acBiasInst, acVarInst = edScan_instrumentalFDCedivd(erange,drange,enum,dnum,False,True,numsamp,numruns,a=a,b=b,c=c,g=g,noise=noise)
        
        if save:
            np.savez(savename + ".npz", cBiasTrue=cBiasTrue, cVarTrue=cVarTrue, aBiasTrue=aBiasTrue, aVarTrue=aVarTrue, acBiasTrue=acBiasTrue, acVarTrue=acVarTrue, cBiasApprox=cBiasApprox, cVarApprox=cVarApprox, aBiasApprox=aBiasApprox, aVarApprox=aVarApprox, acBiasApprox=acBiasApprox, acVarApprox=acVarApprox, cBiasInst=cBiasInst, cVarInst=cVarInst, aBiasInst=aBiasInst, aVarInst=aVarInst, acBiasInst=acBiasInst, acVarInst=acVarInst)
            #np.savez(savename + ".npz", cBiasTrue=cBiasTrue, cVarTrue=cVarTrue, aBiasTrue=aBiasTrue, aVarTrue=aVarTrue, acBiasTrue=acBiasTrue, acVarTrue=acVarTrue, cBiasInst=cBiasInst, cVarInst=cVarInst, aBiasInst=aBiasInst, aVarInst=aVarInst, acBiasInst=acBiasInst, acVarInst=acVarInst)
    
        
    
    fig, ax = plt.subplots(3,3)
    
    evals = np.linspace(erange[0],erange[1],enum)
    dvals = np.linspace(drange[0],drange[1],dnum)
    
    if plottype=="ac":
        print("TRUE")
        
        print(max(aBiasTrue.flatten()))
        print(min(aBiasTrue.flatten()))
        print(np.mean(aBiasTrue.flatten()))
        
        print(max(cBiasTrue.flatten()))
        print(min(cBiasTrue.flatten()))
        print(np.mean(cBiasTrue.flatten()))
        
        acScoreTrue = acBiasTrue + np.sqrt(acVarTrue)
        acScoreApprox = acBiasApprox + np.sqrt(acVarApprox)
        acScoreInst = acBiasInst + np.sqrt(acVarInst)
        
        #acScoreTrue = acBiasTrue * np.sqrt(acVarTrue)
        #acScoreApprox = acBiasApprox * np.sqrt(acVarApprox)
        #acScoreInst = acBiasInst * np.sqrt(acVarInst)
        
    
        acMaxTrue = np.max(np.abs(acBiasTrue))
        acMaxApprox = np.max(np.abs(acBiasApprox))
        #acMaxApprox = 0
        acMaxInst = np.max(np.abs(acBiasInst))
        acMax = np.max([acMaxTrue,acMaxApprox,acMaxInst])
        acLevels = np.linspace(-1 * acMax, acMax, 21)
        
        acMaxVarTrue = np.max(np.abs(acVarTrue))
        acMaxVarApprox = np.max(np.abs(acVarApprox))
        #acMaxVarApprox = 0
        acMaxVarInst = np.max(np.abs(acVarInst))
        acMaxVar = np.max([acMaxVarTrue,acMaxVarApprox,acMaxVarInst])
        acVarLevels = np.linspace(0, acMaxVar, 21)
        
        acMaxScoreTrue = np.max(np.abs(acScoreTrue))
        acMaxScoreApprox = np.max(np.abs(acScoreApprox))
        #acMaxScoreApprox = 0
        acMaxScoreInst = np.max(np.abs(acScoreInst))
        acMaxScore = np.max([acMaxScoreTrue,acMaxScoreApprox,acMaxScoreInst])
        acMinScoreTrue = np.min(acScoreTrue)
        acMinScoreApprox = np.min(acScoreApprox)
        #acMinScoreApprox = 1e10
        acMinScoreInst = np.min(acScoreInst)
        acMinScore = np.min([acMinScoreTrue,acMinScoreApprox,acMinScoreInst])
        acScoreLevels = np.linspace(acMinScore, acMaxScore, 21)
        
        cf00 = ax[0][0].contourf(evals, dvals, acBiasTrue, acLevels, linestyles="solid")
        cf01 = ax[0][1].contourf(evals, dvals, acBiasApprox, acLevels, linestyles="solid")
        cf02 = ax[0][2].contourf(evals, dvals, acBiasInst, acLevels, linestyles="solid")
        cf10 = ax[1][0].contourf(evals, dvals, acVarTrue, acVarLevels, linestyles="solid")
        cf11 = ax[1][1].contourf(evals, dvals, acVarApprox, acVarLevels, linestyles="solid")
        cf12 = ax[1][2].contourf(evals, dvals, acVarInst, acVarLevels, linestyles="solid")
        cf20 = ax[2][0].contourf(evals, dvals, acScoreTrue, acScoreLevels, linestyles="solid")
        cf21 = ax[2][1].contourf(evals, dvals, acScoreApprox, acScoreLevels, linestyles="solid")
        cf22 = ax[2][2].contourf(evals, dvals, acScoreInst, acScoreLevels, linestyles="solid")
        
        print(acBiasTrue)
        #print(acBiasApprox)
        print(acBiasInst)
        
        
    if plottype=="a":
    
        aMaxTrue = np.max(np.abs(aBiasTrue))
        aMaxApprox = np.max(np.abs(aBiasApprox))
        aMaxInst = np.max(np.abs(aBiasInst))
        aMax = np.max([aMaxTrue,aMaxApprox,aMaxInst])
        aLevels = np.linspace(-1 * aMax, aMax, 21)
        
        aMaxVarTrue = np.max(np.abs(aVarTrue))
        aMaxVarApprox = np.max(np.abs(aVarApprox))
        aMaxVarInst = np.max(np.abs(aVarInst))
        aMaxVar = np.max([aMaxVarTrue,aMaxVarApprox,aMaxVarInst])
        aVarLevels = np.linspace(0, aMaxVar, 21)
        
        print("Mean bias(a), True FDC ",np.mean(aBiasTrue))
        print("Mean bias(a), Instrumental FDC with e/d estimator ",np.mean(aBiasApprox))
        print("Mean bias(a), Instrumental FDC with perfect e/d ",np.mean(aBiasInst))
        
        
        cf00 = ax[0][0].contourf(evals, dvals, aBiasTrue, aLevels, linestyles="solid")
        cf01 = ax[0][1].contourf(evals, dvals, aBiasApprox, aLevels, linestyles="solid")
        cf02 = ax[0][2].contourf(evals, dvals, aBiasInst, aLevels, linestyles="solid")
        cf10 = ax[1][0].contourf(evals, dvals, aVarTrue, aVarLevels, linestyles="solid")
        cf11 = ax[1][1].contourf(evals, dvals, aVarApprox, aVarLevels, linestyles="solid")
        cf12 = ax[1][2].contourf(evals, dvals, aVarInst, aVarLevels, linestyles="solid")
        
        print(aBiasTrue)
        print(aBiasApprox)
        print(aBiasInst)

    if plottype=="c":
    
        cMaxTrue = np.max(np.abs(cBiasTrue))
        cMaxApprox = np.max(np.abs(cBiasApprox))
        cMaxInst = np.max(np.abs(cBiasInst))
        cMax = np.max([cMaxTrue,cMaxApprox,cMaxInst])
        cLevels = np.linspace(-1 * cMax, cMax, 21)
        
        cMaxVarTrue = np.max(np.abs(cVarTrue))
        cMaxVarApprox = np.max(np.abs(cVarApprox))
        cMaxVarInst = np.max(np.abs(cVarInst))
        cMaxVar = np.max([cMaxVarTrue,cMaxVarApprox,cMaxVarInst])
        cVarLevels = np.linspace(0, cMaxVar, 21)
       
        print("Mean bias(c), True FDC ",np.mean(cBiasTrue))
        print("Mean bias(c), Instrumental FDC with e/d estimator ",np.mean(cBiasApprox))
        print("Mean bias(c), Instrumental FDC with perfect e/d ",np.mean(cBiasInst))
        
        cf00 = ax[0][0].contourf(evals, dvals, cBiasTrue, cLevels, linestyles="solid")
        cf01 = ax[0][1].contourf(evals, dvals, cBiasApprox, cLevels, linestyles="solid")
        cf02 = ax[0][2].contourf(evals, dvals, cBiasInst, cLevels, linestyles="solid")
        cf10 = ax[1][0].contourf(evals, dvals, cVarTrue, cVarLevels, linestyles="solid")
        cf11 = ax[1][1].contourf(evals, dvals, cVarApprox, cVarLevels, linestyles="solid")
        cf12 = ax[1][2].contourf(evals, dvals, cVarInst, cVarLevels, linestyles="solid")
        
        print(cBiasTrue)
        print(cBiasApprox)
        print(cBiasInst)
        
    if plottype=="debug":
    
        cMaxTrue = np.max(np.abs(cBiasTrue))
        cMaxApprox = np.max(np.abs(cBiasApprox))
        cMax = np.max([cMaxTrue,cMaxApprox])
        cLevels = np.linspace(-1 * cMax, cMax, 21)
        
        cMaxVarTrue = np.max(np.abs(cVarTrue))
        cMaxVarApprox = np.max(np.abs(cVarApprox))
        cMaxVar = np.max([cMaxVarTrue,cMaxVarApprox])
        cVarLevels = np.linspace(0, cMaxVar, 21)
        
        cf00 = ax[0][0].contourf(evals, dvals, aBiasTrue, cLevels, linestyles="solid")
        cf01 = ax[0][1].contourf(evals, dvals, cBiasTrue, cLevels, linestyles="solid")
        cf10 = ax[1][0].contourf(evals, dvals, acBiasTrue, cVarLevels, linestyles="solid")
        #cf11 = ax[1][1].contourf(avals, cvals, cVarApprox, cVarLevels, linestyles="solid")
    
        
    #colorlabel = 'Probability Density'
    ax[2][0].set_xlabel("e")
    ax[2][1].set_xlabel("e")
    ax[2][2].set_xlabel("e")
    ax[0][0].set_ylabel("d")
    ax[1][0].set_ylabel("d")
    ax[2][0].set_ylabel("d")
    
    ax[0][0].set_title('True FDC', fontstyle='italic')
    ax[0][1].set_title('Instrumental FDC with e/d estimator', fontstyle='italic')
    ax[0][2].set_title('Instrumental FDC with perfect e/d', fontstyle='italic')
    
    axins0 = inset_axes(ax[0][2],
           width="5%", # width = 10% of parent_bbox width
           height="100%", # height : 50%
           loc=6,
           bbox_to_anchor=(1.05, 0., 1, 1),
           bbox_transform=ax[0][2].transAxes,
           borderpad=0,
       )
    axins1 = inset_axes(ax[1][2],
           width="5%", # width = 10% of parent_bbox width
           height="100%", # height : 50%
           loc=6,
           bbox_to_anchor=(1.05, 0., 1, 1),
           bbox_transform=ax[1][2].transAxes,
           borderpad=0,
       )
    
    if plottype=="ac":
        axins2 = inset_axes(ax[2][2],
               width="5%", # width = 10% of parent_bbox width
               height="100%", # height : 50%
               loc=6,
               bbox_to_anchor=(1.05, 0., 1, 1),
               bbox_transform=ax[2][2].transAxes,
               borderpad=0,
           )

    cbar0 = plt.colorbar(cf02, cax=axins0)
    cbar0.set_label("Bias")
    
    cbar1 = fig.colorbar(cf12, cax=axins1)
    cbar1.set_label("Variance")
    
    if plottype=="ac":
        cbar2 = fig.colorbar(cf22, cax=axins2)
        cbar2.set_label("Score")

    #ax[0][1].collections[0].colorbar.set_label("Bias")
    #ax[1][1].collections[0].colorbar.set_label("Variance")
    
    
    fig.set_size_inches((22,16), forward=False)
    #fig.tight_layout()
    fig.savefig(ofile + '.pdf', format='pdf', dpi=1200)
    
def acScanres_plot(ofile,plottype):

    
    arange = crange = [-2,2]
    anum = cnum = 3
    numsamp = int(1e3)
    numruns = int(1e2)
    e = 1
    b=1
    d=1
    g=1
    noise = 1
    
    
    
    save=True
    #savename="acdatainstresadj_1ksamp_100run_10div_e1"
    savename="acdatainstresadj_1ksamp_100run_3div_e1_alphascan_subedivd"
    #savename="acdatainst_2ksamp_500run_10div"
    load=not save
    
    tperfedivd = False
    
    if load:
        npzfile = np.load(savename + ".npz")
        
        meanres1var = npzfile['meanres1var']
        meanres2var = npzfile['meanres2var']
        meanres1covX = npzfile['meanres1covX']
        meanres2covX = npzfile['meanres2covX']
        meanres1covM = npzfile['meanres1covM']
        meanres2covM = npzfile['meanres2covM']
        meanres1covW = npzfile['meanres1covW']
        meanres2covW = npzfile['meanres2covW']
        meanadjres2var = npzfile['meanadjres2var']
        meanadjres2covX = npzfile['meanadjres2covX']
        meanadjres2covM = npzfile['meanadjres2covM']
        meanadjres2covW = npzfile['meanadjres2covW']
        meanadjres2varscan = npzfile['meanadjres2varscan']
        meanadjres2covXscan = npzfile['meanadjres2covXscan']
        meanadjres2covMscan = npzfile['meanadjres2covMscan']
        meanadjres2covWscan = npzfile['meanadjres2covWscan']
        
    
    else:
        meanres1var, meanres2var, meanres1covX, meanres2covX, meanres1covM, meanres2covM, meanres1covW, meanres2covW, meanadjres2var, meanadjres2covX, meanadjres2covM, meanadjres2covW, meanadjres2varscan, meanadjres2covXscan, meanadjres2covMscan, meanadjres2covWscan = acScan_IFDCres_ICM(arange,crange,anum,cnum,tperfedivd,numsamp,numruns,e=e,b=b,d=d,g=g,noise=noise)
        
        if save:
            np.savez(savename + ".npz", meanres1var=meanres1var, meanres2var=meanres2var, meanres1covX=meanres1covX, meanres2covX=meanres2covX, meanres1covM=meanres1covM, meanres2covM=meanres2covM, meanres1covW=meanres1covW, meanres2covW=meanres2covW, meanadjres2var=meanadjres2var, meanadjres2covX=meanadjres2covX, meanadjres2covM=meanadjres2covM, meanadjres2covW=meanadjres2covW, meanadjres2varscan=meanadjres2varscan, meanadjres2covXscan=meanadjres2covXscan, meanadjres2covMscan=meanadjres2covMscan, meanadjres2covWscan=meanadjres2covWscan)
    
        
   
    
    if plottype=="vXMW":
    
        fig, ax = plt.subplots(3,4)
    
        avals = np.linspace(arange[0],arange[1],anum)
        cvals = np.linspace(crange[0],crange[1],cnum)
            
        var1Max = np.max(meanres1var)
        covX1Max = np.max(meanres1covX)
        covM1Max = np.max(meanres1covM)
        covW1Max = np.max(meanres1covW)
        Max1 = np.max([var1Max,covX1Max,covM1Max,covW1Max])
        
        var1Min = np.min(meanres1var)
        covX1Min = np.min(meanres1covX)
        covM1Min = np.min(meanres1covM)
        covW1Min = np.min(meanres1covW)
        Min1 = np.min([var1Min,covX1Min,covM1Min,covW1Min])
        Levels1 = np.linspace(Min1, Max1, 101)
        
        var2Max = np.max(meanres2var)
        covX2Max = np.max(meanres2covX)
        covM2Max = np.max(meanres2covM)
        covW2Max = np.max(meanres2covW)
        Max2 = np.max([var2Max,covX2Max,covM2Max,covW2Max])
        
        var2Min = np.min(meanres2var)
        covX2Min = np.min(meanres2covX)
        covM2Min = np.min(meanres2covM)
        covW2Min = np.min(meanres2covW)
        Min2 = np.min([var2Min,covX2Min,covW2Min])
        Levels2 = np.linspace(Min2, Max2, 101)
        
        varadj2Max = np.max(meanadjres2var)
        covXadj2Max = np.max(meanadjres2covX)
        covMadj2Max = np.max(meanadjres2covM)
        covWadj2Max = np.max(meanadjres2covW)
        Maxadj2 = np.max([varadj2Max,covXadj2Max,covMadj2Max,covWadj2Max])
        
        varadj2Min = np.min(meanadjres2var)
        covXadj2Min = np.min(meanadjres2covX)
        covMadj2Min = np.min(meanadjres2covM)
        covWadj2Min = np.min(meanadjres2covW)
        Minadj2 = np.min([varadj2Min,covXadj2Min,covWadj2Min])
        Levelsadj2 = np.linspace(Minadj2, Maxadj2, 101)
        
        Levels = np.linspace(np.min([Min1,Min2,Minadj2,-1.02622768]),np.max([Max1,Max2,Maxadj2,3.111983794]),101)
        print(np.min([Min1,Min2,Minadj2]))
        print(np.max([Max1,Max2,Maxadj2]))
        
       
        
        cf00 = ax[0][0].contourf(avals, cvals, meanres1var, Levels, linestyles="solid")
        cf01 = ax[0][1].contourf(avals, cvals, meanres1covX, Levels, linestyles="solid")
        cf02 = ax[0][2].contourf(avals, cvals, meanres1covM, Levels, linestyles="solid")
        cf03 = ax[0][3].contourf(avals, cvals, meanres1covW, Levels, linestyles="solid")
        cf10 = ax[1][0].contourf(avals, cvals, meanres2var, Levels, linestyles="solid")
        cf11 = ax[1][1].contourf(avals, cvals, meanres2covX, Levels, linestyles="solid")
        cf12 = ax[1][2].contourf(avals, cvals, meanres2covM, Levels, linestyles="solid")
        cf13 = ax[1][3].contourf(avals, cvals, meanres2covW, Levels, linestyles="solid")
        cf20 = ax[2][0].contourf(avals, cvals, meanadjres2var, Levels, linestyles="solid")
        cf21 = ax[2][1].contourf(avals, cvals, meanadjres2covX, Levels, linestyles="solid")
        cf22 = ax[2][2].contourf(avals, cvals, meanadjres2covM, Levels, linestyles="solid")
        cf23 = ax[2][3].contourf(avals, cvals, meanadjres2covW, Levels, linestyles="solid")
        
        var1MaxAv = np.mean(var1Max.flatten())
        var2MaxAv = np.mean(var2Max.flatten())
        covX1MaxAv = np.mean(covX1Max.flatten())
        covX2MaxAv = np.mean(covX2Max.flatten())
        covM1MaxAv = np.mean(covM1Max.flatten())
        covM2MaxAv = np.mean(covM2Max.flatten())
        covW1MaxAv = np.mean(covW1Max.flatten())
        covW2MaxAv = np.mean(covW2Max.flatten())
        
        varadj2MaxAv = np.mean(varadj2Max.flatten())
        covXadj2MaxAv = np.mean(covXadj2Max.flatten())
        covMadj2MaxAv = np.mean(covMadj2Max.flatten())
        covWadj2MaxAv = np.mean(covWadj2Max.flatten())
        
        ax[0][0].text(-1,1.5, "mean: " + '{0:.3f}'.format(var1MaxAv), fontsize = 22, color='r')
        ax[0][1].text(-1,1.5, "mean: " + '{0:.3f}'.format(covX1MaxAv), fontsize = 22, color='r')
        ax[0][2].text(-1,1.5, "mean: " + '{0:.3f}'.format(covM1MaxAv), fontsize = 22, color='r')
        ax[0][3].text(-1,1.5, "mean: " + '{0:.3f}'.format(covW1MaxAv), fontsize = 22, color='r')
        ax[1][0].text(-1,1.5, "mean: " + '{0:.3f}'.format(var2MaxAv), fontsize = 22, color='r')
        ax[1][1].text(-1,1.5, "mean: " + '{0:.3f}'.format(covX2MaxAv), fontsize = 22, color='r')
        ax[1][2].text(-1,1.5, "mean: " + '{0:.3f}'.format(covM2MaxAv), fontsize = 22, color='r')
        ax[1][3].text(-1,1.5, "mean: " + '{0:.3f}'.format(covW2MaxAv), fontsize = 22, color='r')
        ax[2][0].text(-1,1.5, "mean: " + '{0:.3f}'.format(varadj2MaxAv), fontsize = 22, color='r')
        ax[2][1].text(-1,1.5, "mean: " + '{0:.3f}'.format(covXadj2MaxAv), fontsize = 22, color='r')
        ax[2][2].text(-1,1.5, "mean: " + '{0:.3f}'.format(covMadj2MaxAv), fontsize = 22, color='r')
        ax[2][3].text(-1,1.5, "mean: " + '{0:.3f}'.format(covWadj2MaxAv), fontsize = 22, color='r')
        
        
        #print(acBiasTrue)
        #print(acBiasApprox)
        #print(acBiasInst)
        
            #colorlabel = 'Probability Density'
        ax[1][0].set_xlabel("a")
        ax[1][1].set_xlabel("a")
        ax[1][2].set_xlabel("a")
        ax[1][3].set_xlabel("a")
        ax[0][0].set_ylabel("c")
        ax[1][0].set_ylabel("c")
        ax[2][0].set_ylabel("c")
        
        ax[0][0].set_title('Var(res)', fontstyle='italic')
        ax[0][1].set_title('Cov(X,res)', fontstyle='italic')
        ax[0][2].set_title('Cov(M,res)', fontstyle='italic')
        ax[0][3].set_title('Cov(W,res)', fontstyle='italic')
        
        axins0 = inset_axes(ax[0][3],
               width="5%", # width = 10% of parent_bbox width
               height="100%", # height : 50%
               loc=6,
               bbox_to_anchor=(1.05, 0., 1, 1),
               bbox_transform=ax[0][3].transAxes,
               borderpad=0,
           )
        axins1 = inset_axes(ax[1][3],
               width="5%", # width = 10% of parent_bbox width
               height="100%", # height : 50%
               loc=6,
               bbox_to_anchor=(1.05, 0., 1, 1),
               bbox_transform=ax[1][3].transAxes,
               borderpad=0,
           )
        axins2 = inset_axes(ax[2][3],
               width="5%", # width = 10% of parent_bbox width
               height="100%", # height : 50%
               loc=6,
               bbox_to_anchor=(1.05, 0., 1, 1),
               bbox_transform=ax[2][3].transAxes,
               borderpad=0,
           )
        

        cbar0 = plt.colorbar(cf03, cax=axins0)
        cbar0.set_label("Direct residual")
        
        cbar1 = fig.colorbar(cf13, cax=axins1)
        cbar1.set_label("Perfect e/d residual")
        
        cbar2 = fig.colorbar(cf23, cax=axins2)
        cbar2.set_label("Subtracted e/d residual")

        #ax[0][1].collections[0].colorbar.set_label("Bias")
        #ax[1][1].collections[0].colorbar.set_label("Variance")
        
        
        fig.set_size_inches((30,24), forward=False)
        #fig.tight_layout()
        fig.savefig(ofile + '.pdf', format='pdf', dpi=1200)
    
    if plottype=="alphascan":
    
        fig, ax = plt.subplots(1,2)
    
        avals = np.linspace(arange[0],arange[1],anum)
        cvals = np.linspace(crange[0],crange[1],cnum)
    
        numalpha = 10
        
        alphavals = np.linspace(0,1,numalpha)
        
        for ai in range(anum):
            for ci in range(cnum):
                
                ax0 = ax[0].plot(alphavals, meanadjres2covXscan[ai,ci,:], 'r')
                ax1 = ax[1].plot(alphavals, meanadjres2covWscan[ai,ci,:], 'b')
                
        ax[0].set_xlabel("alpha")
        ax[1].set_xlabel("alpha")
        ax[0].set_ylabel("Cov(res,X)")
        ax[1].set_ylabel("Cov(res,W)")
        ax[0].grid()
        ax[1].grid()
        
        ax[0].set_title('Cov X Scan', fontstyle='italic')
        ax[1].set_title('Cov W Scan', fontstyle='italic')
        
        fig.set_size_inches((30,16), forward=False)
        #fig.tight_layout()
        fig.savefig(ofile + '.pdf', format='pdf', dpi=1200)
            
   
    if plottype=="debug":
    
        cMaxTrue = np.max(np.abs(cBiasTrue))
        cMaxApprox = np.max(np.abs(cBiasApprox))
        cMax = np.max([cMaxTrue,cMaxApprox])
        cLevels = np.linspace(-1 * cMax, cMax, 21)
        
        cMaxVarTrue = np.max(np.abs(cVarTrue))
        cMaxVarApprox = np.max(np.abs(cVarApprox))
        cMaxVar = np.max([cMaxVarTrue,cMaxVarApprox])
        cVarLevels = np.linspace(0, cMaxVar, 21)
        
        cf00 = ax[0][0].contourf(avals, cvals, aBiasTrue, cLevels, linestyles="solid")
        cf01 = ax[0][1].contourf(avals, cvals, cBiasTrue, cLevels, linestyles="solid")
        cf10 = ax[1][0].contourf(avals, cvals, acBiasTrue, cVarLevels, linestyles="solid")
        #cf11 = ax[1][1].contourf(avals, cvals, cVarApprox, cVarLevels, linestyles="solid")
    
  


def edivdScan_allmethods(erange,drange,enum,dnum,numsamp=1e4,numruns=1e2,a=1,b=1,c=1,g=1,noise=1):

    evals = np.linspace(erange[0],erange[1],enum)
    dvals = np.linspace(drange[0],drange[1],dnum)
    print(evals)
    print(dvals)
    
    #b = 1
    #d = 1
    
    edivdBias1 = np.empty([enum,dnum])
    edivdVar1 = np.empty([enum,dnum])
    edivdBias2 = np.empty([enum,dnum])
    edivdVar2 = np.empty([enum,dnum])
    edivdBias3 = np.empty([enum,dnum])
    edivdVar3 = np.empty([enum,dnum])
    edivdBias4 = np.empty([enum,dnum])
    edivdVar4 = np.empty([enum,dnum])
    eBias = np.empty([enum,dnum])
    eVar = np.empty([enum,dnum])
    dBias = np.empty([enum,dnum])
    dVar = np.empty([enum,dnum])
    gBias = np.empty([enum,dnum])
    gVar = np.empty([enum,dnum])
    
    for ei,e in enumerate(evals):
        for di,d in enumerate(dvals):

            # graph edges: [("V","X"),("W","X"),("X","M"),("M","Y"),("W","Y"),("W","M")]
            confmodel = models.generate_InstrumentModel([d,c,a,b,g,e],noise=noise)

            runvalues = []

            for run in range(numruns):
            
                #BEGIN (numruns) THREADS, VARS runvalues, ?samples
                
                os.system('cls||clear')
                print("Running Instrumental e/d Estimators")
                print("e = ", e, " in range [", erange[0], ", ", erange[1], "]")
                print("d = ", d, " in range [", drange[0], ", ", drange[1], "]")
                print("Run: ",run," of ",numruns)
                
                #if run>1:
                #    print("Mean g estimates:")
                #    print( np.mean(XVslope[run-1]))
                #    
                #    print("Mean c estimates:")
                #    print( np.mean(cInstEstimators[run-1]) )
                    
                
                confsamples = confmodel.sample(numsamp)
                #for samp in confsamples[-3:-1]:
                #    print(samp)
                
                values = [[x for x in samp.values()] for samp in confsamples]
                valuesT = np.array(values).T.tolist()
                runvalues.append(valuesT)
                
                #CONT THREADS

            nodedict = dict([(node,confmodel.nodes.index(node)) for node in confmodel.nodes])
            
            #CONT THREADS
            
            sampledicts = [dict([(node,runvalues[run][nodedict[node]]) for node in confmodel.nodes]) for run in range(numruns)]

            Vsamples = [sampledicts[run]["V"] for run in range(numruns)]
            Wsamples = [sampledicts[run]["W"] for run in range(numruns)]
            Xsamples = [sampledicts[run]["X"] for run in range(numruns)]
            Msamples = [sampledicts[run]["M"] for run in range(numruns)]
            Ysamples = [sampledicts[run]["Y"] for run in range(numruns)]

            # Calculate the regression slope from x -> m (c)
            cEstimators = [np.dot(Xsamples[run],Msamples[run])/np.dot(Xsamples[run],Xsamples[run]) for run in range(numruns)]
            
            # Calculate the residuals um for each datapoint {x,m}
            uMsamples = [[Msamples[run][samp] - cEstimators[run] * Xsamples[run][samp] for samp in range(numsamp)] for run in range(numruns)]
            
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
           
            # Now we must compare several averaging strategies:
            
            #edivdEst1 = [np.sum(eWplusuMsamples[run]) / np.sum(dWplusuXsamples[run]) for run in range(numruns)]
                    
            edivdEst1 = [cEstimators[run] - cInstEstimators[run] for run in range(numruns)]

            edivdEst2 = [np.linalg.norm(eWplusuMsamples[run]) / np.linalg.norm(dWplusuXsamples[run]) for run in range(numruns)]
            
            edivdEst3 = [np.sum(np.array(eWplusuMsamples[run]) / np.array(dWplusuXsamples[run])) / numsamp for run in range(numruns)]
            
            eEst = [np.linalg.norm(eWplusuMsamples[run]) for run in range(numruns)]
            dEst = [np.linalg.norm(dWplusuXsamples[run]) for run in range(numruns)]
           
            edivdMean1 = stat.mean(edivdEst1)
            edivdBias1[ei,di] = edivdMean1 - (e/d)
            edivdVar1[ei,di] = stat.variance(edivdEst1)

            edivdMean2 = stat.mean(edivdEst2)
            edivdBias2[ei,di] = edivdMean2 - (e/d)
            edivdVar2[ei,di] = stat.variance(edivdEst2)
            
            edivdMean3 = stat.mean(edivdEst3)
            edivdBias3[ei,di] = edivdMean3 - (e/d)
            edivdVar3[ei,di] = stat.variance(edivdEst3)
            
            eMean = stat.mean(eEst)
            eBias[ei,di] = eMean
            eVar[ei,di] = stat.variance(eEst)

            dMean = stat.mean(dEst)
            dBias[ei,di] = dMean
            dVar[ei,di] = stat.variance(dEst)
            
            edivdMean4 = eMean / dMean
            edivdBias4[ei,di] = edivdMean4 - (e/d)
            edivdVar4[ei,di] = 0
            
            gMean = stat.mean(XVslope)
            gBias[ei,di] = gMean - g
            gVar[ei,di] = stat.variance(XVslope)
            
            
    return edivdBias1, edivdVar1, edivdBias2, edivdVar2, edivdBias3, edivdVar3, edivdBias4, edivdVar4, eBias, eVar, dBias, dVar, gBias, gVar

def edivdScan_slides(erange,drange,enum,dnum,numsamp=1e4,numruns=1e2,a=1,b=1,c=1,g=1,noise=1):

    evals = np.linspace(erange[0],erange[1],enum)
    dvals = np.linspace(drange[0],drange[1],dnum)
    print(evals)
    print(dvals)
    #b = 1
    #d = 1
    
    edivdBias2 = np.empty([enum,dnum])
    
    for ei,e in enumerate(evals):
        for di,d in enumerate(dvals):

            # graph edges: [("V","X"),("W","X"),("X","M"),("M","Y"),("W","Y"),("W","M")]
            confmodel = models.generate_InstrumentModel([d,c,a,b,g,e],noise=noise)

            runvalues = []

            for run in range(numruns):
            
                #BEGIN (numruns) THREADS, VARS runvalues, ?samples
                
                os.system('cls||clear')
                print("Running Instrumental e/d Estimators")
                print("e = ", e, " in range [", erange[0], ", ", erange[1], "]")
                print("d = ", d, " in range [", drange[0], ", ", drange[1], "]")
                print("Run: ",run," of ",numruns)
                
                #if run>1:
                #    print("Mean g estimates:")
                #    print( np.mean(XVslope[run-1]))
                #    
                #    print("Mean c estimates:")
                #    print( np.mean(cInstEstimators[run-1]) )
                    
                
                confsamples = confmodel.sample(numsamp)
                #for samp in confsamples[-3:-1]:
                #    print(samp)
                
                values = [[x for x in samp.values()] for samp in confsamples]
                valuesT = np.array(values).T.tolist()
                runvalues.append(valuesT)
                
                #CONT THREADS

            nodedict = dict([(node,confmodel.nodes.index(node)) for node in confmodel.nodes])
            
            #CONT THREADS
            
            sampledicts = [dict([(node,runvalues[run][nodedict[node]]) for node in confmodel.nodes]) for run in range(numruns)]

            Vsamples = [sampledicts[run]["V"] for run in range(numruns)]
            Wsamples = [sampledicts[run]["W"] for run in range(numruns)]
            Xsamples = [sampledicts[run]["X"] for run in range(numruns)]
            Msamples = [sampledicts[run]["M"] for run in range(numruns)]
            Ysamples = [sampledicts[run]["Y"] for run in range(numruns)]

            # Calculate the regression slope from x -> m (c)
            cEstimators = [np.dot(Xsamples[run],Msamples[run])/np.dot(Xsamples[run],Xsamples[run]) for run in range(numruns)]
            
            # Calculate the residuals um for each datapoint {x,m}
            uMsamples = [[Msamples[run][samp] - cEstimators[run] * Xsamples[run][samp] for samp in range(numsamp)] for run in range(numruns)]
            
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
           
            # Now we must compare several averaging strategies:
            

            edivdEst2 = [np.linalg.norm(eWplusuMsamples[run]) / np.linalg.norm(dWplusuXsamples[run]) for run in range(numruns)]
            

            edivdMean2 = stat.mean(edivdEst2)
            edivdBias2[ei,di] = edivdMean2 - (e/d)
            
            
    return edivdBias2


def fmt(x):
    s = f"{x:.1f}"
    if s.endswith("0"):
        s = f"{x:.0f}"
    return rf"{s}" if plt.rcParams["text.usetex"] else f"{s}"


def edivdScan_plot(ofile,plottype):

    
    erange = drange = [0.1,2]
    enum = dnum = 20
    numsamp = int(1e3)
    numruns = int(1e2)
    thisa = 1
    thisc = 1
    thisb = 1
    thisg = 1
    thisnoise = 1
    
    save=False
    savename="edivdscan_1ksamp_100run_20div_noise1"
    #savename="acdatainst_2ksamp_500run_10div"
    load= not save
    
    if load:
        npzfile = np.load(savename + ".npz")
        
        edivdBias1 = npzfile['edivdBias1']
        edivdVar1 = npzfile['edivdVar1']
        edivdBias2 = npzfile['edivdBias2']
        edivdVar2 = npzfile['edivdVar2']
        edivdBias3 = npzfile['edivdBias3']
        edivdVar3 = npzfile['edivdVar3']
        edivdBias4 = npzfile['edivdBias4']
        edivdVar4 = npzfile['edivdVar4']
        eBias = npzfile['eBias']
        eVar = npzfile['eVar']
        dBias = npzfile['dBias']
        dVar = npzfile['dVar']
        gBias = npzfile['gBias']
        gVar = npzfile['gVar']
        
        
    
    else:
        edivdBias1, edivdVar1, edivdBias2, edivdVar2, edivdBias3, edivdVar3, edivdBias4, edivdVar4, eBias, eVar, dBias, dVar, gBias, gVar = edivdScan_allmethods(erange,drange,enum,dnum,numsamp,numruns,a=thisa,b=thisb,c=thisc,g=thisg,noise=thisnoise)
        
        if save:
            np.savez(savename + ".npz", edivdBias1=edivdBias1, edivdVar1=edivdVar1, edivdBias2=edivdBias2, edivdVar2=edivdVar2, edivdBias3=edivdBias3, edivdVar3=edivdVar3, edivdBias4=edivdBias4, edivdVar4=edivdVar4, eBias=eBias, eVar=eVar, dBias=dBias, dVar=dVar, gBias=gBias, gVar=gVar)
    
        
    
    #fig, ax = plt.subplots(3,3)
    
    evals = np.linspace(erange[0],erange[1],enum)
    dvals = np.linspace(drange[0],drange[1],dnum)
    
    if plottype=="edivd":
        fig, ax = plt.subplots(2,2)
    
        edivdMax1 = np.max(edivdBias1)
        edivdMax2 = np.max(edivdBias2)
        edivdMax3 = np.max(edivdBias3)
        
        edivdMin1 = np.min(edivdBias1)
        edivdMin2 = np.min(edivdBias2)
        edivdMin3 = np.min(edivdBias3)

        edivdMax = np.max([edivdMax1,edivdMax2])
        edivdMin = np.min([edivdMin1,edivdMin2])
        edivdLevels = np.linspace(edivdMin, edivdMax, 101)

        edivdMaxVar1 = np.max(np.abs(edivdVar1))
        edivdMaxVar2 = np.max(np.abs(edivdVar2))
        edivdMaxVar3 = np.max(np.abs(edivdVar3))

        edivdMaxVar = np.max([edivdMaxVar1,edivdMaxVar2])
        edivdVarLevels = np.linspace(0, edivdMaxVar, 21)
       
        print("Mean bias(e/d), Mean(|e|)/Mean(|d|) estimator ",np.mean(edivdBias2))
        print("Mean bias(e/d), Mean(e/d) estimator ",np.mean(edivdBias3))
        print("Mean bias(e/d), c - (c+e/d) estimator ",np.mean(edivdBias1))
        
        
        biaslevels = [-10,-8,-6,-4,-2,-1,-0.8,-0.6,-0.4,-0.2,-0.1,0,0.1,0.2,0.4,0.6,0.8,1]
        
        cf00 = ax[0][0].contour(evals, dvals, edivdBias1, biaslevels, linestyles="solid")
        #cf01 = ax[0][1].contourf(evals, dvals, edivdBias2, edivdLevels, linestyles="solid")
        cf02 = ax[0][1].contour(evals, dvals, edivdBias2, biaslevels, linestyles="solid")
        cf10 = ax[1][0].contourf(evals, dvals, edivdVar1, edivdVarLevels, linestyles="solid")
        #cf11 = ax[1][1].contourf(evals, dvals, edivdVar2, edivdVarLevels, linestyles="solid")
        cf12 = ax[1][1].contourf(evals, dvals, edivdVar2, edivdVarLevels, linestyles="solid")
        
        ax[0][0].clabel(cf00, cf00.levels, inline=True, fmt=fmt, fontsize=10)
        ax[0][1].clabel(cf02, cf02.levels, inline=True, fmt=fmt, fontsize=10)
        
        print(edivdBias2)
        print(edivdBias3)
        print(edivdBias4)
        
    if plottype=="ed":
    
        eMax = np.max(np.abs(eBias))
        dMax = np.max(np.abs(dBias))
        
        edMax = np.max([eMax,dMax])
        
        edLevels = np.linspace(-1 * edMax, edMax, 21)
        
        eMaxVar = np.max(np.abs(eVar))
        dMaxVar = np.max(np.abs(dVar))

        edMaxVar = np.max([eMaxVar,dMaxVar])
        edVarLevels = np.linspace(0, edMaxVar, 21)
       
        print("Mean bias(e), residual estimator ",np.mean(eBias))
        print("Mean bias(d), residual estimator ",np.mean(dBias))
        print("Mean bias(e/d), Mean|e|/Mean|d| estimator ",np.mean(edivdBias4))
        
        print("Mean bias(g), residual estimator ",np.mean(gBias))
        
        cf00 = ax[0][0].contourf(evals, dvals, eBias, edLevels, linestyles="solid")
        cf01 = ax[0][1].contourf(evals, dvals, dBias, edLevels, linestyles="solid")
        cf02 = ax[0][2].contourf(evals, dvals, edivdBias4, edLevels, linestyles="solid")
        cf10 = ax[1][0].contourf(evals, dvals, eVar, edVarLevels, linestyles="solid")
        cf11 = ax[1][1].contourf(evals, dvals, dVar, edVarLevels, linestyles="solid")
        cf12 = ax[1][2].contourf(evals, dvals, edivdVar4, edVarLevels, linestyles="solid")
        
        print(eBias)
        print(dBias)
        print(edivdBias4)
        print(gBias)
        
    if plottype=="debug":
    
        cMaxTrue = np.max(np.abs(cBiasTrue))
        cMaxApprox = np.max(np.abs(cBiasApprox))
        cMax = np.max([cMaxTrue,cMaxApprox])
        cLevels = np.linspace(-1 * cMax, cMax, 21)
        
        cMaxVarTrue = np.max(np.abs(cVarTrue))
        cMaxVarApprox = np.max(np.abs(cVarApprox))
        cMaxVar = np.max([cMaxVarTrue,cMaxVarApprox])
        cVarLevels = np.linspace(0, cMaxVar, 21)
        
        cf00 = ax[0][0].contourf(avals, cvals, aBiasTrue, cLevels, linestyles="solid")
        cf01 = ax[0][1].contourf(avals, cvals, cBiasTrue, cLevels, linestyles="solid")
        cf10 = ax[1][0].contourf(avals, cvals, acBiasTrue, cVarLevels, linestyles="solid")
        #cf11 = ax[1][1].contourf(avals, cvals, cVarApprox, cVarLevels, linestyles="solid")
    
        
    #colorlabel = 'Probability Density'
    ax[1][0].set_xlabel("e")
    ax[1][1].set_xlabel("e")
    #ax[1][2].set_xlabel("e")
    ax[0][0].set_ylabel("d")
    ax[1][0].set_ylabel("d")
    #ax[2][0].set_ylabel("d")
    
    ax[0][0].set_title('c - (c+e/d) estimator', fontstyle='italic')
    ax[0][1].set_title('|e|/|d| estimator', fontstyle='italic')
    #ax[0][1].set_title('Mean(e/d) estimator', fontstyle='italic')
    
    #axins0 = inset_axes(ax[0][1],
    #       width="5%", # width = 10% of parent_bbox width
    #       height="100%", # height : 50%
    #       loc=6,
    #       bbox_to_anchor=(1.05, 0., 1, 1),
    #       bbox_transform=ax[0][1].transAxes,
    #       borderpad=0,
    #   )
    axins1 = inset_axes(ax[1][1],
           width="5%", # width = 10% of parent_bbox width
           height="100%", # height : 50%
           loc=6,
           bbox_to_anchor=(1.05, 0., 1, 1),
           bbox_transform=ax[1][1].transAxes,
           borderpad=0,
       )
    

    #cbar0 = plt.colorbar(cf02, cax=axins0)
    #cbar0.set_label("Bias")
    
    cbar1 = fig.colorbar(cf12, cax=axins1)
    cbar1.set_label("Variance")
    #ax[0][1].collections[0].colorbar.set_label("Bias")
    #ax[1][1].collections[0].colorbar.set_label("Variance")
    
    
    fig.set_size_inches((16,16), forward=False)
    #fig.tight_layout()
    fig.savefig(ofile + '.pdf', format='pdf', dpi=1200)


def edivdScan_plot_slides(ofile,plottype):

    
    erange = drange = [-2,2]
    enum = dnum = 20
    numsamp = int(1e3)
    numruns = int(1e2)
    thisa = 1
    thisc = 1
    thisb = 1
    thisg = 1
    thisnoise = 1
    
    save=False
    savename="edivdscan_slides"
    #savename="acdatainst_2ksamp_500run_10div"
    load= not save
    
    if load:
        npzfile = np.load(savename + ".npz")
        
        edivdBias2 = npzfile['edivdBias1']
        
        
    
    else:
        edivdBias1 = edivdScan_slides(erange,drange,enum,dnum,numsamp,numruns,a=thisa,b=thisb,c=thisc,g=thisg,noise=thisnoise)
        
        if save:
            np.savez(savename + ".npz", edivdBias1=edivdBias1)
    
        
    
    #fig, ax = plt.subplots(3,3)
    
    evals = np.linspace(erange[0],erange[1],enum)
    dvals = np.linspace(drange[0],drange[1],dnum)
    
    cmap = clr.LinearSegmentedColormap.from_list('custom green', ['#ffffff','#1ed760','#000000'], N=256)
    cmap2 = clr.LinearSegmentedColormap.from_list('custom green', ['#ffffff','#1db954','#000000'], N=256)

    
    if plottype=="edivd":
        fig, ax = plt.subplots(1,1)
    
        edivdMax = np.max(edivdBias2)
        edivdMin = np.min(edivdBias2)
        print(edivdMin,edivdMax)
        edivdLevels = np.linspace(-20,20, 101)
       
        print("Mean bias(e/d), Mean(|e|)/Mean(|d|) estimator ",np.mean(edivdBias2))

        cf = ax.contourf(evals, dvals, edivdBias2, edivdLevels, linestyles="solid", cmap=cmap)
   

    ax.axis('off')
    fig.set_size_inches((16,16), forward=False)
    #fig.tight_layout()
    fig.savefig(ofile + '.jpg', format='jpg', dpi=1200,bbox_inches='tight',pad_inches = 0)


parser = argparse.ArgumentParser()

parser.add_argument("--outfile", "-o", help="output file name", default="marg_out")
parser.add_argument("--plottype", "-t", help="plot type (a, c, ac, edivd, or ed)", default="ac")


# Read arguments from the command line
args = parser.parse_args()
ofile = args.outfile
plottype = args.plottype


# List of colormap options: https://matplotlib.org/3.1.1/gallery/color/colormap_reference.html

#acScan_fdc_ifdc(ofile,plottype)
#acScan_ifdc_jointx(ofile,plottype)
#acScan_plot(ofile,plottype)
#acScan_edifdc_cm(ofile,plottype)
#acScan_edifdc_icm(ofile,plottype)
#acScan_edifdc_icm_withalpha(ofile,plottype)
#edScan_plot(ofile,plottype)
#acScanres_plot(ofile,plottype)
#edivdScan_plot(ofile,plottype)
edivdScan_plot_slides(ofile,plottype)
    
        
    


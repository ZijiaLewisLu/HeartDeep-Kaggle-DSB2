"""
Calculates a mean EF by samplinf from a SV and DV distro. 
"""

from scipy import stats
import numpy as np


def sampledEF(dv,sv,N):
    #rv_discrete only handles integers... making a dict to translate the range into the central bin value
    vals = np.arange(0.5,599.5,1.0)
    valDict = {i:vals[i] for i in range(len(vals))}
    sv_rand = stats.rv_discrete(name='sv_rand',values=(range(len(vals)),sv))  #creating a RNG using the PDF as a weighting function
    dv_rand = stats.rv_discrete(name='dv_rand',values=(range(len(vals)),dv))
    randSVs = sv_rand.rvs(size=N) #sampling 10k systolic volumes 
    randDVs = dv_rand.rvs(size=N)
    randSVs = [valDict[x] for x in randSVs] # converting integer to central bin values
    randDVs = [valDict[x] for x in randDVs]
    randEFs= [(float(x)-float(y))/float(x) for x,y in zip(randDVs,randSVs)]  # array of EFs generated from the SV and DV samples
    (binCont,bins)=np.histogram(randEFs,bins=101,range=(0,1.0))   # histogramming the EFs
    EF_pdf = np.array(binCont,dtype=np.float)  
    EF_bin_cent = [(bins[i]+bins[i+1])/2.0 for i in range(len(bins)-1)]  # central values of EF bins
    EF_pdf = EF_pdf/np.sum(EF_pdf)    # normalizing the EF PDF
    # Calculating Expectation Value of EF  EV_EF
    mean_EF = 0.0
    for i in range(len(EF_bin_cent)):
        mean_EF += EF_bin_cent[i]*EF_pdf[i]
    return mean_EF 



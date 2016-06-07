"""
Script for processing 2nd DSB submission files and generating metrics commonly used in clinical research. Accepts ordered submission files. Author : J Mulholland. 
"""

#default main structure

import sys
import getopt
import csv
import numpy as np
from sklearn.metrics import  confusion_matrix 
from sklearn.metrics import mean_squared_error
import random
import plots
from sampledEF import sampledEF


def process(arg):
    print "This script accepts no arguments"

def main():
    #parse command line
    try:
        opts, args = getopt.getopt(sys.argv[1:],"h",["help"])
    except getopt.error, msg:
        print msg
        print "for help use -help"
        sys.exit(2)
    # process options
    for o, a in opts:
        if o in ("-h","--help"):
            print __doc__
            print "List of input files must be entered by hand into script. Solutions must be in working directory.\n"
            sys.exit(0)
    for arg in args:
        process(arg)
    ###########
    #
    # 
    #  This script requires the studies in the submission files to be ordered in the same manner
    #  as the file with the true volume values (answers.csv). This is an additional requirement
    #  on top of the requirements for the submission files for the Kaggle scoring. 
    #
    ###########
    
    
    answerFile = "answers.csv" #ordered list of true values for the studies
    #  Format
    #  ID,Volume
    #  case#_Diastole, true diastolic volume
    #  case#_Systloe,  true systolic volume  
    #
    ###############
    #   
    #   Read in answers. Format is same as submission with single colume of volumes
    #   replacing the CDF
    #
    fa = open(answerFile,"r")
    readera = csv.reader(fa,delimiter=',')
    i_line = 0
    sv_true = []
    dv_true = []
    for line in readera:
        if i_line>0:
            label = line[0]
            if 'Systole' in label:
                sv_true.append(float(line[1]))
            else:
                dv_true.append(float(line[1]))
        i_line+=1
        
    ##############
    # arrays storing the true volume values-- idexed by
    # order of appearance in ansers.csv
    #
    sv_true = np.array(sv_true) 
    dv_true = np.array(dv_true)
    fa.close() 
    
    ####################
    #
    #  Reading in competition submission files
    #  
    #  Arrays store file input and output names as well as team names
    #  use for headers and titles on plots
    #
    path = "./"
    #subTeamName = ["Team 1","Team 2","Team 3","Team 4","Team 5"]
    # subInFile = ["submission1.csv","submission2.csv","submission3.csv","submission4.csv","submission5.csv"]
    #subOutFile = ["Team_1","Team_2","Team_3","Team_4","Team_5"]

    subTeamName = ["Team 1"]
    subInFile = ["submission1.csv"]
    subOutFile = ["Team_1"]

    subInFile = [path+x for x in subInFile]
    
    ######
    #
    #  Random list of entries for displaying PDFs
    #
    np.random.seed()
    eventList = random.sample(range(440), 100)
    
    ####################
    #
    #  Array for storing the RMS errors for each entry
    #
    errorTable = np.zeros((len(subInFile),3 )) #table for avg error, stdev of error
    
    
    # Begin loop over submission files
    for i_sub in range(len(subInFile)):
    
        print "Processing %s" % subTeamName[i_sub]
    
        #########################
        #
        #   Read in the submission
        #
    
        fs = open(subInFile[i_sub],"r")
        readers = csv.reader(fs,delimiter=',')
        i_line = 0
        sv_cdf = []
        dv_cdf = []
        for line in readers:
            if i_line>0:
                label = line[0]
                if 'Systole' in label:
                    sv_cdf.append([float(x) for x in line[1:]])
                else:
                    dv_cdf.append([float(x) for x in line[1:]])
            i_line+=1
    
        ############
        #
        #  arrays containing all cdfs for each submission
        #
        sv_cdf = np.vstack(sv_cdf)    # first index is the study
        dv_cdf = np.vstack(dv_cdf)    # second index is the CDF value (index from 0 to 599)
        n_cdf_bins = len(sv_cdf[0])   # number of values in each CDF
        fs.close() 
    
        
        n_entry = len(sv_cdf)                       # number of studies in submission files
        dv_err = np.zeros(n_entry)                  # diastolic volume error, indexed by study
        sv_err = np.zeros(n_entry)                  # systolic volume error, indexed by study
        dv_conf = np.zeros(n_entry)                 # standard deviation of diastolic volume PDF (ie confidence estimate), indexed by study
        sv_conf = np.zeros(n_entry)                 # standard deviation of systolic volume PDF (ie confidence estimate), indexed by study
        ef_true = (dv_true - sv_true)/dv_true       # true ejection fraction calculated from the answer.csv file
        EV_EF =  np.zeros(n_entry)                  # predicted ejection fraction value, indexed by study (EV is expectation value)    
    
        ##########
        #
        # Arrays for accessing pdfs and prediction values on a study by study basis
        #
        dv_pdf_arr = np.zeros((n_entry,len(dv_cdf[0])-1))
        sv_pdf_arr = np.zeros((n_entry,len(sv_cdf[0])-1))
        sv_pred_arr = np.zeros(n_entry)
        dv_pred_arr = np.zeros(n_entry)
    
    
        ####################
        #
        #  Looping over entries in submission file
    
        
        for N in range(n_entry):
            dv_pdf_mean = 0.0  # expectation value of DV PDF
            dv_pdf_var = 0.0   # standard deviation of DV PDF
            sv_pdf_mean = 0.0  
            sv_pdf_var = 0.0
            
            sv_pdf_mean = 0.0
            dv_pdf_mean = 0.0
    
            #"differentiating" the cdf
            dv_pdf = np.array([dv_cdf[N][i]-dv_cdf[N][i-1] for i in range(1,n_cdf_bins) ],dtype=np.float)
            sv_pdf = np.array([sv_cdf[N][i]-sv_cdf[N][i-1] for i in range(1,n_cdf_bins) ],dtype=np.float)
    
            #calculating expectation value for dv and sv based on pdf
            for i in range(len(dv_pdf)):
                dv_pdf_mean += dv_pdf[i]*float(i+0.5) # note : value of dv at index 0 is 1 ml
                sv_pdf_mean += sv_pdf[i]*float(i+0.5)
            dv_pdf_arr[N] = dv_pdf
            sv_pdf_arr[N] = sv_pdf
    
            dv_pred_arr[N] = dv_pdf_mean
            sv_pred_arr[N] = sv_pdf_mean
            dv_err[N] = dv_pdf_mean - dv_true[N]     #absolute error in volume
            sv_err[N] = sv_pdf_mean - sv_true[N]
            for i in range(len(dv_pdf)):    # loop for calculating variance of pdf around prediction
                dv_pdf_var += dv_pdf[i]*(float(i+0.5)-dv_pdf_mean)*(float(i+0.5)-dv_pdf_mean)
                sv_pdf_var += sv_pdf[i]*(float(i+0.5)-sv_pdf_mean)*(float(i+0.5)-sv_pdf_mean)        
            ##########
            #
            #  Defining the confidence of this prediction to be the stdev of pdf about the
            #  predicted value 
            #
            dv_conf[N] = np.sqrt(dv_pdf_var)
            sv_conf[N] = np.sqrt(sv_pdf_var)
    
            ##############3
            #
            #  Given a SV and DV distribution, sample from it to generate 
            #  a EF distro and take the mean of that distro
            #
            EV_EF[N] = sampledEF(dv_pdf,sv_pdf,10000)
    
        
        ########################
        #
        #  generating a confusion/contingency matrix
        #  based on Ejection Fraction bounds provided by Dr Arai in private correspondence
        #
        ef_clinic_bins = np.array([0.0,0.35,0.45,0.55,0.73,1.0])
        ef_clinic_bin_names = ["Sev. Abnorm.\n   <35%","Mod. Abnorm.\n 35% to 45%","Mild. Abnorm.\n 45% to 55%","Normal EF\n 55% to 73% ","Hyperdynamic\n   >73%"]
        # these are the possible classification values of each EF. "-1" is for those that do not fall within
        # the 0 to 100% range. They get removed from the matrix later on
        ef_clinic_bin_index = [-1,0,1,2,3,4]
        ef_pred_class = np.zeros(n_entry,dtype=np.int)
        ef_true_class = np.zeros(n_entry,dtype=np.int)
        ###########
        #
        # looping over studies and assigning a class. -1 is out of bounds
        #
        for N in range(n_entry):
            found_bin = 0  # did we bin this ?
            for i in range(1,len(ef_clinic_bins)):
                if EV_EF[N] >= ef_clinic_bins[i-1] and EV_EF[N] < ef_clinic_bins[i]:
                    ef_pred_class[N] = i-1
                    found_bin = 1
                if ef_true[N] >= ef_clinic_bins[i-1] and ef_true[N] < ef_clinic_bins[i]:
                    ef_true_class[N] = i-1
            if found_bin == 0:
                ef_pred_class[N] = -1
    
        cm_clinic = confusion_matrix(ef_true_class,ef_pred_class,ef_clinic_bin_index) #sklearn confusion matrix 
        ################################
        #
        #  filling in a table with RMS Error for the volumes and the Ejection Fraction
        #
        errorTable[i_sub][0] = np.sqrt(mean_squared_error(dv_true,dv_pred_arr))
        errorTable[i_sub][1] = np.sqrt(mean_squared_error(sv_true,sv_pred_arr))
        errorTable[i_sub][2] = np.sqrt(mean_squared_error(ef_true,(dv_pred_arr-sv_pred_arr)/dv_pred_arr))
    
        #########################################
        #
        #  Plotting all this stuff
        #
    
        ####################
        # 
        # Correlation plots for true and predicted values of sv, dv, and ef
        plots.corrPlotV(dv_true,dv_pred_arr,sv_true,sv_pred_arr,subTeamName[i_sub],subOutFile[i_sub])
        plots.corrPlotEF(ef_true,EV_EF,subTeamName[i_sub],subOutFile[i_sub])
        ##################
        #
        #  Bland - Altman plots
        #
        #plotting Bland-Altman
        plots.BAPlotV(dv_true,dv_pred_arr,sv_true,sv_pred_arr,subTeamName[i_sub],subOutFile[i_sub])
        plots.BAPlotEF(ef_true,EV_EF,subTeamName[i_sub],subOutFile[i_sub])
        ################
        #
        # Confusion Matrix Plot
        #
        plots.cmPlot(cm_clinic,ef_clinic_bin_names,subTeamName[i_sub],subOutFile[i_sub])
        ##################
        #
        # Random PDFs for each submission 
        #
        plots.randomPDFs(eventList,dv_pdf_arr,sv_pdf_arr,dv_true,sv_true,subTeamName[i_sub],subOutFile[i_sub])
        ##########################
        #
        #   Scatter plot of PDF standard deviation vs absolute error of prediction
        #
        #only do this for one submission
        if i_sub == 0:
            plots.scatterErrConf(dv_err,sv_err,dv_conf,sv_conf,subTeamName[i_sub],subOutFile[i_sub])
             
    #################
    #
    # RMS Error table is written to a csv
    # you're on your own if you want to display
    # in a pretty fashion
    # 
    #
    
    fout = open("errorTable.csv","w")
    fout.write("team name,RMS error for D/S/EF\n")
    for i in range(len(subTeamName)):
        fout.write("%s,Diastole,%.4f\n" % (subTeamName[i],errorTable[i][0]))
        fout.write("%s,Systole,%.4f\n" % (subTeamName[i],errorTable[i][1]))
        fout.write("%s,E Fraction,%.2f \n" % (subTeamName[i],100.0*errorTable[i][2]))
    fout.close()
    


    
    
if __name__ == "__main__":
    main()


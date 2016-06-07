"""
Created on Sat Apr 02 17:30:23 2016

@author: 582788
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

####################
# 
# Correlation plots

#plotting prediction vs truth... both volumes
def corrPlotV(dv_true,dv_pred,sv_true,sv_pred,teamName,outName):
    slope, intercept, rvalue, pvalue, std_err = stats.linregress(np.append(dv_true,sv_true),np.append(dv_pred,sv_pred))
    plt.scatter(dv_true,dv_pred,label='Diastolic Volume',marker='o',facecolors='none',edgecolors='r')
    plt.scatter(sv_true,sv_pred,label='Systolic Volume',marker='o',facecolors='none',edgecolors='b')
    plt.xlabel("True Volume (mL)")
    plt.ylabel("Predicted Volume (mL)")
    plt.title("%s\nCorrelation of Volume Predictions with Test Values" % teamName)
    x = np.linspace(0,500,10)
    plt.plot(x,x,color='k',label='guide y = x')
    plt.plot(x,x*slope+intercept,color = 'k',linestyle='--',label='y=%.2fx+%.2f\n$R^2=$%.3f p=%.2e' % (slope,intercept,rvalue**2,pvalue))
    plt.gca().set_xlim((0,500))
    plt.gca().set_ylim((0,500))
    plt.legend(loc='upper left')
    plt.grid()
    plt.savefig("%sCorrVols.png" % outName)
    plt.close()

#plotting prediction vs truth... EF    
def corrPlotEF(true,pred,teamName,outName):  
    slope, intercept, rvalue, pvalue, std_err = stats.linregress(true*100.0,pred*100.0)
    plt.scatter(true*100.0,pred*100.0,marker='x',color='#990099',label="Ejection Fraction")
    x = np.linspace(0,90,10)
    plt.plot(x,x,color='k',label='guide y = x')
    plt.plot(x,x*slope+intercept,color = 'k',linestyle='--',label='y=%.2fx+%.2f\n$R^2=$%.3f p=%.2e' % (slope,intercept,rvalue**2,pvalue))
    plt.gca().set_xlim((0,90))
    plt.gca().set_ylim((0,90))
    plt.xlabel("True Ejection Fraction (%)")
    plt.ylabel("Predicted Ejection Fraction (%)")
    plt.title("%s\nCorrelation of EF Predictions with Test Values" % teamName)
    plt.legend(loc='upper left')
    plt.grid()
    plt.savefig("%sCorrEF.png" % outName)
    plt.close()
    
##################
#
#  Bland - Altman plots
#
#plotting Bland-Altman for dv and sv   
def BAPlotV(dv_true,dv_pred,sv_true,sv_pred,teamName,outName):
    avgVold = (dv_true+dv_pred) /2.0
    difVold = (dv_true-dv_pred)
    plt.scatter(avgVold,difVold,label='Diastolic Volume',marker='o',facecolors='none',edgecolors='r')
    avgVols = (sv_true+sv_pred) /2.0
    difVols = (sv_true-sv_pred)
    plt.scatter(avgVols,difVols,label='Systolic Volume',marker='o',facecolors='none',edgecolors='b')
    totalMean = np.mean(np.append(difVold,difVols))
    totalstd = np.std(np.append(difVold,difVols))
    plt.gca().set_xlim((0,500))
    plt.gca().set_ylim((-100.0,100.0))
    x = np.linspace(0,500,10)
    y = np.array([totalMean]*10)
    plt.plot(x,y,label="Mean = %2.3f" % totalMean)
    y = np.array([totalstd]*10)
    plt.plot(x,1.96*y,linestyle='--',label="1.96*StdDev = %2.3f" % (1.96*totalstd))
    plt.plot(x,-1.96*y,linestyle='--')
    plt.xlabel("Mean of Prediction and Test Value (mL)")
    plt.ylabel("Dif of Prediction and Test Value (mL)")
    plt.title("%s\nBland-Altman Plot for Volume Prediction" % teamName)
    plt.legend(loc='lower right')
    plt.grid()
    plt.savefig("%sVolBlandAltman.png" % outName)
    plt.close()
    
def BAPlotEF(ef_true,ef_pred,teamName,outName):
     #plotting Bland-Altman for EF
    avgVol = (ef_true+ef_pred) /2.0
    difVol = (ef_true-ef_pred)
    plt.scatter(avgVol,difVol,label='Ejection Fraction',marker='o',color='m')
    totalMean = np.mean(difVol)
    totalstd = np.std(difVol)
    plt.gca().set_xlim((0.1,0.9))
    plt.gca().set_ylim((-0.6,0.6))
    x = np.linspace(0,1,10)
    y = np.array([totalMean]*10)
    plt.plot(x,y,label="Mean = %2.3f" % totalMean)
    y = np.array([totalstd]*10)
    plt.plot(x,1.96*y,linestyle='--',label="1.96*StdDev = %2.3f" % (1.96*totalstd))
    plt.plot(x,-1.96*y,linestyle='--')
    plt.xlabel("Mean of Prediction and Test Value (mL)")
    plt.ylabel("Dif of Prediction and Test Value (mL)")
    plt.title("%s\nBland-Altman Plot for Ejection Fractions" % (teamName))
    plt.legend(loc='lower right')
    plt.grid()
    plt.savefig("%sEFBlandAltman.png" % (outName) )
    plt.close()
#################
#
#  Scatter plot of confidence vs error
#
def scatterErrConf(dv_err,sv_err,dv_conf,sv_conf,teamName,outName):
    #########################
    #
    #  Pulling out the worst/best 5% in both absolute error and pdf stdev for
    #  both systolic and diastolic volumes
    #  plotting them on the same chart to see what the outliers look like
    #

    dv_err_25 = np.percentile(dv_err,2.5)
    dv_err_975 = np.percentile(dv_err,97.5)
    dv_err_worst_err = np.append(dv_err[dv_err <= dv_err_25] ,dv_err[dv_err >= dv_err_975] )
    dv_err_worst_conf = np.append(dv_conf[dv_err <= dv_err_25] ,dv_conf[dv_err >= dv_err_975] )

    dv_conf_5 = np.percentile(dv_conf,2.5)
    dv_conf_best_err = dv_err[dv_conf <= dv_conf_5]
    dv_conf_best_conf = dv_conf[dv_conf <= dv_conf_5]

    dv_err_475 = np.percentile(dv_err,47.5)
    dv_err_525 = np.percentile(dv_err,52.5)
    bool_array = np.logical_and(dv_err >= dv_err_475,dv_err <= dv_err_525)
    dv_err_best_err = dv_err[bool_array]
    dv_err_best_conf = dv_conf[bool_array] 
    
    dv_conf_95 = np.percentile(dv_conf,95)
    dv_conf_worst_err = dv_err[dv_conf >= dv_conf_95] 
    dv_conf_worst_conf = dv_conf[dv_conf >= dv_conf_95] 


    sv_err_25 = np.percentile(sv_err,2.5)
    sv_err_975 = np.percentile(sv_err,97.5)
    sv_err_worst_err = np.append(sv_err[sv_err <= sv_err_25] ,sv_err[sv_err >= sv_err_975] )
    sv_err_worst_conf = np.append(sv_conf[sv_err <= sv_err_25] ,sv_conf[sv_err >= sv_err_975] )

    sv_conf_5 = np.percentile(sv_conf,2.5)
    sv_conf_best_err = sv_err[sv_conf <= sv_conf_5]
    sv_conf_best_conf = sv_conf[sv_conf <= sv_conf_5]

    sv_err_475 = np.percentile(sv_err,47.5)
    sv_err_525 = np.percentile(sv_err,52.5)
    bool_array = np.logical_and(sv_err >= sv_err_475,sv_err <= sv_err_525)
    sv_err_best_err = sv_err[bool_array] 
    sv_err_best_conf = sv_conf[bool_array] 

    sv_conf_95 = np.percentile(sv_conf,95)
    sv_conf_worst_err = sv_err[sv_conf >= sv_conf_95] 
    sv_conf_worst_conf = sv_conf[sv_conf >= sv_conf_95] 

    figScat = plt.figure(figsize=(14,7.5))
    axDV = figScat.add_subplot(121)
    axSV = figScat.add_subplot(122)
    figScat.suptitle(" %s  :  Individual prediction error and associated PDF StDev" % (teamName))

    axSV.scatter(abs(sv_err),sv_conf,color='#bfbfbf',s=6,marker='o',label='All Predictions')
    axSV.scatter(abs(sv_err_best_err),sv_err_best_conf,color='c',marker='*',s=55,label='5% Smallest Abs Error')
    axSV.scatter(abs(sv_conf_best_err),sv_conf_best_conf,color='b',marker='*',s=55,label= '5% Most Confident')
    axSV.scatter(abs(sv_err_worst_err),sv_err_worst_conf,color='#ff9900',marker='v',s=55, label = '5% Biggest Abs Error')
    axSV.scatter(abs(sv_conf_worst_err),sv_conf_worst_conf,color='#993300',marker='v',s=55, label = '5% Least Confident')
    axSV.set_xlabel("Absolute Value of Error (ml)")
    axSV.set_ylabel("StDev on PDF (ml)")
    axSV.set_title("Systolic Volume (ml)")
    axSV.legend(loc = 'lower right',markerscale = 1.2, prop = {'size':14} )
    axSV.set_xlim((0,80))
    axSV.set_ylim((0,25))
    axSV.grid()

    axDV.scatter(abs(dv_err),dv_conf,color='#bfbfbf',s=6,marker='o', label = "All Predictions")
    axDV.scatter(abs(dv_err_best_err),dv_err_best_conf,color='c',marker='*',s=55,label='5% Smallest Abs Error')
    axDV.scatter(abs(dv_conf_best_err),dv_conf_best_conf,color='b',marker='*',s=55,label= '5% Most Confident')
    axDV.scatter(abs(dv_err_worst_err),dv_err_worst_conf,color='#ff9900',marker='v',s=55, label = '5% Biggest Abs Error')
    axDV.scatter(abs(dv_conf_worst_err),dv_conf_worst_conf,color='#993300',marker='v',s=55, label = '5% Least Confident')
    axDV.set_xlabel("Absolute Value of Error (ml)")
    axDV.set_ylabel("StDev on PDF (ml)")
    axDV.set_title("Diastolic Volume (ml)")
    axDV.legend(loc = 'lower right',markerscale = 1.2, prop = {'size':14} )
    axDV.set_xlim((0,80))
    axDV.set_ylim((0,25))
    axDV.grid()

    figScat.savefig("%sScatter.png" %(outName))
    plt.close()
    
def randomPDFs(eventList,dv_pdf,sv_pdf,dv_true,sv_true,teamName,outName):
    ######################
    #
    #  plotting PDFs along with true SV and DV values.
    #  100 study indices are chosen randomly and put into eventList
    #  Here they are plotted in a large grid of plots 
    # 
    #
    if len(eventList) != 100:
        print"eventList passed to randomPDFs must have exactly 100 events in it\n"
        return None
    figPDF,ax = plt.subplots(10,10,sharex=False,sharey=False)
    figPDF.set_size_inches(20,20)
    figPDF.suptitle(" %s\nRandomly selected PDFS and asssociated test values\nRed : Systolic PDF        Blue : Diastolic PDF\ndashed line : True Systolic        solid line : True Diastolic" % (teamName),fontsize="20")
    for i in range(10):
        for j in range(10):
            ax[i,j].tick_params(labelbottom="off",labelleft="off")
            N = eventList[j + i*10]
            ax[i,j].plot(range(599),sv_pdf[N],color='r',label="Systolic Vol PDF")
            ax[i,j].plot(range(599),dv_pdf[N],color='b',label="Diastolic Vol PDF")
            ylimits = ax[i,j].get_ylim() # make sure the PDFS fit within the plot 
            Y = np.linspace(ylimits[0],ylimits[1],10)
            X = np.array([sv_true[N]]*10)
            ax[i,j].plot(X,Y,color='k',linestyle='--',label='Systolic True')
            X = np.array([dv_true[N]]*10)
            ax[i,j].plot(X,Y,color='k',label='Diastolic True')
            ax[i,j].plot(range(599),sv_pdf[N],color='r',label="Systolic Vol PDF")
            ax[i,j].plot(range(599),dv_pdf[N],color='b',label="Diastolic Vol PDF")
            ax[i,j].set_title("index = %d" % N)
            # for each plot I wanted the PDF and true SV and DV values to be seen as
            #clearly as possible, so I'm fiddling with the x bounds here
            moveX = list(ax[i,j].get_xlim())
            testX = np.max([sv_true[N],dv_true[N]])    #
            moveX[1] = np.min([testX+75,600])
            ax[i,j].set_xlim(moveX)
    figPDF.savefig("%s100PDFs.png" % (outName))
    plt.close()    
    
    ################
    #
    #
    # Confusion Matrix Plot
    #
    #
def cmPlot(cmatrix,class_names,teamName,outName):  
    cm_temp = cmatrix[1:,1:] #(get rid of overflow bins)
    plt.imshow(cm_temp,interpolation='nearest',cmap=plt.cm.Reds)
    plt.title("%s\nConfusion Matrix" % (teamName))
    cbar=plt.colorbar()
    cbar.set_label("# of predictions")
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=70)
    plt.yticks(tick_marks, class_names)
    plt.tight_layout()
    plt.ylabel('True Category')
    plt.xlabel('Predicted Category')
    width, height = cm_temp.shape
    for x in range(width):
        for y in range(height):
            plt.gca().annotate(str(cm_temp[x,y]),xy=(y,x),horizontalalignment = 'center',verticalalignment='center')
    plt.gcf().set_size_inches((8.0,6.5))
    plt.savefig("%scmClinicSub.png" % (outName))
    plt.close()

# Get species richness vs. EPAW-W data (important features scaling laws)
# 
# 1. NPDESDensWs --> 15
# 2. TRIDensWs   --> 17
# 3. PctUrbHiWs  --> 110
# 4. PctUrbLoWs  --> 111
# 5. PctUrbMdWs  --> 112
# 6. PctUrbOpWs  --> 113 
#
#   Important features identified include (dataIndex --> FeatureName): A total of 24 features
#       2   --> 0.282 0.273 DRNAREA
#       11  --> 0.393 0.416 DamDensWs
#       12  --> 0.285 0.381 DamNIDStorM3Ws
#       13  --> 0.281 0.372 DamNrmStorM3Ws
#       15  --> 0.358 0.615 NPDESDensWs
#       16  --> 0.395 0.426 SuperfundDensWs
#       17  --> 0.421 0.611 TRIDensWs
#       31  --> 0.264 0.299 PctFrstLoss2003Ws
#       37  --> 0.294 0.418 PctFrstLoss2009Ws
#       54  --> 0.286 0.547 PctImp2006Ws
#       71  --> 0.246 0.205 PctNonCarbResidWs
#       75  --> 0.337 0.407 MineDensWs
#       76  --> 0.351 0.345 NABD_DensWs
#       77  --> 0.293 0.334 NABD_NIDStorM3Ws
#       78  --> 0.29 0.324 NABD_NrmStorM3Ws
#       92  --> 0.37 0.357 PctOw2006Ws
#       94  --> 0.288 0.575 PctUrbHi2006Ws
#       110 --> 0.301 0.56 PctUrbHi2011Ws
#       111 --> 0.306 0.559 PctUrbLo2011Ws
#       112 --> 0.254 0.578 PctUrbMd2011Ws
#       113 --> 0.376 0.419 PctUrbOp2011Ws
#       115 --> 0.325 0.43 PctNonAgIntrodManagVegWs
#       122 --> 0.312 0.323 Tmax8110Ws
#       123 --> 0.337 0.346 Tmean8110Ws
#
# AUTHOR -- Maruti Kumar Mudunuru
#   https://stackoverflow.com/questions/3433486/how-to-do-exponential-and-logarithmic-curve-fitting-in-python-i-found-only-poly
# p-value from a curvefit
#    https://stats.stackexchange.com/questions/362520/how-to-know-if-a-parameter-is-statistically-significant-in-a-curve-fit-estimat

import os
import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import linregress
from scipy.optimize import curve_fit
import scipy.odr
import scipy.stats
#
np.set_printoptions(precision=2)

#==================================;
#  Function-1: Curve fit function  ;
#==================================;
def func(x, b, z):
    return b * np.power(x, z)

#========================================;
#  Function-2: ODR for p-value function  ;
#========================================;
def f_wrapper_for_odr(beta, x): #parameter order for odr
    return func(x, *beta)

#===============================;
#  Function-3: p-value function ;
#===============================;
def get_pvalue(i, j):

    ind      = np.argwhere(~np.isnan(hs_arr[:,j]))[:,0]
    parameters, cov = curve_fit(func, hs_arr[ind,j], sr_arr[ind,i])
    #
    model    = scipy.odr.odrpack.Model(f_wrapper_for_odr)
    data     = scipy.odr.odrpack.Data(hs_arr[ind,j], sr_arr[ind,i])
    myodr    = scipy.odr.odrpack.ODR(data, model, beta0=parameters,  maxit=0)
    myodr.set_job(fit_type=2)
    parameterStatistics = myodr.run()
    df_e     = len(hs_arr[ind,j]) - len(parameters) # degrees of freedom, error
    cov_beta = parameterStatistics.cov_beta # parameter covariance matrix from ODR
    sd_beta  = parameterStatistics.sd_beta * parameterStatistics.sd_beta
    ci       = []
    t_df     = scipy.stats.t.ppf(0.975, df_e)
    ci       = []
    for k in range(len(parameters)):
        ci.append([parameters[k] - t_df * parameterStatistics.sd_beta[k], parameters[k] + t_df * parameterStatistics.sd_beta[k]])

    tstat_beta = parameters / parameterStatistics.sd_beta # coeff t-statistics
    pstat_beta = (1.0 - scipy.stats.t.cdf(np.abs(tstat_beta), df_e)) * 2.0    # coef. p-values

    for l in range(len(parameters)):
        print('parameter:', parameters[l])
        print('   conf interval:', ci[l][0], ci[l][1])
        print('   tstat:', tstat_beta[l])
        print('   pstat:', '{0:.3g}'.format(pstat_beta[l]))
        print()

    return parameters, pstat_beta

#******************************;
#  1. Set pathsfor .csv files  ;
#******************************;
path           = os.getcwd() + '/'
#
df_sr          = pd.read_csv(path + "Inputs_Outputs_v4/1_WAT_SR_54s_11f.csv", index_col = 1).iloc[:,1:] #[54 rows x 10 columns]
df_sd          = pd.read_csv(path + "Inputs_Outputs_v4/2_WAT_SHANN_54s_11f.csv", index_col = 1).iloc[:,1:] #[54 rows x 10 columns]
#
df_acw_all     = pd.read_csv(path + "Inputs_Outputs_v4/5_ACW_EPAWaters_54s_141f.csv", index_col = 1).iloc[:,1:] #[54 rows x 140 columns]
acw_list       = np.delete(np.arange(0,140), [2,3,126])
df_acw         = df_acw_all.iloc[:,acw_list] #[54 rows x 137 columns]
#
comp_list      = df_sr.columns.to_list() #10
acw_ftrs_list  = df_acw.columns.to_list() #137
#
sr_arr         = df_sr.values #(54, 10)
sd_arr         = df_sd.values #(54, 10)
acw_arr        = df_acw.values #(54, 137)
#
epaacw_index_list = [2, 11, 12, 13, 15, 16, 17, 31, 37, 54, 71, 75, 76, 77, 78, \
                        92, 94, 110, 111, 112, 113, 115, 122, 123] #24 -- imp ftrs
epaacw_ftrs_list  = ['DRNAREA',
                        'DamDensWs',
                        'DamNIDStorM3Ws',
                        'DamNrmStorM3Ws',
                        'NPDESDensWs',
                        'SuperfundDensWs',
                        'TRIDensWs',
                        'PctFrstLoss2003Ws',
                        'PctFrstLoss2009Ws',
                        'PctImp2006Ws',
                        'PctNonCarbResidWs',
                        'MineDensWs',
                        'NABD_DensWs',
                        'NABD_NIDStorM3Ws',
                        'NABD_NrmStorM3Ws',
                        'PctOw2006Ws',
                        'PctUrbHi2006Ws',
                        'PctUrbHi2011Ws',
                        'PctUrbLo2011Ws',
                        'PctUrbMd2011Ws',
                        'PctUrbOp2011Ws',
                        'PctNonAgIntrodManagVegWs',
                        'Tmax8110Ws',
                        'Tmean8110Ws'] #24
#
for i in range(0,len(epaacw_index_list)): #24
    countx = epaacw_index_list[i]
    print(countx, acw_ftrs_list[countx], epaacw_ftrs_list[i])
#
#hs_arr          = copy.deepcopy(acw_arr[:,epaacw_index_list]) #(54, 6)
hs_arr           = np.zeros((acw_arr.shape[0],len(epaacw_index_list)), dtype = float) #(54, 24)
for i in range(0,len(epaacw_index_list)): #24
    hs_arr[:,i] = copy.deepcopy(acw_arr[:,epaacw_index_list[i]])
#
marker_list    = ['o', 'v', '8', 's', 'p', '*', 'h', '+', 'x', '^'] #10
color_list     = ['b', 'k', 'r', 'c', 'm', 'g', 'y', 'tab:purple', 'tab:brown', 'tab:orange'] #10

#*****************************************************************;
#  2. log(SR) vs. log(extrinsic factors) for 9 compounds and sum  ;
#      log(SR) = log(b) + z*log(EF)                               ;
#*****************************************************************;
logb_list = np.zeros((len(comp_list), len(epaacw_index_list)), dtype = float) #(10,24)
logz_list = np.zeros((len(comp_list), len(epaacw_index_list)), dtype = float) #(10,24)
#
b_list    = np.zeros((len(comp_list), len(epaacw_index_list)), dtype = float) #(10,24)
z_list    = np.zeros((len(comp_list), len(epaacw_index_list)), dtype = float) #(10,24)
#
for j in range(0,len(epaacw_index_list)): #Top-24 features
    for i in range(0,len(comp_list)): #Compounds i = 0 to 10
        ind        = np.argwhere(~(hs_arr[:,j] == 0))[:,0]
        #
        m_sr, c_sr, \
        r_value_sr, \
        p_value_sr,\
        std_err_sr  = linregress(np.log10(hs_arr[ind,j]), np.log(sr_arr[ind,i]))
        print('i, j, Comp name, Extrinsic feature, logz_sr, logb_sr, z_sr, b_sr = ', \
                i, j, comp_list[i], epaacw_ftrs_list[j], \
                '{0:.3g}'.format(m_sr), '{0:.3g}'.format(c_sr), \
                '{0:.3g}'.format(10**m_sr), '{0:.3g}'.format(10**c_sr))
        #
        logb_list[i,j] = c_sr
        logz_list[i,j] = m_sr
        #
        b_list[i,j]    = 10**c_sr
        z_list[i,j]    = 10**m_sr

#********************************************************************;
#  3a. log(SR) vs. log10(extrinsic factors) for 9 compounds and sum  ;
#********************************************************************;
imp_ftrs_list  = epaacw_ftrs_list
#
ymin_list  = [-0.1, -2, -0.5, -0.5]
ymax_list  = [8.5, 8.5, 8.5, 8.5]
#
for j in range(0,len(epaacw_index_list)): #Extrinsic features 0 to 24
    legend_properties = {'weight':'bold'}
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    ax.set_xlabel(epaacw_ftrs_list[j] + ' (in log10)', fontsize = 12, fontweight = 'bold')
    ax.set_ylabel('Species richness (in log10)', fontsize = 12, fontweight = 'bold')
    plt.title('log(SR) vs. log(' + epaacw_ftrs_list[j] + ')')
    #ax.set_ylim([ymin_list[j], ymax_list[j]])
    #
    for i in range(0,len(comp_list)): #Compounds i = 0 to 10
        color   = color_list[i]
        marker  = marker_list[i]
        #
        c_sr    = logb_list[i,j]
        m_sr    = logz_list[i,j]
        #
        ind        = np.argwhere(~(hs_arr[:,j] == 0))[:,0]
        yfit_sr = np.asarray([c_sr + m_sr*xi for xi in np.log10(hs_arr[ind,j])])
        #
        ax.scatter(np.log10(hs_arr[ind,j]), np.log(sr_arr[ind,i]), \
                    s = 20, c = color, marker = marker, edgecolor = 'face')
        ax.plot(np.log10(hs_arr[ind,j]), yfit_sr, \
                color = color, label = comp_list[i])
    ax.legend(bbox_to_anchor=(1.04, 1.0), loc = 'upper left')
    fig.tight_layout()
    plt.savefig(path + 'Plots_EPAWaters_ACW/Scaling_Laws/logSR_' + str(j) + '_' + imp_ftrs_list[j] + '.png')
    plt.close(fig)

#********************************************************;
#  3b. SR vs. extrinsic factors for 9 compounds and sum  ;
#********************************************************;
for j in range(0,len(epaacw_index_list)): #Extrinsic features 0 to 24
    legend_properties = {'weight':'bold'}
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    ax.set_xlabel(epaacw_ftrs_list[j], fontsize = 12, fontweight = 'bold')
    ax.set_ylabel('Species richness', fontsize = 12, fontweight = 'bold')
    plt.title('SR vs. ' + epaacw_ftrs_list[j])
    #ax.set_ylim([ymin_list[j], ymax_list[j]])
    #
    for i in range(0,len(comp_list)): #Compounds i = 0 to 10
        color      = color_list[i]
        marker     = marker_list[i]
        #
        ind        = np.argwhere(~np.isnan(hs_arr[:,j]))[:,0]
        popt, pcov = curve_fit(func, hs_arr[ind,j], sr_arr[ind,i])
        print(epaacw_ftrs_list[j], comp_list[i], popt)
        #
        ax.scatter(hs_arr[:,j], sr_arr[:,i], \
                    s = 20, c = color, marker = marker, edgecolor = 'face')
        ax.plot(np.sort(hs_arr[ind,j]), func(np.sort(hs_arr[ind,j]), *popt), \
                color = color, label = comp_list[i] + ',  SE = ' + str('{0:.3g}'.format(popt[1])))
    #ax.legend(bbox_to_anchor=(1.04, 1.0), loc = 'upper left')
    fig.tight_layout()
    plt.savefig(path + 'Plots_EPAWaters_ACW/Scaling_Laws/SR_' + str(j) + '_' + imp_ftrs_list[j] + '.png')
    plt.close(fig)

#*****************************************************************;
#  3c. SR vs. extrinsic factors for 9 compounds and sum (labels)  ;
#*****************************************************************;
for j in range(0,len(epaacw_index_list)): #Extrinsic features 0 to 24
    legend_properties = {'weight':'bold'}
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    ax.set_axis_off()
    for i in range(0,len(comp_list)): #Compounds i = 0 to 10
        color      = color_list[i]
        marker     = marker_list[i]
        #
        popt, pval = get_pvalue(i,j)
        print(imp_ftrs_list[j], comp_list[i], pval[0], pval[1], popt[1])
        #
        ax.plot([], [], color = color, marker = marker, \
                label = comp_list[i] + ',  SE = ' + str('{0:.3g}'.format(popt[1])) + \
                ',  pv-b = ' + str('{0:.3g}'.format(pval[0])) + \
                ',  pv-z = ' + str('{0:.3g}'.format(pval[1])))
    ax.legend(bbox_to_anchor=(1.05, 1.0), loc = 'upper left')
    fig.tight_layout()
    plt.savefig(path + 'Plots_EPAWaters_ACW/Scaling_Laws/LegendSR_' + str(j) + '_' + imp_ftrs_list[j] + '.png')
    plt.close(fig)
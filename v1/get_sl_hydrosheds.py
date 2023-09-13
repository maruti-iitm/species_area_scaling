# Get species richness vs. HYDROSHEDS data (important features scaling laws)
# 
# 1. Actual ET - aet_mm_s04         --> 87
# 2. Potential ET - pet_mm_s04      --> 73
# 3. Air temp - tmp_dc_s05          --> 46
# 4. Natural discharge - dis_m3_pmn --> 10
# 5. Inundation extent - inu_pc_slt --> 17
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

#*******************************;
#  1. Set paths for .csv files  ;
#*******************************;
path           = os.getcwd() + '/'
#
df_sr           = pd.read_csv(path + "Inputs_Outputs_v4/1_WAT_SR_54s_11f.csv", index_col = 1).iloc[:,1:] #[54 rows x 10 columns]
df_sd           = pd.read_csv(path + "Inputs_Outputs_v4/2_WAT_SHANN_54s_11f.csv", index_col = 1).iloc[:,1:] #[54 rows x 10 columns]
#
df_huc12        = pd.read_csv(path + "Inputs_Outputs_v4/8_Hydrosheds_huc12_54s_294f.csv", index_col = 1).iloc[:,1:-1] #[54 rows x 292 columns]
#
comp_list       = df_sr.columns.to_list() #10
huc12_ftrs_list = df_huc12.columns.to_list() #292
#
sr_arr          = df_sr.values #(54, 10)
sd_arr          = df_sd.values #(54, 10)
huc12_arr       = df_huc12.values #(54, 292)
#
hydrosheds_index_list = [10, 17, 46, 73, 87] #5 -- imp ftrs indices
hydrosheds_ftrs_list  = ['Natural discharge - dis_m3_pmn', 'Inundation extent - inu_pc_slt', \
                            'Air temp - tmp_dc_s05', 'Potential ET - pet_mm_s04', \
                            'Actual ET - aet_mm_s04'] #5
#
hs_arr           = copy.deepcopy(huc12_arr[:,hydrosheds_index_list]) #(54, 5)
#
marker_list    = ['o', 'v', '8', 's', 'p', '*', 'h', '+', 'x', '^'] #10
color_list     = ['b', 'k', 'r', 'c', 'm', 'g', 'y', 'tab:purple', 'tab:brown', 'tab:orange'] #10

#*****************************************************************;
#  2. log(SR) vs. log(extrinsic factors) for 9 compounds and sum  ;
#      log(SR) = log(b) + z*log(EF)                               ;
#*****************************************************************;
logb_list = np.zeros((len(comp_list), 5), dtype = float) #(10,5)
logz_list = np.zeros((len(comp_list), 5), dtype = float) #(10,5)
#
b_list    = np.zeros((len(comp_list), 5), dtype = float) #(10,5)
z_list    = np.zeros((len(comp_list), 5), dtype = float) #(10,5)
#
for j in range(0,5): #Top-5 features
    for i in range(0,len(comp_list)): #Compounds i = 0 to 10
        ind         = np.argwhere(~np.isnan(hs_arr[:,j]))[:,0]
        #
        m_sr, c_sr, \
        r_value_sr, \
        p_value_sr,\
        std_err_sr  = linregress(np.log10(hs_arr[ind,j] + 1e-8), np.log(sr_arr[ind,i]+ 1e-8))
        print('i, j, Comp name, Extrinsic feature, logz_sr, logb_sr, z_sr, b_sr = ', \
                i, j, comp_list[i], hydrosheds_ftrs_list[j], \
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
imp_ftrs_list  = ['Discharge', 'InundationExtent', 'AirTemp', 'PotentialET', 'ActualET']
#
ymin_list  = [-0.1, -0.1, -2, -0.5, -0.5]
ymax_list  = [8.5, 8.5, 8.5, 8.5, 8.5]
#
for j in range(0,5): #Extrinsic features 0 to 5
    legend_properties = {'weight':'bold'}
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    ax.set_xlabel(hydrosheds_ftrs_list[j] + ' (in log10)', fontsize = 12, fontweight = 'bold')
    ax.set_ylabel('Species richness (in log10)', fontsize = 12, fontweight = 'bold')
    plt.title('log(SR) vs. log(' + hydrosheds_ftrs_list[j] + ')')
    ax.set_ylim([ymin_list[j], ymax_list[j]])
    #
    for i in range(0,len(comp_list)): #Compounds i = 0 to 10
        color   = color_list[i]
        marker  = marker_list[i]
        #
        c_sr    = logb_list[i,j]
        m_sr    = logz_list[i,j]
        #
        yfit_sr = np.asarray([c_sr + m_sr*xi for xi in np.log10(hs_arr[:,j])])
        #
        ax.scatter(np.log10(hs_arr[:,j]), np.log(sr_arr[:,i]), \
                    s = 20, c = color, marker = marker, edgecolor = 'face')
        ax.plot(np.log10(hs_arr[:,j] + 1e-8), yfit_sr, \
                color = color, label = comp_list[i])
    ax.legend(bbox_to_anchor=(1.04, 1.0), loc = 'upper left')
    fig.tight_layout()
    plt.savefig(path + 'Plots_HYDROSHEDS/Scaling_Laws/logSR_' + str(j) + '_' + imp_ftrs_list[j] + '.png')
    plt.close(fig)

#********************************************************;
#  3b. SR vs. extrinsic factors for 9 compounds and sum  ;
#********************************************************;
for j in range(0,5): #Extrinsic features 0 to 5
    legend_properties = {'weight':'bold'}
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    ax.set_xlabel(hydrosheds_ftrs_list[j], fontsize = 12, fontweight = 'bold')
    ax.set_ylabel('Species richness', fontsize = 12, fontweight = 'bold')
    plt.title('SR vs. ' + hydrosheds_ftrs_list[j])
    #ax.set_ylim([ymin_list[j], ymax_list[j]])
    #
    for i in range(0,len(comp_list)): #Compounds i = 0 to 10
        color      = color_list[i]
        marker     = marker_list[i]
        #
        ind        = np.argwhere(~np.isnan(hs_arr[:,j]))[:,0]
        popt, pcov = curve_fit(func, hs_arr[ind,j], sr_arr[ind,i])
        print(hydrosheds_ftrs_list[j], comp_list[i], popt)
        #
        ax.scatter(hs_arr[:,j], sr_arr[:,i], \
                    s = 20, c = color, marker = marker, edgecolor = 'face')
        ax.plot(np.sort(hs_arr[ind,j]), func(np.sort(hs_arr[ind,j]), *popt), \
                color = color, label = comp_list[i] + ',  SE = ' + str('{0:.3g}'.format(popt[1])))
    #ax.legend(bbox_to_anchor=(1.04, 1.0), loc = 'upper left')
    fig.tight_layout()
    plt.savefig(path + 'Plots_HYDROSHEDS/Scaling_Laws/SR_' + str(j) + '_' + imp_ftrs_list[j] + '.png')
    plt.close(fig)

#*****************************************************************;
#  3c. SR vs. extrinsic factors for 9 compounds and sum (labels)  ;
#*****************************************************************;
b_arr    = np.zeros((len(comp_list),5), dtype = float) #(10,5)
se_arr   = np.zeros((len(comp_list),5), dtype = float) #(10,5)
pv_b_arr = np.zeros((len(comp_list),5), dtype = float) #(10,5)
pv_z_arr = np.zeros((len(comp_list),5), dtype = float) #(10,5)
#
for j in range(0,5): #Extrinsic features 0 to 5
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
        b_arr[i,j]    = popt[0]
        se_arr[i,j]   = popt[1]
        pv_b_arr[i,j] = pval[0]
        pv_z_arr[i,j] = pval[1]
    ax.legend(bbox_to_anchor=(1.05, 1.0), loc = 'upper left')
    fig.tight_layout()
    plt.savefig(path + 'Plots_HYDROSHEDS/Scaling_Laws/LegendSR_' + str(j) + '_' + imp_ftrs_list[j] + '.png')
    plt.close(fig)

df_b_arr    = pd.DataFrame(b_arr, index = comp_list, columns = imp_ftrs_list) #b
df_se_arr   = pd.DataFrame(se_arr, index = comp_list, columns = imp_ftrs_list) #z
df_pv_b_arr = pd.DataFrame(pv_b_arr, index = comp_list, columns = imp_ftrs_list) #p-value for b
df_pv_z_arr = pd.DataFrame(pv_z_arr, index = comp_list, columns = imp_ftrs_list) #p-value for z
#
df_b_arr.to_csv('Plots_HYDROSHEDS/Scaling_Laws/b_array.csv')
df_se_arr.to_csv('Plots_HYDROSHEDS/Scaling_Laws/scaling_exponent_array.csv')
df_pv_b_arr.to_csv('Plots_HYDROSHEDS/Scaling_Laws/p-value_b_array.csv')
df_pv_z_arr.to_csv('Plots_HYDROSHEDS/Scaling_Laws/p-value_scaling_expoenent_array.csv')

#*********************************************************************;
#  4a. Raw and normalized exponents for 9 compounds and sum (labels)  ;
#*********************************************************************;
se_list      = np.zeros((len(comp_list), 5), dtype = float) #(10,5)
cvalue_list  = np.zeros((len(comp_list), 5), dtype = float) #(10,5)
#
for j in range(0,5): #Extrinsic features 0 to 4
    for i in range(0,len(comp_list)): #Compounds i = 0 to 10
        popt, pval = get_pvalue(i,j)
        print(imp_ftrs_list[j], comp_list[i], popt[1])
        #
        cvalue_list[i,j] = popt[0]
        se_list[i,j]     = popt[1]
#
norm_cvalue_list = copy.deepcopy(cvalue_list/np.max(cvalue_list, axis = 0))
norm_se_list     = copy.deepcopy(se_list/np.max(se_list, axis = 0))

#*************************************************************************************;
#  4b. Bar plots (Normalized c-value and exponents) for 9 compounds and sum (labels)  ;
#*************************************************************************************;
for j in range(0,5): #Extrinsic features 0 to 4
    legend_properties = {'weight':'bold'}
    fig = plt.figure(figsize=(12,5))
    ax  = fig.add_subplot(111)
    ax.set_xlabel('Compound type', fontsize = 12, fontweight = 'bold')
    ax.set_ylabel('Normalized exponent (z/z-max)', fontsize = 12, fontweight = 'bold')
    plt.title('Normalized exponent vs. Compound type for ' + imp_ftrs_list[j])
    ax.bar(comp_list, norm_se_list[:,j], color = color_list)
    fig.tight_layout()
    plt.savefig(path + 'Plots_HYDROSHEDS/Scaling_Laws/Bar_NormExp_' + str(j) + '_' + imp_ftrs_list[j] + '.png')
    plt.close(fig)
    #
    legend_properties = {'weight':'bold'}
    fig = plt.figure(figsize=(12,5))
    ax  = fig.add_subplot(111)
    ax.set_xlabel('Compound type', fontsize = 12, fontweight = 'bold')
    ax.set_ylabel('Normalized c-value (c/c-max)', fontsize = 12, fontweight = 'bold')
    plt.title('Normalized c-value vs. Compound type for ' + imp_ftrs_list[j])
    ax.bar(comp_list, norm_cvalue_list[:,j], color = color_list)
    fig.tight_layout()
    plt.savefig(path + 'Plots_HYDROSHEDS/Scaling_Laws/Bar_NormCval_' + str(j) + '_' + imp_ftrs_list[j] + '.png')
    plt.close(fig)    

#******************************************************************************;
#  4c. Bar plots (Raw c-value and exponents) for 9 compounds and sum (labels)  ;
#******************************************************************************;
for j in range(0,5): #Extrinsic features 0 to 4
    legend_properties = {'weight':'bold'}
    fig = plt.figure(figsize=(12,5))
    ax  = fig.add_subplot(111)
    ax.set_xlabel('Compound type', fontsize = 12, fontweight = 'bold')
    ax.set_ylabel('Raw exponent (z-value)', fontsize = 12, fontweight = 'bold')
    plt.title('Raw exponent vs. Compound type for ' + imp_ftrs_list[j])
    ax.bar(comp_list, se_list[:,j], color = color_list)
    fig.tight_layout()
    plt.savefig(path + 'Plots_HYDROSHEDS/Scaling_Laws/Bar_RawExp_' + str(j) + '_' + imp_ftrs_list[j] + '.png')
    plt.close(fig)
    #
    legend_properties = {'weight':'bold'}
    fig = plt.figure(figsize=(12,5))
    ax  = fig.add_subplot(111)
    ax.set_xlabel('Compound type', fontsize = 12, fontweight = 'bold')
    ax.set_ylabel('Raw c-value', fontsize = 12, fontweight = 'bold')
    plt.title('Raw c-value vs. Compound type for ' + imp_ftrs_list[j])
    ax.bar(comp_list, cvalue_list[:,j], color = color_list)
    fig.tight_layout()
    plt.savefig(path + 'Plots_HYDROSHEDS/Scaling_Laws/Bar_RawCval_' + str(j) + '_' + imp_ftrs_list[j] + '.png')
    plt.close(fig) 
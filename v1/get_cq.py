# Get c-q plots for WHONDRS, StreamStats, HYDROSHEDS, EPA-ACC, and EPA-ACW
# Catchment controls on solute export -- https://www.sciencedirect.com/science/article/pii/S030917081500233X
# 
# Species richness in water samples (9 comp and sum of all comps)  -- 1_WAT_SR.csv
# Shannon diversity in water samples (9 comp and sum of all comps) -- 2_WAT_SHANN.csv
#
# WHONDRS     -- Metadata pre-processed from Eric for PCA analysis -- 7_PCA_Eric_WHONDRS.csv
# HYDROSHEDS  -- Pre-processed data from Erika                     -- 8_Hydrosheds.xlsx
# StreamStats -- Pre-processed data from Michelle                  -- 6_StreamStats.xlsx
# EPAWaters   -- Pre-processed data from Michelle                  -- 5_ACC_EPAWaters.csv and 5_ACW_EPAWaters.csv
#
# WHONDRS data
#   1. SW_Temp_degC --> 11
#   2. DO_mg.per.L  --> 10
#   3. Avg water column height      --> avg of 2, 4, 6
#         US_Water.Column.Height_cm --> 2
#         MS_Water.Column.Height_cm --> 4
#         DS_Water.Column.Height_cm --> 6
#   4. SW_pH       --> 8
#   5. DO_perc.sat --> 9
#
# StreamStats data
#   1. Latitude      --> 1
#   2. Drainage area --> 2
#   3. Elev          --> 3
#   4. Precip        --> 4
#   5. Forest cover  --> 7
#
# HYDROSHEDS data
#   1. Actual ET - aet_mm_s04         --> 87
#   2. Potential ET - pet_mm_s04      --> 73
#   3. Air temp - tmp_dc_s05          --> 46
#   4. Natural discharge - dis_m3_pmn --> 10
#
# EPAWaters-C (Catchment) data
#   1. Percent forest cover loss - PctFrstLoss --> 34
#   2. Precipitation gradient - SN             --> 84
#   3. PctNonAgIntrodManagVegCat               --> 92
#   4. Mean annual stream temperature -- MAST  --> 115
#   5. Mean summer stream temperature - MSST   --> 126
#
# EPAWaters-W (Watersheds) data
#   1. NPDESDensWs --> 15
#   2. TRIDensWs   --> 17
#   3. PctUrbHiWs  --> 110
#   4. PctUrbLoWs  --> 111
#   5. PctUrbMdWs  --> 112
#   6. PctUrbOpWs  --> 113 
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

#******************************************;
#  1a. Set paths for .csv files (WHONDRS)  ;
#******************************************;
path           = os.getcwd() + '/'
#
df_sr          = pd.read_csv(path + "Inputs_Outputs_v4/1_WAT_SR_54s_11f.csv", index_col = 1).iloc[:,1:] #[54 rows x 10 columns]
df_sd          = pd.read_csv(path + "Inputs_Outputs_v4/2_WAT_SHANN_54s_11f.csv", index_col = 1).iloc[:,1:] #[54 rows x 10 columns]
#
df_wdrs        = pd.read_csv(path + 'Inputs_Outputs_v4/7_PCA_Eric_WHONDRS_54s_13f.csv', index_col = 1).iloc[:,1:] #[54 rows x 12 columns]
#
comp_list      = df_sr.columns.to_list() #10
wdrs_ftrs_list = df_wdrs.columns.to_list() #12
#
sr_arr         = df_sr.values #(54, 10) #species richness
sd_arr         = df_sd.values #(54, 10) #shannon diversity
wdrs_arr       = df_wdrs.values #(54, 12) #ftrs
#
marker_list    = ['o', 'v', '8', 's', 'p', '*', 'h', '+', 'x', '^'] #10
color_list     = ['b', 'k', 'r', 'c', 'm', 'g', 'y', 'tab:purple', 'tab:brown', 'tab:orange'] #10
#
wdrs_ftimp_arr        = np.zeros((wdrs_arr.shape[0], 5), dtype = float) #(54,5)
wdrs_ftimp_arr[:,0]   = copy.deepcopy(np.mean(wdrs_arr[:,[2,4,6]], axis = 1)) #Avg water column height
wdrs_ftimp_arr[:,1:5] = copy.deepcopy(wdrs_arr[:,[8,9,10,11]]) #Copy imp ftrs
#
imp_ftrs_list  = ['Avg_Water.Column.Height_cm', \
                    'SW_pH', 'DO_perc.sat', 'DO_mg.per.L', 'SW_Temp_degC'] #Important features

#********************************************************;
#  1b. SR vs. extrinsic factors for 9 compounds and sum  ;
#      (WHONDRS data); C = aQ^b the c-q eqn              ;
#********************************************************;
b_wdrs_list  = np.zeros((len(comp_list), 5), dtype = float) #(10,5) #scaling exp
cv_wdrs_list = np.zeros((len(comp_list), 5), dtype = float) #(10,5) #coeff of variation
#
for j in range(0,5): #Extrinsic features 0 to 5 (imp only)
    for i in range(0,len(comp_list)): #Compounds i = 0 to 10
        ind        = np.argwhere(~np.isnan(wdrs_ftimp_arr[:,j]))[:,0] #get non-nan samples
        popt, pcov = curve_fit(func, wdrs_ftimp_arr[ind,j], sr_arr[ind,i]) #get scaling exponent
        cv_sr      = np.std(sr_arr[ind,i])/np.mean(sr_arr[ind,i]) #coeff of variation for SR
        cv_ftrs    = np.std(wdrs_ftimp_arr[ind,j])/np.mean(wdrs_ftimp_arr[ind,j]) #coeff of variation for ftrs
        cv_ratio   = cv_sr/cv_ftrs #cv ratio
        #
        print(imp_ftrs_list[j], comp_list[i], popt[1], cv_sr, cv_ftrs, cv_ratio)
        #
        b_wdrs_list[i,j]  = popt[1]
        cv_wdrs_list[i,j] = cv_ratio

#*******************************;
#  1c. Plot c-q (WHONDRS data)  ;
#*******************************;
legend_properties = {'weight':'bold'}
fig = plt.figure()
ax  = fig.add_subplot(111)
ax.set_xlabel(r'$\frac{CV_{sr}}{CV_{ftrs}}$', fontsize = 12, fontweight = 'bold')
ax.set_ylabel('Exponent (z)', fontsize = 12, fontweight = 'bold')
plt.title('z vs CV (WHONDRS)')
#ax.set_xlim([np.min(cv_wdrs_list), np.max(cv_wdrs_list)])
#ax.set_ylim([np.min(b_wdrs_list), np.max(b_wdrs_list)])
#
ax.plot([0, 2], [0,0], \
        linestyle = 'dashed', linewidth = 0.5, marker = None, color = 'k') #One-to-One line
ax.plot([1,1], [-1, 1], \
        linestyle = 'dashed', linewidth = 0.5, marker = None, color = 'k') #One-to-One line
ax.plot([0,1], [0,1], \
        linestyle = 'dashed', linewidth = 1.5, marker = None, color = 'k') #One-to-One line
ax.plot([0,1], [0,-1], \
        linestyle = 'dashed', linewidth = 1.5, marker = None, color = 'k') #One-to-One line
for i in range(0,len(comp_list)): #Compounds i = 0 to 10
    color      = color_list[i]
    marker     = marker_list[i]
    #
    ax.scatter(cv_wdrs_list[i,:], b_wdrs_list[i,:], \
                s = 20, c = color, marker = marker, edgecolor = 'face')
fig.tight_layout()
plt.savefig(path + 'Plots_WHONDRS/Scaling_Laws/cq_wdrs.png')
plt.close(fig)

#**********************************************;
#  2a. Set paths for .csv files (StreamStats)  ;
#**********************************************;
path           = os.getcwd() + '/'
#
df_sr          = pd.read_csv(path + "Inputs_Outputs_v4/1_WAT_SR_54s_11f.csv", index_col = 1).iloc[:,1:] #[54 rows x 10 columns]
df_sd          = pd.read_csv(path + "Inputs_Outputs_v4/2_WAT_SHANN_54s_11f.csv", index_col = 1).iloc[:,1:] #[54 rows x 10 columns]
#
df_strmstats   = pd.read_csv(path + "Inputs_Outputs_v4/6_StreamStats_54s_9f.csv", index_col = 3).iloc[:,1:] #[54 rows x 8 columns]
#
comp_list      = df_sr.columns.to_list() #10
strm_ftrs_list = df_strmstats.columns.to_list() #8
#
sr_arr         = df_sr.values #(54, 10)
strm_arr       = df_strmstats.values #(54, 8)
#
s_index_list = [1, 2, 3, 4, 7] #imp ftrs indices
s_ftrs_list  = ['Latitude', 'DrainageArea', 'Elevation', 'Precipitation', 'ForestCover'] #imp features
#
hs_arr       = copy.deepcopy(strm_arr[:,s_index_list]) #(54, 5)

#******************************;
#  2b. c-q (StreamStats data)  ;
#******************************;
b_strm_list  = np.zeros((len(comp_list), 5), dtype = float) #(10,5) #scaling exp
cv_strm_list = np.zeros((len(comp_list), 5), dtype = float) #(10,5) #coeff of variation
#
for j in range(0,5): #Extrinsic features 0 to 5 (imp only)
    for i in range(0,len(comp_list)): #Compounds i = 0 to 10
        ind        = np.argwhere(~np.isnan(hs_arr[:,j]))[:,0] #Non-nan values
        popt, pcov = curve_fit(func, hs_arr[ind,j], sr_arr[ind,i]) #scaling law exp
        cv_sr      = np.std(sr_arr[ind,i])/np.mean(sr_arr[ind,i])
        cv_ftrs    = np.std(hs_arr[ind,j])/np.mean(hs_arr[ind,j])
        cv_ratio   = cv_sr/cv_ftrs
        #
        print(s_ftrs_list[j], comp_list[i], popt[1], cv_sr, cv_ftrs, cv_ratio)
        #
        b_strm_list[i,j]  = popt[1]
        cv_strm_list[i,j] = cv_ratio

#***********************************;
#  2c. Plot c-q (StreamStats data)  ;
#***********************************;
legend_properties = {'weight':'bold'}
fig = plt.figure()
ax  = fig.add_subplot(111)
ax.set_xlabel(r'$\frac{CV_{sr}}{CV_{ftrs}}$', fontsize = 12, fontweight = 'bold')
ax.set_ylabel('Exponent (z)', fontsize = 12, fontweight = 'bold')
plt.title('z vs CV (StreamStats)')
#ax.set_xlim([np.min(cv_strm_list), np.max(cv_strm_list)])
#ax.set_ylim([np.min(b_strm_list), np.max(b_strm_list)])
#
ax.plot([0, 2], [0,0], \
        linestyle = 'dashed', linewidth = 0.5, marker = None, color = 'k') #One-to-One line
ax.plot([1,1], [-1, 1], \
        linestyle = 'dashed', linewidth = 0.5, marker = None, color = 'k') #One-to-One line
ax.plot([0,1], [0,1], \
        linestyle = 'dashed', linewidth = 1.5, marker = None, color = 'k') #One-to-One line
ax.plot([0,1], [0,-1], \
        linestyle = 'dashed', linewidth = 1.5, marker = None, color = 'k') #One-to-One line
for i in range(0,len(comp_list)): #Compounds i = 0 to 10
    color      = color_list[i]
    marker     = marker_list[i]
    #
    ax.scatter(cv_strm_list[i,:], b_strm_list[i,:], \
                s = 25, c = color, marker = marker, edgecolor = 'face')
fig.tight_layout()
plt.savefig(path + 'Plots_StreamStats/Scaling_Laws/cq_strm.png')
plt.close(fig)

#*********************************************;
#  3a. Set paths for .csv files (HYDROSHEDS)  ;
#*********************************************;
path            = os.getcwd() + '/'
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
hydrosheds_index_list = [10, 46, 73, 87] #4 -- imp ftrs indices
hydrosheds_ftrs_list  = ['Natural discharge - dis_m3_pmn', 'Air temp - tmp_dc_s05', \
                            'Potential ET - pet_mm_s04', 'Actual ET - aet_mm_s04'] #4
#
hs_arr          = copy.deepcopy(huc12_arr[:,hydrosheds_index_list]) #(54, 4)

#*****************************;
#  3b. c-q (HYDROSHEDS data)  ;
#*****************************;
b_hydshds_list  = np.zeros((len(comp_list), 4), dtype = float) #(10,4)
cv_hydshds_list = np.zeros((len(comp_list), 4), dtype = float) #(10,4)
#
for j in range(0,4): #Extrinsic features 0 to 4; imp only
    for i in range(0,len(comp_list)): #Compounds i = 0 to 10
        ind        = np.argwhere(~np.isnan(hs_arr[:,j]))[:,0]
        popt, pcov = curve_fit(func, hs_arr[ind,j], sr_arr[ind,i])
        cv_sr      = np.std(sr_arr[ind,i])/np.mean(sr_arr[ind,i])
        cv_ftrs    = np.std(hs_arr[ind,j])/np.mean(hs_arr[ind,j])
        cv_ratio   = cv_sr/cv_ftrs
        #
        print(s_ftrs_list[j], comp_list[i], popt[1], cv_sr, cv_ftrs, cv_ratio)
        #
        b_hydshds_list[i,j]  = popt[1]
        cv_hydshds_list[i,j] = cv_ratio

#***********************************;
#  3c. Plot c-q (StreamStats data)  ;
#***********************************;
legend_properties = {'weight':'bold'}
fig = plt.figure()
ax  = fig.add_subplot(111)
ax.set_xlabel(r'$\frac{CV_{sr}}{CV_{ftrs}}$', fontsize = 12, fontweight = 'bold')
ax.set_ylabel('Exponent (z)', fontsize = 12, fontweight = 'bold')
plt.title('z vs CV (HYDROSHEDS)')
#ax.set_xlim([np.min(cv_hydshds_list), np.max(cv_hydshds_list)])
#ax.set_ylim([np.min(b_hydshds_list), np.max(b_hydshds_list)])
#
ax.plot([0, 2], [0,0], \
        linestyle = 'dashed', linewidth = 0.5, marker = None, color = 'k') #One-to-One line
ax.plot([1,1], [-1, 1], \
        linestyle = 'dashed', linewidth = 0.5, marker = None, color = 'k') #One-to-One line
ax.plot([0,1], [0,1], \
        linestyle = 'dashed', linewidth = 1.5, marker = None, color = 'k') #One-to-One line
ax.plot([0,1], [0,-1], \
        linestyle = 'dashed', linewidth = 1.5, marker = None, color = 'k') #One-to-One line
for i in range(0,len(comp_list)): #Compounds i = 0 to 10
    color      = color_list[i]
    marker     = marker_list[i]
    #
    ax.scatter(cv_hydshds_list[i,:], b_hydshds_list[i,:], \
                s = 20, c = color, marker = marker, edgecolor = 'face')
fig.tight_layout()
plt.savefig(path + 'Plots_HYDROSHEDS/Scaling_Laws/cq_hydshds.png')
plt.close(fig)

#******************************************;
#  4a. Set paths for .csv files (EPA-ACC)  ;
#******************************************;
path           = os.getcwd() + '/'
#
df_sr          = pd.read_csv(path + "Inputs_Outputs_v4/1_WAT_SR_54s_11f.csv", index_col = 1).iloc[:,1:] #[54 rows x 10 columns]
df_sd          = pd.read_csv(path + "Inputs_Outputs_v4/2_WAT_SHANN_54s_11f.csv", index_col = 1).iloc[:,1:] #[54 rows x 10 columns]
#
df_acc_all     = pd.read_csv(path + "Inputs_Outputs_v4/5_ACC_EPAWaters_54s_141f.csv", index_col = 1).iloc[:,1:] #[54 rows x 140 columns]
acc_list       = np.delete(np.arange(0,140), [2,3,126])
df_acc         = df_acc_all.iloc[:,acc_list] #[54 rows x 137 columns]
#
comp_list      = df_sr.columns.to_list() #10
acc_ftrs_list  = df_acc.columns.to_list() #137
#
sr_arr         = df_sr.values #(54, 10)
acc_arr        = df_acc.values #(54, 137)
#
epaacc_index_list = [34, 82, 115, 126, 131] #5 -- imp features
epaacc_ftrs_list  = ['Percent forest cover loss - PctFrstLoss', \
                        'Precipitation gradient - SN', \
                        'PctNonAgIntrodManagVegCat', \
                        'Mean annual stream temperature -- MAST', \
                        'Mean summer stream temperature - MSST'] #5
#
hs_arr          = copy.deepcopy(acc_arr[:,epaacc_index_list]) #(54, 6)

#**************************;
#  4b. c-q (EPA-ACC data)  ;
#**************************;
b_acc_list  = np.zeros((len(comp_list), 5), dtype = float) #(10,5)
cv_acc_list = np.zeros((len(comp_list), 5), dtype = float) #(10,5)
#
for j in range(0,5): #Extrinsic features 0 to 5; impr only
    for i in range(0,len(comp_list)): #Compounds i = 0 to 10
        ind        = np.argwhere(~np.isnan(hs_arr[:,j]))[:,0]
        popt, pcov = curve_fit(func, hs_arr[ind,j], sr_arr[ind,i])
        cv_sr      = np.std(sr_arr[ind,i])/np.mean(sr_arr[ind,i])
        cv_ftrs    = np.std(hs_arr[ind,j])/np.mean(hs_arr[ind,j])
        cv_ratio   = cv_sr/cv_ftrs
        #
        print(s_ftrs_list[j], comp_list[i], popt[1], cv_sr, cv_ftrs, cv_ratio)
        #
        b_acc_list[i,j]  = popt[1]
        cv_acc_list[i,j] = cv_ratio

#*******************************;
#  4c. Plot c-q (EPA-ACC data)  ;
#*******************************;
legend_properties = {'weight':'bold'}
fig = plt.figure()
ax  = fig.add_subplot(111)
ax.set_xlabel(r'$\frac{CV_{sr}}{CV_{ftrs}}$', fontsize = 12, fontweight = 'bold')
ax.set_ylabel('Exponent (z)', fontsize = 12, fontweight = 'bold')
plt.title('z vs CV (EPA-ACC)')
#ax.set_xlim([np.min(cv_acc_list), np.max(cv_acc_list)])
#ax.set_ylim([np.min(b_acc_list), np.max(b_acc_list)])
#
ax.plot([0, 2], [0,0], \
        linestyle = 'dashed', linewidth = 0.5, marker = None, color = 'k') #One-to-One line
ax.plot([1,1], [-1, 1], \
        linestyle = 'dashed', linewidth = 0.5, marker = None, color = 'k') #One-to-One line
ax.plot([0,1], [0,1], \
        linestyle = 'dashed', linewidth = 1.5, marker = None, color = 'k') #One-to-One line
ax.plot([0,1], [0,-1], \
        linestyle = 'dashed', linewidth = 1.5, marker = None, color = 'k') #One-to-One line
for i in range(0,len(comp_list)): #Compounds i = 0 to 10
    color      = color_list[i]
    marker     = marker_list[i]
    #
    ax.scatter(cv_acc_list[i,:], b_acc_list[i,:], \
                s = 25, c = color, marker = marker, edgecolor = 'face')
fig.tight_layout()
plt.savefig(path + 'Plots_EPAWaters_ACC/Scaling_Laws/cq_acc.png')
plt.close(fig)

#******************************************;
#  5a. Set paths for .csv files (EPA-ACW)  ;
#******************************************;
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
epaacw_index_list = [15, 17, 110, 111, 112, 113] #6 -- imp ftrs
epaacw_ftrs_list  = ['NPDESDensWs','TRIDensWs', 'PctUrbHiWs', 'PctUrbLoWs', 'PctUrbMdWs', 'PctUrbOpWs'] #6
#
hs_arr          = copy.deepcopy(acw_arr[:,epaacw_index_list]) #(54, 6)

#**************************;
#  5b. c-q (EPA-ACW data)  ;
#**************************;
b_acw_list  = np.zeros((len(comp_list), 6), dtype = float) #(10,6)
cv_acw_list = np.zeros((len(comp_list), 6), dtype = float) #(10,6)
#
for j in range(0,6): #Extrinsic features 0 to 6 -- impr ftrs
    for i in range(0,len(comp_list)): #Compounds i = 0 to 10
        ind        = np.argwhere(~np.isnan(hs_arr[:,j]))[:,0]
        popt, pcov = curve_fit(func, hs_arr[ind,j], sr_arr[ind,i])
        cv_sr      = np.std(sr_arr[ind,i])/np.mean(sr_arr[ind,i])
        cv_ftrs    = np.std(hs_arr[ind,j])/np.mean(hs_arr[ind,j])
        cv_ratio   = cv_sr/cv_ftrs
        #
        print(epaacw_ftrs_list[j], comp_list[i], popt[1], cv_sr, cv_ftrs, cv_ratio)
        #
        b_acw_list[i,j]  = popt[1]
        cv_acw_list[i,j] = cv_ratio

#*******************************;
#  5c. Plot c-q (EPA-ACW data)  ;
#*******************************;
legend_properties = {'weight':'bold'}
fig = plt.figure()
ax  = fig.add_subplot(111)
ax.set_xlabel(r'$\frac{CV_{sr}}{CV_{ftrs}}$', fontsize = 12, fontweight = 'bold')
ax.set_ylabel('Exponent (z)', fontsize = 12, fontweight = 'bold')
plt.title('z vs CV (EPA-ACW)')
#ax.set_xlim([np.min(cv_acw_list), np.max(cv_acw_list)])
#ax.set_ylim([np.min(b_acw_list), np.max(b_acw_list)])
#
ax.plot([0, 2], [0,0], \
        linestyle = 'dashed', linewidth = 0.5, marker = None, color = 'k') #One-to-One line
ax.plot([1,1], [-1, 1], \
        linestyle = 'dashed', linewidth = 0.5, marker = None, color = 'k') #One-to-One line
ax.plot([0,1], [0,1], \
        linestyle = 'dashed', linewidth = 1.5, marker = None, color = 'k') #One-to-One line
ax.plot([0,1], [0,-1], \
        linestyle = 'dashed', linewidth = 1.5, marker = None, color = 'k') #One-to-One line
for i in range(0,len(comp_list)): #Compounds i = 0 to 10
    color      = color_list[i]
    marker     = marker_list[i]
    #
    ax.scatter(cv_acw_list[i,:], b_acw_list[i,:], \
                s = 30, c = color, marker = marker, edgecolor = 'face')
fig.tight_layout()
plt.savefig(path + 'Plots_EPAWaters_ACW/Scaling_Laws/cq_acw.png')
plt.close(fig)

#**************************;
#  6. Plot c-q (all data)  ;
#**************************;
legend_properties = {'weight':'bold'}
fig = plt.figure()
ax  = fig.add_subplot(111)
ax.set_xlabel(r'$\frac{CV_{sr}}{CV_{ftrs}}$', fontsize = 12, fontweight = 'bold')
ax.set_ylabel('Exponent (z)', fontsize = 12, fontweight = 'bold')
plt.title('z vs CV (All datasets)')
#ax.set_xlim([np.min(cv_acw_list), np.max(cv_acw_list)])
#ax.set_ylim([np.min(b_acw_list), np.max(b_acw_list)])
#
ax.plot([0, 2], [0,0], \
        linestyle = 'dashed', linewidth = 0.5, marker = None, color = 'k') #One-to-One line
ax.plot([1,1], [-1, 1], \
        linestyle = 'dashed', linewidth = 0.5, marker = None, color = 'k') #One-to-One line
ax.plot([0,1], [0,1], \
        linestyle = 'dashed', linewidth = 1.5, marker = None, color = 'k') #One-to-One line
ax.plot([0,1], [0,-1], \
        linestyle = 'dashed', linewidth = 1.5, marker = None, color = 'k') #One-to-One line
for i in range(0,len(comp_list)): #Compounds i = 0 to 10
    color      = color_list[i]
    marker     = marker_list[i]
    #
    ax.scatter(cv_wdrs_list[i,:], b_wdrs_list[i,:], \
                s = 10, c = color, marker = marker, edgecolor = 'face')
    ax.scatter(cv_strm_list[i,:], b_strm_list[i,:], \
                s = 15, c = color, marker = marker, edgecolor = 'face')
    ax.scatter(cv_hydshds_list[i,:], b_hydshds_list[i,:], \
                s = 20, c = color, marker = marker, edgecolor = 'face')
    ax.scatter(cv_acc_list[i,:], b_acc_list[i,:], \
                s = 25, c = color, marker = marker, edgecolor = 'face')
    ax.scatter(cv_acw_list[i,:], b_acw_list[i,:], \
                s = 30, c = color, marker = marker, edgecolor = 'face')
fig.tight_layout()
plt.savefig(path + 'CQ_all/cq_all.png')
plt.close(fig)

#****************************************************************;
#  7. SR vs. extrinsic factors for 9 compounds and sum (labels)  ;
#****************************************************************;
legend_properties = {'weight':'bold'}
fig = plt.figure()
ax  = fig.add_subplot(111)
ax.set_axis_off()
for i in range(0,len(comp_list)): #Compounds i = 0 to 10
    color      = color_list[i]
    marker     = marker_list[i]
    #
    ax.plot([], [], color = color, marker = marker, linestyle = 'None', \
            label = comp_list[i])
ax.legend(bbox_to_anchor=(0.0,0.0), loc = 'upper left')
fig.tight_layout()
plt.savefig(path + 'CQ_all/legend_cq.png')
plt.close(fig)
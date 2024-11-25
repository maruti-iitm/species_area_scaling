# Perform PCA analysis on WHONDRS, StreamStats, HYDROSHEDS, EPA-ACC, and EPA-ACW 
#   1. on imp features based on Table-1
#
#   WHONDRS features
#       'DS_Water.Column.Height_cm' --> 6
#       'SW_Temp_degC'              --> 11
#
#   StreamStats features
#       'lat'     --> 1
#       'DRNAREA' --> 2
#
#   HydroSheds features
#       'Natural discharge - dis_m3_pmn' --> 10
#       'Air temp - tmp_dc_s05'          --> 46
#       'Potential ET - pet_mm_s04'      --> 73 
#       'Actual ET - aet_mm_s04'         --> 87
#
#   EPAW-Catchment features
#       'Mean summer stream temperature - MSST' --> 131
#
#   EPAW-Watershed features
#       'PctImpWs' --> 54
#       'TmeanWs'  --> 123       
#
# https://stackoverflow.com/questions/45333733/plotting-pca-output-in-scatter-plot-whilst-colouring-according-to-to-label-pytho
# AUTHOR -- Maruti Kumar Mudunuru
#  Loadings:
#    https://www.nxn.se/valent/loadings-with-scikit-learn-pca
#    https://stackoverflow.com/questions/21217710/factor-loadings-using-sklearn
#    https://stackoverflow.com/questions/39216897/plot-pca-loadings-and-loading-in-biplot-in-sklearn-like-rs-autoplot

import os
import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import itertools
#
import sklearn #'1.0.2'
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
#
np.set_printoptions(precision=2)

#******************************************;
#  1a. Set paths for .csv files (WHONDRS)  ;
#******************************************;
path              = os.getcwd() + '/'
#
df_sr             = pd.read_csv(path + "Inputs_Outputs_v4/1_WAT_SR_54s_11f.csv", index_col = 1).iloc[:,1:] #[54 rows x 10 columns]
df_sd             = pd.read_csv(path + "Inputs_Outputs_v4/2_WAT_SHANN_54s_11f.csv", index_col = 1).iloc[:,1:] #[54 rows x 10 columns]
#
df_wdrs           = pd.read_csv(path + 'Inputs_Outputs_v4/7_PCA_Eric_WHONDRS_54s_13f.csv', index_col = 1).iloc[:,1:] #[54 rows x 12 columns]
#
comp_list         = df_sr.columns.to_list() #10
wdrs_ftrs_list    = df_wdrs.columns.to_list() #12
#
sr_arr            = df_sr.values #(54, 10) #species richness
sd_arr            = df_sd.values #(54, 10) #shannon diversity
wdrs_arr          = df_wdrs.fillna(method="ffill").values[:,[6,11]] #(54, 2) #ftrs and fill NaNs for SW_Temp_degC
wdrs_id_list      = [6, 11] #wdrs ids that are non-NaNs
wdrs_ftrs_id_list = [wdrs_ftrs_list[i] for i in wdrs_id_list] #2
#print(np.min(wdrs_arr, axis=0))
#print(np.max(wdrs_arr, axis=0))

#**********************************************;
#  1b. Set paths for .csv files (StreamStats)  ;
#**********************************************;
df_strmstats      = pd.read_csv(path + "Inputs_Outputs_v4/6_StreamStats_54s_9f.csv", index_col = 3).iloc[:,1:]#[54 rows x 8 columns]
#
comp_list         = df_sr.columns.to_list() #10
strm_ftrs_list    = df_strmstats.columns.to_list() #8
#
strm_arr          = df_strmstats.values[:,[1,2]] #(54, 2) #ftrs
strm_id_list      = [1, 2] #strm ids that are non-NaNs
strm_ftrs_id_list = [strm_ftrs_list[i] for i in strm_id_list] #2
#print(np.min(strm_arr, axis=0))
#print(np.max(strm_arr, axis=0))

#*********************************************;
#  1c. Set paths for .csv files (HYDROSHEDS)  ;
#*********************************************;
df_huc12              = pd.read_csv(path + "Inputs_Outputs_v4/8_Hydrosheds_huc12_54s_294f.csv", index_col = 1).iloc[:,1:-1] #[54 rows x 292 columns]
#
comp_list             = df_sr.columns.to_list() #10
huc12_ftrs_list       = df_huc12.columns.to_list() #292
#
huc12_arr             = df_huc12.values #(54, 292)
#
hydrosheds_index_list = [10, 46, 73, 87] #4
hydrosheds_ftrs_list  = ['Natural discharge - dis_m3_pmn', 'Air temp - tmp_dc_s05', \
                            'Potential ET - pet_mm_s04', 'Actual ET - aet_mm_s04'] #4
#
hs_arr                = copy.deepcopy(huc12_arr[:,hydrosheds_index_list]) #(54, 4)
#print(np.min(hs_arr, axis=0))
#print(np.max(hs_arr, axis=0))

#******************************************;
#  1d. Set paths for .csv files (EPA-ACC)  ;
#******************************************;
df_acc_all     = pd.read_csv(path + "Inputs_Outputs_v4/5_ACC_EPAWaters_54s_141f.csv", index_col = 1).iloc[:,1:] #[54 rows x 140 columns]
acc_list       = np.delete(np.arange(0,140), [2,3,126])
df_acc         = df_acc_all.iloc[:,acc_list] #[54 rows x 137 columns]
#
comp_list      = df_sr.columns.to_list() #10
acc_ftrs_list  = df_acc.columns.to_list() #137
#
acc_arr        = df_acc.values #(54, 137)
#
epaacc_index_list = [131] #1
epaacc_ftrs_list  = ['Mean summer stream temperature - MSST'] #1
#
epac_arr          = copy.deepcopy(acc_arr[:,epaacc_index_list]) #(54, 1)
#print(np.min(epac_arr, axis=0))
#print(np.max(epac_arr, axis=0))

#******************************************;
#  1e. Set paths for .csv files (EPA-ACW)  ;
#******************************************;
df_acw_all     = pd.read_csv(path + "Inputs_Outputs_v4/5_ACW_EPAWaters_54s_141f.csv", index_col = 1).iloc[:,1:] #[54 rows x 140 columns]
acw_list       = np.delete(np.arange(0,140), [2,3,126])
df_acw         = df_acw_all.iloc[:,acw_list] #[54 rows x 137 columns]
#
comp_list      = df_sr.columns.to_list() #10
acw_ftrs_list  = df_acw.columns.to_list() #137
#
acw_arr        = df_acw.values #(54, 137)
#
epaacw_index_list = [54, 123] #2
epaacw_ftrs_list  = ['PctImpWs', 'TmeanWs'] #2
#
epaw_arr          = copy.deepcopy(acw_arr[:,epaacw_index_list]) #(54, 2)
#print(np.min(epaw_arr, axis=0))
#print(np.max(epaw_arr, axis=0))

#*******************************************;
#  2a. Construct the full data matrix (raw) ;
#*******************************************;
num_samples       = wdrs_arr.shape[0] #54
num_ftrs          = wdrs_arr.shape[1] + strm_arr.shape[1] + \
                    hs_arr.shape[1] + epac_arr.shape[1] + \
                    epaw_arr.shape[1] #11
full_arr          = np.zeros((num_samples, num_ftrs), dtype = float) #(54, 11)
#
ftrs_list         = [wdrs_ftrs_id_list, strm_ftrs_id_list, \
                        hydrosheds_ftrs_list, epaacc_ftrs_list, \
                        epaacw_ftrs_list] #11
features_list     = list(itertools.chain.from_iterable(ftrs_list)) #54
#
full_arr[:,0:2]   = copy.deepcopy(wdrs_arr) #whondrs #2
full_arr[:,2:4]   = copy.deepcopy(strm_arr) #streamstats #2
full_arr[:,4:8]   = copy.deepcopy(hs_arr) #hydrosheds #4
full_arr[:,8:9]   = copy.deepcopy(epac_arr) #epa-catchment #1
full_arr[:,9:11]  = copy.deepcopy(epaw_arr) #epa-watershed #2
#
nonnan_rows_list  = [] #find non-nan samples = 54
#
for i in range(0,num_samples): #Iterate over number of samples
    if len(np.argwhere(np.isnan(full_arr[i,:]))[:,0]) == 0:
        print(i, len(np.argwhere(np.isnan(full_arr[i,:]))[:,0]))
        nonnan_rows_list.append(i) #Non-nan sample indices

X = copy.deepcopy(full_arr[nonnan_rows_list,:]) #(54, 11); full data features with non-nan samples
y = copy.deepcopy(sr_arr[nonnan_rows_list,:]) #(54, 10); sr corresponding to non-nan samples

#************************************************************;
#  2b. Normalize the full data matrix using Standard Scalar  ;
#************************************************************;
fdm_ss = StandardScaler() #Full-data-matrix (fdm) standard-scalar
fdm_ss.fit(X) #Fit standard-scalar for full-data-matrix (fdm)
#
X_ss   = fdm_ss.transform(X) #Transform full-data-matrix (54, 54); (num_samples, num_ftrs)
X_ss_t = X_ss.T #Transpose the transformed full-data-matrix (54, 54); (num_ftrs, n_samples)

#******************************************************************************;
#  2c. Perform PCA analysis on (n_ftrs, n_samples) --> (n_ftrs, n_components)  ;
#******************************************************************************;
pca_ftrs   = PCA(n_components=2, random_state = 1337)
pca_ftrs.fit(X_ss_t) #PCA on ftrs
X_pca_ss_t = pca_ftrs.fit_transform(X_ss_t) #(54, 2)

#********************************************************************************;
#  2d. Perform PCA analysis on (n_samples,n_ftrs) --> (n_samples, n_components)  ;
#********************************************************************************;
pca_samples = PCA(n_components=2, random_state = 1337)
pca_samples.fit(X_ss) #PCA on samples
X_pca_ss    = pca_samples.fit_transform(X_ss) #(54, 2)
scores      = copy.deepcopy(X_pca_ss[:,:2]) #(54, 2)
loadings    = pca_samples.components_[:2].T #(54, 2)
pvars       = pca_samples.explained_variance_ratio_[:2] * 100 #(2,)
arrows_list = loadings * np.abs(scores).max(axis=0)
arrows_list = loadings * np.ptp(scores, axis=0)

#********************************************************************************************;
#  3a. Plot PCA components with labels = WHONDRS, StreamStats, HYDROSHEDS, EPA-C, and EPA-W  ;
#      (Variation across data samples)                                                       ;
#********************************************************************************************;
legend_properties = {'weight':'bold'}
fig = plt.figure(figsize=(5,5))
plt.rc('legend', fontsize = 8)
ax  = fig.add_subplot(111)
ax.set_xlabel('PC 1 (%.2f%%)' % (pca_samples.explained_variance_ratio_[0]*100))
ax.set_ylabel('PC 2 (%.2f%%)' % (pca_samples.explained_variance_ratio_[1]*100))
ax.scatter(X_pca_ss[:,0], X_pca_ss[:,1], c = 'k', s = 25, edgecolor = ['none'])
fig.tight_layout()
plt.savefig(path + 'PCA_all/PCA_samples_all_table1.png', dpi = 300)
plt.savefig(path + 'PCA_all/SVG_figs/PCA_samples_all_table1.svg')
plt.close(fig)

#********************************************************************************************;
#  3c. Plot PCA components with labels = WHONDRS, StreamStats, HYDROSHEDS, EPA-C, and EPA-W  ;
#      (Variation across data samples)                                                       ;
#********************************************************************************************;
legend_properties = {'weight':'bold'}
fig = plt.figure(figsize=(5,5))
plt.rc('legend', fontsize = 8)
ax  = fig.add_subplot(111)
ax.set_xlabel('PC 1 (%.2f%%)' % (pca_samples.explained_variance_ratio_[0]*100))
ax.set_ylabel('PC 2 (%.2f%%)' % (pca_samples.explained_variance_ratio_[1]*100))
ax.scatter(X_pca_ss[:,0], X_pca_ss[:,1], c = 'k', s = 25, edgecolor = ['none'])
#
#empirical formula to determine arrow width
width = -0.0075 * np.min([np.subtract(*plt.xlim()), np.subtract(*plt.ylim())])
#
# features as arrows
for feature, arrow in zip(features_list, arrows_list):
    ax.arrow(0, 0, *arrow, color='k', alpha=0.5, width=width, ec='none',
                length_includes_head=True)
    ax.text(*(arrow * 1.05), feature,
                ha='center', va='center', fontsize = 'xx-small')
fig.tight_layout()
plt.savefig(path + 'PCA_all/PCA_samples_all_loadings_impftrs.png', dpi = 300)
plt.savefig(path + 'PCA_all/SVG_figs/PCA_samples_all_loadings_impftrs.svg')
plt.close(fig)
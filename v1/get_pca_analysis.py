# Perform PCA analysis on WHONDRS, StreamStats, HYDROSHEDS, EPA-ACC, and EPA-ACW on imp features
#
#   WHONDRS features removed for only PCA analysis because of NaNs
#       DO_perc.sat  --> 10
#       DO_mg.per.L  --> 11
#       SW_Temp_degC --> 12
#
#   StreamStats features removed for only PCA analysis because of NaNs
#       ELEV     --> 3
#       PRECIP   --> 4
#       ELEVMAX  --> 5
#       MINBELEV --> 6
#       FOREST   --> 7
#
# https://stackoverflow.com/questions/45333733/plotting-pca-output-in-scatter-plot-whilst-colouring-according-to-to-label-pytho
# AUTHOR -- Maruti Kumar Mudunuru

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
wdrs_arr       = df_wdrs.values[:,0:9] #(54, 9) #ftrs
wdrs_id_list   = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] #wdrs ids that are non-NaNs

#**********************************************;
#  1b. Set paths for .csv files (StreamStats)  ;
#**********************************************;
df_strmstats   = pd.read_csv(path + "Inputs_Outputs_v4/6_StreamStats_54s_9f.csv", index_col = 3).iloc[:,1:]#[54 rows x 8 columns]
#
comp_list      = df_sr.columns.to_list() #10
strm_ftrs_list = df_strmstats.columns.to_list() #8
#
strm_arr       = df_strmstats.values[:,0:3] #(54, 3) #ftrs
strm_id_list   = [0, 1, 2] #strm ids that are non-NaNs

#*********************************************;
#  1c. Set paths for .csv files (HYDROSHEDS)  ;
#*********************************************;
df_huc12        = pd.read_csv(path + "Inputs_Outputs_v4/8_Hydrosheds_huc12_54s_294f.csv", index_col = 1).iloc[:,1:-1] #[54 rows x 292 columns]
#
comp_list       = df_sr.columns.to_list() #10
huc12_ftrs_list = df_huc12.columns.to_list() #292
#
huc12_arr       = df_huc12.values #(54, 292)
#
hydrosheds_index_list = [10, 17, 24, 26, 37, 46, 73, 87] #8
hydrosheds_ftrs_list  = ['Natural discharge - dis_m3_pmn', 'Inundation extent - inu_pc_slt', \
                            'River area - ria_ha_ssu', 'River volume - riv_tc_ssu', \
                            'Climate strata - cls_cl_smj', 'Air temp - tmp_dc_s05', \
                            'Potential ET - pet_mm_s04', 'Actual ET - aet_mm_s04'] #8
#
hs_arr           = copy.deepcopy(huc12_arr[:,hydrosheds_index_list]) #(54, 8)

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
epaacc_index_list = [34, 52, 54, 82, 92, 97, 114, 115, 119, 122, 126, 131] #12
epaacc_ftrs_list  = ['Percent forest cover loss - PctFrstLoss', \
                        'Mean hydraulic conductivity in catchment -- HydrlCondCat', \
                        'Mean imperviousness of anthropogenic surfaces within catchment - PctImp', \
                        'Precipitation gradient - SN', \
                        'Open water land cover (in a catchment) - PctOw', \
                        'Developed catchment area (Land use) - PctUrbOp', \
                        'Woody wetland land cover -- PctWdWet', \
                        'PctNonAgIntrodManagVegCat', \
                        '30 year mean normal temperature - Tmean', \
                        '30 year max normal temperature - Tmax', \
                        'Mean annual stream temperature -- MAST', \
                        'Mean summer stream temperature - MSST'] #12
#
epac_arr          = copy.deepcopy(acc_arr[:,epaacc_index_list]) #(54, 12)

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
epaacw_index_list = [71, 112, 13, 2, 12, 54, 78, 77, 37, 110, 111, 122, 115, 123, 75, 76, 15, 92, 113, 11, 16, 17] #22
epaacw_ftrs_list  = ['PctNonCarbResidWs',
                        'PctUrbMdWs',
                        'DamNrmStorM3Ws',
                        'DRNAREA',
                        'DamNIDStorM3Ws',
                        'PctImpWs',
                        'NABD_NrmStorM3Ws',
                        'NABD_NIDStorM3Ws',
                        'PctFrstLossWs',
                        'PctUrbHiWs',
                        'PctUrbLoWs',
                        'TmaxWs',
                        'PctNonAgIntrodManagVegWs',
                        'TmeanWs',
                        'MineDensWs',
                        'NABD_DensWs',
                        'NPDESDensWs',
                        'PctOwWs',
                        'PctUrbOpWs',
                        'DamDensWs',
                        'SuperfundDensWs',
                        'TRIDensWs'] #22
#
epaw_arr          = copy.deepcopy(acw_arr[:,epaacw_index_list]) #(54, 22)

#*******************************************;
#  2a. Construct the full data matrix (raw) ;
#*******************************************;
num_samples       = wdrs_arr.shape[0] #54
num_ftrs          = wdrs_arr.shape[1] + strm_arr.shape[1] + \
                    hs_arr.shape[1] + epac_arr.shape[1] + \
                    epaw_arr.shape[1] #54
full_arr          = np.zeros((num_samples, num_ftrs), dtype = float)
#
full_arr[:,0:9]   = copy.deepcopy(wdrs_arr) #whondrs #9
full_arr[:,9:12]  = copy.deepcopy(strm_arr) #streamstats #3
full_arr[:,12:20] = copy.deepcopy(hs_arr) #hydrosheds #8
full_arr[:,20:32] = copy.deepcopy(epac_arr) #epa-catchment #12
full_arr[:,32:54] = copy.deepcopy(epaw_arr) #epa-watershed #22
#
nonnan_rows_list  = [] #find non-nan samples = 54
#
for i in range(0,num_samples): #Iterate over number of samples
    if len(np.argwhere(np.isnan(full_arr[i,:]))[:,0]) == 0:
        print(i, len(np.argwhere(np.isnan(full_arr[i,:]))[:,0]))
        nonnan_rows_list.append(i) #Non-nan sample indices

X = copy.deepcopy(full_arr[nonnan_rows_list,:]) #(54, 54); full data features with non-nan samples
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
pca        = PCA(n_components=2)
pca.fit(X_ss_t)
X_pca_ss_t = pca.fit_transform(X_ss_t) #(54, 2)

#*******************************************************************************************;
#  3. Plot PCA components with labels = WHONDRS, StreamStats, HYDROSHEDS, EPA-C, and EPA-W  ;
#*******************************************************************************************;
marker_list  = ['o', 'v', '8', 's', 'p', '*', 'h', '+', 'x', '^'] #10
color_list   = ['b', 'k', 'r', 'c', 'm', 'g', 'y', 'tab:purple', 'tab:brown', 'tab:orange'] #10
#
wdrs_cvec    = ['b' for i in range(0,wdrs_arr.shape[1])] #Color vector creation for wdrs
strm_cvec    = ['k' for i in range(0,strm_arr.shape[1])] #Color vector creation for strm
hs_cvec      = ['r' for i in range(0,hs_arr.shape[1])] #Color vector creation for hs
epac_cvec    = ['c' for i in range(0,epac_arr.shape[1])] #Color vector creation for epac
epaw_cvec    = ['m' for i in range(0,epaw_arr.shape[1])] #Color vector creation for epaw
#
cvec_sublist = [wdrs_cvec, strm_cvec, hs_cvec, epac_cvec, epaw_cvec]
cvec_full    = list(itertools.chain.from_iterable(cvec_sublist)) #Color vector creation for full ftrs
#
wdrs_mvec    = ['o' for i in range(0,wdrs_arr.shape[1])] #Marker vector creation for wdrs
strm_mvec    = ['v' for i in range(0,strm_arr.shape[1])] #Marker vector creation for strm
hs_mvec      = ['8' for i in range(0,hs_arr.shape[1])] #Marker vector creation for hs
epac_mvec    = ['s' for i in range(0,epac_arr.shape[1])] #Marker vector creation for epac
epaw_mvec    = ['p' for i in range(0,epaw_arr.shape[1])] #Marker vector creation for epaw
#
mvec_sublist = [wdrs_mvec, strm_mvec, hs_mvec, epac_mvec, epaw_mvec]
mvec_full    = list(itertools.chain.from_iterable(mvec_sublist)) #Marker vector creation for full ftrs
#
legend_properties = {'weight':'bold'}
fig = plt.figure(figsize=(5,5))
plt.rc('legend', fontsize = 8)
ax  = fig.add_subplot(111)
ax.set_xlabel('PC 1 (%.2f%%)' % (pca.explained_variance_ratio_[0]*100))
ax.set_ylabel('PC 2 (%.2f%%)' % (pca.explained_variance_ratio_[1]*100))
ax.scatter(X_pca_ss_t[:,0], X_pca_ss_t[:,1], c = cvec_full, s = 25, edgecolor = ['none'])
ax.plot([], [], color = 'b', marker = 'o', linestyle = 'None', label = 'WHONDRS')
ax.plot([], [], color = 'k', marker = 'o', linestyle = 'None', label = 'StreamStats')
ax.plot([], [], color = 'r', marker = 'o', linestyle = 'None', label = 'Hydrosheds')
ax.plot([], [], color = 'c', marker = 'o', linestyle = 'None', label = 'EPAW-C')
ax.plot([], [], color = 'm', marker = 'o', linestyle = 'None', label = 'EPAW-W')
ax.legend(loc = 'upper left')
fig.tight_layout()
plt.savefig(path + 'PCA_all/PCA_ftrs_all.png', dpi = 300)
plt.close(fig)
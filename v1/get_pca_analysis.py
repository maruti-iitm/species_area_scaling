# Perform PCA analysis on WHONDRS, StreamStats, HYDROSHEDS, EPA-ACC, and EPA-ACW
#
# AUTHOR -- Maruti Kumar Mudunuru

import os
import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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
wdrs_arr       = df_wdrs.values #(54, 12) #ftrs

#**********************************************;
#  1b. Set paths for .csv files (StreamStats)  ;
#**********************************************;
df_strmstats   = pd.read_csv(path + "Inputs_Outputs_v4/6_StreamStats_54s_9f.csv", index_col = 3).iloc[:,1:]#[54 rows x 8 columns]
#
comp_list      = df_sr.columns.to_list() #10
strm_ftrs_list = df_strmstats.columns.to_list() #8
#
strm_arr       = df_strmstats.values #(54, 8) #ftrs

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
num_samples = wdrs_arr.shape[0] #54
num_ftrs    = wdrs_arr.shape[1] + strm_arr.shape[1] + \
                hs_arr.shape[1] + epac_arr.shape[1] + \
                epaw_arr.shape[1] #62
full_arr    = np.zeros((num_samples, num_ftrs), dtype = float)
#
full_arr[:,0:12]  = copy.deepcopy(wdrs_arr) #whondrs
full_arr[:,12:20] = copy.deepcopy(strm_arr) #streamstats
full_arr[:,20:28] = copy.deepcopy(hs_arr) #hydrosheds
full_arr[:,28:40] = copy.deepcopy(epac_arr) #epa-catchment
full_arr[:,40:62] = copy.deepcopy(epaw_arr) #epa-watershed
#
nonnan_rows_list = [] #find non-nan samples = 23
#
for i in range(0,num_samples): #Iterate over number of samples
    if len(np.argwhere(np.isnan(full_arr[i,:]))[:,0]) == 0:
        print(i, len(np.argwhere(np.isnan(full_arr[i,:]))[:,0]))
        nonnan_rows_list.append(i) #Non-nan sample indices

X = copy.deepcopy(full_arr[nonnan_rows_list,:]) #(3, 62); full data features with non-nan samples
y = copy.deepcopy(sr_arr[nonnan_rows_list,:]) #(3, 10); sr corresponding to non-nan samples

#************************************************************;
#  2b. Normalize the full data matrix using Standard Scalar  ;
#************************************************************;
fdm_ss = StandardScaler() #Full-data-matrix (fdm) standard-scalar
fdm_ss.fit(X) #Fit standard-scalar for full-data-matrix (fdm)
#
X_ss  = fdm_ss.transform(X) #Transform full-data-matrix (3, 62)

#****************************;
#  2c. Perform PCA analysis  ;
#****************************;
pca      = PCA(n_components=2)
pca.fit(X_ss)
X_pca_tr = pca.fit_transform(X_ss) #(3, 2)
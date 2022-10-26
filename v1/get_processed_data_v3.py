# Species richness in water samples (9 comp and sum of all comps)  -- 1_WAT_SR.csv
# Shannon diversity in water samples (9 comp and sum of all comps) -- 2_WAT_SHANN.csv
#
# 1. WHONDRS     -- Metadata pre-processed from Eric for PCA analysis -- 7_PCA_Eric_WHONDRS.csv
# 2. HYDROSHEDS  -- Pre-processed data from Erika                     -- 8_Hydrosheds.xlsx
# 3. StreamStats -- Pre-processed data from Michelle                  -- 6_StreamStats.xlsx
# 4. EPAWaters   -- Pre-processed data from Michelle                  -- 5_ACC_EPAWaters.csv and 5_ACW_EPAWaters.csv
#
# Process species richness and shannon diversity for scaling laws, feature importance, and PCA analysis
# AUTHOR -- Maruti Kumar Mudunuru

import pandas as pd
import re
import numpy as np
import copy

#**************************************************************;
#  1. Set paths, create directories, and dump huc12.csv files  ;
#**************************************************************;
path           = '/Users/mudu605/OneDrive - PNNL/Desktop/Papers_PNNL/18_Crowd_ML/Python_Scripts/0-AlphaDiversity/'
#
df_sr          = pd.read_csv(path + 'Inputs_Outputs_v1/1_WAT_SR.csv') #[78 rows x 11 columns]
df_sd          = pd.read_csv(path + 'Inputs_Outputs_v1/2_WAT_SHANN.csv') #[78 rows x 11 columns]
#
df_whondrs     = pd.read_csv(path + 'Inputs_Outputs_v1/7_PCA_Eric_WHONDRS.csv').iloc[:,[0,1,2,3,6,7,10,11,14,15,16,17,18]] #[97 rows x 12 columns]
df_hydrosheds  = pd.read_excel(path + 'Inputs_Outputs_v1/8_Hydrosheds.xlsx') #[1147 rows x 294 columns]
df_streamstats = pd.read_excel(path + 'Inputs_Outputs_v1/6_StreamStats.xlsx').iloc[:,[3,4,8,9,11,12,15,16,17]]  #[73 rows x 9 columns]
df_acc         = pd.read_csv(path + 'Inputs_Outputs_v1/5_ACC_EPAWaters.csv') #[74 rows x 141 columns]
df_acw         = pd.read_csv(path + 'Inputs_Outputs_v1/5_ACW_EPAWaters.csv') #[74 rows x 141 columns]
#
huc_list       = df_hydrosheds.iloc[:,-1].values.tolist()
temp_list      = []
#
for i in range(0,len(huc_list)):
	huc_num = int(re.findall(r'\d+', huc_list[i].split('_')[2])[0])
	temp_list.append(huc_num)
	#print(i,huc_num)

huc_level_list = copy.deepcopy(np.asarray(temp_list, dtype = int)) #(1147,)
huc12_ind_list = np.argwhere(huc_level_list == 12)[:,0] #(95,)
df_hs12        = df_hydrosheds.iloc[huc12_ind_list,:] #[95 rows x 294 columns]
df_hs12.to_csv(path + "Inputs_Outputs_v1/8_Hydrosheds_huc12.csv") #[95 rows x 294 columns]

#****************************************************************************************************;
#  2. Extract common samples between SR/SHANN with whondrs, hydrosheds-huc12, streamstats, acc, acw  ;
#****************************************************************************************************;
### SPECIES RICHNESS DATA -- Sample_ID
sr_samp_list = df_sr.iloc[:,0].tolist() #(78,) 
temp_list    = []
#
for i in range(0,len(sr_samp_list)):
	samp_num = int(re.findall(r'\d+', sr_samp_list[i].split('_')[1])[0])
	temp_list.append(samp_num)
	#print(i,samp_num)
#
sr_sampid_list = copy.deepcopy(np.asarray(temp_list, dtype = int)) #(78,)
print('Species Richness = ', len(sr_sampid_list)) #(78,)
#
### SHANNON DIVERSITY DATA -- Sample_ID
sd_samp_list = df_sd.iloc[:,0].tolist() #(78,) 
temp_list    = []
#
for i in range(0,len(sd_samp_list)):
	sd_samp_num = int(re.findall(r'\d+', sd_samp_list[i].split('_')[1])[0])
	temp_list.append(sd_samp_num)
	#print(i,sd_samp_num)
#
sd_sampid_list = copy.deepcopy(np.asarray(temp_list, dtype = int)) #(78,)
print('Shannon Diversity = ', len(sd_sampid_list)) #(78,)
#
### WHONDRS-META-DATA -- Sample_ID
wdrs_samp_list = df_whondrs.iloc[:,0].tolist() #(97,)
temp_list      = []
#
for i in range(0,len(wdrs_samp_list)):
	wdrs_samp_num = int(re.findall(r'\d+', wdrs_samp_list[i].split('_')[1])[0])
	temp_list.append(wdrs_samp_num)
	#print(i,wdrs_samp_num)
#
wdrs_sampid_list = copy.deepcopy(np.asarray(temp_list, dtype = int)) #(97,)
print('WHONDRS Data = ', len(wdrs_sampid_list)) #(97,)
#
### HYDROSHEDS-HUC12 DATA -- Sample_ID
huc12_samp_list = df_hs12.iloc[:,0].tolist() #(95,)
temp_list       = []
#
for i in range(0,len(huc12_samp_list)):
	huc12_samp_num = int(re.findall(r'\d+', huc12_samp_list[i].split('_')[1])[0])
	temp_list.append(huc12_samp_num)
	#print(i,huc12_samp_num)
#
huc12_sampid_list = copy.deepcopy(np.asarray(temp_list, dtype = int)) #(95,)
print('HYDROSHEDS Data = ', len(huc12_sampid_list)) #(95,)
#
### STREAMSTATS DATA -- Sample_ID
strmstats_samp_list = df_streamstats.iloc[:,2].tolist() #(73,)
temp_list           = []
#
for i in range(0,len(strmstats_samp_list)):
	strmstats_samp_num = int(re.findall(r'\d+', strmstats_samp_list[i].split('_')[1])[0])
	temp_list.append(strmstats_samp_num)
	#print(i,strmstats_samp_num)
#
strmstats_sampid_list = copy.deepcopy(np.asarray(temp_list, dtype = int)) #(73,)
print('STREAMSTATS Data = ', len(strmstats_sampid_list)) #(73,)
#
### EPAWaters-ACC DATA -- Sample_ID
acc_samp_list = df_acc.iloc[:,0].tolist() #(74,)
temp_list     = []
#
for i in range(0,len(acc_samp_list)):
	acc_samp_num = int(re.findall(r'\d+', acc_samp_list[i].split('_')[1])[0])
	temp_list.append(acc_samp_num)
	#print(i,acc_samp_num)
#
acc_sampid_list = copy.deepcopy(np.asarray(temp_list, dtype = int)) #(74,)
print('EPAWaters-ACC Data = ', len(acc_sampid_list)) #(74,)
#
### EPAWaters-ACW DATA -- Sample_ID
acw_samp_list = df_acw.iloc[:,0].tolist() #(74,)
temp_list     = []
#
for i in range(0,len(acw_samp_list)):
	acw_samp_num = int(re.findall(r'\d+', acw_samp_list[i].split('_')[1])[0])
	temp_list.append(acw_samp_num)
	#print(i,acw_samp_num)
#
acw_sampid_list = copy.deepcopy(np.asarray(temp_list, dtype = int)) #(74,)
print('EPAWaters-ACW Data = ', len(acw_sampid_list)) #(74,)

#********************************************************************************************;
#  3. Common samples between SR/SHANN with whondrs, hydrosheds-huc12, streamstats, acc, acw  ;
#********************************************************************************************;
s1_list             = np.intersect1d(sr_sampid_list,sd_sampid_list) #(78,)
s2_list             = np.intersect1d(s1_list,wdrs_sampid_list) #(78,)
s3_list             = np.intersect1d(s2_list,huc12_sampid_list) #(76,)
s4_list             = np.intersect1d(s3_list,strmstats_sampid_list) #(60,)
s5_list             = np.intersect1d(s4_list,acc_sampid_list) #(54,)
s6_list             = np.intersect1d(s5_list,acw_sampid_list) #(54,)
common_samples_list = copy.deepcopy(s5_list) #(54,)
#
sr_argwhr_list        = []
sd_argwhr_list        = []
wdrs_argwhr_list      = []
huc12_argwhr_list     = []
strmstats_argwhr_list = []
acc_argwhr_list       = []
acw_argwhr_list       = []

for i in range(0,len(common_samples_list)):
	sr_comm_id        = np.argwhere(sr_sampid_list == common_samples_list[i])[0,0]
	sd_comm_id        = np.argwhere(sd_sampid_list == common_samples_list[i])[0,0]
	wdrs_comm_id      = np.argwhere(wdrs_sampid_list == common_samples_list[i])[0,0]
	huc12_comm_id     = np.argwhere(huc12_sampid_list == common_samples_list[i])[0,0]
	strmstats_comm_id = np.argwhere(strmstats_sampid_list == common_samples_list[i])[0,0]
	acc_comm_id       = np.argwhere(acc_sampid_list == common_samples_list[i])[0,0]
	acw_comm_id       = np.argwhere(acw_sampid_list == common_samples_list[i])[0,0]
	#
	#print(i,sr_comm_id,sd_comm_id,wdrs_comm_id,huc12_comm_id,strmstats_comm_id,acc_comm_id,acw_comm_id)
	sr_argwhr_list.append(sr_comm_id)
	sd_argwhr_list.append(sd_comm_id)
	wdrs_argwhr_list.append(wdrs_comm_id)
	huc12_argwhr_list.append(huc12_comm_id)
	strmstats_argwhr_list.append(strmstats_comm_id)
	acc_argwhr_list.append(acc_comm_id)
	acw_argwhr_list.append(acw_comm_id)

df_sr_comm        = df_sr.iloc[sr_argwhr_list,:].copy(deep=True) #(54, 11)
df_sd_comm        = df_sd.iloc[sd_argwhr_list,:].copy(deep=True) #(54, 11)
df_wdrs_comm      = df_whondrs.iloc[wdrs_argwhr_list,:].copy(deep=True) #(54, 13)
df_huc12_comm     = df_hs12.iloc[huc12_argwhr_list,:].copy(deep=True) #(54, 294)
df_strmstats_comm = df_streamstats.iloc[strmstats_argwhr_list,:].copy(deep=True) #(54, 9)
df_acc_comm       = df_acc.iloc[acc_argwhr_list,:].copy(deep=True) #(54, 141)
df_acw_comm       = df_acw.iloc[acw_argwhr_list,:].copy(deep=True) #(54, 141)
#
df_sr_comm.to_csv(path + "Inputs_Outputs_v4/1_WAT_SR_54s_11f.csv") #(54, 11)
df_sd_comm.to_csv(path + "Inputs_Outputs_v4/2_WAT_SHANN_54s_11f.csv") #(54, 11)
df_wdrs_comm.to_csv(path + "Inputs_Outputs_v4/7_PCA_Eric_WHONDRS_54s_13f.csv") #(54, 13)
df_huc12_comm.to_csv(path + "Inputs_Outputs_v4/8_Hydrosheds_huc12_54s_294f.csv") #(54, 294)
df_strmstats_comm.to_csv(path + "Inputs_Outputs_v4/6_StreamStats_54s_9f.csv") #(54, 9)
df_acc_comm.to_csv(path + "Inputs_Outputs_v4/5_ACC_EPAWaters_54s_141f.csv") #(54, 141)
df_acw_comm.to_csv(path + "Inputs_Outputs_v4/5_ACW_EPAWaters_54s_141f.csv") #(54, 141)
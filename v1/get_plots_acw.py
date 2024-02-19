# Get species richness vs. EPAWaters acw data
# Get shannon diversity vs. EPAWaters acw data
#
#71 0.246 0.205 PctNonCarbResidWs
#112 0.254 0.578 PctUrbMd2011Ws
#31 0.264 0.299 PctFrstLoss2003Ws
#13 0.281 0.372 DamNrmStorM3Ws
#2 0.282 0.273 DRNAREA
#12 0.285 0.381 DamNIDStorM3Ws
#54 0.286 0.547 PctImp2006Ws
#94 0.288 0.575 PctUrbHi2006Ws
#78 0.29 0.324 NABD_NrmStorM3Ws
#77 0.293 0.334 NABD_NIDStorM3Ws
#37 0.294 0.418 PctFrstLoss2009Ws
#110 0.301 0.56 PctUrbHi2011Ws
#111 0.306 0.559 PctUrbLo2011Ws
#122 0.312 0.323 Tmax8110Ws
#115 0.325 0.43 PctNonAgIntrodManagVegWs
#123 0.337 0.346 Tmean8110Ws
#75 0.337 0.407 MineDensWs
#76 0.351 0.345 NABD_DensWs
#15 0.358 0.615 NPDESDensWs
#92 0.37 0.357 PctOw2006Ws
#113 0.376 0.419 PctUrbOp2011Ws
#11 0.393 0.416 DamDensWs
#16 0.395 0.426 SuperfundDensWs
#17 0.421 0.611 TRIDensWs
#
# Process species richness and shannon diversity for scaling laws, feature importance, and PCA analysis
# AUTHOR -- Maruti Kumar Mudunuru

import os
import math
import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import linregress
import itertools
from scipy import stats
#
np.set_printoptions(precision=2)

#******************************;
#  1. Set pathsfor .csv files  ;
#******************************;
path           = os.getcwd() + '/'
#
df_sr          = pd.read_csv(path + "Inputs_Outputs_v4/1_WAT_SR_54s_11f.csv", index_col = 1).iloc[:,1:] #(54, 10)
df_sd          = pd.read_csv(path + "Inputs_Outputs_v4/2_WAT_SHANN_54s_11f.csv", index_col = 1).iloc[:,1:] #(54, 10)
#
df_acw_all     = pd.read_csv(path + "Inputs_Outputs_v4/5_ACW_EPAWaters_54s_141f.csv", index_col = 1).iloc[:,1:] #(54, 140)
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
marker_small   = [['o', 'v', '8', 's', 'p', '*', 'h', '+', 'x', '^'] for i in range(0,14)] #10
color_small    = [['b', 'k', 'r', 'c', 'm', 'g', 'y', 'tab:purple', 'tab:brown', 'tab:orange'] for i in range(0,14)] #10
#
marker_list    = list(itertools.chain.from_iterable(marker_small))
color_list     = list(itertools.chain.from_iterable(color_small))

#*******************************************************;
#  2a. ACW correlation coefficients (all 137 features)  ;
#*******************************************************;
for i in range(0,acw_arr.shape[1]):
	if len(np.argwhere(np.isnan(acw_arr[:,i]))[:,0]) > 0:
		print(i)

pcorr_list = []
scorr_list = []
count_list = []
#
for i in range(0,acw_arr.shape[1]):
	if len(np.argwhere(np.isnan(acw_arr[:,i]))[:,0]) == 0:
		pcorr, _ = pearsonr(acw_arr[:,i], sr_arr[:,-1])
		scorr, _ = spearmanr(acw_arr[:,i], sr_arr[:,-1])
		print('Pearsons correlation (SR vs. i): %.3f' % pcorr)
		print('Spearmans correlation (SR vs. i): %.3f' % scorr)
		pcorr_list.append(pcorr)
		scorr_list.append(scorr)
		count_list.append(i)
	else: #Ignoring features with NaN values
		pcorr, scorr = -1000, -1000
		pcorr_list.append(pcorr)
		scorr_list.append(scorr)
		count_list.append(i)

pcorr_list = np.asarray(pcorr_list, dtype = float)
scorr_list = np.asarray(scorr_list, dtype = float)
count_list = np.asarray(count_list, dtype = int)
#
ascend_argsort_pcorr = np.argsort(pcorr_list)[105:135] ##pcorr_list[np.argsort(pcorr_list)]
ascend_argsort_scorr = np.argsort(pcorr_list)[105:135]
ascend_argsort_count = np.argsort(pcorr_list)[105:135]

ascend_argsort_pcorr = np.argsort(pcorr_list)[105:135] ##pcorr_list[np.argsort(pcorr_list)]
ascend_argsort_scorr = np.argsort(pcorr_list)[105:135]
ascend_argsort_count = np.argsort(pcorr_list)[105:135]
#
print(pcorr_list[ascend_argsort_pcorr])
print(scorr_list[ascend_argsort_scorr])
print(count_list[ascend_argsort_count])
#
correlated_ftrs_list  = [acw_ftrs_list[i] for i in count_list[ascend_argsort_count]] #acw_ftrs_list[count_list[ascend_argsort_count]] 
correlated_count_list = [i for i in count_list[ascend_argsort_count]]
#
print('\n')
#
for i in range(0,len(correlated_count_list)):
	print(i, correlated_count_list[i], '{0:.3g}'.format(pcorr_list[ascend_argsort_pcorr][i]), \
		'{0:.3g}'.format(scorr_list[ascend_argsort_pcorr][i]), \
		correlated_ftrs_list[i])

#*******************************************************************************;
#  2b. Correlation values for SR vs. extrinsic factors for 9 compounds and sum  ;
#      (Along with p-values)                                                    ;
#*******************************************************************************;
full_pcorr_list        = [] 
full_scorr_list        = []
full_pcorr_pvalue_list = [] 
full_scorr_pvalue_list = []
i_name_list            = []
j_name_list            = []
#
for j in range(0,len(acw_ftrs_list)): #Extrinsic features 0 to 137
	for i in range(0,len(comp_list)): #Compounds i = 0 to 10
		print('Comp name, Extrinsic feature = ', comp_list[i], acw_ftrs_list[j])
		#print(np.argwhere(np.isnan(acw_arr[:,j]))[:,0])
		ind      = np.argwhere(~np.isnan(acw_arr[:,j]))[:,0]
		#
		pcorr_sr, pcorr_sr_pvalue = pearsonr(acw_arr[ind,j], sr_arr[ind,i])
		scorr_sr, scorr_sr_pvalue = spearmanr(acw_arr[ind,j], sr_arr[ind,i])
		print('i, j, Pearsons_sr, Spearmans_sr, pcorr-p-value, scorr-p-value = ', i, j, \
			'{0:.3g}'.format(pcorr_sr), '{0:.3g}'.format(scorr_sr), \
			'{0:.3g}'.format(pcorr_sr_pvalue), '{0:.3g}'.format(scorr_sr_pvalue))
		#
		i_name_list.append(comp_list[i])
		j_name_list.append(acw_ftrs_list[j])
		full_pcorr_list.append(pcorr_sr)
		full_scorr_list.append(scorr_sr)
		full_pcorr_pvalue_list.append(pcorr_sr_pvalue)
		full_scorr_pvalue_list.append(scorr_sr_pvalue)

df_ps = pd.DataFrame({
    'Compound_Name': i_name_list,
    'EPA-ACW_Features': j_name_list,
    'Pearsons_Correlation': full_pcorr_list,
    'Spearsman_Correlation': full_pcorr_list,
    'Pearsons_Correlation_pvalue': full_pcorr_pvalue_list,
    'Spearsman_Correlation_pvalue': full_pcorr_pvalue_list,
})

print(df_ps)
df_ps.to_csv(path + "Plots_EPAWaters_ACW/PS_Feature_Filtering/EPAACW_Pearson_Spearsman_Coeff_Values.csv") #[1370 rows x 6 columns]

for value in full_pcorr_list:
    if math.isnan(value):
        print('NaN value found!')

#************************************************;
#  3. Best epaacw correlation coefficients list  ;
#************************************************;
epaacw_index_list = [71, 112, 31, 13, 2, 12, 54, 94, 78, 77, 37, 110, 111, 122, 115, 123, 75, 76, 15, 92, 113, 11, 16, 17]
epaacw_ftrs_list  = ['PctNonCarbResidWs',
						'PctUrbMd2011Ws',
						'PctFrstLoss2003Ws',
						'DamNrmStorM3Ws',
						'DRNAREA',
						'DamNIDStorM3Ws',
						'PctImp2006Ws',
						'PctUrbHi2006Ws',
						'NABD_NrmStorM3Ws',
						'NABD_NIDStorM3Ws',
						'PctFrstLoss2009Ws',
						'PctUrbHi2011Ws',
						'PctUrbLo2011Ws',
						'Tmax8110Ws',
						'PctNonAgIntrodManagVegWs',
						'Tmean8110Ws',
						'MineDensWs',
						'NABD_DensWs',
						'NPDESDensWs',
						'PctOw2006Ws',
						'PctUrbOp2011Ws',
						'DamDensWs',
						'SuperfundDensWs',
						'TRIDensWs']

#**************************************************************;
#  4. SR and SD vs. extrinsic factors for 9 compounds and sum  ;
#**************************************************************;
for j in range(0,len(epaacw_ftrs_list)): #Extrinsic features 0 to 12
	for i in range(0,len(comp_list)): #Compounds i = 0 to 10
		h_index = epaacw_index_list[j]
		#
		color  = color_list[i]
		marker = marker_list[i]
		#
		print('Comp name, Extrinsic feature = ', comp_list[i], epaacw_ftrs_list[j])
		print(np.argwhere(np.isnan(acw_arr[:,h_index]))[:,0])
		ind      = np.argwhere(~np.isnan(acw_arr[:,h_index]))[:,0]
		#
		pcorr_sr, _ = pearsonr(acw_arr[ind,h_index], sr_arr[ind,i])
		scorr_sr, _ = spearmanr(acw_arr[ind,h_index], sr_arr[ind,i])
		print('i, j, Pearsons_sr, Spearmans_sr = ', i, j, '{0:.3g}'.format(pcorr_sr), '{0:.3g}'.format(scorr_sr))
		#
		pcorr_sd, _ = pearsonr(acw_arr[ind,h_index], sd_arr[ind,i])
		scorr_sd, _ = spearmanr(acw_arr[ind,h_index], sd_arr[ind,i])
		print('i, j, Pearsons_sd, Spearmans_sd = ', i, j, '{0:.3g}'.format(pcorr_sd), '{0:.3g}'.format(scorr_sd))
		#
		legend_properties = {'weight':'bold'}
		fig = plt.figure()
		ax  = fig.add_subplot(111)
		ax.set_xlabel(epaacw_ftrs_list[j], fontsize = 12, fontweight = 'bold')
		ax.set_ylabel('Species richness of ' + comp_list[i], fontsize = 12, fontweight = 'bold')
		plt.title('Pearsons correlation: ' + str('{0:.3g}'.format(pcorr_sr)) +  \
					', Spearmans correlation: ' + str('{0:.3g}'.format(scorr_sr)))
		ax.scatter(acw_arr[:,h_index], sr_arr[:,i], s = 20, c = color, marker = marker, edgecolor = 'face')
		plt.savefig(path + 'Plots_EPAWaters_ACW/Species_Richness/SR_' + \
					str(i) + '_' + str(j) + '_' + comp_list[i] + '_' + epaacw_ftrs_list[j] + '.png')
		plt.close(fig)
		#
		legend_properties = {'weight':'bold'}
		fig = plt.figure()
		ax  = fig.add_subplot(111)
		ax.set_xlabel(epaacw_ftrs_list[j], fontsize = 12, fontweight = 'bold')
		ax.set_ylabel('Shann. diversity of ' + comp_list[i], fontsize = 12, fontweight = 'bold')
		plt.title('Pearsons correlation: ' + str('{0:.3g}'.format(pcorr_sd)) +  \
					', Spearmans correlation: ' + str('{0:.3g}'.format(scorr_sd)))
		ax.scatter(acw_arr[:,h_index], sd_arr[:,i], s = 20, c = color, marker = marker, edgecolor = 'face')
		plt.savefig(path + 'Plots_EPAWaters_ACW/Shannon_Diversity/SD_' + \
					str(i) + '_' + str(j) + '_' + comp_list[i] + '_' + epaacw_ftrs_list[j] + '.png')
		plt.close(fig)
		print('\n')

#**************************************************************;
#  4. SR and SD vs. extrinsic factors for 9 compounds and sum  ;
#**************************************************************;
for j in range(0,len(epaacw_ftrs_list)): #Extrinsic features 0 to 12
	for i in range(0,len(comp_list)): #Compounds i = 0 to 10
		h_index = epaacw_index_list[j]
		#
		color  = color_list[i]
		marker = marker_list[i]
		#
		print('Comp name, Extrinsic feature = ', comp_list[i], epaacw_ftrs_list[j])
		print(np.argwhere(np.isnan(acw_arr[:,h_index]))[:,0])
		ind      = np.argwhere(~np.isnan(acw_arr[:,h_index]))[:,0]
		#
		pcorr_sr, _ = pearsonr(np.log10(acw_arr[ind,h_index] + 1e-8), sr_arr[ind,i])
		scorr_sr, _ = spearmanr(np.log10(acw_arr[ind,h_index] + 1e-8), sr_arr[ind,i])
		m_sr, c_sr, \
		r_value_sr, \
		p_value_sr,\
		std_err_sr  = linregress(np.log10(acw_arr[ind,h_index] + 1e-8), sr_arr[ind,i])
		print('i, j, Pearsons_sr, Spearmans_sr, m_sr = ', i, j, \
				'{0:.3g}'.format(pcorr_sr), '{0:.3g}'.format(scorr_sr), '{0:.3g}'.format(m_sr))
		#
		pcorr_sd, _ = pearsonr(np.log10(acw_arr[ind,h_index] + 1e-8), sd_arr[ind,i])
		scorr_sd, _ = spearmanr(np.log10(acw_arr[ind,h_index] + 1e-8), sd_arr[ind,i])
		m_sd, c_sd, \
		r_value_sd, \
		p_value_sd,\
		std_err_sd  = linregress(np.log10(acw_arr[ind,h_index] + 1e-8), sd_arr[ind,i])
		print('i, j, Pearsons_sd, Spearmans_sd, m_sd = ', i, j, \
				'{0:.3g}'.format(pcorr_sd), '{0:.3g}'.format(scorr_sd), '{0:.3g}'.format(m_sd))
		#
		legend_properties = {'weight':'bold'}
		fig = plt.figure()
		ax  = fig.add_subplot(111)
		ax.set_xlabel(epaacw_ftrs_list[j] + ' (in log10)', fontsize = 12, fontweight = 'bold')
		ax.set_ylabel('Species richness of ' + comp_list[i], fontsize = 12, fontweight = 'bold')
		plt.title('Pearsons correlation: ' + str('{0:.3g}'.format(pcorr_sr)) +  \
					', Spearmans correlation: ' + str('{0:.3g}'.format(scorr_sr)))
		ax.scatter(np.log10(acw_arr[ind,h_index] + 1e-8), sr_arr[:,i], s = 20, c = color, marker = marker, edgecolor = 'face')
		yfit_sr = np.asarray([c_sr + m_sr*xi for xi in np.log10(acw_arr[ind,h_index] + 1e-8)])
		ax.plot(np.log10(acw_arr[ind,h_index] + 1e-8), yfit_sr, color = color, label = 'Slope = ' + str('{0:.3g}'.format(m_sr)))
		ax.legend(loc = 'best')
		plt.savefig(path + 'Plots_EPAWaters_ACW/Species_Richness_log/SR_' + \
					str(i) + '_' + str(j) + '_' + comp_list[i] + '_' + epaacw_ftrs_list[j] + '.png')
		plt.close(fig)
		#
		legend_properties = {'weight':'bold'}
		fig = plt.figure()
		ax  = fig.add_subplot(111)
		ax.set_xlabel(epaacw_ftrs_list[j] + ' (in log10)', fontsize = 12, fontweight = 'bold')
		ax.set_ylabel('Shann. diversity of ' + comp_list[i], fontsize = 12, fontweight = 'bold')
		plt.title('Pearsons correlation: ' + str('{0:.3g}'.format(pcorr_sd)) +  \
					', Spearmans correlation: ' + str('{0:.3g}'.format(scorr_sd)))
		ax.scatter(np.log10(acw_arr[ind,h_index] + 1e-8), sd_arr[:,i], s = 20, c = color, marker = marker, edgecolor = 'face')
		yfit_sd = np.asarray([c_sd + m_sd*xi for xi in np.log10(acw_arr[ind,h_index] + 1e-8)])
		ax.plot(np.log10(acw_arr[ind,h_index] + 1e-8), yfit_sd, color = color, label = 'Slope = ' + str('{0:.3g}'.format(m_sd)))
		ax.legend(loc = 'best')
		plt.savefig(path + 'Plots_EPAWaters_ACW/Shannon_Diversity_log/SD_' + \
					str(i) + '_' + str(j) + '_' + comp_list[i] + '_' + epaacw_ftrs_list[j] + '.png')
		plt.close(fig)
		print('\n')

#*********************************************************************************************;
#  5a. Correlation values for SR and SD vs. log10(extrinsic factors) for 9 compounds and sum  ;
#*********************************************************************************************;
pcorr_sr_list = np.zeros((len(comp_list), len(epaacw_ftrs_list)), dtype = float) #(10,12)
scorr_sr_list = np.zeros((len(comp_list), len(epaacw_ftrs_list)), dtype = float) #(10,12)
#
pcorr_sd_list = np.zeros((len(comp_list), len(epaacw_ftrs_list)), dtype = float) #(10,12)
scorr_sd_list = np.zeros((len(comp_list), len(epaacw_ftrs_list)), dtype = float) #(10,12)
#
for j in range(0,len(epaacw_ftrs_list)): #Extrinsic features 0 to 12
	for i in range(0,len(comp_list)): #Compounds i = 0 to 10
		h_index = epaacw_index_list[j]
		#
		#print('Comp name, Extrinsic feature = ', comp_list[i], epaacw_ftrs_list[j])
		#print(np.argwhere(np.isnan(acw_arr[:,h_index]))[:,0])
		ind      = np.argwhere(~np.isnan(acw_arr[:,h_index]))[:,0]
		#
		pcorr_sr, _ = pearsonr(np.log10(acw_arr[ind,h_index] + 1e-8), sr_arr[ind,i])
		scorr_sr, _ = spearmanr(np.log10(acw_arr[ind,h_index] + 1e-8), sr_arr[ind,i])
		m_sr, c_sr, \
		r_value_sr, \
		p_value_sr,\
		std_err_sr  = linregress(np.log10(acw_arr[ind,h_index] + 1e-8), sr_arr[ind,i])
		print('i, j, Pearsons_sr, Spearmans_sr, m_sr = ', i, j, \
				'{0:.3g}'.format(pcorr_sr), '{0:.3g}'.format(scorr_sr), '{0:.3g}'.format(m_sr))
		#
		pcorr_sd, _ = pearsonr(np.log10(acw_arr[ind,h_index] + 1e-8), sd_arr[ind,i])
		scorr_sd, _ = spearmanr(np.log10(acw_arr[ind,h_index] + 1e-8), sd_arr[ind,i])
		m_sd, c_sd, \
		r_value_sd, \
		p_value_sd,\
		std_err_sd  = linregress(np.log10(acw_arr[ind,h_index] + 1e-8), sd_arr[ind,i])
		print('i, j, Pearsons_sd, Spearmans_sd, m_sd = ', i, j, \
				'{0:.3g}'.format(pcorr_sd), '{0:.3g}'.format(scorr_sd), '{0:.3g}'.format(m_sd))
		#
		#
		pcorr_sr_list[i,j] = pcorr_sr
		scorr_sr_list[i,j] = scorr_sr
		#
		pcorr_sd_list[i,j] = pcorr_sd
		scorr_sd_list[i,j] = scorr_sd

#*******************************************************************************************;
#  5b. Correlation plots of SR and SD vs. log10(extrinsic factors) for 9 compounds and sum  ;
#*******************************************************************************************;
#https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
fig = plt.figure(figsize=(15,15))
ax  = fig.add_subplot(111)
im = ax.imshow(pcorr_sr_list, cmap = 'coolwarm')
ax.set_xticks(np.arange(len(epaacw_ftrs_list)), labels=epaacw_ftrs_list)
ax.set_yticks(np.arange(len(comp_list)), labels=comp_list)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
#
for i in range(len(comp_list)): # Loop over data dimensions and create text annotations.
	for j in range(len(epaacw_ftrs_list)):
		#print(j, i, pcorr_sr_list[i,j])
		text = ax.text(j, i, '{0:.2g}'.format(pcorr_sr_list[i,j]), \
						ha="center", va="center", color="k")
ax.set_title("Pearsons correlation: Species richness vs. log(EPAWaters_ACW) data")
fig.tight_layout()
plt.savefig(path + 'Plots_EPAWaters_ACW/pcorr_SR_vs_logEPAWaters_ACW.png')
plt.close(fig)
#
fig = plt.figure(figsize=(15,15))
ax  = fig.add_subplot(111)
im = ax.imshow(scorr_sr_list, cmap = 'coolwarm')
ax.set_xticks(np.arange(len(epaacw_ftrs_list)), labels=epaacw_ftrs_list)
ax.set_yticks(np.arange(len(comp_list)), labels=comp_list)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
#
for i in range(len(comp_list)): # Loop over data dimensions and create text annotations.
	for j in range(len(epaacw_ftrs_list)):
		#print(j, i, scorr_sr_list[i,j])
		text = ax.text(j, i, '{0:.2g}'.format(scorr_sr_list[i,j]), \
						ha="center", va="center", color="k")
ax.set_title("Spearmans correlation: Species richness vs. log(EPAWaters_ACW) data")
fig.tight_layout()
plt.savefig(path + 'Plots_EPAWaters_ACW/scorr_SR_vs_logEPAWaters_ACW.png')
plt.close(fig)
#
fig = plt.figure(figsize=(15,15))
ax  = fig.add_subplot(111)
im = ax.imshow(pcorr_sd_list, cmap = 'coolwarm')
ax.set_xticks(np.arange(len(epaacw_ftrs_list)), labels=epaacw_ftrs_list)
ax.set_yticks(np.arange(len(comp_list)), labels=comp_list)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
#
for i in range(len(comp_list)): # Loop over data dimensions and create text annotations.
	for j in range(len(epaacw_ftrs_list)):
		#print(j, i, pcorr_sd_list[i,j])
		text = ax.text(j, i, '{0:.2g}'.format(pcorr_sd_list[i,j]), \
						ha="center", va="center", color="k")
ax.set_title("Pearsons correlation: Shannon diversity vs. log(EPAWaters_ACW) data")
fig.tight_layout()
plt.savefig(path + 'Plots_EPAWaters_ACW/pcorr_SD_vs_logEPAWaters_ACW.png')
plt.close(fig)
#
fig = plt.figure(figsize=(15,15))
ax  = fig.add_subplot(111)
im = ax.imshow(scorr_sd_list, cmap = 'coolwarm')
ax.set_xticks(np.arange(len(epaacw_ftrs_list)), labels=epaacw_ftrs_list)
ax.set_yticks(np.arange(len(comp_list)), labels=comp_list)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
#
for i in range(len(comp_list)): # Loop over data dimensions and create text annotations.
	for j in range(len(epaacw_ftrs_list)):
		#print(j, i, scorr_sd_list[i,j])
		text = ax.text(j, i, '{0:.2g}'.format(scorr_sd_list[i,j]), \
						ha="center", va="center", color="k")
ax.set_title("Spearmans correlation: Shannon diversity vs. log(EPAWaters_ACW) data")
fig.tight_layout()
plt.savefig(path + 'Plots_EPAWaters_ACW/scorr_SD_vs_logEPAWaters_ACW.png')
plt.close(fig)
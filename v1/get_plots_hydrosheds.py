# Get species richness vs. HYDROSHEDS data
# Get shannon diversity vs. HYDROSHEDS data
#
#10 'Natural discharge - dis_m3_pmn'
#17 'Inundation extent - inu_pc_slt'
#24 'River area - ria_ha_ssu'
#26 'River volume - riv_tc_ssu'
#37 'Climate strata - cls_cl_smj'
#46 'Air temp - tmp_dc_s05'
#73 'Potential ET - pet_mm_s04'
#87 'Actual ET - aet_mm_s04'
#
# Process species richness and shannon diversity for scaling laws, feature importance, and PCA analysis
# AUTHOR -- Maruti Kumar Mudunuru

import os
import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import linregress
from scipy import stats
#
np.set_printoptions(precision=2)

#******************************;
#  1. Set pathsfor .csv files  ;
#******************************;
path            = os.getcwd() + '/'
#
df_sr           = pd.read_csv(path + "Inputs_Outputs_v4/1_WAT_SR_54s_11f.csv", index_col = 1).iloc[:,1:] #(54, 10)
df_sd           = pd.read_csv(path + "Inputs_Outputs_v4/2_WAT_SHANN_54s_11f.csv", index_col = 1).iloc[:,1:] #(54, 10)
#
df_huc12        = pd.read_csv(path + "Inputs_Outputs_v4/8_Hydrosheds_huc12_54s_294f.csv", index_col = 1).iloc[:,1:-1] #(54, 292)
#
comp_list       = df_sr.columns.to_list() #10
huc12_ftrs_list = df_huc12.columns.to_list() #292
#
sr_arr          = df_sr.values #(54, 10)
sd_arr          = df_sd.values #(54, 10)
huc12_arr       = df_huc12.values #(54, 12)
#
marker_list     = ['o', 'v', '8', 's', 'p', '*', 'h', '+', 'x', '^'] #10
color_list      = ['b', 'k', 'r', 'c', 'm', 'g', 'y', 'tab:purple', 'tab:brown', 'tab:orange'] #10

#**************************************************************;
#  2a. Hydrosheds correlation coefficients (all 292 features)  ;
#**************************************************************;
pcorr_list = []
scorr_list = []
count_list = []
#
for i in range(0,huc12_arr.shape[1]):
	pcorr, _ = pearsonr(huc12_arr[:,i], sr_arr[:,-1])
	scorr, _ = spearmanr(huc12_arr[:,i], sr_arr[:,-1])
	print('Pearsons correlation (SR vs. i): %.3f' % pcorr)
	print('Spearmans correlation (SR vs. i): %.3f' % scorr)
	pcorr_list.append(pcorr)
	scorr_list.append(scorr)
	count_list.append(i)

pcorr_list = np.asarray(pcorr_list, dtype = float)
scorr_list = np.asarray(scorr_list, dtype = float)
count_list = np.asarray(count_list, dtype = int)
#
ascend_argsort_pcorr = np.argsort(pcorr_list)[218:248] ##pcorr_list[np.argsort(pcorr_list)]
ascend_argsort_scorr = np.argsort(pcorr_list)[218:248]
ascend_argsort_count = np.argsort(pcorr_list)[218:248]
#
print(pcorr_list[ascend_argsort_pcorr])
print(scorr_list[ascend_argsort_scorr])
print(count_list[ascend_argsort_count])
#
correlated_ftrs_list  = [huc12_ftrs_list[i] for i in count_list[ascend_argsort_count]] #huc12_ftrs_list[count_list[ascend_argsort_count]] 
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
for j in range(0,len(huc12_ftrs_list)): #Extrinsic features 0 to 292
	for i in range(0,len(comp_list)): #Compounds i = 0 to 10
		print('Comp name, Extrinsic feature = ', comp_list[i], huc12_ftrs_list[j])
		#print(np.argwhere(np.isnan(huc12_arr[:,j]))[:,0])
		ind      = np.argwhere(~np.isnan(huc12_arr[:,j]))[:,0]
		#
		pcorr_sr, pcorr_sr_pvalue = pearsonr(huc12_arr[ind,j], sr_arr[ind,i])
		scorr_sr, scorr_sr_pvalue = spearmanr(huc12_arr[ind,j], sr_arr[ind,i])
		print('i, j, Pearsons_sr, Spearmans_sr, pcorr-p-value, scorr-p-value = ', i, j, \
			'{0:.3g}'.format(pcorr_sr), '{0:.3g}'.format(scorr_sr), \
			'{0:.3g}'.format(pcorr_sr_pvalue), '{0:.3g}'.format(scorr_sr_pvalue))
		#
		i_name_list.append(comp_list[i])
		j_name_list.append(huc12_ftrs_list[j])
		full_pcorr_list.append(pcorr_sr)
		full_scorr_list.append(scorr_sr)
		full_pcorr_pvalue_list.append(pcorr_sr_pvalue)
		full_scorr_pvalue_list.append(scorr_sr_pvalue)

df_ps = pd.DataFrame({
    'Compound_Name': i_name_list,
    'HYDROSHEDS_Features': j_name_list,
    'Pearsons_Correlation': full_pcorr_list,
    'Spearsman_Correlation': full_pcorr_list,
    'Pearsons_Correlation_pvalue': full_pcorr_pvalue_list,
    'Spearsman_Correlation_pvalue': full_pcorr_pvalue_list,
})

print(df_ps)
df_ps.to_csv(path + "Plots_HYDROSHEDS/PS_Feature_Filtering/HYDROSHEDS_Pearson_Spearsman_Coeff_Values.csv") #[2920 rows x 6 columns]

#****************************************************;
#  3. Best hydrosheds correlation coefficients list  ;
#****************************************************;
hydrosheds_index_list = [10, 17, 24, 26, 37, 46, 73, 87]
hydrosheds_ftrs_list  = ['Natural discharge - dis_m3_pmn', 'Inundation extent - inu_pc_slt', \
							'River area - ria_ha_ssu', 'River volume - riv_tc_ssu', \
							'Climate strata - cls_cl_smj', 'Air temp - tmp_dc_s05', \
							'Potential ET - pet_mm_s04', 'Actual ET - aet_mm_s04']

#**************************************************************;
#  4. SR and SD vs. extrinsic factors for 9 compounds and sum  ;
#**************************************************************;
for j in range(0,len(hydrosheds_ftrs_list)): #Extrinsic features 0 to 12
	for i in range(0,len(comp_list)): #Compounds i = 0 to 10
		h_index = hydrosheds_index_list[j]
		#
		color  = color_list[i]
		marker = marker_list[i]
		#
		print('Comp name, Extrinsic feature = ', comp_list[i], hydrosheds_ftrs_list[j])
		print(np.argwhere(np.isnan(huc12_arr[:,h_index]))[:,0])
		ind      = np.argwhere(~np.isnan(huc12_arr[:,h_index]))[:,0]
		#
		pcorr_sr, _ = pearsonr(huc12_arr[ind,h_index], sr_arr[ind,i])
		scorr_sr, _ = spearmanr(huc12_arr[ind,h_index], sr_arr[ind,i])
		print('i, j, Pearsons_sr, Spearmans_sr = ', i, j, '{0:.3g}'.format(pcorr_sr), '{0:.3g}'.format(scorr_sr))
		#
		pcorr_sd, _ = pearsonr(huc12_arr[ind,h_index], sd_arr[ind,i])
		scorr_sd, _ = spearmanr(huc12_arr[ind,h_index], sd_arr[ind,i])
		print('i, j, Pearsons_sd, Spearmans_sd = ', i, j, '{0:.3g}'.format(pcorr_sd), '{0:.3g}'.format(scorr_sd))
		#
		legend_properties = {'weight':'bold'}
		fig = plt.figure()
		ax  = fig.add_subplot(111)
		ax.set_xlabel(hydrosheds_ftrs_list[j], fontsize = 12, fontweight = 'bold')
		ax.set_ylabel('Species richness of ' + comp_list[i], fontsize = 12, fontweight = 'bold')
		plt.title('Pearsons correlation: ' + str('{0:.3g}'.format(pcorr_sr)) +  \
					', Spearmans correlation: ' + str('{0:.3g}'.format(scorr_sr)))
		ax.scatter(huc12_arr[:,h_index], sr_arr[:,i], s = 20, c = color, marker = marker, edgecolor = 'face')
		plt.savefig(path + 'Plots_HYDROSHEDS/Species_Richness/SR_' + \
					str(i) + '_' + str(j) + '_' + comp_list[i] + '_' + hydrosheds_ftrs_list[j] + '.png')
		plt.close(fig)
		#
		legend_properties = {'weight':'bold'}
		fig = plt.figure()
		ax  = fig.add_subplot(111)
		ax.set_xlabel(hydrosheds_ftrs_list[j], fontsize = 12, fontweight = 'bold')
		ax.set_ylabel('Shann. diversity of ' + comp_list[i], fontsize = 12, fontweight = 'bold')
		plt.title('Pearsons correlation: ' + str('{0:.3g}'.format(pcorr_sd)) +  \
					', Spearmans correlation: ' + str('{0:.3g}'.format(scorr_sd)))
		ax.scatter(huc12_arr[:,h_index], sd_arr[:,i], s = 20, c = color, marker = marker, edgecolor = 'face')
		plt.savefig(path + 'Plots_HYDROSHEDS/Shannon_Diversity/SD_' + \
					str(i) + '_' + str(j) + '_' + comp_list[i] + '_' + hydrosheds_ftrs_list[j] + '.png')
		plt.close(fig)
		print('\n')

#**************************************************************;
#  4. SR and SD vs. extrinsic factors for 9 compounds and sum  ;
#**************************************************************;
for j in range(0,len(hydrosheds_ftrs_list)): #Extrinsic features 0 to 12
	for i in range(0,len(comp_list)): #Compounds i = 0 to 10
		h_index = hydrosheds_index_list[j]
		#
		color  = color_list[i]
		marker = marker_list[i]
		#
		print('Comp name, Extrinsic feature = ', comp_list[i], hydrosheds_ftrs_list[j])
		print(np.argwhere(np.isnan(huc12_arr[:,h_index]))[:,0])
		ind      = np.argwhere(~np.isnan(huc12_arr[:,h_index]))[:,0]
		#
		pcorr_sr, _ = pearsonr(np.log10(huc12_arr[ind,h_index] + 1e-8), sr_arr[ind,i])
		scorr_sr, _ = spearmanr(np.log10(huc12_arr[ind,h_index] + 1e-8), sr_arr[ind,i])
		m_sr, c_sr, \
		r_value_sr, \
		p_value_sr,\
		std_err_sr  = linregress(np.log10(huc12_arr[ind,h_index] + 1e-8), sr_arr[ind,i])
		print('i, j, Pearsons_sr, Spearmans_sr, m_sr = ', i, j, \
				'{0:.3g}'.format(pcorr_sr), '{0:.3g}'.format(scorr_sr), '{0:.3g}'.format(m_sr))
		#
		pcorr_sd, _ = pearsonr(np.log10(huc12_arr[ind,h_index] + 1e-8), sd_arr[ind,i])
		scorr_sd, _ = spearmanr(np.log10(huc12_arr[ind,h_index] + 1e-8), sd_arr[ind,i])
		m_sd, c_sd, \
		r_value_sd, \
		p_value_sd,\
		std_err_sd  = linregress(np.log10(huc12_arr[ind,h_index] + 1e-8), sd_arr[ind,i])
		print('i, j, Pearsons_sd, Spearmans_sd, m_sd = ', i, j, \
				'{0:.3g}'.format(pcorr_sd), '{0:.3g}'.format(scorr_sd), '{0:.3g}'.format(m_sd))
		#
		legend_properties = {'weight':'bold'}
		fig = plt.figure()
		ax  = fig.add_subplot(111)
		ax.set_xlabel(hydrosheds_ftrs_list[j] + ' (in log10)', fontsize = 12, fontweight = 'bold')
		ax.set_ylabel('Species richness of ' + comp_list[i], fontsize = 12, fontweight = 'bold')
		plt.title('Pearsons correlation: ' + str('{0:.3g}'.format(pcorr_sr)) +  \
					', Spearmans correlation: ' + str('{0:.3g}'.format(scorr_sr)))
		ax.scatter(np.log10(huc12_arr[ind,h_index] + 1e-8), sr_arr[:,i], s = 20, c = color, marker = marker, edgecolor = 'face')
		yfit_sr = np.asarray([c_sr + m_sr*xi for xi in np.log10(huc12_arr[ind,h_index] + 1e-8)])
		ax.plot(np.log10(huc12_arr[ind,h_index] + 1e-8), yfit_sr, color = color, label = 'Slope = ' + str('{0:.3g}'.format(m_sr)))
		ax.legend(loc = 'best')
		plt.savefig(path + 'Plots_HYDROSHEDS/Species_Richness_log/SR_' + \
					str(i) + '_' + str(j) + '_' + comp_list[i] + '_' + hydrosheds_ftrs_list[j] + '.png')
		plt.close(fig)
		#
		legend_properties = {'weight':'bold'}
		fig = plt.figure()
		ax  = fig.add_subplot(111)
		ax.set_xlabel(hydrosheds_ftrs_list[j] + ' (in log10)', fontsize = 12, fontweight = 'bold')
		ax.set_ylabel('Shann. diversity of ' + comp_list[i], fontsize = 12, fontweight = 'bold')
		plt.title('Pearsons correlation: ' + str('{0:.3g}'.format(pcorr_sd)) +  \
					', Spearmans correlation: ' + str('{0:.3g}'.format(scorr_sd)))
		ax.scatter(np.log10(huc12_arr[ind,h_index] + 1e-8), sd_arr[:,i], s = 20, c = color, marker = marker, edgecolor = 'face')
		yfit_sd = np.asarray([c_sd + m_sd*xi for xi in np.log10(huc12_arr[ind,h_index] + 1e-8)])
		ax.plot(np.log10(huc12_arr[ind,h_index] + 1e-8), yfit_sd, color = color, label = 'Slope = ' + str('{0:.3g}'.format(m_sd)))
		ax.legend(loc = 'best')
		plt.savefig(path + 'Plots_HYDROSHEDS/Shannon_Diversity_log/SD_' + \
					str(i) + '_' + str(j) + '_' + comp_list[i] + '_' + hydrosheds_ftrs_list[j] + '.png')
		plt.close(fig)
		print('\n')

#*********************************************************************************************;
#  5a. Correlation values for SR and SD vs. log10(extrinsic factors) for 9 compounds and sum  ;
#*********************************************************************************************;
pcorr_sr_list = np.zeros((len(comp_list), len(hydrosheds_ftrs_list)), dtype = float) #(10,8)
scorr_sr_list = np.zeros((len(comp_list), len(hydrosheds_ftrs_list)), dtype = float) #(10,8)
#
pcorr_sd_list = np.zeros((len(comp_list), len(hydrosheds_ftrs_list)), dtype = float) #(10,8)
scorr_sd_list = np.zeros((len(comp_list), len(hydrosheds_ftrs_list)), dtype = float) #(10,8)
#
for j in range(0,len(hydrosheds_ftrs_list)): #Extrinsic features 0 to 8
	for i in range(0,len(comp_list)): #Compounds i = 0 to 10
		h_index = hydrosheds_index_list[j]
		#
		#print('Comp name, Extrinsic feature = ', comp_list[i], hydrosheds_ftrs_list[j])
		#print(np.argwhere(np.isnan(huc12_arr[:,h_index]))[:,0])
		ind      = np.argwhere(~np.isnan(huc12_arr[:,h_index]))[:,0]
		#
		pcorr_sr, _ = pearsonr(np.log10(huc12_arr[ind,h_index] + 1e-8), sr_arr[ind,i])
		scorr_sr, _ = spearmanr(np.log10(huc12_arr[ind,h_index] + 1e-8), sr_arr[ind,i])
		m_sr, c_sr, \
		r_value_sr, \
		p_value_sr,\
		std_err_sr  = linregress(np.log10(huc12_arr[ind,h_index] + 1e-8), sr_arr[ind,i])
		print('i, j, Pearsons_sr, Spearmans_sr, m_sr = ', i, j, \
				'{0:.3g}'.format(pcorr_sr), '{0:.3g}'.format(scorr_sr), '{0:.3g}'.format(m_sr))
		#
		pcorr_sd, _ = pearsonr(np.log10(huc12_arr[ind,h_index] + 1e-8), sd_arr[ind,i])
		scorr_sd, _ = spearmanr(np.log10(huc12_arr[ind,h_index] + 1e-8), sd_arr[ind,i])
		m_sd, c_sd, \
		r_value_sd, \
		p_value_sd,\
		std_err_sd  = linregress(np.log10(huc12_arr[ind,h_index] + 1e-8), sd_arr[ind,i])
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
fig = plt.figure(figsize=(10,10))
ax  = fig.add_subplot(111)
im = ax.imshow(pcorr_sr_list, cmap = 'coolwarm')
ax.set_xticks(np.arange(len(hydrosheds_ftrs_list)), labels=hydrosheds_ftrs_list)
ax.set_yticks(np.arange(len(comp_list)), labels=comp_list)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
#
for i in range(len(comp_list)): # Loop over data dimensions and create text annotations.
	for j in range(len(hydrosheds_ftrs_list)):
		#print(j, i, pcorr_sr_list[i,j])
		text = ax.text(j, i, '{0:.2g}'.format(pcorr_sr_list[i,j]), \
						ha="center", va="center", color="k")
ax.set_title("Pearsons correlation: Species richness vs. log(HYDROSHEDS) data")
fig.tight_layout()
plt.savefig(path + 'Plots_HYDROSHEDS/pcorr_SR_vs_logHYDROSHEDS.png')
plt.close(fig)
#
fig = plt.figure(figsize=(10,10))
ax  = fig.add_subplot(111)
im = ax.imshow(scorr_sr_list, cmap = 'coolwarm')
ax.set_xticks(np.arange(len(hydrosheds_ftrs_list)), labels=hydrosheds_ftrs_list)
ax.set_yticks(np.arange(len(comp_list)), labels=comp_list)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
#
for i in range(len(comp_list)): # Loop over data dimensions and create text annotations.
	for j in range(len(hydrosheds_ftrs_list)):
		#print(j, i, scorr_sr_list[i,j])
		text = ax.text(j, i, '{0:.2g}'.format(scorr_sr_list[i,j]), \
						ha="center", va="center", color="k")
ax.set_title("Spearmans correlation: Species richness vs. log(HYDROSHEDS) data")
fig.tight_layout()
plt.savefig(path + 'Plots_HYDROSHEDS/scorr_SR_vs_logHYDROSHEDS.png')
plt.close(fig)
#
fig = plt.figure(figsize=(10,10))
ax  = fig.add_subplot(111)
im = ax.imshow(pcorr_sd_list, cmap = 'coolwarm')
ax.set_xticks(np.arange(len(hydrosheds_ftrs_list)), labels=hydrosheds_ftrs_list)
ax.set_yticks(np.arange(len(comp_list)), labels=comp_list)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
#
for i in range(len(comp_list)): # Loop over data dimensions and create text annotations.
	for j in range(len(hydrosheds_ftrs_list)):
		#print(j, i, pcorr_sd_list[i,j])
		text = ax.text(j, i, '{0:.2g}'.format(pcorr_sd_list[i,j]), \
						ha="center", va="center", color="k")
ax.set_title("Pearsons correlation: Shannon diversity vs. log(HYDROSHEDS) data")
fig.tight_layout()
plt.savefig(path + 'Plots_HYDROSHEDS/pcorr_SD_vs_logHYDROSHEDS.png')
plt.close(fig)
#
fig = plt.figure(figsize=(10,10))
ax  = fig.add_subplot(111)
im = ax.imshow(scorr_sd_list, cmap = 'coolwarm')
ax.set_xticks(np.arange(len(hydrosheds_ftrs_list)), labels=hydrosheds_ftrs_list)
ax.set_yticks(np.arange(len(comp_list)), labels=comp_list)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
#
for i in range(len(comp_list)): # Loop over data dimensions and create text annotations.
	for j in range(len(hydrosheds_ftrs_list)):
		#print(j, i, scorr_sd_list[i,j])
		text = ax.text(j, i, '{0:.2g}'.format(scorr_sd_list[i,j]), \
						ha="center", va="center", color="k")
ax.set_title("Spearmans correlation: Shannon diversity vs. log(HYDROSHEDS) data")
fig.tight_layout()
plt.savefig(path + 'Plots_HYDROSHEDS/scorr_SD_vs_logHYDROSHEDS.png')
plt.close(fig)
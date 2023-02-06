# Get species richness vs. EPAWaters acc data
# Get shannon diversity vs. EPAWaters acc data
#
# Important features identified include (dataIndex --> FeatureName):
#	34  --> 0.275 0.307 'Percent forest cover loss - PctFrstLoss'
#   52  --> 0.243 -0.0237 'Mean hydraulic conductivity in catchment -- HydrlCondCat'
#   54  --> 0.259 0.343 'Mean imperviousness of anthropogenic surfaces within catchment - PctImp'
#   82  --> 0.212 0.181 'Precipitation gradient - SN'
#   92  --> 0.235 0.378 'Open water land cover (in a catchment) - PctOw'
#   97  --> 0.283 0.289 'Developed catchment area (Land use) - PctUrbOp'
#   114 --> 0.249 0.27 'Woody wetland land cover -- PctWdWet'
#   115 --> 0.225 0.28 'PctNonAgIntrodManagVegCat'
#   119 --> 0.308 0.326 '30 year mean normal temperature - Tmean'
#   122 --> 0.281 0.283 '30 year max normal temperature - Tmax'
#   126 --> 0.44 0.426 'Mean annual stream temperature -- MAST'
#   131 --> 0.414 0.437 'Mean summer stream temperature - MSST'
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
import itertools
#
np.set_printoptions(precision=2)

#******************************;
#  1. Set pathsfor .csv files  ;
#******************************;
#path           = os.getcwd() + '/'
path           = '/Users/mudu605/OneDrive - PNNL/Desktop/Papers_PNNL/18_Crowd_ML/Python_Scripts/0-AlphaDiversity/'
#
df_sr          = pd.read_csv(path + "Inputs_Outputs_v4/1_WAT_SR_54s_11f.csv", index_col = 1).iloc[:,1:] #(54, 10)
df_sd          = pd.read_csv(path + "Inputs_Outputs_v4/2_WAT_SHANN_54s_11f.csv", index_col = 1).iloc[:,1:] #(54, 10)
#
df_acc_all     = pd.read_csv(path + "Inputs_Outputs_v4/5_ACC_EPAWaters_54s_141f.csv", index_col = 1).iloc[:,1:] #(54, 140)
acc_list       = np.delete(np.arange(0,140), [2,3,126])
df_acc         = df_acc_all.iloc[:,acc_list] #[54 rows x 137 columns]
#
comp_list      = df_sr.columns.to_list() #10
acc_ftrs_list  = df_acc.columns.to_list() #137
#
sr_arr         = df_sr.values #(54, 10)
sd_arr         = df_sd.values #(54, 10)
acc_arr        = df_acc.values #(54, 137)
#
marker_small   = [['o', 'v', '8', 's', 'p', '*', 'h', '+', 'x', '^'] for i in range(0,14)] #10
color_small    = [['b', 'k', 'r', 'c', 'm', 'g', 'y', 'tab:purple', 'tab:brown', 'tab:orange'] for i in range(0,14)] #10
#
marker_list    = list(itertools.chain.from_iterable(marker_small))
color_list     = list(itertools.chain.from_iterable(color_small))

#***********************************;
#  2. ACC correlation coefficients  ;
#***********************************;
for i in range(0,acc_arr.shape[1]):
	if len(np.argwhere(np.isnan(acc_arr[:,i]))[:,0]) > 0:
		print(i)

pcorr_list = []
scorr_list = []
count_list = []
#
for i in range(0,acc_arr.shape[1]):
	if len(np.argwhere(np.isnan(acc_arr[:,i]))[:,0]) == 0:
		pcorr, _ = pearsonr(acc_arr[:,i], sr_arr[:,-1])
		scorr, _ = spearmanr(acc_arr[:,i], sr_arr[:,-1])
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
ascend_argsort_pcorr = np.argsort(pcorr_list)[91:121] ##pcorr_list[np.argsort(pcorr_list)]
ascend_argsort_scorr = np.argsort(pcorr_list)[91:121]
ascend_argsort_count = np.argsort(pcorr_list)[91:121]

ascend_argsort_pcorr = np.argsort(pcorr_list)[91:121] ##pcorr_list[np.argsort(pcorr_list)]
ascend_argsort_scorr = np.argsort(pcorr_list)[91:121]
ascend_argsort_count = np.argsort(pcorr_list)[91:121]
#
print(pcorr_list[ascend_argsort_pcorr])
print(scorr_list[ascend_argsort_scorr])
print(count_list[ascend_argsort_count])
#
correlated_ftrs_list  = [acc_ftrs_list[i] for i in count_list[ascend_argsort_count]] #acc_ftrs_list[count_list[ascend_argsort_count]] 
correlated_count_list = [i for i in count_list[ascend_argsort_count]]
#
print('\n')
#
for i in range(0,len(correlated_count_list)):
	print(i, correlated_count_list[i], '{0:.3g}'.format(pcorr_list[ascend_argsort_pcorr][i]), \
		'{0:.3g}'.format(scorr_list[ascend_argsort_pcorr][i]), \
		correlated_ftrs_list[i])

#************************************************;
#  3. Best epaacc correlation coefficients list  ;
#************************************************;
epaacc_index_list = [34, 52, 54, 82, 92, 97, 114, 115, 119, 122, 126, 131]
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
						'Mean summer stream temperature - MSST']

#**************************************************************;
#  4. SR and SD vs. extrinsic factors for 9 compounds and sum  ;
#**************************************************************;
for j in range(0,len(epaacc_ftrs_list)): #Extrinsic features 0 to 12
	for i in range(0,len(comp_list)): #Compounds i = 0 to 10
		h_index = epaacc_index_list[j]
		#
		color  = color_list[i]
		marker = marker_list[i]
		#
		print('Comp name, Extrinsic feature = ', comp_list[i], epaacc_ftrs_list[j])
		print(np.argwhere(np.isnan(acc_arr[:,h_index]))[:,0])
		ind      = np.argwhere(~np.isnan(acc_arr[:,h_index]))[:,0]
		#
		pcorr_sr, _ = pearsonr(acc_arr[ind,h_index], sr_arr[ind,i])
		scorr_sr, _ = spearmanr(acc_arr[ind,h_index], sr_arr[ind,i])
		print('i, j, Pearsons_sr, Spearmans_sr = ', i, j, '{0:.3g}'.format(pcorr_sr), '{0:.3g}'.format(scorr_sr))
		#
		pcorr_sd, _ = pearsonr(acc_arr[ind,h_index], sd_arr[ind,i])
		scorr_sd, _ = spearmanr(acc_arr[ind,h_index], sd_arr[ind,i])
		print('i, j, Pearsons_sd, Spearmans_sd = ', i, j, '{0:.3g}'.format(pcorr_sd), '{0:.3g}'.format(scorr_sd))
		#
		legend_properties = {'weight':'bold'}
		fig = plt.figure()
		ax  = fig.add_subplot(111)
		ax.set_xlabel(epaacc_ftrs_list[j], fontsize = 12, fontweight = 'bold')
		ax.set_ylabel('Species richness of ' + comp_list[i], fontsize = 12, fontweight = 'bold')
		plt.title('Pearsons correlation: ' + str('{0:.3g}'.format(pcorr_sr)) +  \
					', Spearmans correlation: ' + str('{0:.3g}'.format(scorr_sr)))
		ax.scatter(acc_arr[:,h_index], sr_arr[:,i], s = 20, c = color, marker = marker, edgecolor = 'face')
		plt.savefig(path + 'Plots_EPAWaters_ACC/Species_Richness/SR_' + \
					str(i) + '_' + str(j) + '_' + comp_list[i] + '_' + epaacc_ftrs_list[j] + '.png')
		plt.close(fig)
		#
		legend_properties = {'weight':'bold'}
		fig = plt.figure()
		ax  = fig.add_subplot(111)
		ax.set_xlabel(epaacc_ftrs_list[j], fontsize = 12, fontweight = 'bold')
		ax.set_ylabel('Shann. diversity of ' + comp_list[i], fontsize = 12, fontweight = 'bold')
		plt.title('Pearsons correlation: ' + str('{0:.3g}'.format(pcorr_sd)) +  \
					', Spearmans correlation: ' + str('{0:.3g}'.format(scorr_sd)))
		ax.scatter(acc_arr[:,h_index], sd_arr[:,i], s = 20, c = color, marker = marker, edgecolor = 'face')
		plt.savefig(path + 'Plots_EPAWaters_ACC/Shannon_Diversity/SD_' + \
					str(i) + '_' + str(j) + '_' + comp_list[i] + '_' + epaacc_ftrs_list[j] + '.png')
		plt.close(fig)
		print('\n')

#**************************************************************;
#  4. SR and SD vs. extrinsic factors for 9 compounds and sum  ;
#**************************************************************;
for j in range(0,len(epaacc_ftrs_list)): #Extrinsic features 0 to 12
	for i in range(0,len(comp_list)): #Compounds i = 0 to 10
		h_index = epaacc_index_list[j]
		#
		color  = color_list[i]
		marker = marker_list[i]
		#
		print('Comp name, Extrinsic feature = ', comp_list[i], epaacc_ftrs_list[j])
		print(np.argwhere(np.isnan(acc_arr[:,h_index]))[:,0])
		ind      = np.argwhere(~np.isnan(acc_arr[:,h_index]))[:,0]
		#
		pcorr_sr, _ = pearsonr(np.log10(acc_arr[ind,h_index] + 1e-8), sr_arr[ind,i])
		scorr_sr, _ = spearmanr(np.log10(acc_arr[ind,h_index] + 1e-8), sr_arr[ind,i])
		m_sr, c_sr, \
		r_value_sr, \
		p_value_sr,\
		std_err_sr  = linregress(np.log10(acc_arr[ind,h_index] + 1e-8), sr_arr[ind,i])
		print('i, j, Pearsons_sr, Spearmans_sr, m_sr = ', i, j, \
				'{0:.3g}'.format(pcorr_sr), '{0:.3g}'.format(scorr_sr), '{0:.3g}'.format(m_sr))
		#
		pcorr_sd, _ = pearsonr(np.log10(acc_arr[ind,h_index] + 1e-8), sd_arr[ind,i])
		scorr_sd, _ = spearmanr(np.log10(acc_arr[ind,h_index] + 1e-8), sd_arr[ind,i])
		m_sd, c_sd, \
		r_value_sd, \
		p_value_sd,\
		std_err_sd  = linregress(np.log10(acc_arr[ind,h_index] + 1e-8), sd_arr[ind,i])
		print('i, j, Pearsons_sd, Spearmans_sd, m_sd = ', i, j, \
				'{0:.3g}'.format(pcorr_sd), '{0:.3g}'.format(scorr_sd), '{0:.3g}'.format(m_sd))
		#
		legend_properties = {'weight':'bold'}
		fig = plt.figure()
		ax  = fig.add_subplot(111)
		ax.set_xlabel(epaacc_ftrs_list[j] + ' (in log10)', fontsize = 12, fontweight = 'bold')
		ax.set_ylabel('Species richness of ' + comp_list[i], fontsize = 12, fontweight = 'bold')
		plt.title('Pearsons correlation: ' + str('{0:.3g}'.format(pcorr_sr)) +  \
					', Spearmans correlation: ' + str('{0:.3g}'.format(scorr_sr)))
		ax.scatter(np.log10(acc_arr[ind,h_index] + 1e-8), sr_arr[:,i], s = 20, c = color, marker = marker, edgecolor = 'face')
		yfit_sr = np.asarray([c_sr + m_sr*xi for xi in np.log10(acc_arr[ind,h_index] + 1e-8)])
		ax.plot(np.log10(acc_arr[ind,h_index] + 1e-8), yfit_sr, color = color, label = 'Slope = ' + str('{0:.3g}'.format(m_sr)))
		ax.legend(loc = 'best')
		plt.savefig(path + 'Plots_EPAWaters_ACC/Species_Richness_log/SR_' + \
					str(i) + '_' + str(j) + '_' + comp_list[i] + '_' + epaacc_ftrs_list[j] + '.png')
		plt.close(fig)
		#
		legend_properties = {'weight':'bold'}
		fig = plt.figure()
		ax  = fig.add_subplot(111)
		ax.set_xlabel(epaacc_ftrs_list[j] + ' (in log10)', fontsize = 12, fontweight = 'bold')
		ax.set_ylabel('Shann. diversity of ' + comp_list[i], fontsize = 12, fontweight = 'bold')
		plt.title('Pearsons correlation: ' + str('{0:.3g}'.format(pcorr_sd)) +  \
					', Spearmans correlation: ' + str('{0:.3g}'.format(scorr_sd)))
		ax.scatter(np.log10(acc_arr[ind,h_index] + 1e-8), sd_arr[:,i], s = 20, c = color, marker = marker, edgecolor = 'face')
		yfit_sd = np.asarray([c_sd + m_sd*xi for xi in np.log10(acc_arr[ind,h_index] + 1e-8)])
		ax.plot(np.log10(acc_arr[ind,h_index] + 1e-8), yfit_sd, color = color, label = 'Slope = ' + str('{0:.3g}'.format(m_sd)))
		ax.legend(loc = 'best')
		plt.savefig(path + 'Plots_EPAWaters_ACC/Shannon_Diversity_log/SD_' + \
					str(i) + '_' + str(j) + '_' + comp_list[i] + '_' + epaacc_ftrs_list[j] + '.png')
		plt.close(fig)
		print('\n')

#*********************************************************************************************;
#  5a. Correlation values for SR and SD vs. log10(extrinsic factors) for 9 compounds and sum  ;
#*********************************************************************************************;
pcorr_sr_list = np.zeros((len(comp_list), len(epaacc_ftrs_list)), dtype = float) #(10,12)
scorr_sr_list = np.zeros((len(comp_list), len(epaacc_ftrs_list)), dtype = float) #(10,12)
#
pcorr_sd_list = np.zeros((len(comp_list), len(epaacc_ftrs_list)), dtype = float) #(10,12)
scorr_sd_list = np.zeros((len(comp_list), len(epaacc_ftrs_list)), dtype = float) #(10,12)
#
for j in range(0,len(epaacc_ftrs_list)): #Extrinsic features 0 to 12
	for i in range(0,len(comp_list)): #Compounds i = 0 to 10
		h_index = epaacc_index_list[j]
		#
		#print('Comp name, Extrinsic feature = ', comp_list[i], epaacc_ftrs_list[j])
		#print(np.argwhere(np.isnan(acc_arr[:,h_index]))[:,0])
		ind      = np.argwhere(~np.isnan(acc_arr[:,h_index]))[:,0]
		#
		pcorr_sr, _ = pearsonr(np.log10(acc_arr[ind,h_index] + 1e-8), sr_arr[ind,i])
		scorr_sr, _ = spearmanr(np.log10(acc_arr[ind,h_index] + 1e-8), sr_arr[ind,i])
		m_sr, c_sr, \
		r_value_sr, \
		p_value_sr,\
		std_err_sr  = linregress(np.log10(acc_arr[ind,h_index] + 1e-8), sr_arr[ind,i])
		print('i, j, Pearsons_sr, Spearmans_sr, m_sr = ', i, j, \
				'{0:.3g}'.format(pcorr_sr), '{0:.3g}'.format(scorr_sr), '{0:.3g}'.format(m_sr))
		#
		pcorr_sd, _ = pearsonr(np.log10(acc_arr[ind,h_index] + 1e-8), sd_arr[ind,i])
		scorr_sd, _ = spearmanr(np.log10(acc_arr[ind,h_index] + 1e-8), sd_arr[ind,i])
		m_sd, c_sd, \
		r_value_sd, \
		p_value_sd,\
		std_err_sd  = linregress(np.log10(acc_arr[ind,h_index] + 1e-8), sd_arr[ind,i])
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
ax.set_xticks(np.arange(len(epaacc_ftrs_list)), labels=epaacc_ftrs_list)
ax.set_yticks(np.arange(len(comp_list)), labels=comp_list)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
#
for i in range(len(comp_list)): # Loop over data dimensions and create text annotations.
	for j in range(len(epaacc_ftrs_list)):
		#print(j, i, pcorr_sr_list[i,j])
		text = ax.text(j, i, '{0:.2g}'.format(pcorr_sr_list[i,j]), \
						ha="center", va="center", color="k")
ax.set_title("Pearsons correlation: Species richness vs. log(EPAWaters_ACC) data")
fig.tight_layout()
plt.savefig(path + 'Plots_EPAWaters_ACC/pcorr_SR_vs_logEPAWaters_ACC.png')
plt.close(fig)
#
fig = plt.figure(figsize=(10,10))
ax  = fig.add_subplot(111)
im = ax.imshow(scorr_sr_list, cmap = 'coolwarm')
ax.set_xticks(np.arange(len(epaacc_ftrs_list)), labels=epaacc_ftrs_list)
ax.set_yticks(np.arange(len(comp_list)), labels=comp_list)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
#
for i in range(len(comp_list)): # Loop over data dimensions and create text annotations.
	for j in range(len(epaacc_ftrs_list)):
		#print(j, i, scorr_sr_list[i,j])
		text = ax.text(j, i, '{0:.2g}'.format(scorr_sr_list[i,j]), \
						ha="center", va="center", color="k")
ax.set_title("Spearmans correlation: Species richness vs. log(EPAWaters_ACC) data")
fig.tight_layout()
plt.savefig(path + 'Plots_EPAWaters_ACC/scorr_SR_vs_logEPAWaters_ACC.png')
plt.close(fig)
#
fig = plt.figure(figsize=(10,10))
ax  = fig.add_subplot(111)
im = ax.imshow(pcorr_sd_list, cmap = 'coolwarm')
ax.set_xticks(np.arange(len(epaacc_ftrs_list)), labels=epaacc_ftrs_list)
ax.set_yticks(np.arange(len(comp_list)), labels=comp_list)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
#
for i in range(len(comp_list)): # Loop over data dimensions and create text annotations.
	for j in range(len(epaacc_ftrs_list)):
		#print(j, i, pcorr_sd_list[i,j])
		text = ax.text(j, i, '{0:.2g}'.format(pcorr_sd_list[i,j]), \
						ha="center", va="center", color="k")
ax.set_title("Pearsons correlation: Shannon diversity vs. log(EPAWaters_ACC) data")
fig.tight_layout()
plt.savefig(path + 'Plots_EPAWaters_ACC/pcorr_SD_vs_logEPAWaters_ACC.png')
plt.close(fig)
#
fig = plt.figure(figsize=(10,10))
ax  = fig.add_subplot(111)
im = ax.imshow(scorr_sd_list, cmap = 'coolwarm')
ax.set_xticks(np.arange(len(epaacc_ftrs_list)), labels=epaacc_ftrs_list)
ax.set_yticks(np.arange(len(comp_list)), labels=comp_list)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
#
for i in range(len(comp_list)): # Loop over data dimensions and create text annotations.
	for j in range(len(epaacc_ftrs_list)):
		#print(j, i, scorr_sd_list[i,j])
		text = ax.text(j, i, '{0:.2g}'.format(scorr_sd_list[i,j]), \
						ha="center", va="center", color="k")
ax.set_title("Spearmans correlation: Shannon diversity vs. log(EPAWaters_ACC) data")
fig.tight_layout()
plt.savefig(path + 'Plots_EPAWaters_ACC/scorr_SD_vs_logEPAWaters_ACC.png')
plt.close(fig)
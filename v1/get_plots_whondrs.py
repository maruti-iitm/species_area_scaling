# Get species richness vs. WHONDRS data
# Get shannon diversity vs. WHONDRS data
#
# Process species richness and shannon diversity for scaling laws, feature importance, and PCA analysis
# AUTHOR -- Maruti Kumar Mudunuru

import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import linregress
#
np.set_printoptions(precision=2)

#******************************;
#  1. Set pathsfor .csv files  ;
#******************************;
path           = '/Users/mudu605/OneDrive - PNNL/Desktop/Papers_PNNL/18_Crowd_ML/Python_Scripts/0-AlphaDiversity/'
#
df_sr          = pd.read_csv(path + "Inputs_Outputs_v4/1_WAT_SR_54s_11f.csv", index_col = 1).iloc[:,1:] #(54, 10)
df_sd          = pd.read_csv(path + "Inputs_Outputs_v4/2_WAT_SHANN_54s_11f.csv", index_col = 1).iloc[:,1:] #(54, 10)
#
df_wdrs        = pd.read_csv(path + 'Inputs_Outputs_v4/7_PCA_Eric_WHONDRS_54s_13f.csv', index_col = 1).iloc[:,1:] #[54 rows x 12 columns]
#
comp_list      = df_sr.columns.to_list() #10
wdrs_ftrs_list = df_wdrs.columns.to_list() #12
#
sr_arr         = df_sr.values #(54, 10)
sd_arr         = df_sd.values #(54, 10)
wdrs_arr       = df_wdrs.values #(54, 12)
#
marker_list    = ['o', 'v', '8', 's', 'p', '*', 'h', '+', 'x', '^'] #10
color_list     = ['b', 'k', 'r', 'c', 'm', 'g', 'y', 'tab:purple', 'tab:brown', 'tab:orange'] #10

#**************************************************************;
#  2. SR and SD vs. extrinsic factors for 9 compounds and sum  ;
#**************************************************************;
for j in range(0,len(wdrs_ftrs_list)): #Extrinsic features 0 to 12
	for i in range(0,len(comp_list)): #Compounds i = 0 to 10
		color = color_list[i]
		marker = marker_list[i]
		#
		print('Comp name, Extrinsic feature = ', comp_list[i], wdrs_ftrs_list[j])
		print(np.argwhere(np.isnan(wdrs_arr[:,j]))[:,0])
		ind      = np.argwhere(~np.isnan(wdrs_arr[:,j]))[:,0]
		#
		pcorr_sr, _ = pearsonr(wdrs_arr[ind,j], sr_arr[ind,i])
		scorr_sr, _ = spearmanr(wdrs_arr[ind,j], sr_arr[ind,i])
		print('i, j, Pearsons_sr, Spearmans_sr = ', i, j, '{0:.3g}'.format(pcorr_sr), '{0:.3g}'.format(scorr_sr))
		#
		pcorr_sd, _ = pearsonr(wdrs_arr[ind,j], sd_arr[ind,i])
		scorr_sd, _ = spearmanr(wdrs_arr[ind,j], sd_arr[ind,i])
		print('i, j, Pearsons_sd, Spearmans_sd = ', i, j, '{0:.3g}'.format(pcorr_sd), '{0:.3g}'.format(scorr_sd))
		#
		legend_properties = {'weight':'bold'}
		fig = plt.figure()
		ax  = fig.add_subplot(111)
		ax.set_xlabel(wdrs_ftrs_list[j], fontsize = 12, fontweight = 'bold')
		ax.set_ylabel('Species richness of ' + comp_list[i], fontsize = 12, fontweight = 'bold')
		plt.title('Pearsons correlation: ' + str('{0:.3g}'.format(pcorr_sr)) +  \
					', Spearmans correlation: ' + str('{0:.3g}'.format(scorr_sr)))
		ax.scatter(wdrs_arr[:,j], sr_arr[:,i], s = 20, c = color, marker = marker, edgecolor = 'face')
		plt.savefig(path + 'Plots_WHONDRS/Species_Richness/SR_' + \
					str(i) + '_' + str(j) + '_' + comp_list[i] + '_' + wdrs_ftrs_list[j] + '.png')
		plt.close(fig)
		#
		legend_properties = {'weight':'bold'}
		fig = plt.figure()
		ax  = fig.add_subplot(111)
		ax.set_xlabel(wdrs_ftrs_list[j], fontsize = 12, fontweight = 'bold')
		ax.set_ylabel('Shann. diversity of ' + comp_list[i], fontsize = 12, fontweight = 'bold')
		plt.title('Pearsons correlation: ' + str('{0:.3g}'.format(pcorr_sd)) +  \
					', Spearmans correlation: ' + str('{0:.3g}'.format(scorr_sd)))
		ax.scatter(wdrs_arr[:,j], sd_arr[:,i], s = 20, c = color, marker = marker, edgecolor = 'face')
		plt.savefig(path + 'Plots_WHONDRS/Shannon_Diversity/SD_' + \
					str(i) + '_' + str(j) + '_' + comp_list[i] + '_' + wdrs_ftrs_list[j] + '.png')
		plt.close(fig)
		print('\n')

#*********************************************************************;
#  3. SR and SD vs. log10(extrinsic factors) for 9 compounds and sum  ;
#*********************************************************************;
for j in range(0,len(wdrs_ftrs_list)): #Extrinsic features 0 to 12
	for i in range(0,len(comp_list)): #Compounds i = 0 to 10
		color = color_list[i]
		marker = marker_list[i]
		#
		print('Comp name, Extrinsic feature = ', comp_list[i], wdrs_ftrs_list[j])
		print(np.argwhere(np.isnan(wdrs_arr[:,j]))[:,0])
		ind      = np.argwhere(~np.isnan(wdrs_arr[:,j]))[:,0]
		#
		pcorr_sr, _ = pearsonr(np.log10(wdrs_arr[ind,j] + 1e-8), sr_arr[ind,i])
		scorr_sr, _ = spearmanr(np.log10(wdrs_arr[ind,j] + 1e-8), sr_arr[ind,i])
		m_sr, c_sr, \
		r_value_sr, \
		p_value_sr,\
		std_err_sr  = linregress(np.log10(wdrs_arr[ind,j] + 1e-8), sr_arr[ind,i])
		print('i, j, Pearsons_sr, Spearmans_sr, m_sr = ', i, j, \
				'{0:.3g}'.format(pcorr_sr), '{0:.3g}'.format(scorr_sr), '{0:.3g}'.format(m_sr))
		#
		pcorr_sd, _ = pearsonr(np.log10(wdrs_arr[ind,j] + 1e-8), sd_arr[ind,i])
		scorr_sd, _ = spearmanr(np.log10(wdrs_arr[ind,j] + 1e-8), sd_arr[ind,i])
		m_sd, c_sd, \
		r_value_sd, \
		p_value_sd,\
		std_err_sd  = linregress(np.log10(wdrs_arr[ind,j] + 1e-8), sd_arr[ind,i])
		print('i, j, Pearsons_sd, Spearmans_sd, m_sd = ', i, j, \
				'{0:.3g}'.format(pcorr_sd), '{0:.3g}'.format(scorr_sd), '{0:.3g}'.format(m_sd))
		#
		legend_properties = {'weight':'bold'}
		fig = plt.figure()
		ax  = fig.add_subplot(111)
		ax.set_xlabel(wdrs_ftrs_list[j] + ' (in log10)', fontsize = 12, fontweight = 'bold')
		ax.set_ylabel('Species richness of ' + comp_list[i], fontsize = 12, fontweight = 'bold')
		plt.title('Pearsons correlation: ' + str('{0:.3g}'.format(pcorr_sr)) +  \
					', Spearmans correlation: ' + str('{0:.3g}'.format(scorr_sr)))
		ax.scatter(np.log10(wdrs_arr[:,j] + 1e-8), sr_arr[:,i], s = 20, c = color, marker = marker, edgecolor = 'face')
		yfit_sr = np.asarray([c_sr + m_sr*xi for xi in np.log10(wdrs_arr[:,j] + 1e-8)])
		ax.plot(np.log10(wdrs_arr[:,j] + 1e-8), yfit_sr, color = color, label = 'Slope = ' + str('{0:.3g}'.format(m_sr)))
		ax.legend(loc = 'best')
		plt.savefig(path + 'Plots_WHONDRS/Species_Richness_log/SR_' + \
					str(i) + '_' + str(j) + '_' + comp_list[i] + '_' + wdrs_ftrs_list[j] + '.png')
		plt.close(fig)
		#
		legend_properties = {'weight':'bold'}
		fig = plt.figure()
		ax  = fig.add_subplot(111)
		ax.set_xlabel(wdrs_ftrs_list[j] + ' (in log10)', fontsize = 12, fontweight = 'bold')
		ax.set_ylabel('Shann. diversity of ' + comp_list[i], fontsize = 12, fontweight = 'bold')
		plt.title('Pearsons correlation: ' + str('{0:.3g}'.format(pcorr_sd)) +  \
					', Spearmans correlation: ' + str('{0:.3g}'.format(scorr_sd)))
		ax.scatter(np.log10(wdrs_arr[:,j] + 1e-8), sd_arr[:,i], s = 20, c = color, marker = marker, edgecolor = 'face')
		yfit_sd = np.asarray([c_sd + m_sd*xi for xi in np.log10(wdrs_arr[:,j] + 1e-8)])
		ax.plot(np.log10(wdrs_arr[:,j] + 1e-8), yfit_sd, color = color, label = 'Slope = ' + str('{0:.3g}'.format(m_sd)))
		ax.legend(loc = 'best')
		plt.savefig(path + 'Plots_WHONDRS/Shannon_Diversity_log/SD_' + \
					str(i) + '_' + str(j) + '_' + comp_list[i] + '_' + wdrs_ftrs_list[j] + '.png')
		plt.close(fig)
		print('\n')

#*********************************************************************************************;
#  4a. Correlation values for SR and SD vs. log10(extrinsic factors) for 9 compounds and sum  ;
#*********************************************************************************************;
pcorr_sr_list = np.zeros((len(comp_list), len(wdrs_ftrs_list)), dtype = float) #(10,12)
scorr_sr_list = np.zeros((len(comp_list), len(wdrs_ftrs_list)), dtype = float) #(10,12)
#
pcorr_sd_list = np.zeros((len(comp_list), len(wdrs_ftrs_list)), dtype = float) #(10,12)
scorr_sd_list = np.zeros((len(comp_list), len(wdrs_ftrs_list)), dtype = float) #(10,12)
#
for j in range(0,len(wdrs_ftrs_list)): #Extrinsic features 0 to 12
	for i in range(0,len(comp_list)): #Compounds i = 0 to 10
		#print('Comp name, Extrinsic feature = ', comp_list[i], wdrs_ftrs_list[j])
		#print(np.argwhere(np.isnan(wdrs_arr[:,j]))[:,0])
		ind      = np.argwhere(~np.isnan(wdrs_arr[:,j]))[:,0]
		#
		pcorr_sr, _ = pearsonr(np.log10(wdrs_arr[ind,j] + 1e-8), sr_arr[ind,i])
		scorr_sr, _ = spearmanr(np.log10(wdrs_arr[ind,j] + 1e-8), sr_arr[ind,i])
		m_sr, c_sr, \
		r_value_sr, \
		p_value_sr,\
		std_err_sr  = linregress(np.log10(wdrs_arr[ind,j] + 1e-8), sr_arr[ind,i])
		print('i, j, Comp name, Extrinsic feature, Pearsons_sr, Spearmans_sr, m_sr = ', \
				i, j, comp_list[i], wdrs_ftrs_list[j], \
				'{0:.3g}'.format(pcorr_sr), '{0:.3g}'.format(scorr_sr), '{0:.3g}'.format(m_sr))
		#
		pcorr_sd, _ = pearsonr(np.log10(wdrs_arr[ind,j] + 1e-8), sd_arr[ind,i])
		scorr_sd, _ = spearmanr(np.log10(wdrs_arr[ind,j] + 1e-8), sd_arr[ind,i])
		m_sd, c_sd, \
		r_value_sd, \
		p_value_sd,\
		std_err_sd  = linregress(np.log10(wdrs_arr[ind,j] + 1e-8), sd_arr[ind,i])
		print('i, j, Comp name, Extrinsic feature, Pearsons_sd, Spearmans_sd, m_sd = ', \
				i, j, comp_list[i], wdrs_ftrs_list[j], \
				'{0:.3g}'.format(pcorr_sd), '{0:.3g}'.format(scorr_sd), '{0:.3g}'.format(m_sd))
		#
		pcorr_sr_list[i,j] = pcorr_sr
		scorr_sr_list[i,j] = scorr_sr
		#
		pcorr_sd_list[i,j] = pcorr_sd
		scorr_sd_list[i,j] = scorr_sd

#*******************************************************************************************;
#  4b. Correlation plots of SR and SD vs. log10(extrinsic factors) for 9 compounds and sum  ;
#*******************************************************************************************;
#https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
fig = plt.figure(figsize=(10,10))
ax  = fig.add_subplot(111)
im = ax.imshow(pcorr_sr_list, cmap = 'coolwarm')
ax.set_xticks(np.arange(len(wdrs_ftrs_list)), labels=wdrs_ftrs_list)
ax.set_yticks(np.arange(len(comp_list)), labels=comp_list)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
#
for i in range(len(comp_list)): # Loop over data dimensions and create text annotations.
	for j in range(len(wdrs_ftrs_list)):
		#print(j, i, pcorr_sr_list[i,j])
		text = ax.text(j, i, '{0:.2g}'.format(pcorr_sr_list[i,j]), \
						ha="center", va="center", color="k")
ax.set_title("Pearsons correlation: Species richness vs. log(WHONDRS) data")
fig.tight_layout()
plt.savefig(path + 'Plots_WHONDRS/pcorr_SR_vs_logWHONDRS.png')
plt.close(fig)
#
fig = plt.figure(figsize=(10,10))
ax  = fig.add_subplot(111)
im = ax.imshow(scorr_sr_list, cmap = 'coolwarm')
ax.set_xticks(np.arange(len(wdrs_ftrs_list)), labels=wdrs_ftrs_list)
ax.set_yticks(np.arange(len(comp_list)), labels=comp_list)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
#
for i in range(len(comp_list)): # Loop over data dimensions and create text annotations.
	for j in range(len(wdrs_ftrs_list)):
		#print(j, i, scorr_sr_list[i,j])
		text = ax.text(j, i, '{0:.2g}'.format(scorr_sr_list[i,j]), \
						ha="center", va="center", color="k")
ax.set_title("Spearmans correlation: Species richness vs. log(WHONDRS) data")
fig.tight_layout()
plt.savefig(path + 'Plots_WHONDRS/scorr_SR_vs_logWHONDRS.png')
plt.close(fig)
#
fig = plt.figure(figsize=(10,10))
ax  = fig.add_subplot(111)
im = ax.imshow(pcorr_sd_list, cmap = 'coolwarm')
ax.set_xticks(np.arange(len(wdrs_ftrs_list)), labels=wdrs_ftrs_list)
ax.set_yticks(np.arange(len(comp_list)), labels=comp_list)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
#
for i in range(len(comp_list)): # Loop over data dimensions and create text annotations.
	for j in range(len(wdrs_ftrs_list)):
		#print(j, i, pcorr_sd_list[i,j])
		text = ax.text(j, i, '{0:.2g}'.format(pcorr_sd_list[i,j]), \
						ha="center", va="center", color="k")
ax.set_title("Pearsons correlation: Shannon diversity vs. log(WHONDRS) data")
fig.tight_layout()
plt.savefig(path + 'Plots_WHONDRS/pcorr_SD_vs_logWHONDRS.png')
plt.close(fig)
#
fig = plt.figure(figsize=(10,10))
ax  = fig.add_subplot(111)
im = ax.imshow(scorr_sd_list, cmap = 'coolwarm')
ax.set_xticks(np.arange(len(wdrs_ftrs_list)), labels=wdrs_ftrs_list)
ax.set_yticks(np.arange(len(comp_list)), labels=comp_list)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
#
for i in range(len(comp_list)): # Loop over data dimensions and create text annotations.
	for j in range(len(wdrs_ftrs_list)):
		#print(j, i, scorr_sd_list[i,j])
		text = ax.text(j, i, '{0:.2g}'.format(scorr_sd_list[i,j]), \
						ha="center", va="center", color="k")
ax.set_title("Spearmans correlation: Shannon diversity vs. log(WHONDRS) data")
fig.tight_layout()
plt.savefig(path + 'Plots_WHONDRS/scorr_SD_vs_logWHONDRS.png')
plt.close(fig)
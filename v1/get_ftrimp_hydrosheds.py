# HYDROSHEDS data -- Get feature importance using the following methods:
#   1. F-test
#   2. MI
#   3. Random Forests
#      https://mljar.com/blog/feature-importance-in-random-forest/
#      https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
#   4. SHAP
#      https://mljar.com/blog/feature-importance-in-random-forest/
#      https://www.investopedia.com/terms/s/shapley-value.asp#:~:text=Essentially%2C%20the%20Shapley%20value%20is,or%20less%20than%20the%20others.
#      https://towardsdatascience.com/the-shapley-value-for-ml-models-f1100bff78d1
#   5. Pearson correlation
#   6. Spearsman correlation
#
#   Important features identified include (dataIndex --> FeatureName):
#       10 --> 'Natural discharge - dis_m3_pmn'
#       17 --> 'Inundation extent - inu_pc_slt'
#       24 --> 'River area - ria_ha_ssu'
#       26 --> 'River volume - riv_tc_ssu'
#       37 --> 'Climate strata - cls_cl_smj'
#       46 --> 'Air temp - tmp_dc_s05'
#       73 --> 'Potential ET - pet_mm_s04'
#       87 --> 'Actual ET - aet_mm_s04'
#
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
#
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
#
from sklearn.inspection import permutation_importance
import shap
#
np.set_printoptions(precision=2)

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

#******************************************;
#  2. Hydrosheds correlation coefficients  ;
#******************************************;
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

#****************************************************;
#  3. Best hydrosheds correlation coefficients list  ;
#****************************************************;
hydrosheds_index_list = [10, 17, 24, 26, 37, 46, 73, 87] #8
hydrosheds_ftrs_list  = ['Natural discharge - dis_m3_pmn', 'Inundation extent - inu_pc_slt', \
                            'River area - ria_ha_ssu', 'River volume - riv_tc_ssu', \
                            'Climate strata - cls_cl_smj', 'Air temp - tmp_dc_s05', \
                            'Potential ET - pet_mm_s04', 'Actual ET - aet_mm_s04'] #8
#
hs_arr           = copy.deepcopy(huc12_arr[:,hydrosheds_index_list]) #(54, 8)
#
X = copy.deepcopy(hs_arr) #(54, 8); hydrosheds features with non-nan samples
y = copy.deepcopy(sr_arr) #(54, 10); sr corresponding to non-nan samples

#**********************************************************************;
#  4a. Feature importance using F-test and MI (a total of 8 features)  ;
#**********************************************************************;
ftest_sr_list = np.zeros((len(comp_list), len(hydrosheds_ftrs_list)), dtype = float) #(10,8)
mi_sr_list    = np.zeros((len(comp_list), len(hydrosheds_ftrs_list)), dtype = float) #(10,8)
#
for i in range(0,len(comp_list)):
    f_test, _          = f_regression(X, y[:,i]) #(8,)
    f_test            /= np.max(f_test) #(8,)
    #
    mi                 = mutual_info_regression(X, y[:,i], n_neighbors = 3, random_state = 0) #(8,)
    mi                /= np.max(mi) #(8,)
    #
    ftest_sr_list[i,:] = copy.deepcopy(f_test)
    mi_sr_list[i,:]    = copy.deepcopy(mi)

#******************************************************************************;
#  4b. Feature importance using RF and SHAPley values (a total of 8 features)  ;
#******************************************************************************;
rf_sr_list       = np.zeros((len(comp_list), len(hydrosheds_ftrs_list)), dtype = float) #(10,8)
shap_sr_list     = np.zeros((len(comp_list), len(hydrosheds_ftrs_list)), dtype = float) #(10,8)
rfnorm_sr_list   = np.zeros((len(comp_list), len(hydrosheds_ftrs_list)), dtype = float) #(10,8)
shapnorm_sr_list = np.zeros((len(comp_list), len(hydrosheds_ftrs_list)), dtype = float) #(10,8)
#
for i in range(0,len(comp_list)):
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y[:,i])
    perm_importance = permutation_importance(rf, X, y[:,i], random_state=42)
    print(i, perm_importance.importances_mean)
    #
    #plt.barh(hydrosheds_ftrs_list, perm_importance.importances_mean)
    #plt.show()
    #
    explainer   = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X)
    np.abs(shap_values.sum(1) + explainer.expected_value - rf.predict(X)).max()
    #
    shap_interaction_values = explainer.shap_interaction_values(X)
    shap_interaction_values[0]
    np.abs(shap_interaction_values.sum((1,2)) + explainer.expected_value - rf.predict(X)).max()
    #
    #shap.summary_plot(shap_values, X, plot_type="bar")
    #plt.barh(hydrosheds_ftrs_list, np.mean(np.abs(shap_values), axis = 0))
    #plt.show()
    #
    #shap.summary_plot(shap_interaction_values, X, plot_type="bar")
    #
    rf_sr_list[i,:]   = copy.deepcopy(perm_importance.importances_mean)
    shap_sr_list[i,:] = copy.deepcopy(np.mean(np.abs(shap_values), axis = 0))
    #
    rfnorm_sr_list[i,:]   = copy.deepcopy(perm_importance.importances_mean/np.max(perm_importance.importances_mean))
    shapnorm_sr_list[i,:] = copy.deepcopy(np.mean(np.abs(shap_values), axis = 0)/np.max(np.mean(np.abs(shap_values), axis = 0)))

#*******************************************************************************;
#  4c. Correlation values for SR vs. extrinsic factors for 9 compounds and sum  ;
#*******************************************************************************;
pcorr_sr_list = np.zeros((len(comp_list), len(hydrosheds_ftrs_list)), dtype = float) #(10,8)
scorr_sr_list = np.zeros((len(comp_list), len(hydrosheds_ftrs_list)), dtype = float) #(10,8)
#
for j in range(0,len(hydrosheds_ftrs_list)): #Extrinsic features 0 to 8
    for i in range(0,len(comp_list)): #Compounds i = 0 to 10
        h_index = hydrosheds_index_list[j]
        #
        #print('Comp name, Extrinsic feature = ', comp_list[i], hydrosheds_ftrs_list[j])
        #print(np.argwhere(np.isnan(huc12_arr[:,h_index]))[:,0])
        ind      = np.argwhere(~np.isnan(huc12_arr[:,h_index]))[:,0]
        #
        pcorr_sr, _ = pearsonr(huc12_arr[ind,h_index] + 1e-8, sr_arr[ind,i])
        scorr_sr, _ = spearmanr(huc12_arr[ind,h_index] + 1e-8, sr_arr[ind,i])
        m_sr, c_sr, \
        r_value_sr, \
        p_value_sr,\
        std_err_sr  = linregress(huc12_arr[ind,h_index] + 1e-8, sr_arr[ind,i])
        print('i, j, Pearsons_sr, Spearmans_sr, m_sr = ', i, j, \
                '{0:.3g}'.format(pcorr_sr), '{0:.3g}'.format(scorr_sr), '{0:.3g}'.format(m_sr))
        #
        pcorr_sr_list[i,j] = pcorr_sr
        scorr_sr_list[i,j] = scorr_sr

#*********************************************************************************************************;
#  4d. Feature importance among F-test, MI, RF, SHAPley values, pcorr, and scorr (a total of 8 features)  ;
#*********************************************************************************************************;
ftest_imp     = np.sum(ftest_sr_list, axis = 0)/np.max(np.sum(ftest_sr_list, axis = 0)) #(8,)
mi_imp        = np.sum(mi_sr_list, axis = 0)/np.max(np.sum(mi_sr_list, axis = 0)) #(8,)
rfnorm_imp    = np.sum(rfnorm_sr_list, axis = 0)/np.max(np.sum(rfnorm_sr_list, axis = 0)) #(8,)
shapnorm_imp  = np.sum(shapnorm_sr_list, axis = 0)/np.max(np.sum(shapnorm_sr_list, axis = 0)) #(8,)
pcorr_imp     = np.sum(np.abs(pcorr_sr_list), axis = 0)/np.max(np.sum(np.abs(pcorr_sr_list), axis = 0)) #(8,)
scorr_imp     = np.sum(np.abs(scorr_sr_list), axis = 0)/np.max(np.sum(np.abs(scorr_sr_list), axis = 0)) #(8,)
#
ftrs_imp      = np.zeros((7,len(hydrosheds_ftrs_list)), dtype = float) #(7,8)
ftrs_imp[0,:] = copy.deepcopy(ftest_imp) #F-test
ftrs_imp[1,:] = copy.deepcopy(mi_imp) #MI
ftrs_imp[2,:] = copy.deepcopy(rfnorm_imp) #RF
ftrs_imp[3,:] = copy.deepcopy(shapnorm_imp) #SHAPley
ftrs_imp[4,:] = copy.deepcopy(pcorr_imp) #pcorr
ftrs_imp[5,:] = copy.deepcopy(scorr_imp) #scorr
#
#ftrs_imp[6,:] = np.sum(ftrs_imp[0:6,:], axis = 0)/np.max(np.sum(ftrs_imp[0:6], axis = 0))
ftrs_imp[6,:] = np.mean(ftrs_imp[0:6,:], axis = 0) #Avg-value

#***************************************************;
#  5a. F-test and MI plots for 9 compounds and sum  ;
#***************************************************;
#https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
fig = plt.figure(figsize=(10,10))
ax  = fig.add_subplot(111)
im  = ax.imshow(ftest_sr_list, cmap = 'coolwarm', alpha = 0.5)
ax.set_xticks(np.arange(len(hydrosheds_ftrs_list)), labels=hydrosheds_ftrs_list)
ax.set_yticks(np.arange(len(comp_list)), labels=comp_list)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
#
for i in range(len(comp_list)): # Loop over data dimensions and create text annotations.
    for j in range(len(hydrosheds_ftrs_list)):
        #print(j, i, ftest_sr_list[i,j])
        text = ax.text(j, i, '{0:.2g}'.format(ftest_sr_list[i,j]), \
                        ha="center", va="center", color="k")
ax.set_title("F-test based feature importance on species richness vs. HYDROSHEDS data features")
fig.tight_layout()
plt.savefig(path + 'Plots_HYDROSHEDS/ftest_SR_vs_HYDROSHEDS_ftrs.png')
plt.close(fig)
#
#https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
fig = plt.figure(figsize=(10,10))
ax  = fig.add_subplot(111)
im  = ax.imshow(mi_sr_list, cmap = 'coolwarm', alpha = 0.5)
ax.set_xticks(np.arange(len(hydrosheds_ftrs_list)), labels=hydrosheds_ftrs_list)
ax.set_yticks(np.arange(len(comp_list)), labels=comp_list)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
#
for i in range(len(comp_list)): # Loop over data dimensions and create text annotations.
    for j in range(len(hydrosheds_ftrs_list)):
        #print(j, i, mi_sr_list[i,j])
        text = ax.text(j, i, '{0:.2g}'.format(mi_sr_list[i,j]), \
                        ha="center", va="center", color="k")
ax.set_title("MI-based feature importance on species richness vs. HYDROSHEDS data features")
fig.tight_layout()
plt.savefig(path + 'Plots_HYDROSHEDS/mi_SR_vs_HYDROSHEDS_ftrs.png')
plt.close(fig)

#**********************************************************;
#  5b. RF and SHAPley value plots for 9 compounds and sum  ;
#**********************************************************;
#https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
fig = plt.figure(figsize=(10,10))
ax  = fig.add_subplot(111)
im  = ax.imshow(rf_sr_list, cmap = 'coolwarm', alpha = 0.5)
ax.set_xticks(np.arange(len(hydrosheds_ftrs_list)), labels=hydrosheds_ftrs_list)
ax.set_yticks(np.arange(len(comp_list)), labels=comp_list)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
#
for i in range(len(comp_list)): # Loop over data dimensions and create text annotations.
    for j in range(len(hydrosheds_ftrs_list)):
        #print(j, i, rf_sr_list[i,j])
        text = ax.text(j, i, '{0:.2g}'.format(rf_sr_list[i,j]), \
                        ha="center", va="center", color="k")
ax.set_title("RF-based feature importance on species richness vs. HYDROSHEDS data features")
fig.tight_layout()
plt.savefig(path + 'Plots_HYDROSHEDS/rf_SR_vs_HYDROSHEDS_ftrs.png')
plt.close(fig)
#
#https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
fig = plt.figure(figsize=(10,10))
ax  = fig.add_subplot(111)
im  = ax.imshow(shap_sr_list, cmap = 'coolwarm', alpha = 0.5)
ax.set_xticks(np.arange(len(hydrosheds_ftrs_list)), labels=hydrosheds_ftrs_list)
ax.set_yticks(np.arange(len(comp_list)), labels=comp_list)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
#
for i in range(len(comp_list)): # Loop over data dimensions and create text annotations.
    for j in range(len(hydrosheds_ftrs_list)):
        #print(j, i, shap_sr_list[i,j])
        text = ax.text(j, i, '{0:.2g}'.format(shap_sr_list[i,j]), \
                        ha="center", va="center", color="k")
ax.set_title("SHAPley-based feature importance on species richness vs. HYDROSHEDS data features")
fig.tight_layout()
plt.savefig(path + 'Plots_HYDROSHEDS/shap_SR_vs_HYDROSHEDS_ftrs.png')
plt.close(fig)

#***********************************************************************;
#  5c. RF and SHAPley value plots for 9 compounds and sum (normalized)  ;
#***********************************************************************;
#https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
fig = plt.figure(figsize=(10,10))
ax  = fig.add_subplot(111)
im  = ax.imshow(rfnorm_sr_list, cmap = 'coolwarm', alpha = 0.5)
ax.set_xticks(np.arange(len(hydrosheds_ftrs_list)), labels=hydrosheds_ftrs_list)
ax.set_yticks(np.arange(len(comp_list)), labels=comp_list)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
#
for i in range(len(comp_list)): # Loop over data dimensions and create text annotations.
    for j in range(len(hydrosheds_ftrs_list)):
        #print(j, i, rfnorm_sr_list[i,j])
        text = ax.text(j, i, '{0:.2g}'.format(rfnorm_sr_list[i,j]), \
                        ha="center", va="center", color="k")
ax.set_title("RF-based feature importance (normalized) on species richness vs. HYDROSHEDS data features")
fig.tight_layout()
plt.savefig(path + 'Plots_HYDROSHEDS/rfnorm_SR_vs_HYDROSHEDS_ftrs.png')
plt.close(fig)
#
#https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
fig = plt.figure(figsize=(10,10))
ax  = fig.add_subplot(111)
im  = ax.imshow(shapnorm_sr_list, cmap = 'coolwarm', alpha = 0.5)
ax.set_xticks(np.arange(len(hydrosheds_ftrs_list)), labels=hydrosheds_ftrs_list)
ax.set_yticks(np.arange(len(comp_list)), labels=comp_list)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
#
for i in range(len(comp_list)): # Loop over data dimensions and create text annotations.
    for j in range(len(hydrosheds_ftrs_list)):
        #print(j, i, shapnorm_sr_list[i,j])
        text = ax.text(j, i, '{0:.2g}'.format(shapnorm_sr_list[i,j]), \
                        ha="center", va="center", color="k")
ax.set_title("SHAPley-based feature importance (normalized) on species richness vs. HYDROSHEDS data features")
fig.tight_layout()
plt.savefig(path + 'Plots_HYDROSHEDS/shapnorm_SR_vs_HYDROSHEDS_ftrs.png')
plt.close(fig)

#********************************************************************************;
#  5d. F-test, MI, RF, SHAPley, pcorr, scorr, and avg-value feature importances  ;
#      (Each one is normalized from 0 to 1)                                      ;
#********************************************************************************;
#https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
fig = plt.figure(figsize=(10,10))
ax  = fig.add_subplot(111)
im  = ax.imshow(ftrs_imp, cmap = 'coolwarm', alpha = 0.5)
ax.set_xticks(np.arange(len(hydrosheds_ftrs_list)), labels=hydrosheds_ftrs_list)
ax.set_yticks(np.arange(7), labels=['F-test', 'MI', 'RF', 'SHAPley', 'Pearson', 'Spearsman', 'Avg-value'])
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
#
for i in range(7): # Loop over data dimensions and create text annotations.
    for j in range(len(hydrosheds_ftrs_list)):
        #print(j, i, ftrs_imp[i,j])
        text = ax.text(j, i, '{0:.2g}'.format(ftrs_imp[i,j]), \
                        ha="center", va="center", color="k")
ax.set_title("Feature importance (normalized) -- species richness vs. HYDROSHEDS data features")
fig.tight_layout()
plt.savefig(path + 'Plots_HYDROSHEDS/Z_imp_SR_vs_HYDROSHEDS_ftrs.png')
plt.savefig(path + 'Plots_HYDROSHEDS/SVG_figs/Z_imp_SR_vs_HYDROSHEDS_ftrs.svg')
plt.savefig(path + 'Plots_HYDROSHEDS/SVG_figs/Z_imp_SR_vs_HYDROSHEDS_ftrs.pdf')
plt.close(fig)

#*********************************************************************************************;
#  5e. Min, Max, Mean, and STD for F-test, MI, RF, SHAPley, pcorr, scorr feature importances  ;
#      (Each one is normalized from 0 to 1)                                                   ;
#*********************************************************************************************;
temp_data             = copy.deepcopy(ftrs_imp[:-1,:]) #(6, 8)
mean_ftrs_imp         = np.mean(temp_data, axis = 0) #(8,)
std_ftrs_imp          = np.std(temp_data, axis = 0) #(8,)
min_ftrs_imp          = np.min(temp_data, axis = 0) #(8,)
max_ftrs_imp          = np.max(temp_data, axis = 0) #(8,)
#
metrics_ftrs_imp      = np.zeros((4,len(hydrosheds_ftrs_list)), dtype = float) #(4, 8)
metrics_ftrs_imp[0,:] = copy.deepcopy(mean_ftrs_imp) #mean #(8,)
metrics_ftrs_imp[1,:] = copy.deepcopy(std_ftrs_imp) #std #(8,)
metrics_ftrs_imp[2,:] = copy.deepcopy(min_ftrs_imp) #min #(8,)
metrics_ftrs_imp[3,:] = copy.deepcopy(max_ftrs_imp) #max #(8,)
#
row_names             = ['mean', 'std', 'min', 'max']
df_metrics_ftrs_imp   = pd.DataFrame(metrics_ftrs_imp, columns = hydrosheds_ftrs_list, index = row_names)
df_metrics_ftrs_imp.to_csv(path+ 'Plots_HYDROSHEDS/Feature_List/Hydrosheds_ftrs_min_max_mean_std.csv')
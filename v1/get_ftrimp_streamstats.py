# StreamStats data -- Get feature importance using the following methods:
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
#  StreamStats features:
#       'long', 
#       'lat', 
#       'DRNAREA', 
#       'ELEV', 
#       'PRECIP', 
#       'ELEVMAX', 
#       'MINBELEV', 
#       'FOREST'
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
df_sr          = pd.read_csv(path + "Inputs_Outputs_v4/1_WAT_SR_54s_11f.csv", index_col = 1).iloc[:,1:] #[54 rows x 10 columns]
df_sd          = pd.read_csv(path + "Inputs_Outputs_v4/2_WAT_SHANN_54s_11f.csv", index_col = 1).iloc[:,1:] #[54 rows x 10 columns]
#
df_strmstats   = pd.read_csv(path + "Inputs_Outputs_v4/6_StreamStats_54s_9f.csv", index_col = 3).iloc[:,1:]#[54 rows x 8 columns]
#
comp_list      = df_sr.columns.to_list() #10
strm_ftrs_list = df_strmstats.columns.to_list() #8
#
sr_arr         = df_sr.values #(54, 10) #species richness
sd_arr         = df_sd.values #(54, 10) #shannon diversity
strm_arr       = df_strmstats.values #(54, 8) #ftrs
#
nonnan_rows_list = [] #find non-nan samples = 12
#
for i in range(0,strm_arr.shape[0]): #Iterate over number of samples
    if len(np.argwhere(np.isnan(strm_arr[i,:]))[:,0]) == 0:
        print(i, len(np.argwhere(np.isnan(strm_arr[i,:]))[:,0]))
        nonnan_rows_list.append(i) #Non-nan sample indices

X = copy.deepcopy(strm_arr[nonnan_rows_list,:]) #(12, 8); streamstats features with non-nan samples
y = copy.deepcopy(sr_arr[nonnan_rows_list,:]) #(12, 10); sr corresponding to non-nan samples

#**********************************************************************;
#  2a. Feature importance using F-test and MI (a total of 8 features)  ;
#**********************************************************************;
ftest_sr_list = np.zeros((len(comp_list), len(strm_ftrs_list)), dtype = float) #(10,8)
mi_sr_list    = np.zeros((len(comp_list), len(strm_ftrs_list)), dtype = float) #(10,8)
#
for i in range(0,len(comp_list)):
    f_test, _          = f_regression(X, y[:,i]) #(8,)
    f_test            /= np.max(f_test) #(8,)
    #
    mi                 = mutual_info_regression(X, y[:,i], n_neighbors = 3, random_state = 0) #(8,)
    mi                /= np.max(mi) #(8,)
    #
    ftest_sr_list[i,:] = copy.deepcopy(f_test) #(10,8)
    mi_sr_list[i,:]    = copy.deepcopy(mi) #(10,8)

#******************************************************************************;
#  2b. Feature importance using RF and SHAPley values (a total of 8 features)  ;
#      (Raw and normalized feature importance values)                          ;
#******************************************************************************;
rf_sr_list       = np.zeros((len(comp_list), len(strm_ftrs_list)), dtype = float) #(10,8)
shap_sr_list     = np.zeros((len(comp_list), len(strm_ftrs_list)), dtype = float) #(10,8)
rfnorm_sr_list   = np.zeros((len(comp_list), len(strm_ftrs_list)), dtype = float) #(10,8)
shapnorm_sr_list = np.zeros((len(comp_list), len(strm_ftrs_list)), dtype = float) #(10,8)
#
for i in range(0,len(comp_list)):
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y[:,i])
    perm_importance = permutation_importance(rf, X, y[:,i], random_state=42)
    print(i, perm_importance.importances_mean)
    #
    #plt.barh(strm_ftrs_list, perm_importance.importances_mean)
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
    #plt.barh(strm_ftrs_list, np.mean(np.abs(shap_values), axis = 0))
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
#  2c. Correlation values for SR vs. extrinsic factors for 9 compounds and sum  ;
#*******************************************************************************;
pcorr_sr_list     = np.zeros((len(comp_list), len(strm_ftrs_list)), dtype = float) #(10,8)
scorr_sr_list     = np.zeros((len(comp_list), len(strm_ftrs_list)), dtype = float) #(10,8)
#
for j in range(0,len(strm_ftrs_list)): #Extrinsic features 0 to 8
    for i in range(0,len(comp_list)): #Compounds i = 0 to 10
        #print('Comp name, Extrinsic feature = ', comp_list[i], strm_ftrs_list[j])
        #print(np.argwhere(np.isnan(strm_arr[:,j]))[:,0])
        ind      = np.argwhere(~np.isnan(strm_arr[:,j]))[:,0]
        #
        pcorr_sr, _ = pearsonr(strm_arr[ind,j] + 1e-8, sr_arr[ind,i])
        scorr_sr, _ = spearmanr(strm_arr[ind,j] + 1e-8, sr_arr[ind,i])
        m_sr, c_sr, \
        r_value_sr, \
        p_value_sr,\
        std_err_sr  = linregress(strm_arr[ind,j] + 1e-8, sr_arr[ind,i])
        print('i, j, Comp name, Extrinsic feature, Pearsons_sr, Spearmans_sr, m_sr = ', \
                i, j, comp_list[i], strm_ftrs_list[j], \
                '{0:.3g}'.format(pcorr_sr), '{0:.3g}'.format(scorr_sr), '{0:.3g}'.format(m_sr))
        #
        pcorr_sr_list[i,j] = pcorr_sr
        scorr_sr_list[i,j] = scorr_sr

#*********************************************************************************************************;
#  2d. Feature importance among F-test, MI, RF, SHAPley values, pcorr, and scorr (a total of 8 features)  ;
#*********************************************************************************************************;
ftest_imp     = np.sum(ftest_sr_list, axis = 0)/np.max(np.sum(ftest_sr_list, axis = 0)) #(8,)
mi_imp        = np.sum(mi_sr_list, axis = 0)/np.max(np.sum(mi_sr_list, axis = 0)) #(8,)
rfnorm_imp    = np.sum(rfnorm_sr_list, axis = 0)/np.max(np.sum(rfnorm_sr_list, axis = 0)) #(8,)
shapnorm_imp  = np.sum(shapnorm_sr_list, axis = 0)/np.max(np.sum(shapnorm_sr_list, axis = 0)) #(8,)
pcorr_imp     = np.sum(np.abs(pcorr_sr_list), axis = 0)/np.max(np.sum(np.abs(pcorr_sr_list), axis = 0)) #(8,)
scorr_imp     = np.sum(np.abs(scorr_sr_list), axis = 0)/np.max(np.sum(np.abs(scorr_sr_list), axis = 0)) #(8,)
#
ftrs_imp      = np.zeros((7,len(strm_ftrs_list)), dtype = float) #(7, 8)
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
#  3a. F-test and MI plots for 9 compounds and sum  ;
#***************************************************;
#https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
fig = plt.figure(figsize=(10,10))
ax  = fig.add_subplot(111)
im  = ax.imshow(ftest_sr_list, cmap = 'coolwarm', alpha = 0.5)
ax.set_xticks(np.arange(len(strm_ftrs_list)), labels=strm_ftrs_list)
ax.set_yticks(np.arange(len(comp_list)), labels=comp_list)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
#
for i in range(len(comp_list)): # Loop over data dimensions and create text annotations.
    for j in range(len(strm_ftrs_list)):
        #print(j, i, ftest_sr_list[i,j])
        text = ax.text(j, i, '{0:.2g}'.format(ftest_sr_list[i,j]), \
                        ha="center", va="center", color="k")
ax.set_title("F-test based feature importance on species richness vs. StreamStats data features")
fig.tight_layout()
plt.savefig(path + 'Plots_StreamStats/ftest_SR_vs_StreamStats_ftrs.png')
plt.close(fig)
#
#https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
fig = plt.figure(figsize=(10,10))
ax  = fig.add_subplot(111)
im  = ax.imshow(mi_sr_list, cmap = 'coolwarm', alpha = 0.5)
ax.set_xticks(np.arange(len(strm_ftrs_list)), labels=strm_ftrs_list)
ax.set_yticks(np.arange(len(comp_list)), labels=comp_list)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
#
for i in range(len(comp_list)): # Loop over data dimensions and create text annotations.
    for j in range(len(strm_ftrs_list)):
        #print(j, i, mi_sr_list[i,j])
        text = ax.text(j, i, '{0:.2g}'.format(mi_sr_list[i,j]), \
                        ha="center", va="center", color="k")
ax.set_title("MI-based feature importance on species richness vs. StreamStats data features")
fig.tight_layout()
plt.savefig(path + 'Plots_StreamStats/mi_SR_vs_StreamStats_ftrs.png')
plt.close(fig)

#**********************************************************;
#  3b. RF and SHAPley value plots for 9 compounds and sum  ;
#**********************************************************;
#https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
fig = plt.figure(figsize=(10,10))
ax  = fig.add_subplot(111)
im  = ax.imshow(rf_sr_list, cmap = 'coolwarm', alpha = 0.5)
ax.set_xticks(np.arange(len(strm_ftrs_list)), labels=strm_ftrs_list)
ax.set_yticks(np.arange(len(comp_list)), labels=comp_list)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
#
for i in range(len(comp_list)): # Loop over data dimensions and create text annotations.
    for j in range(len(strm_ftrs_list)):
        #print(j, i, rf_sr_list[i,j])
        text = ax.text(j, i, '{0:.2g}'.format(rf_sr_list[i,j]), \
                        ha="center", va="center", color="k")
ax.set_title("RF-based feature importance on species richness vs. StreamStats data features")
fig.tight_layout()
plt.savefig(path + 'Plots_StreamStats/rf_SR_vs_StreamStats_ftrs.png')
plt.close(fig)
#
#https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
fig = plt.figure(figsize=(10,10))
ax  = fig.add_subplot(111)
im  = ax.imshow(shap_sr_list, cmap = 'coolwarm', alpha = 0.5)
ax.set_xticks(np.arange(len(strm_ftrs_list)), labels=strm_ftrs_list)
ax.set_yticks(np.arange(len(comp_list)), labels=comp_list)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
#
for i in range(len(comp_list)): # Loop over data dimensions and create text annotations.
    for j in range(len(strm_ftrs_list)):
        #print(j, i, shap_sr_list[i,j])
        text = ax.text(j, i, '{0:.2g}'.format(shap_sr_list[i,j]), \
                        ha="center", va="center", color="k")
ax.set_title("SHAPley-based feature importance on species richness vs. StreamStats data features")
fig.tight_layout()
plt.savefig(path + 'Plots_StreamStats/shap_SR_vs_StreamStats_ftrs.png')
plt.close(fig)

#***********************************************************************;
#  3c. RF and SHAPley value plots for 9 compounds and sum (normalized)  ;
#***********************************************************************;
#https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
fig = plt.figure(figsize=(10,10))
ax  = fig.add_subplot(111)
im  = ax.imshow(rfnorm_sr_list, cmap = 'coolwarm', alpha = 0.5)
ax.set_xticks(np.arange(len(strm_ftrs_list)), labels=strm_ftrs_list)
ax.set_yticks(np.arange(len(comp_list)), labels=comp_list)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
#
for i in range(len(comp_list)): # Loop over data dimensions and create text annotations.
    for j in range(len(strm_ftrs_list)):
        #print(j, i, rfnorm_sr_list[i,j])
        text = ax.text(j, i, '{0:.2g}'.format(rfnorm_sr_list[i,j]), \
                        ha="center", va="center", color="k")
ax.set_title("RF-based feature importance (normalized) on species richness vs. StreamStats data features")
fig.tight_layout()
plt.savefig(path + 'Plots_StreamStats/rfnorm_SR_vs_StreamStats_ftrs.png')
plt.close(fig)
#
#https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
fig = plt.figure(figsize=(10,10))
ax  = fig.add_subplot(111)
im  = ax.imshow(shapnorm_sr_list, cmap = 'coolwarm', alpha = 0.5)
ax.set_xticks(np.arange(len(strm_ftrs_list)), labels=strm_ftrs_list)
ax.set_yticks(np.arange(len(comp_list)), labels=comp_list)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
#
for i in range(len(comp_list)): # Loop over data dimensions and create text annotations.
    for j in range(len(strm_ftrs_list)):
        #print(j, i, shapnorm_sr_list[i,j])
        text = ax.text(j, i, '{0:.2g}'.format(shapnorm_sr_list[i,j]), \
                        ha="center", va="center", color="k")
ax.set_title("SHAPley-based feature importance (normalized) on species richness vs. StreamStats data features")
fig.tight_layout()
plt.savefig(path + 'Plots_StreamStats/shapnorm_SR_vs_StreamStats_ftrs.png')
plt.close(fig)

#********************************************************************************;
#  3d. F-test, MI, RF, SHAPley, pcorr, scorr, and avg-value feature importances  ;
#      (Each one is normalized from 0 to 1)                                      ;
#********************************************************************************;
#https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
fig = plt.figure(figsize=(10,10))
ax  = fig.add_subplot(111)
im  = ax.imshow(ftrs_imp, cmap = 'coolwarm', alpha = 0.5)
ax.set_xticks(np.arange(len(strm_ftrs_list)), labels=strm_ftrs_list)
ax.set_yticks(np.arange(7), labels=['F-test', 'MI', 'RF', 'SHAPley', 'Pearson', 'Spearsman', 'Avg-value'])
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
#
for i in range(7): # Loop over data dimensions and create text annotations.
    for j in range(len(strm_ftrs_list)):
        #print(j, i, ftrs_imp[i,j])
        text = ax.text(j, i, '{0:.2g}'.format(ftrs_imp[i,j]), \
                        ha="center", va="center", color="k")
ax.set_title("Feature importance (normalized) -- species richness vs. StreamStats data features")
fig.tight_layout()
plt.savefig(path + 'Plots_StreamStats/Z_imp_SR_vs_StreamStats_ftrs.png')
plt.savefig(path + 'Plots_StreamStats/SVG_figs/Z_imp_SR_vs_StreamStats_ftrs.svg')
plt.savefig(path + 'Plots_StreamStats/SVG_figs/Z_imp_SR_vs_StreamStats_ftrs.pdf')
plt.close(fig)

#*********************************************************************************************;
#  3e. Min, Max, Mean, and STD for F-test, MI, RF, SHAPley, pcorr, scorr feature importances  ;
#      (Each one is normalized from 0 to 1)                                                   ;
#*********************************************************************************************;
temp_data             = copy.deepcopy(ftrs_imp[:-1,:]) #(6, 8)
mean_ftrs_imp         = np.mean(temp_data, axis = 0) #(8,)
std_ftrs_imp          = np.std(temp_data, axis = 0) #(8,)
min_ftrs_imp          = np.min(temp_data, axis = 0) #(8,)
max_ftrs_imp          = np.max(temp_data, axis = 0) #(8,)
#
metrics_ftrs_imp      = np.zeros((4,len(strm_ftrs_list)), dtype = float) #(4, 8)
metrics_ftrs_imp[0,:] = copy.deepcopy(mean_ftrs_imp) #mean #(8,)
metrics_ftrs_imp[1,:] = copy.deepcopy(std_ftrs_imp) #std #(8,)
metrics_ftrs_imp[2,:] = copy.deepcopy(min_ftrs_imp) #min #(8,)
metrics_ftrs_imp[3,:] = copy.deepcopy(max_ftrs_imp) #max #(8,)
#
row_names             = ['mean', 'std', 'min', 'max']
df_metrics_ftrs_imp   = pd.DataFrame(metrics_ftrs_imp, columns = strm_ftrs_list, index = row_names)
df_metrics_ftrs_imp.to_csv(path+ 'Plots_StreamStats/Feature_List/StreamStats_ftrs_min_max_mean_std.csv')
#!/usr/bin/env python
# coding: utf-8

# # Model training and testing for Magneticum cluster mass estimates
# 
# In this notebook, we load in our relevant data points: masses, luminosities, and morphological parameters.
# The luminosities are processed in another notebook in order to get core-excised luminosities (15% inner) from the FITS files without background.
# 
# We use Ridge Regression, Ordinary Linear Regression, Lasso Regression, and Random Forest Regression.
# 
# http://www.magneticum.org/simulations.html

# RF model is greatly improved by using logarithmic quantities, verify if it is affected by scaling the data first.

# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import scatter_matrix
import hashlib
from useful_functions.plotter import plot, loglogplot
from random import shuffle
from collections import defaultdict
from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import RFE


# In[3]:


data_dir= '../data/'
fig_dir = '../figs/'


# In[4]:


#TODO: once we get the box0 data, make sure that the uids used for the originals remain the same,
#and that the box0 uids are different, so that our test set items aren't moved over.

#could we consider adding errors in as features?


# In[5]:


#global flags
use_rfe = False
run_rf_cv = False
scale_features = True


# In[6]:


#various functions used throughout

def scatter(M1,M2, robust=True):
    if(robust):
        return (np.percentile(np.log(M1/M2),84) - np.percentile(np.log(M1/M2),16)) / 2.,                (np.percentile(np.log(M1/M2),97.72) - np.percentile(np.log(M1/M2),2.28)) / 2.
    else:
        return np.std(np.log(M1/M2))

def scatter_prelog(lnM1, lnM2, robust=True):
    if(robust):
        return (np.percentile(lnM1 - lnM2,84) - np.percentile(lnM1 - lnM2,16)) / 2.,                (np.percentile(lnM1 - lnM2,97.72) - np.percentile(lnM1 - lnM2,2.28)) / 2.
    else:
        return np.std(lnM1 - lnM2)

def test_set_check(identifier, test_ratio, hash):
    return hash(np.int64(identifier)).digest()[-1] > 256 * (1. - test_ratio)

#assumes the index of the dataframe is the unique identifier, which it is in the present case
def split_train_test_by_id(data, test_ratio, id_column, hash=hashlib.md5):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
    return data.loc[~in_test_set], data.loc[in_test_set]

def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} dex.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    return accuracy


# In[7]:


magneticum_data = np.load(data_dir+'clusterList.npy')
magneticum_data = pd.DataFrame(data=magneticum_data, columns=magneticum_data.dtype.names)
magneticum_data = magneticum_data.set_index('id')

#load in excised Lx
excised_lx = np.loadtxt(data_dir+'excised_lx.txt')
#make sure that this is the up to date one, we may want to eventually change naming once we have mock obs Lx vals
excised_lx = pd.DataFrame(data=excised_lx, columns=['id', 'excised_Lx_ergs'])
excised_lx = excised_lx.set_index('id')

magneticum_data = magneticum_data.join(excised_lx)

N_clusters = len(magneticum_data.index)
N_unique = len(np.unique(magneticum_data['uid']))
print(N_clusters)
print(N_unique)


# In[8]:


#load the morphological parameters and line them up properly with the dataframe

morph_parms = []

for name in magneticum_data.index:
    morph_parms.append(np.insert(np.loadtxt(data_dir+'parameters/%d.txt' % name, 
                                            skiprows=1, usecols=(1,2)).flatten(),0,name))
    
mp_names = [x.split(' ')[0] for x in open(data_dir+'parameters/%d.txt' % magneticum_data.index[0]).readlines()][1:]
full_mp_names = ['id']
for nm in mp_names:
    full_mp_names.append(nm)
    full_mp_names.append(nm+'_error')

assert(full_mp_names[1] == "Xray-peak")
assert(full_mp_names[2] == "Xray-peak_error")
assert(full_mp_names[3] == "centroid")
assert(full_mp_names[4] == "centroid_error")
full_mp_names[1] = "Xray-peak_x"
full_mp_names[2] = "Xray-peak_y"
full_mp_names[3] = "centroid_x"
full_mp_names[4] = "centroid_y"
    
morph_parms = np.array(morph_parms)
morph_parms = pd.DataFrame(data=morph_parms, columns=full_mp_names).set_index('id')


# ### Morphological Parameter Definitions
# 
# + concentration
# + ww = centroid-shift 
# + P30 = third moment of the power ratio
# + P40 = fourth moment of the power ratio
# + ell = ellipticity
# + PA = position angle
# + P20 = second moment of the power ratio (an alternative of the ellipticity estimate)
# + P10 = first moment of the power ratio
# + M20 = the normalized second order moment of the brightest 20% of the cluster flux
# + Cas = concentration as defined in Lotz+04
# + cAs = asymmetry as defined in Lotz+04
# + caS = smoothness as defined in Lotz+04

# In[9]:


#all the data for each cluster is stored in a properly-labelled pandas dataframe, so combine
magneticum_data = magneticum_data.join(morph_parms)

#xray-peak-centroid offset in kpch, correlates with r=0.26 to logM500
#TODO: verify that we want to add this in as a parameter
magneticum_data['centroid_xr_dist'] = (np.sqrt((magneticum_data['centroid_x'] - magneticum_data['Xray-peak_x'])**2 +
         (magneticum_data['centroid_y'] - magneticum_data['Xray-peak_y'])**2) 
 / magneticum_data['R500_pixel']) * magneticum_data['r500_kpch'] / (1. + magneticum_data['redshift']) #physical kpch


# In[10]:


np.corrcoef(np.log10(magneticum_data['M500_msolh'][magneticum_data['redshift']==0.1]), np.log10(magneticum_data['R500_pixel'][magneticum_data['redshift']==0.1]))


# In[11]:


np.corrcoef(np.log10(magneticum_data['M500_msolh'][magneticum_data['redshift']==0.1]), magneticum_data['caS'][magneticum_data['redshift']==0.1])


# In[13]:


plt.plot(np.log10(magneticum_data['M500_msolh'][magneticum_data['redshift']==0.1]), magneticum_data['concentration'][magneticum_data['redshift']==0.1], '.')
plt.xlabel('log mass')
plt.ylabel('$c$')
#plt.savefig('c_vs_mass_z0.1.pdf', bbox_inches='tight')


# In[14]:


plt.plot(np.log10(magneticum_data['M500_msolh'][magneticum_data['redshift']==0.1]), magneticum_data['caS'][magneticum_data['redshift']==0.1], '.')
plt.xlabel('log mass')
plt.ylabel('caS')
#plt.savefig('cas_vs_mass_z0.1.pdf', bbox_inches='tight')


# In[15]:


#magneticum_data[magneticum_data['redshift'] == 0.10].sort_values(by='ww')[['concentration','caS','ww', 'M500_msolh', 'R500_pixel', 'redshift']][-200:]


# In[16]:


#morphological parameter distributions
plt.rc('text', usetex=False)
morph_parms.hist(bins=50, figsize=(20,20), histtype="step",color='k', linewidth=1.2);
plt.savefig(fig_dir+'param_dists.pdf',bbox_inches='tight')
plt.show()


# In[17]:


#generate the relevant logged categories: mass, Lx, power ratios
magneticum_data['logM500'] = np.log10(magneticum_data['M500_msolh'])
magneticum_data['logLx_ex'] = np.log10(magneticum_data['excised_Lx_ergs'])
magneticum_data['logLx'] = np.log10(magneticum_data['Lx_ergs'])
magneticum_data['logPR10'] = np.log10(magneticum_data['P10'])
magneticum_data['logPR20'] = np.log10(magneticum_data['P20'])
magneticum_data['logPR30'] = np.log10(magneticum_data['P30'])
magneticum_data['logPR40'] = np.log10(magneticum_data['P40'])


# In[496]:


corr_matrix = magneticum_data.corr()
corr_matrix['M500_msolh'].sort_values(ascending=False)


# In[18]:


#need to make sure this lines up with the tex columns
cats_to_keep = ['uid', 'logM500', 'logLx_ex', 'logLx', 'logPR10', 'logPR20', 'logPR30', 'logPR40',
                'centroid_xr_dist', 'concentration', 'M20', 'Cas', 'cAs', 'caS', 'ww', 'ell', 'PA', 'redshift']
train_columns_tex = pd.Index([r'Excised $L_x$', r'$L_x$', r'$\log{P_{10}}$', r'$\log{P_{20}}$', r'$\log{P_{30}}$', 
                              r'$\log{P_{40}}$', r'$d_{xc}$', '$c$', r'$M_{20}$', r'$C$', r'$A$', r'$S$', r'$w$', 
                              r'$e$', r'$\eta$', r'$z$'])
#TODO: Make this a dictionary so there are no issues with indices being mismatched

#only keep the categories from above
magneticum_data = magneticum_data[cats_to_keep]


# In[19]:


lin_reg = LinearRegression()
lin_reg.fit(magneticum_data['logLx_ex'].reshape(-1,1), magneticum_data['logM500'])
predictions = lin_reg.predict(magneticum_data['logLx_ex'].reshape(-1,1))
resids = predictions - magneticum_data['logM500'] #pred minus true
mg_dat_v2 = magneticum_data.copy()
mg_dat_v2['resids'] = resids
corr_matrix = mg_dat_v2.corr()
print(corr_matrix['logM500'].sort_values(ascending=True))
print(corr_matrix['resids'].sort_values(ascending=True))
#the mass itself correlates strongly with residual, which means that it's not unbiased
#expect this to go down once we get a truly flat mass function


# In[20]:


#print distributions of parameters that will be used in analysis
plt.rc('text', usetex=False)
magneticum_data.hist(bins=50, figsize=(20,20), histtype="step",color='k', linewidth=1.2);
plt.savefig(fig_dir+'feature_dists.pdf',bbox_inches='tight')
plt.show()


# In[21]:


#generate train and test set based on uids to keep same cluster across snapshots in either train or test
train_set, test_set = split_train_test_by_id(magneticum_data, test_ratio=0.2, id_column='uid')
print(train_set.shape, test_set.shape)

#let's make copies of the relevant parts of the train and test sets
train = train_set.drop(['logM500', 'uid'], axis=1)
train_columns = train.columns
train_labels = train_set['logM500']
train_uids = train_set['uid']

test = test_set.drop(['logM500', 'uid'], axis=1)
test_labels = test_set['logM500']

#Generate groups on the train set for group k-fold CV (s.t. each uid halo is only in one group)
n_folds = 10
fold = train_uids % n_folds
logo_iter = list(LeaveOneGroupOut().split(train,train_labels,fold))


# In[22]:


fig,ax = plot()

print("\n Fold\t", end=" ")
for i in range(0,n_folds):
    print(str(i)+'\t', end =" ")
print("\n Count\t", end=" ")
for i in range(0, n_folds):
    print(str(list(fold).count(i))+'\t', end=" ")
    plt.hist(train_labels[fold == i], density='normed', alpha=0.6)


# In[23]:


fig, ax = plot()
plt.hist(magneticum_data['logM500'],30,alpha=0.5, density='normed',label='Full dataset')
plt.hist(test_set['logM500'],30,alpha=0.5, density='normed', label='Test set')
plt.legend()
plt.xlabel(r'$\log_{10}(M)$')
plt.ylabel('Normed pdf');
#is this an unbiased enough sample?


# In[24]:


#scale columns of train set based on median and 16/84 quanties, scale test set by same numbers
if(scale_features):
    rs = RobustScaler(quantile_range=(16.0, 84.0))
    train = rs.fit_transform(train)
    test = rs.transform(test)


# In[25]:


#RFE
#TODO: try RandomForestRegression() instead
reg_alg = LinearRegression()

rfe = RFE(reg_alg,1,1)
rfe.fit(train, train_labels)
rnk = rfe.ranking_
print(train_columns[rfe.support_])
print(rfe.ranking_)

print(train_columns[np.argsort(rnk)])

r2_vals = []

for i in range(1, len(train_columns)+1):
    rfe = RFE(reg_alg,i,1)
    rfe.fit(train, train_labels)
    r2_vals.extend([rfe.score(train, train_labels)])
    
fig, ax = plot(figsize=(10,5))
plt.plot(range(1,len(train_columns)+1), r2_vals)
plt.xlabel('Number of features')
plt.ylabel(r'$r^2$')

ax2 = ax.twiny()
ax2.set_xlim(ax.get_xlim())
ax2.set_xticks(range(1,len(train_columns)+1));
ax2.set_xticklabels(train_columns_tex[np.argsort(rnk)]);
plt.xticks(rotation=50)
#plt.savefig('rfe.pdf',bbox_inches='tight')


# In[26]:


#code to remove non-important features from our listing
if(use_rfe):
    n_features_to_keep = 10
    reg_alg = LinearRegression()

    rfe = RFE(reg_alg,n_features_to_keep,1)
    rfe.fit(train, train_labels)
    train = train[:,rfe.support_]
    train_columns = train_columns[rfe.support_]
    test = test[:,rfe.support_]


# ## Model fitting and testing
# 
# Part 1: $M-L_x$ Relationship
# 
# ### TODO: Generate list of Maughan curves and plot scatter distribution like in Ntampaka et al. 2018
# 
# ### TODO: Try different cost functions for minimization instead of mean squared error.

# In[27]:


#let's get the final number for a simple mass-Lx regression on the training data used to predict the test data
#NOTE: we have not yet introduced redshift dependence here, as is done in Maughan
lin_reg = LinearRegression()
lin_reg.fit(train[:,train_columns == 'logLx_ex'], train_labels)
test_predictions = lin_reg.predict(test[:,train_columns == 'logLx_ex'])
train_predictions = lin_reg.predict(train[:, train_columns == 'logLx_ex'])
m_lx_scatter_robust, m_lx_sc_rb_2sig = scatter_prelog(test_predictions, test_labels)
m_lx_scatter_std = scatter_prelog(test_predictions, test_labels, False)
print("The (robust) scatter for a simple mass-luminosity relationship is %.3f dex" % m_lx_scatter_robust)
print("Robust 2-sigma scatter: %.3f dex" % m_lx_sc_rb_2sig)
print("The (std) scatter for a simple mass-luminosity relationship is %.3f dex" % m_lx_scatter_std)


# Part 2: Ordinary Linear Regression

# In[28]:


lin_reg = LinearRegression()
lin_reg.fit(train, train_labels)
test_predictions = lin_reg.predict(test)
lin_reg_scatter_robust, lin_reg_sc_rb_2sig = scatter_prelog(test_predictions, test_labels)
lin_reg_scatter_std = scatter_prelog(test_predictions, test_labels, False)
print("The (robust) scatter for a LR model is %.3f dex" % lin_reg_scatter_robust)
print("Robust 2-sigma scatter: %.3f dex" % lin_reg_sc_rb_2sig)
print("The (std) scatter for a LR model is %.3f dex" % lin_reg_scatter_std)

#TODO: we need to quantify how this changes once we only use the n most important parameters
#TODO: do a plot similar to RFE, but which shows the *scatter* as a function of features remaining


# Part 3: Ridge Regression

# In[29]:


#k-fold CV
rr = Ridge()
param_grid = [{'alpha': np.logspace(-4,5,50)}]
grid_search = GridSearchCV(rr, param_grid, 
                           cv=logo_iter, scoring='neg_mean_squared_error')
grid_search.fit(train, train_labels)
print(grid_search.best_params_)
best_alpha_rr = grid_search.best_params_['alpha']

#train and test final optimized model
rr = Ridge(alpha=best_alpha_rr)
rr.fit(train, train_labels)
rr_preds = rr.predict(test)
rr_scatter_robust, rr_sc_rb_2sig = scatter_prelog(rr_preds, test_labels)
rr_scatter_std = scatter_prelog(rr_preds, test_labels, False)
print("The (robust) scatter for a RR model is %.3f dex" % rr_scatter_robust)
print("Robust 2-sigma scatter: %.3f dex" % rr_sc_rb_2sig)
print("The (std) scatter for a RR model is %.3f dex" % rr_scatter_std)


# Part 4: Lasso Regression

# In[30]:


#k-fold CV
lr = Lasso()
param_grid = [{'alpha': np.logspace(-8,5,50)}]
grid_search = GridSearchCV(lr, param_grid, 
                           cv=logo_iter, scoring='neg_mean_squared_error')
grid_search.fit(train, train_labels)
print(grid_search.best_params_)
best_alpha_lr = grid_search.best_params_['alpha']

#train and test final optimized model
lr = Lasso(alpha=best_alpha_lr)
lr.fit(train, train_labels)
lr_preds = lr.predict(test)
lr_scatter_robust, lr_sc_rb_2sig = scatter_prelog(lr_preds, test_labels)
lr_scatter_std = scatter_prelog(lr_preds, test_labels, False)
print("The (robust) scatter for a LR model is %.3f dex" % lr_scatter_robust)
print("Robust 2-sigma scatter: %.3f dex" % lr_sc_rb_2sig)
print("The (std) scatter for a LR model is %.3f dex" % lr_scatter_std)


# Part 5: Random Forest Regression

# In[31]:


#randomized search CV for optimizing RF hyperparameters
#NOTE: this takes a long time to run, so will re-run it again once we have final feature sets and data
#notes on cross-validation: https://machinelearningmastery.com/train-final-machine-learning-model/
rf = RandomForestRegressor()
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
max_depth.append(5)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 20]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4, 8, 10]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


# In[33]:


#run the randomized search
if(run_rf_cv):
    rand_search = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, 
                                 cv = logo_iter, verbose=2, random_state=42, n_jobs = -1, 
                                 scoring='neg_mean_squared_error')
    rand_search.fit(train, train_labels)
    rf_params = rand_search.best_params_
    best_model = rand_search.best_estimator_
    print(rand_search.best_params_)
#TODO: after we run randomized search on our full dataset,
#try narrowing down one more step with an additional grid search


# In[34]:


#RF without cross-validation, default hyperparameters
rf = RandomForestRegressor(random_state=42)

rf.fit(train, train_labels)
rf_preds = rf.predict(test)
rf_scatter_robust, rf_sc_rb_2sig = scatter_prelog(rf_preds, test_labels)
rf_scatter_std = scatter_prelog(rf_preds, test_labels, False)
print("The (robust) scatter for a RF model is %.3f dex" % rf_scatter_robust)
print("Robust 2-sigma scatter: %.3f dex" % rf_sc_rb_2sig)
print("The (std) scatter for a RF model is %.3f dex" % rf_scatter_std)


# In[35]:


#use the best set of hyperparameters to train RF model on whole train set, test on test set
#TODO: go back and do RFE using RF since we have an idea of the hyperparameter values
#rf = RandomForestRegressor(random_state=42, bootstrap=True, max_depth=90, max_features='auto', min_samples_leaf=2,
#                           min_samples_split=2, n_estimators=2000)
rf = RandomForestRegressor(random_state=42, bootstrap=False, max_depth=70, max_features='sqrt', min_samples_leaf=2,
                           min_samples_split=2, n_estimators=2000)

rf.fit(train, train_labels)
rf_preds = rf.predict(test)
rf_scatter_robust, rf_sc_rb_2sig = scatter_prelog(rf_preds, test_labels)
rf_scatter_std = scatter_prelog(rf_preds, test_labels, False)
print("The (robust) scatter for a RF model is %.3f dex" % rf_scatter_robust)
print("Robust 2-sigma scatter: %.3f dex" % rf_sc_rb_2sig)
print("The (std) scatter for a RF model is %.3f dex" % rf_scatter_std)
#no optimization, 4.5% scatter on the test set...
#now we can use grid-search CV on the train set to get the best value
#10.2% scatter if we just use logLx_ex
#still 4.5% scatter on test set if we use the cross-validated model
#going from 8.9 to 5% scatter is a pretty solid reduction, and this is with only 7 features... what about all?
#only a slight change depending on if we use 7 features or all (4.8 vs 4.5% scatter)
#if we use all features, cross-validated, we get 4.5% scatter!!!
#If we replace cshift with my centroid-xray peak distance in physical units, we get !!3.4% scatter with bias -0.002
#4.43% scatter with best_estimator_
#when we include redshift and my centroid-xray peak distance we get 2.8% scatter with bias of -0.0011
#if we then also add the non-core-excised value, we get scatter of 2.7% w/ bias -0.0021

#model is greatly improved if we log the necessary columns


# In[36]:


#look at feature importance for RF
for i in range(0,len(train_columns)):
    print('%.4f \t %s' % (rf.feature_importances_[i], train_columns[i]))


# In[37]:


#calculate RF model performance (bias from mean vs. median?)
evaluate(rf, test, test_labels)
robust_bias = np.median(rf_preds - test_labels)
bias = np.mean(rf_preds - test_labels)
print('The final RF model applied to test set had a (robust) bias of %.4f dex' % robust_bias)
print('The final RF model applied to test set had a bias of %.4f dex' % bias)


# In[38]:


#compare RF predicted masses to true
fig, ax = plot()
plt.plot(test_labels, rf_preds, '.', markersize=10, alpha=0.75)
plt.xlabel(r'$\log[M_{500c,\textrm{true}} / (M_\odot h^{-1})]$');
plt.ylabel(r'$\log[M_{500c,\textrm{pred}} / (M_\odot h^{-1})]$');
plt.plot([12,16], [12,16], 'k')

#see how close to 1-1 the predictions are, i.e., what is the slope?
lin_reg = LinearRegression()
lin_reg.fit(test_labels.values.reshape(-1,1), rf_preds)

#should add the scatter and the regression fit
plt.text(13.6,14.7,r'$\delta_{RF} = %.4f$' % rf_scatter_robust)
plt.text(13.6,14.6,r'$\mu_{RF} = %.4f$' % robust_bias)
plt.text(13.6,14.5,r'$A = %.2f$' % lin_reg.coef_[0])
plt.text(13.6,14.4,r'$B = %.2f$' % lin_reg.intercept_)

xv = np.linspace(12, 16, 50)
plt.plot(xv, xv*lin_reg.coef_[0] + lin_reg.intercept_, 'r', linestyle='dashed')

plt.xlim(13.4, 14.9)
plt.ylim(13.4, 14.9)

print("The predicted-to-true intercept is %.3f" % lin_reg.intercept_)
print("The predicted-to-true slope is %.3f" % lin_reg.coef_[0])
if(lin_reg.coef_[0] < 1):
    print("High masses will be underpredicted!")
else:
    print("High masses will be overpredicted!")
    
plt.savefig(fig_dir+'mpred_vs_mtrue.pdf', bbox_inches='tight')

#TODO: figure out what is going on with the outliers


# In[39]:


#look at distribution of residuals for final RF model
fig, ax = plot()
plt.hist(rf_preds - test_labels, histtype='step', color='k', 
         linewidth=1.2, bins=30, density="normed", label="4.2\%")
plt.xlabel(r'$\log(M_{500c,\textrm{pred}}) - \log(M_{500c,\textrm{true}})$')
plt.ylabel('PDF');
plt.legend()

plt.savefig(fig_dir+'error_dist.pdf', bbox_inches='tight')


# ## Up next: AdaBoost and Gradient Boosted Trees
# 
# They've been "implemented" below, but have not yet been tuned using CV to find the best hyperparameters.

# In[40]:


gb = GradientBoostingRegressor(random_state=42)

gb.fit(train, train_labels)
gb_preds = gb.predict(test)
gb_scatter_robust = scatter_prelog(gb_preds, test_labels)
gb_scatter_std = scatter_prelog(gb_preds, test_labels, False)
print("The (robust) scatter for a GB model is %.3f dex" % gb_scatter_robust)
print("The (std) scatter for a GB model is %.3f dex" % gb_scatter_std)

#in the cross-validation phase, need to try early stopping here (either using warm-start or staged predict)


# In[ ]:


ab = AdaBoostRegressor(random_state=42)

ab.fit(train, train_labels)
ab_preds = ab.predict(test)
ab_scatter_robust = scatter_prelog(ab_preds, test_labels)
ab_scatter_std = scatter_prelog(ab_preds, test_labels, False)
print("The (robust) scatter for a AB model is %.3f dex" % ab_scatter_robust)
print("The (std) scatter for a AB model is %.3f dex" % ab_scatter_std)


# ## Could also consider using PCA instead of RFE, then keep the first N PCs and run RF on these... May be worth seeing if this reduces model complexity and scatter?
# 
# ## Compare the out-of-bag accuracy on the RandomForest to the CV error and see how similar they are
# 
# ## Look into ExtraTreesRegressor for a model that uses random thresholds for each feature rather than searching for the best possible threshold, trades even further a larger bias for lower variance, which is what we want!
# 
# ## Should we look at the cross-validation error on the best hyperparameters and compare that to the test scatter? Is picking the model with the best test scatter the best way to do the final model selection?
# 
# ## TODO: Implement CV method for hyperparameter optimizing for RR, LR, AB, and GB

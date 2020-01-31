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

# ## Setup and Preprocessing

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
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
from astropy.io import fits
import matplotlib.cm as cm
from pathlib import Path
from os.path import expanduser


# In[2]:


#could we consider adding errors in as features?
#could also consider adding back in some of the features that we threw out for use in Chandra set
#could look into weighting the samples by some metric inversely proportional to errors?


# In[3]:


#global flags
use_rfe = False
run_rf_cv = True
run_gb_cv = True
scale_features = True
idealized = True
compare_dists = False
instrument = 'erosita'


# In[4]:


data_dir= Path('../data/')
fig_dir = Path('../figs/')
home_dir = Path(expanduser('~'))
chandra_dir = home_dir / 'magneticum_chandra'
erosita_no_bg_dir = home_dir / 'magneticum_no_bg'

if(compare_dists):
    instrument = 'chandra'
    idealized  = True #overriding anything above

if(idealized):
    if(instrument == 'chandra'):
        param_dir = 'parchandra/'
    else:
        param_dir = 'parameters/' #change this to a folder called "idealized_parameters" or something...
else:
    param_dir = 'parameters1500/'

print("Using %s" %param_dir)

if(instrument == 'chandra'):
    print('Using Chandra')
    use_dir = chandra_dir
elif(instrument == 'erosita'):
    print('Using eROSITA no bg')
    use_dir = erosita_no_bg_dir


# ### Generating cluster catalog file for Lorenzo

# In[33]:


data = np.load(data_dir / (instrument+'_clusterList.npy'))


# In[34]:


#no scatter
h=0.704 #from Magneticum data
r500_kpc_phys = (data['r500_kpch'] / h) / (1. + data['redshift']) #was originally comoving kpc/h, so divide by h*(1+z)
for_lorenzo = np.column_stack((data['id'],r500_kpc_phys, data['R500_pixel'], data['redshift']))
np.savetxt(data_dir / (instrument+'_cluster_data.dat'),for_lorenzo,fmt=['%d','%f','%f','%f'])


# ### Computing core-excised luminosities
# 
# Only would need to be run if we added more clusters, since it computes the core-excised luminosity from the Magneticum luminosity using the no-background mock image.

# In[6]:


cluster_directory = use_dir
magneticum_data = np.load(use_dir / 'clusterList.npy')
# make this into a pandas data frame
magneticum_data = pd.DataFrame(data=magneticum_data, columns=magneticum_data.dtype.names)
magneticum_data = magneticum_data.set_index('id')
Nclusters = len(magneticum_data.index)


# In[7]:


def distance(x1, x2):
    return np.sqrt((x1[0] - x2[0])**2 + (x1[1] - x2[1])**2)


# In[9]:


#make an array that is 384 x 384, and each element contains the 4 corners in the cell

#so we say that we are within the radius if part of the pixel is within the radius,
#not if its center is within radius
#TODO: we should see how much things differ when we do this...
img_shape_x, img_shape_y = fits.open(cluster_directory / '922603.fits')[0].data.shape
center = np.array([img_shape_x, img_shape_y]) / 2.
cell_dists = np.ndarray((img_shape_x, img_shape_y))
for i in range(0, img_shape_x):
    print(i)
    for j in range(0, img_shape_y):
        #cell_dists[i,j] = distance(center, [i,j])
        cell_dists[i,j] = np.min([distance(center,[i,j]), distance(center,[i+1,j]), 
                                  distance(center,[i,j+1]), distance(center,[i+1,j+1])])


# In[12]:


def core_excised_Lx(cluster_id, excise_rad_frac):
    dat = fits.open(cluster_directory / ('%d.fits' % cluster_id))[0].data
    r500 = magneticum_data['R500_pixel'].loc[cluster_id]
    total_mask = (cell_dists <= r500)
    excise_mask = (cell_dists <= excise_rad_frac * r500)
    Ntot = np.sum(dat[total_mask])
    Ncore = np.sum(dat[excise_mask])
    #print((Ntot-Ncore)/Ntot)
    return (Ntot-Ncore)/Ntot * magneticum_data['Lx_ergs'].loc[cluster_id]

def core_excised_counts(cluster_id, excise_rad_frac):
    dat = fits.open(cluster_directory / ('%d.fits' % cluster_id))[0].data
    r500 = magneticum_data['R500_pixel'].loc[cluster_id]
    total_mask = (cell_dists <= r500)
    excise_mask = (cell_dists <= excise_rad_frac * r500)
    Ntot = np.sum(dat[total_mask])
    Ncore = np.sum(dat[excise_mask])
    return(Ntot-Ncore)


# In[11]:


excised_counts = np.zeros(Nclusters)
for i,cluster_id in enumerate(magneticum_data.index.values):
    if(i % 100 == 0):
        print(i)
    excised_counts[i] = core_excised_counts(cluster_id, 0.15)


# In[13]:


pix_corner_excised_Lx = np.zeros(Nclusters)
for i,cluster_id in enumerate(magneticum_data.index.values):
    if(i % 100 == 0):
        print(i)
    pix_corner_excised_Lx[i] = core_excised_Lx(cluster_id, 0.15)


# In[16]:


#now, we want to save a list of the excised Lx values and the respective cluster ids,
#to be loaded into our main file
output = np.column_stack((magneticum_data.index.values, pix_corner_excised_Lx))
np.savetxt(data_dir / (instrument+'_excised_lx.txt'),output,fmt='%d %e')


# #### Comparison between Chandra and eROSITA core-excised luminosities

# In[10]:


# let's do a comparison between chandra and erosita core-excised
erosita_ex_lx = np.loadtxt(data_dir / 'erosita_excised_lx.txt')
chandra_ex_lx = np.loadtxt(data_dir / 'chandra_excised_lx.txt')
print(erosita_ex_lx.shape)
print(chandra_ex_lx.shape)


# In[11]:


msk = np.isin(chandra_ex_lx[:,0], erosita_ex_lx[:,0])
chandra_ex_lx = chandra_ex_lx[msk]
msk = np.isin(erosita_ex_lx[:,0], chandra_ex_lx[:,0])
erosita_ex_lx = erosita_ex_lx[msk]
chandra_ex_lx = chandra_ex_lx[np.argsort(erosita_ex_lx[:,0])]
erosita_ex_lx = erosita_ex_lx[np.argsort(erosita_ex_lx[:,0])]
print(erosita_ex_lx.shape)
print(chandra_ex_lx.shape)
for i in range(0, len(chandra_ex_lx)):
    #print(i)
    assert(chandra_ex_lx[i,0] == erosita_ex_lx[i,0])


# In[12]:


loglogplot()
plt.plot(chandra_ex_lx[:,1],erosita_ex_lx[:,1],'.')

rel_err = (erosita_ex_lx[:,1] - chandra_ex_lx[:,1]) / chandra_ex_lx[:,1]

plot(semilogx=True)
plt.plot(chandra_ex_lx[:,1],rel_err,'.')

# this tells us that the chandra luminosities are basically always larger, and this should be no surprise
# since we have more pixels, we can throw out less light at the boundary

# however, i'm surprised at the percentage-level difference between the two


# ### Back to preprocessing

# In[5]:


#various functions used throughout

def scatter(M1,M2, robust=True):
    if(robust):
        return (np.percentile(np.log10(M1/M2),84) - np.percentile(np.log10(M1/M2),16)) / 2.,                (np.percentile(np.log10(M1/M2),97.72) - np.percentile(np.log10(M1/M2),2.28)) / 2.
    else:
        return np.std(np.log10(M1/M2))

def scatter_prelog(logM1, logM2, robust=True):
    if(robust):
        return (np.percentile(logM1 - logM2,84) - np.percentile(logM1 - logM2,16)) / 2.,                (np.percentile(logM1 - logM2,97.72) - np.percentile(logM1 - logM2,2.28)) / 2.
    else:
        return np.std(logM1 - logM2)

def test_set_check(identifier, test_ratio, hash):
    return hash(np.int64(identifier)).digest()[-1] > 256 * (1. - test_ratio)

# returns the negative of the 16/84 scatter, for potential use in hyperparameter grid-searching
def scorer(estimator, X, y):
    y_pred = estimator.predict(X)
    scatter_robust, scatter_robust_2sig = scatter_prelog(y, y_pred)
    return -1.* scatter_robust

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


# In[6]:


# start by finding the set of clusters that is present in both eROSITA and Chandra datasets
# we will then make an array of these IDs and make sure that either loaded in will be the same
# this way our train and test sets and everything is the exact same between datasets (since indexed by ID)

chandra_ids = np.load(data_dir / 'chandra_clusterList.npy')['id']
erosita_ids = np.load(data_dir / 'erosita_clusterList.npy')['id']
shared_ids = np.intersect1d(chandra_ids,erosita_ids)


# In[7]:


# load in data
magneticum_data = np.load(data_dir / (instrument+'_clusterList.npy'))
print(len(magneticum_data))
magneticum_data = pd.DataFrame(data=magneticum_data, columns=magneticum_data.dtype.names)
magneticum_data = magneticum_data.set_index('id')

# only use the ones present in both eROSITA and Chandra datasets for analysis
magneticum_data = magneticum_data.loc[shared_ids]

#load in excised Lx
excised_lx = np.loadtxt(data_dir / (instrument+'_excised_lx.txt'))
#make sure that this is the up to date one, we may want to eventually change naming once we have mock obs Lx vals
excised_lx = pd.DataFrame(data=excised_lx, columns=['id', 'excised_Lx_ergs'])
excised_lx = excised_lx.set_index('id')
excised_lx = excised_lx.loc[shared_ids]

magneticum_data = magneticum_data.join(excised_lx)

md = magneticum_data.copy()

N_clusters = len(magneticum_data.index)
N_unique = len(np.unique(magneticum_data['uid']))
print(N_clusters)
print(N_unique)
print(np.unique(magneticum_data['redshift']))


# In[8]:


# need to generate the mass function here and then throw stuff out...
bin_count = 13
binseq = np.logspace(13.5,14.8,bin_count + 1)
bincenters = 0.5*(binseq[1:]+binseq[:-1])
binsize=np.zeros(len(bincenters))
for j in range(0,len(bincenters)):
    binsize[j] = binseq[j+1] - binseq[j]

bins_to_use = np.log10(binseq)
    
masses = magneticum_data['M500_msolh']
print(len(masses))
massFunc = np.zeros(bin_count) #i.e. number per bin
for i in range(0,len(massFunc)):
    massFunc[i] = len(masses[np.logical_and(masses >= binseq[i], masses < binseq[i+1])])

#now, loop over the bins and throw out a certain number of clusters in each mass range from
#13.7 to 14.1
np.random.seed(42)
max_in_bin = int(massFunc[0]) #226
print(np.log10(binseq))
for i in range(1,7):
    #get indices of relevant clusters
    indices_to_drop = np.random.choice(masses[np.logical_and(masses >= binseq[i], masses < binseq[i+1])].index, replace=False, size=int(massFunc[i] - max_in_bin))
    magneticum_data = magneticum_data.drop(indices_to_drop)


    


# In[9]:


masses = magneticum_data['M500_msolh']
print(len(masses))
N_unique = len(np.unique(magneticum_data['uid']))
print(N_unique)
massFunc = np.zeros(bin_count) #i.e. number per bin
for i in range(0,len(massFunc)):
    massFunc[i] = len(masses[np.logical_and(masses >= binseq[i], masses < binseq[i+1])])
    
fig, ax = plot()
n, bins, patches = plt.hist(np.log10(masses),histtype='step',linewidth=1.2, color='k', bins = bins_to_use)#int(np.sqrt(len(masses))))
print(n)
plt.ylabel(r'Clusters per 0.1 dex mass bin')
plt.xlabel(r'$\log(M_\mathrm{500c})$ $[M_\odot h^{-1}]$')
plt.savefig(fig_dir / 'mass_func.pdf',bbox_inches='tight')


# In[10]:


#load the morphological parameters and line them up properly with the dataframe

morph_parms = []
if(compare_dists):
    morph_parms_erosita = []

for name in magneticum_data.index:
    morph_parms.append(np.insert(np.loadtxt(data_dir / param_dir / ('%d.txt' % name), 
                                            skiprows=1, usecols=(1,2)).flatten(),0,name))
    if(compare_dists):
        morph_parms_erosita.append(np.insert(np.loadtxt(data_dir / 'parameters' / ('%d.txt' % name), 
                                            skiprows=1, usecols=(1,2)).flatten(),0,name))
    
mp_names = [x.split(' ')[0] for x in open(data_dir / param_dir / ('%d.txt' % magneticum_data.index[0])).readlines()][1:]
print(mp_names)
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

if(compare_dists):
    morph_parms_erosita = np.array(morph_parms_erosita)
    morph_parms_erosita = pd.DataFrame(data=morph_parms_erosita, columns=full_mp_names).set_index('id')


# In[11]:


#fixes for the inifinites due to highly concentrated (throw out Cas, degenerate with concentration)
#also, set M20 infinities to -5, which is at the tail end of the distribution
morph_parms['M20'][np.isinf(morph_parms['M20'])] = -6
morph_parms = morph_parms.drop('Cas',axis=1)
morph_parms = morph_parms.drop('M20_error',axis=1)

if(compare_dists):
    morph_parms_erosita['M20'][np.isinf(morph_parms_erosita['M20'])] = -6
    morph_parms_erosita = morph_parms_erosita.drop('Cas',axis=1)
    morph_parms_erosita = morph_parms_erosita.drop('M20_error',axis=1)


# Morphological Parameter Definitions
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

# In[12]:


#all the data for each cluster is stored in a properly-labelled pandas dataframe, so combine
magneticum_data = magneticum_data.join(morph_parms)


# In[13]:


# TODO: See if this works properly with eROSITA... doubt it, probably still encodes too much info

##we don't need this stuff anymore, don't run it
#xray-peak-centroid offset in kpch, correlates with r=0.26 to logM500

'''
magneticum_data['centroid_xr_dist'] = \
(np.sqrt((magneticum_data['centroid_x'] - magneticum_data['Xray-peak_x'])**2 +
         (magneticum_data['centroid_y'] - magneticum_data['Xray-peak_y'])**2) 
 / magneticum_data['R500_pixel']) * magneticum_data['r500_kpch'] / (1. + magneticum_data['redshift']) #physical kpch
'''


# In[13]:


#morphological parameter distributions
plt.rc('text', usetex=False)
morph_parms.hist(bins=50, figsize=(20,20), histtype="step",color='k', linewidth=1.2);
plt.savefig(fig_dir / 'param_dists.pdf',bbox_inches='tight')
plt.show()


# In[14]:


# replacing the core-excised luminosity with the evolution-scaled core-excised luminosity
#WMAP 7 parameters
omega_l = 0.728
omega_m = 0.272
h=0.704

def E(z):
    return np.sqrt((1+z)**3 * omega_m + omega_l)

magneticum_data['excised_Lx_ergs'] = magneticum_data['excised_Lx_ergs'] * E(magneticum_data['redshift'])**(-7./3.)
#from self-similar scaling


# In[15]:


#generate the relevant logged categories: mass, Lx, power ratios
magneticum_data['logM500'] = np.log10(magneticum_data['M500_msolh'])
magneticum_data['logLx_ex'] = np.log10(magneticum_data['excised_Lx_ergs'])
magneticum_data['logLx'] = np.log10(magneticum_data['Lx_ergs'])
magneticum_data['logPR10'] = np.log10(magneticum_data['P10'])
magneticum_data['logPR20'] = np.log10(magneticum_data['P20'])
magneticum_data['logPR30'] = np.log10(magneticum_data['P30'])
magneticum_data['logPR40'] = np.log10(magneticum_data['P40'])
magneticum_data['logw'] = np.log10(magneticum_data['ww'])
#magneticum_data['logdxc'] = np.log10(magneticum_data['centroid_xr_dist'])

if(compare_dists):
    morph_parms_erosita['logPR10'] = np.log10(morph_parms_erosita['P10'])
    morph_parms_erosita['logPR20'] = np.log10(morph_parms_erosita['P20'])
    morph_parms_erosita['logPR30'] = np.log10(morph_parms_erosita['P30'])
    morph_parms_erosita['logPR40'] = np.log10(morph_parms_erosita['P40'])
    morph_parms_erosita['logw'] = np.log10(morph_parms_erosita['ww'])


# ### Identifying clusters for visualization

# In[99]:


# look at fixed redshift for the clusters with the highest and lowest concentration
sset = magneticum_data[mag_redshifts == 0.1] # largest, brightest because closest
sset = sset[np.logical_and(sset['logM500'] > 14.3, sset['logM500'] < 14.6)]
large_c = np.argsort(sset['concentration'].values)[-1]
small_c = np.argsort(sset['concentration'].values)[0]
print(sset['concentration'].iloc[large_c]); print(sset.iloc[large_c])
print(sset['concentration'].iloc[small_c]); print(sset.iloc[small_c])


# In[100]:


# look at fixed redshift for the clusters with high and low asymmetry
sset = magneticum_data[mag_redshifts == 0.1] # largest, brightest because closest
sset = sset[np.logical_and(sset['logM500'] > 14.3, sset['logM500'] < 14.6)]
large_A = np.argsort(sset['cAs'].values)[-2]
small_A = np.argsort(sset['cAs'].values)[1]
print(sset['cAs'].iloc[large_A]); print(sset.iloc[large_A])
print(sset['cAs'].iloc[small_A]); print(sset.iloc[small_A])


# In[101]:


# look at fixed redshift for the clusters with high and low smoothness
sset = magneticum_data[mag_redshifts == 0.1] # largest, brightest because closest
sset = sset[np.logical_and(sset['logM500'] > 14.3, sset['logM500'] < 14.6)]
large_S = np.argsort(sset['caS'].values)[-1]
small_S = np.argsort(sset['caS'].values)[1]
print(sset['caS'].iloc[large_S]); print(sset.iloc[large_S])
print(sset['caS'].iloc[small_S]); print(sset.iloc[small_S])


# ## Parameter correlations and distributions

# In[15]:


corr_matrix = magneticum_data.corr()
corr_matrix['M500_msolh'].sort_values(ascending=False)


# In[16]:


#decide whether or not to throw out redshift?

#ALSO THROWING OUT LX AND REDSHIFT

#need to make sure this lines up with the tex columns
cats_to_keep = ['uid', 'logM500', 'logLx_ex', 'logPR10', 'logPR20', 'logPR30', 'logPR40',
                'concentration', 'M20', 'cAs', 'caS', 'logw', 'ell', 'PA']
#cats_to_keep = ['uid', 'logM500', 'logLx_ex', 'concentration', 'cAs', 'caS']
if(compare_dists):
    cats_to_keep_erosita = ['logPR10', 'logPR20', 'logPR30', 'logPR40',
                            'concentration', 'M20', 'cAs', 'caS', 'logw', 'ell', 'PA']
train_columns_tex = pd.Index([r'Excised $L_x$', r'$\log{P_{10}}$', r'$\log{P_{20}}$', r'$\log{P_{30}}$', 
                              r'$\log{P_{40}}$', '$c$', r'$M_{20}$', r'$A$', r'$S$', r'$\log{w}$', 
                              r'$e$', r'$\eta$'])
#train_columns_tex = pd.Index([r'Excised $L_x$', '$c$', r'$A$', r'$S$'])
#TODO: Make this a dictionary so there are no issues with indices being mismatched

mag_redshifts = magneticum_data['redshift']

#only keep the categories from above
magneticum_data = magneticum_data[cats_to_keep]


# In[31]:


if(compare_dists):
    for i in range(3, len(cats_to_keep)):
        fig, ax = plot()
        cr = np.corrcoef(magneticum_data['logM500'], magneticum_data[cats_to_keep[i]])
        sc = ax.scatter(magneticum_data['logM500'], magneticum_data[cats_to_keep[i]], color='k', s=3, label = 'Chandra')
        ax.scatter(magneticum_data['logM500'], morph_parms_erosita[cats_to_keep_erosita[i-3]], color='r', s=3, label='eROSITA')
        plt.xlabel('logM500')
        plt.ylabel(train_columns_tex[i-2])
        plt.legend()
        if(cats_to_keep[i] == 'cAs'):
            plt.savefig(fig_dir / ('corr_plots/ce_compare_asymmetry.png'))
        else:
            plt.savefig(fig_dir / ('corr_plots/ce_compare_%s.png' % cats_to_keep[i]))
else:
    for i in range(2, len(cats_to_keep)):
        fig, ax = plot()
        cr = np.corrcoef(magneticum_data['logM500'], magneticum_data[cats_to_keep[i]])
        sc = ax.scatter(magneticum_data['logM500'], magneticum_data[cats_to_keep[i]], c=mag_redshifts, cmap=plt.get_cmap('jet'), label = '%0.3f' % cr[0,1])
        cbar = plt.colorbar(sc)
        cbar.set_label(r'$z$')
        plt.xlabel('logM500')
        plt.ylabel(train_columns_tex[i-2])
        plt.legend()
        #if(cats_to_keep[i] == 'cAs'):
        #    plt.savefig(fig_dir / ('corr_plots/asymmetry.png'))
        #else:
        #    plt.savefig(fig_dir / ('corr_plots/%s.png' % cats_to_keep[i]))
    
# could do the same thing for only one redshift if interested

# NOTE: If we end up including these correlation plots in the final paper, we will want to crop them to disregard
# the one to two outliers that are present in any of them


# In[27]:


lin_reg = LinearRegression()
lin_reg.fit(magneticum_data['logLx_ex'].values.reshape(-1,1), magneticum_data['logM500'])
predictions = lin_reg.predict(magneticum_data['logLx_ex'].values.reshape(-1,1))
resids = predictions - magneticum_data['logM500'] #pred minus true
mg_dat_v2 = magneticum_data.copy()
mg_dat_v2['resids'] = resids
corr_matrix = mg_dat_v2.corr()
print(corr_matrix['logM500'].sort_values(ascending=True))
print(corr_matrix['resids'].sort_values(ascending=True))
#the mass itself correlates strongly with residual, which means that it's not unbiased
#expect this to go down once we get a truly flat mass function

# the correlations with caS, cAs are much tighter in the eROSITA case...


# In[19]:


#print distributions of parameters that will be used in analysis
if(compare_dists):
    plt.rc('text', usetex=False)
    ax = magneticum_data.drop(columns=['logM500','logLx_ex','uid']).hist(bins=50, figsize=(20,20), histtype="step",color='k', linewidth=1.2, label='Chandra', density='normed');
    ax = morph_parms_erosita[cats_to_keep_erosita].hist(bins=50, histtype="step",color='r', linewidth=1.2, ax=ax.reshape((12,-1))[0:11], label='eROSITA', density='normed');
    ax[0].legend(loc=2)
    plt.savefig(fig_dir / 'ce_compare_feature_dists.pdf',bbox_inches='tight')
    plt.show()
else:
    plt.rc('text', usetex=False)
    magneticum_data.hist(bins=50, figsize=(20,20), histtype="step",color='k', linewidth=1.2);
    plt.savefig(fig_dir / 'feature_dists.pdf',bbox_inches='tight')
    plt.show()


# ## Train-Test Splitting, Validation, Feature Standardization

# In[17]:


#generate train and test set based on uids to keep same cluster across snapshots in either train or test
train_set, test_set = split_train_test_by_id(magneticum_data, test_ratio=0.2, id_column='uid')
print(train_set.shape, test_set.shape)

#let's make copies of the relevant parts of the train and test sets
train = train_set.drop(['logM500', 'uid'], axis=1)
train_columns = train.columns
train_labels = train_set['logM500']
train_uids = train_set['uid']

#store the redshifts of the test set so we can look at scatter binned by redshift
test_redshifts = mag_redshifts.loc[test_set.index] ##MAKE SURE THIS IS CORRECT

test = test_set.drop(['logM500', 'uid'], axis=1)
test_labels = test_set['logM500']

#Generate groups on the train set for group k-fold CV (s.t. each uid halo is only in one group)
n_folds = 10
fold = train_uids % n_folds
logo_iter = list(LeaveOneGroupOut().split(train,train_labels,fold))


# In[18]:


fig,ax = plot()

print("\n Fold\t", end=" ")
for i in range(0,n_folds):
    print(str(i)+'\t', end =" ")
print("\n Count\t", end=" ")
for i in range(0, n_folds):
    print(str(list(fold).count(i))+'\t', end=" ")
    plt.hist(train_labels[fold == i], density='normed', alpha=0.6)


# In[19]:


fig, ax = plot()
plt.hist(magneticum_data['logM500'],30,alpha=0.5, density='normed',label='Full dataset')
plt.hist(test_set['logM500'],30,alpha=0.5, density='normed', label='Test set')
plt.legend()
plt.xlabel(r'$\log_{10}(M)$')
plt.ylabel('Normed pdf');
#is this an unbiased enough sample?


# In[20]:


#scale columns of train set based on median and 16/84 quanties, scale test set by same numbers
if(scale_features):
    rs = RobustScaler(quantile_range=(16.0, 84.0))
    train = rs.fit_transform(train)
    test = rs.transform(test)
else:
    train = train.values
    test  = test.values
    
# we will need to make the scales for the parameters publicly available if we release the model


# In[21]:


# save everything needed so cross-validation can be performed on Grace
np.savez(data_dir / 'erosita_training_data_and_folds_cas',logo_iter=logo_iter, train=train, train_labels=train_labels)


# ## Recursive Feature Elimination

# In[35]:


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


# In[30]:


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

# ### Part 1: $M-L_x$ Relationship
# 
# TODO: Generate list of Maughan curves and plot scatter distribution like in Ntampaka et al. 2018
# 
# TODO: Try different cost functions for minimization instead of mean squared error.

# In[21]:


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

#the 0.103 is consistent with the 0.11 that probably comes about due to having the extra 300 clusters


# In[24]:


#somehow, the connection between the power ratio and the dxc is important...
#however, with power ratio, dxc, and redshift, we can get it with 0.014 dex... what's going on with that?
#doesn't seem believable... should we throw z out?

#logdxc and logPR10 combined are better than Lx as a tracer for mass... strange...
#very odd that once we add in redshift, we get extremely low scatter though, need to figure that out


# ### Part 2: Ordinary Linear Regression

# In[25]:


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


# ### Bugfixing d_xc, PR10, z relationship

# In[26]:


train_columns


# In[29]:


#pick the relevant columns
train_columns[[1,5,12]]


# In[30]:


fig,ax = plot()
sc = plt.scatter(train[:,1], train[:,5], c=train_labels, cmap=plt.get_cmap('jet'))
plt.xlabel(train_columns[1])
plt.ylabel(train_columns[5]);
cbar = plt.colorbar(sc);
cbar.set_label(r'logM500');


# In[40]:


plt.hist(train[:,14])
print(np.unique(train[:,14]))
#what's going on with the redshift?
#something is changing it between up there and down here


# In[20]:


fig,ax = plot()
  
sc = plt.scatter(-1.13841608* train[:,2] + 2.27764029*train[:,6], train_labels, c=train[:,14], cmap=plt.get_cmap('jet'))
plt.xlabel('-1.67*logPR10 + 1.59*logdxc')
plt.ylabel(r'logM500');
cbar = plt.colorbar(sc);
cbar.set_label(r'$z$');


# In[25]:


np.corrcoef(train[:,6],train[:,2])


# In[21]:


lin_reg = LinearRegression()
lin_reg.fit(train[:,[2,6,14]], train_labels)
test_predictions = lin_reg.predict(test[:,[2,6,14]])
lin_reg_scatter_robust, lin_reg_sc_rb_2sig = scatter_prelog(test_predictions, test_labels)
lin_reg_scatter_std = scatter_prelog(test_predictions, test_labels, False)
print("The (robust) scatter for a LR model is %.3f dex" % lin_reg_scatter_robust)
print("Robust 2-sigma scatter: %.3f dex" % lin_reg_sc_rb_2sig)
print("The (std) scatter for a LR model is %.3f dex" % lin_reg_scatter_std)
print(lin_reg.coef_, lin_reg.intercept_)


# In[22]:


lin_reg = LinearRegression()
lin_reg.fit(train[:,[6,14]], train_labels)
test_predictions = lin_reg.predict(test[:,[6,14]])
lin_reg_scatter_robust, lin_reg_sc_rb_2sig = scatter_prelog(test_predictions, test_labels)
lin_reg_scatter_std = scatter_prelog(test_predictions, test_labels, False)
print("The (robust) scatter for a LR model is %.3f dex" % lin_reg_scatter_robust)
print("Robust 2-sigma scatter: %.3f dex" % lin_reg_sc_rb_2sig)
print("The (std) scatter for a LR model is %.3f dex" % lin_reg_scatter_std)
print(lin_reg.coef_, lin_reg.intercept_)


# In[23]:


lin_reg = LinearRegression()
lin_reg.fit(train[:,[2,14]], train_labels)
test_predictions = lin_reg.predict(test[:,[2,14]])
lin_reg_scatter_robust, lin_reg_sc_rb_2sig = scatter_prelog(test_predictions, test_labels)
lin_reg_scatter_std = scatter_prelog(test_predictions, test_labels, False)
print("The (robust) scatter for a LR model is %.3f dex" % lin_reg_scatter_robust)
print("Robust 2-sigma scatter: %.3f dex" % lin_reg_sc_rb_2sig)
print("The (std) scatter for a LR model is %.3f dex" % lin_reg_scatter_std)
print(lin_reg.coef_, lin_reg.intercept_)


# In[24]:


lin_reg = LinearRegression()
lin_reg.fit(train[:,[2,6]], train_labels)
test_predictions = lin_reg.predict(test[:,[2,6]])
lin_reg_scatter_robust, lin_reg_sc_rb_2sig = scatter_prelog(test_predictions, test_labels)
lin_reg_scatter_std = scatter_prelog(test_predictions, test_labels, False)
print("The (robust) scatter for a LR model is %.3f dex" % lin_reg_scatter_robust)
print("Robust 2-sigma scatter: %.3f dex" % lin_reg_sc_rb_2sig)
print("The (std) scatter for a LR model is %.3f dex" % lin_reg_scatter_std)
print(lin_reg.coef_, lin_reg.intercept_)


# In[ ]:


#### BUGFIXING ####
# Let's try this with different combinations of dxc, PR10, and z to figure out how the correlation is so good
# Let's also make sure that it isn't the scatter being low but overall completely biased


# In[27]:


fig,ax=plot()
plt.plot(test_predictions, test_labels, '.')
#it's basically a perfect prediction all the way down...


# ### Part 3: Ridge Regression

# In[26]:


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


# ### Part 4: Lasso Regression

# In[27]:


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


# ### Part 5: Random Forest Regression

# In[88]:


#randomized search CV for optimizing RF hyperparameters
#NOTE: this takes a long time to run, so will re-run it again once we have final feature sets and data
#notes on cross-validation: https://machinelearningmastery.com/train-final-machine-learning-model/
rf = RandomForestRegressor()
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 1000, stop = 2000, num = 6)] # more is better for avoiding overfitting
# Number of features to consider at every split
max_features = ['auto', 'sqrt'] # problem specific
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)] # large max_depth is ok for RF, but small max_depth for GB
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 3, 5, 7, 10] # larger value here is better for avoiding overfitting
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4, 8] # larger value is better for avoiding overfitting
# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


# In[89]:


#run the randomized search
if(run_rf_cv):
    rand_search = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, 
                                 cv = logo_iter, verbose=2, random_state=42, n_jobs = -1, 
                                 scoring='neg_mean_squared_error')
    rand_search.fit(train, train_labels)
    rf_params = rand_search.best_params_
    best_model = rand_search.best_estimator_
    print(rand_search.best_params_)
    best_params = rand_search.best_params_
else:
    if(idealized):
        print()
    else:
        # same best parameters using the Chandra data
        # now, are there more parameters that we could optimize?
        best_params = {'n_estimators': 1600, 'min_samples_split': 2, 'min_samples_leaf': 2, 'max_features': 'auto', 'max_depth': 60, 'bootstrap': True}
#TODO: after we run randomized search on our full dataset,
#try narrowing down one more step with an additional grid search


# In[43]:


#if(idealized):
#    print()
#else:
#    best_params = {'n_estimators': 1600, 'min_samples_split': 2, 'min_samples_leaf': 2, 'max_features': 'auto', 'max_depth': 60, 'bootstrap': True}

# now, let's do a grid-search CV
#rf = RandomForestRegressor(random_state=42)
#param_grid = [{'n_estimators': [1500,1600,1700], 'min_samples_split':[2,3,4], 'min_samples_leaf':[1,2,3],'max_features':['auto','sqrt'], 'max_depth':[50,60,70], 'bootstrap':[True,False]}]
#grid_search = GridSearchCV(rf, param_grid, n_jobs=-1,
#                           cv=logo_iter, scoring='neg_mean_squared_error')
#grid_search.fit(train, train_labels)
#print(grid_search.best_params_)


# In[28]:


#RF without cross-validation, default hyperparameters
rf = RandomForestRegressor(random_state=42)

rf.fit(train, train_labels)
rf_preds = rf.predict(test)
rf_scatter_robust, rf_sc_rb_2sig = scatter_prelog(rf_preds, test_labels)
rf_scatter_std = scatter_prelog(rf_preds, test_labels, False)
print("The (robust) scatter for a RF model is %.3f dex" % rf_scatter_robust)
print("Robust 2-sigma scatter: %.3f dex" % rf_sc_rb_2sig)
print("The (std) scatter for a RF model is %.3f dex" % rf_scatter_std)


# In[30]:


rf = RandomForestRegressor(random_state=42, bootstrap=False, max_depth=50, max_features=0.5, min_samples_leaf=2,
                           min_samples_split=2, n_estimators=1000)

#{'bootstrap': False, 'min_samples_leaf': 2, 'n_estimators': 1000, 'min_samples_split': 2, 'max_features': 0.5, 'max_depth': 50}

# let's play a bit with max features to see if something in between 3 and 12 is good
print(np.mean(cross_val_score(rf, train, train_labels, cv=logo_iter, scoring='neg_mean_squared_error')))

# current best parameters: -0.0071652676657399435
# best parameters with max_features=0.5: -0.006762182586815648 (this is close to the gradient boosted regressor one)
# best parameters with max_features=0.75: -0.0068384746423482915
# NOTE: the cross-validation error is actually reduced considerably more in our best fit
# reduced min_samples_leaf to 1, keep max_features=0.5: -0.006736532845819686


# In[21]:


#use the best set of hyperparameters to train RF model on whole train set, test on test set
#TODO: go back and do RFE using RF since we have an idea of the hyperparameter values
#rf = RandomForestRegressor(random_state=42, bootstrap=True, max_depth=90, max_features='auto', min_samples_leaf=2,
#                           min_samples_split=2, n_estimators=2000)
#rf = RandomForestRegressor(random_state=42, bootstrap=True, max_depth=50, max_features='auto', min_samples_leaf=2,
#                           min_samples_split=2, n_estimators=1600)
#rf = RandomForestRegressor(random_state=42, bootstrap=False, max_depth=50, max_features=0.5, min_samples_leaf=2,
#                           min_samples_split=2, n_estimators=1000)
#{'bootstrap': True, 'min_samples_leaf': 1, 'n_estimators': 1200, 'min_samples_split': 2, 'max_features': 0.75, 'max_depth': 30}
rf = RandomForestRegressor(random_state=42, bootstrap=True, min_samples_leaf=1, n_estimators=1200, min_samples_split=2,
                           max_features=0.75, max_depth=30)

rf.fit(train, train_labels)
rf_preds = rf.predict(test)
rf_scatter_robust, rf_sc_rb_2sig = scatter_prelog(rf_preds, test_labels)
rf_scatter_std = scatter_prelog(rf_preds, test_labels, False)
print("The (robust) scatter for a RF model is %.5f dex" % rf_scatter_robust)
print("Robust 2-sigma scatter: %.3f dex" % rf_sc_rb_2sig)
print("The (std) scatter for a RF model is %.5f dex" % rf_scatter_std)
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

# so if we use neg mean sq error as the metric for cross-validation, we get 0.067/0.084 for robust/std
# if we use robust_scatter as scoring: same exact values; it doesn't matter!
# hence, it is strange that gradient boosting trees does another 6% better

# write this out as systematic prediction
np.save('rf_preds_systematic', rf_preds)

#The (robust) scatter for a RF model is 0.06575 dex
#Robust 2-sigma scatter: 0.156 dex
#The (std) scatter for a RF model is 0.07667 dex


# In[25]:


#{'bootstrap': True, 'max_depth': 80, 'max_features': 0.5, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 1800}
rf = RandomForestRegressor(random_state=42, bootstrap=True, min_samples_leaf=1, n_estimators=1800, min_samples_split=5,
                           max_features=0.5, max_depth=80)

rf.fit(train, train_labels)
rf_preds = rf.predict(test)
rf_scatter_robust, rf_sc_rb_2sig = scatter_prelog(rf_preds, test_labels)
rf_scatter_std = scatter_prelog(rf_preds, test_labels, False)
print("The (robust) scatter for a RF model is %.5f dex" % rf_scatter_robust)
print("Robust 2-sigma scatter: %.3f dex" % rf_sc_rb_2sig)
print("The (std) scatter for a RF model is %.5f dex" % rf_scatter_std)

# the cross-validated RF model that performed best for eROSITA using only c, A, S is described above


# In[33]:


import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(rf).fit(test, test_labels)
eli5.show_weights(perm)


# In[35]:


perm = PermutationImportance(rf).fit(train, train_labels)
eli5.show_weights(perm)


# In[21]:


#look at feature importance for RF
for i in range(0,len(train_columns)):
    print('%.4f \t %s' % (rf.feature_importances_[i], train_columns[i]))


# In[22]:


#calculate RF model performance (bias from mean vs. median?)
evaluate(rf, test, test_labels)
robust_bias = np.median(rf_preds - test_labels)
bias = np.mean(rf_preds - test_labels)
print('The final RF model applied to test set had a (robust) bias of %.4f dex' % robust_bias)
print('The final RF model applied to test set had a bias of %.4f dex' % bias)


# ## Results, Plots, Scatter distributions, etc.

# <span style="color:red">We need to copy over the mass-luminosity plots from the other Jupyter notebook to see how our results look with eROSITA vs. Chandra.</span>

# In[28]:


#compare RF predicted masses to true

log10_mass_median = 14 #log10 of 1e14 Msun/h

fig, ax = plot()
plt.plot(test_labels, rf_preds, '.', markersize=8, alpha=0.75)
plt.xlabel(r'$\log[M_{500c,\mathrm{true}} / (M_\odot h^{-1})]$');
plt.ylabel(r'$\log[M_{500c,\mathrm{pred}} / (M_\odot h^{-1})]$');
plt.plot([12,16], [12,16], 'k')
plt.fill_between([12,16], [12,16]-rf_scatter_robust, [12,16]+rf_scatter_robust, zorder=-32, alpha=0.3, color='g')

#see how close to 1-1 the predictions are, i.e., what is the slope?
lin_reg = LinearRegression()
lin_reg.fit(test_labels.values.reshape(-1,1) - log10_mass_median, rf_preds - log10_mass_median) # normalized by 1e14

#should add the scatter and the regression fit
plt.text(13.5,14.7,r'$\delta_\mathrm{RF} = %.4f$' % rf_scatter_robust, fontsize=18)
plt.text(13.5,14.6,r'$\mu_\mathrm{RF} = %.4f$' % robust_bias, fontsize=18)
plt.text(13.5,14.5,r'$a = %.3f$' % lin_reg.coef_[0], fontsize=18)
plt.text(13.5,14.4,r'$b = %.3f$' % lin_reg.intercept_, fontsize=18)
plt.text(14.2,13.8,r'Realistic \textit{eROSITA}', fontsize=18)

xv = np.linspace(12, 16, 50)
plt.plot(xv, xv*lin_reg.coef_[0] + lin_reg.intercept_ + log10_mass_median*(1. - lin_reg.coef_[0]), 'r', linestyle='dashed')

plt.xlim(13.35, 14.9)
plt.ylim(13.35, 14.9)

# Set major ticks for x axis
ticks = np.linspace(13.4, 14.8, 8)
print(ticks)

ax.set_xticks(ticks)
ax.set_yticks(ticks)

print("The predicted-to-true intercept is %.3f" % lin_reg.intercept_)
print("The predicted-to-true slope is %.3f" % lin_reg.coef_[0])
if(lin_reg.coef_[0] < 1):
    print("High masses will be underpredicted!")
else:
    print("High masses will be overpredicted!")
    
plt.savefig(fig_dir / 'mpred_vs_mtrue_erosita.pdf', bbox_inches='tight')

#TODO: figure out what is going on with the outliers

#could perhaps do the intercept in terms of M_14 or something so that the zero point has more meaning..


# In[42]:


lin_reg = LinearRegression()
lin_reg.fit(train[:,train_columns == 'logLx_ex'], train_labels)
mlx_predictions = lin_reg.predict(test[:,train_columns == 'logLx_ex'])
#train_predictions = lin_reg.predict(train[:, train_columns == 'logLx_ex'])
m_lx_scatter_robust, m_lx_sc_rb_2sig = scatter_prelog(mlx_predictions, test_labels)
m_lx_scatter_std = scatter_prelog(mlx_predictions, test_labels, False)
print("The (robust) scatter for a simple mass-luminosity relationship is %.3f dex" % m_lx_scatter_robust)
print("Robust 2-sigma scatter: %.3f dex" % m_lx_sc_rb_2sig)
print("The (std) scatter for a simple mass-luminosity relationship is %.3f dex" % m_lx_scatter_std)


# In[43]:


rf_preds_systematic = np.load('rf_preds_systematic.npy')
rf_scatter_systematic_robust, junk = scatter_prelog(rf_preds_systematic,test_labels)


# In[44]:


#look at distribution of residuals for final RF model, for idealized and non, and for m-lx baseline
fig, ax = plot()
plt.hist(rf_preds - test_labels, histtype='step', color='k', linestyle='solid',
         linewidth=2, bins=30, density="normed", label='RF Ideal \n %.3f dex,\n %.2f percent' %(rf_scatter_robust,(10**rf_scatter_robust - 1)*100)) #change this to the actual %age
#plt.hist(rf_preds_systematic - test_labels, histtype='step', color='r', linestyle='dashdot',
#         linewidth=2, bins=30, density="normed", label='RF Systematic \n %.3f dex,\n %.2f percent' %(rf_scatter_systematic_robust,(10**rf_scatter_systematic_robust - 1)*100))
plt.hist(mlx_predictions - test_labels, histtype='step', color='b', linestyle='dashed',
         linewidth=2, bins=30, density="normed", label='$M-L_{\mathrm{ex}}(z)$ \n %.3f dex,\n %.2f percent' %(m_lx_scatter_robust,(10**m_lx_scatter_robust - 1)*100))
plt.xlabel(r'$\log(M_{500c,\mathrm{pred}}) - \log(M_{500c,\mathrm{true}})$')
plt.ylabel('PDF');
plt.legend()

# slightly less biased perhaps, let's report on the biases between the various ones?

#over-plot mass-luminosity relationship?

plt.savefig(fig_dir / 'error_dist.pdf', bbox_inches='tight')

# there is roughly an 18% improvement regardless of which scatter definition we use...


# ### Redshift dependence on scatter
# 
# Not really worth looking at

# In[63]:


#scatter as a function of redshift

fig, ax = plot()

z,c = np.unique(test_redshifts, return_counts=True)
#for this, we need to actually go back and get the redshifts, since they were dropped from the dataset
for i,z in enumerate(np.unique(test_redshifts)):
    rfs, rf2 = scatter_prelog(rf_preds[test_redshifts == z], test_labels[test_redshifts == z])
    print(z,rfs,c[i])
    plt.hist(rf_preds[test_redshifts == z] - test_labels[test_redshifts == z], histtype='step', 
         linewidth=1.2, bins=30, density="normed", label=r'$z$=%.2f, %.3f dex' %(z,rfs)) #change this to the actual %age

plt.xlabel(r'$\log(M_{500c,\textrm{pred}}) - \log(M_{500c,\textrm{true}})$')
plt.ylabel('PDF');
plt.legend()
plt.savefig(fig_dir / 'error_dist_redshift.pdf', bbox_inches='tight')


# ## Generating Images of Clusters

# In[29]:


if(instrument == 'chandra'):
    center_lower = 1035
    center_upper = 1035
elif(instrument == 'erosita'):
    center_lower = 191
    center_upper = 192
    
clusterlist = np.load(data_dir / (instrument+'_clusterList.npy'))


# In[156]:


def gen_plot(cluster_id, vmin=None, vmax=None):
    head = fits.open(use_dir / ('%d.fits' % cluster_id))
    data = fits.open(use_dir / ('%d.fits' % cluster_id))[0].data

    print(np.sum(data))

    masked_array = np.log10(np.ma.array (data, mask=(data==0)));
    cmap = cm.plasma
    cmap.set_bad('black',1.)
    fig, ax = plot(figsize=(10,10))
    ax.imshow(masked_array, interpolation='nearest', extent=[0,1,0,1], cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)
    plt.axis('off')
    rad = clusterlist['R500_pixel'][clusterlist['id'] == cluster_id]
    #plt.xlim(center_lower-rad,center_upper+rad)
    #plt.ylim(center_lower-rad,center_upper+rad)
    return fig,ax

def photon_count(cluster_id):
    head = fits.open(use_dir / ('%d.fits' % cluster_id))
    data = fits.open(use_dir / ('%d.fits' % cluster_id))[0].data

    return np.sum(data)


# ### Verifying photon count distributions

# In[42]:


ranked_by_mass = magneticum_data.sort_values(by='logM500').index
photon_counts = np.zeros(len(ranked_by_mass))
for i in range(0, len(ranked_by_mass)):
    print(i)
    photon_counts[i] = photon_count(ranked_by_mass[i])


# In[52]:


uz = np.unique(mag_redshifts)


# In[54]:


zs = mag_redshifts.loc[ranked_by_mass]
for i in range(0,len(uz)):
    plt.hist(np.log10(photon_counts[zs == uz[i]]), alpha=0.5, label=uz[i])
plt.legend()

# there don't seem to be any outliers now


# ### Generating images for paper

# In[162]:


#cluster 871048
#high c, c=0.369735

# we will eventually want to control brightness of image by setting up/low bounds for cmap

cluster_id = 871048
gen_plot(cluster_id)
plt.savefig(fig_dir / ('high_c_%d.pdf' % cluster_id), bbox_inches='tight')


# In[144]:


#cluster 870976
#low c, c= 0.039817

cluster_id = 870976
gen_plot(cluster_id)
plt.savefig(fig_dir / ('low_c_%d.pdf' % cluster_id), bbox_inches='tight')


# In[145]:


#cluster 871050
#high A, A=1.49215

cluster_id = 871050
gen_plot(cluster_id)
plt.savefig(fig_dir / ('high_A_%d.pdf' % cluster_id), bbox_inches='tight')


# In[146]:


#cluster 870991
#low A, A=0.927353

cluster_id = 870991
gen_plot(cluster_id)
plt.savefig(fig_dir / ('low_A_%d.pdf' % cluster_id), bbox_inches='tight')


# In[147]:


#cluster 870985
#high S, S=1.03454

cluster_id = 870985
gen_plot(cluster_id)
plt.savefig(fig_dir / ('high_S_%d.pdf' % cluster_id), bbox_inches='tight')


# In[148]:


#cluster 870997
#low S, S=0.612777

cluster_id = 870997
gen_plot(cluster_id)
plt.savefig(fig_dir / ('low_S_%d.pdf' % cluster_id), bbox_inches='tight')


# ## Gradient Boosted Trees Test
# 
# This ended up not being as successful as the random forest method, so this will be omitted from analysis.

# In[52]:


gb = GradientBoostingRegressor()

learning_rate = np.logspace(-3, np.log10(2), 10) # 10 options for learning rate spanning 3 orders of magnitude
n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=5)]
n_estimators.append(100)
n_estimators.append(300)
# Number of features to consider at every split
max_features = ['auto', 'sqrt', 0.5, 0.75] # may want to consider fraction like 1/2, 3/4...
# Maximum number of levels in tree
max_depth = [None, 3, 5, 10, 12, 15, 20]
# Minimum number of samples required to split a node
min_samples_split = [2, 3, 5, 7, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4, 8]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'learning_rate': learning_rate}

rand_search = RandomizedSearchCV(estimator=gb, param_distributions=random_grid, n_iter=100,
                                 cv=logo_iter, verbose=2, random_state=42, n_jobs=-1, scoring='neg_mean_squared_error')
rand_search.fit(train, train_labels)
gb_params = rand_search.best_params_
best_model = rand_search.best_estimator_
print(rand_search.best_params_)
best_params = rand_search.best_params_


# In[116]:


# now with some idea of the best values, let's iterate around this with a grid-search to find the best
learning_rate = [0.028, 0.029, 0.03] # 15 options for learning rate spanning 4 orders of magnitude
n_estimators = [1700, 1800, 1900]
# Number of features to consider at every split
max_features = [train.shape[1], train.shape[1]-1]
# Maximum number of levels in tree
max_depth = [4, 5, 6]
# Minimum number of samples required to split a node
min_samples_split = [2, 3, 5]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2]

# Create the random grid
grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'learning_rate': learning_rate}

rand_search = RandomizedSearchCV(estimator = gb, param_distributions = grid, n_iter = 10, 
                             cv = logo_iter, verbose=2, random_state=42, n_jobs = -1)#, 
rand_search.fit(train, train_labels)
gb_params = rand_search.best_params_
best_model = rand_search.best_estimator_
print(rand_search.best_params_)
best_params = rand_search.best_params_


# In[166]:


#{'learning_rate': 0.029317331822241718, 'min_samples_leaf': 4, 'n_estimators': 200, 'max_features': 0.5, 'min_samples_split': 2, 'max_depth': None}

# the newest GB regression tree hyperparameter setup found in the CV is actually worse, and I don't want to re-run
# let's wait until we see what we get for the RF

#gb = GradientBoostingRegressor(random_state=42, n_estimators=300)
#gb = GradientBoostingRegressor(random_state=42, n_estimators=1800, min_samples_split=5, min_samples_leaf=1, max_features='auto', max_depth=4, learning_rate=0.0293173318)#, tol=1e-7, n_iter_no_change=500)
gb = GradientBoostingRegressor(random_state=42, n_estimators=1550, min_samples_split=2,
                          min_samples_leaf=4, max_features=0.5, max_depth=None, learning_rate=0.005414548164181543)
print(np.mean(cross_val_score(gb, train, train_labels, cv=logo_iter, scoring='neg_mean_squared_error')))

# so the error is really on the order of .007, so the tolerance should be like or 1e-6, 

# neg_mean_square error

# with the new best fit: -0.007161608096827821
# with early stopping: -0.007110618246812547

# TODO: Figure out

# with defaults and n_estimators=300: -0.007541896653092285

# clearly, our identified model does a better job, so it does still minimize the CV error

# cross-validation with early stopping: -0.06305186883787336
# cross-validation without early stopping: -0.06213357328426965
# in conclusion, don't both with early stopping anymore

# new configuration that is best on cross-validation: -0.006585094833285662
# the best cross-validation set actually does worse than random forest!
# this just means we can drop Gradient Boosted Trees altogether, but should still tell Michelle


# In[167]:


#gb = GradientBoostingRegressor(random_state=42, n_estimators=1800, min_samples_split=5,
#                               min_samples_leaf=1, max_features='auto', max_depth=4, learning_rate=0.0293173318)
#gb = GradientBoostingRegressor(random_state=42, n_estimators=1100, min_samples_split=7,
#                               min_samples_leaf=1, max_features=0.5, max_depth=10, learning_rate=0.012599210498948734)
#gb = GradientBoostingRegressor(random_state=42, n_estimators=1550, min_samples_split=2,
#                               min_samples_leaf=4, max_features=0.5, max_depth=None, learning_rate=0.005414548164181543)

gb.fit(train, train_labels)
gb_preds = gb.predict(test)
gb_scatter_robust, gb_scatter_robust_2sig = scatter_prelog(
    gb_preds, test_labels)
gb_scatter_std = scatter_prelog(gb_preds, test_labels, False)
print("The (robust) scatter for a GB model is %.5f dex" % gb_scatter_robust)
print("The (std) scatter for a GB model is %.5f dex" % gb_scatter_std)


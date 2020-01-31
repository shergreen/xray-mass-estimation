#!/usr/bin/env python
# coding: utf-8

# ## Generates a file called excised_lx.txt which contains the 15% R_500 core-excised luminosities of Box2, computed using the no-background images

# In[47]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
from astropy.io import fits
from pathlib import Path
from os.path import expanduser
from useful_functions.plotter import plot, loglogplot


# In this notebook, we're going to recompute the luminosities from the xspec code that Lorenzo sent us, then write functions for core-excision. The output of this notebook will be the following:
# 
# 1. For the N clusters, there will be an L500_obs, L500_ex,obs. (2N data points)
# 2. We will generate a plot of L500_true vs. L500_obs (probably won't go in paper, but just to verify L values)

# From Lorenzo:
# 
# I mean, the fact that we add the background does not change the luminosity of the cluster, which is still the one provided by Klaus. The presence of the background just decreases our ability to properly estimate the luminosity (which is one of the problems with observations); naively if you know perfectly the background (e.g. you take the background map that was added to the erosita image) you perfectly recover the Lx from Klaus.
# 
# So, if you just need to know the intrinsic Lx of your cluster we can use the values from Klaus. **For the excised Lx you can use the erosita images without the background that I guess Michelle can create in a similar way as done with the ones including the background.**
# 
# Instead if you want to simulate the same limitations that will have the observers we need to go though xspec I think. You first determine the background-subtracted count rate and then you use the script that I sent you. 
# 
# **Since we want to simulate what the observers will see, we should generate our own Lx values and then core-excise them here.**
# 
# Steps:
# 
# - count the number of photons in the region of interest 
# - estimate the cluster temperature (maybe from an M-T relation)
# - varying the norm of apec model in XSPEC until the count rate match the value of your cluster 
# - print out the corresponding flux and convert into a Lx
# 
# Question:
# 
# What is the most fair way to do these calculations? We know the cluster temperatures from Magneticum, but a real observer wouldn't know this, so what is the most fair way to truly get observer luminosities?

# In shell you just need to run 
# 
# xspec - lumin.tcl 
# 
# after having initialized heasoft in the shell. In principle it can be translated into a python code but I never did. 
# 
# Before running the script you need to create a file "cluster.par" which include almost all the parameters required by the script: column density, cluster temperature, cluster metallicity, redshift, count rate, exposure. E.g. 
# 
# echo -n 5E20 3.6 0.3 0.12 0.35 1600 >! cluster.par 
# 
# For the hydrogen column density and exposure you should used the values used to run the simulation.
# 
# **Pending: Michelle send these two pieces of information (2000 seconds), Michelle sends no-background images, Michelle/Lorenzo suggest what temperature to use, Klaus responds with necessary data files**

# In[2]:


#in the mean time, we can write our functions to perform the core-excision
#we load in the fits files and have the cluster information available
#need the radii in pixels
#then we count up all within the radii
#we cound up the amount within 15% of the radii
#simple math
data_dir = Path('../data')
home_dir = Path(expanduser('~'))

fig_dir = Path('../figs/')
chandra_dir = home_dir / 'magneticum_chandra'
erosita_no_bg_dir = home_dir / 'magneticum_no_bg'

cluster_directory = chandra_dir


# In[3]:


magneticum_data = np.load(cluster_directory / 'clusterList.npy')
# make this into a pandas data frame
magneticum_data = pd.DataFrame(data=magneticum_data, columns=magneticum_data.dtype.names)
magneticum_data = magneticum_data.set_index('id')
Nclusters = len(magneticum_data.index)


# In[4]:


def distance(x1, x2):
    return np.sqrt((x1[0] - x2[0])**2 + (x1[1] - x2[1])**2)


# In[7]:


#make an array that is 384 x 384, and each element contains the 4 corners in the cell

#so we say that we are within the radius if part of the pixel is within the radius,
#not if its center is within radius
#TODO: we should see how much things differ when we do this...
img_shape_x, img_shape_y = fits.open(cluster_directory / '922603.fits')[0].data.shape
center = np.array([img_shape_x, img_shape_y]) / 2.
cell_dists = np.ndarray((img_shape_x, img_shape_y))
for i in range(0, img_shape_x):
    for j in range(0, img_shape_y):
#        #cell_dists[i,j] = distance(center, [i,j])
        cell_dists[i,j] = np.min([distance(center,[i,j]), distance(center,[i+1,j]), 
                                  distance(center,[i,j+1]), distance(center,[i+1,j+1])])


# In[8]:


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


# In[9]:


excised_counts = np.zeros(Nclusters)
for i,cluster_id in enumerate(magneticum_data.index.values):
    print(i)
    excised_counts[i] = core_excised_counts(cluster_id, 0.15)


# In[10]:


plt.hist(np.log10(excised_counts))
plt.xlabel(r'$\log N_{ex}$')

# much higher for the Chandra data, wonder what is going on down at the low end...


# In[9]:


#the dead center is the bottom corner of pixel 192 (counting from 0 as in python.)
#this is same as saying (starting from 1) pixels (192-193) is the center, if pixels are defined from center
#if we want to 


# In[12]:


pix_corner_excised_Lx = np.zeros(Nclusters)
for i,cluster_id in enumerate(magneticum_data.index.values):
    if(i % 100 == 0):
        print(i)
    pix_corner_excised_Lx[i] = core_excised_Lx(cluster_id, 0.15)


# In[14]:


#this is if we want to measure within R500 based on the pixel centers instead of nearest edge
#pix_center_excised_Lx = np.zeros(Nclusters)
#for i,cluster_id in enumerate(magneticum_data.index.values):
#    pix_center_excised_Lx[i] = core_excised_Lx(cluster_id, 0.15)


# In[15]:


#plt.hist(np.log10(pix_corner_excised_Lx),alpha=0.5,label='Corner', bins=100)
#plt.hist(np.log10(pix_center_excised_Lx),alpha=0.5,label='Center', bins=100)
#plt.legend()

#looks like the center values are larger, only slightly
#this is because we are more likely to excise an extra ring of pixels if we use the center
#i'll use the center, because it seems more reasonable to me, even though it makes very little difference


# In[13]:


#now, we want to save a list of the excised Lx values and the respective cluster ids,
#to be loaded into our main file
output = np.column_stack((magneticum_data.index.values, pix_corner_excised_Lx))
np.savetxt(data_dir / 'chandra_excised_lx.txt',output,fmt='%d %e')


# In[44]:


# let's do a comparison between chandra and erosita core-excised
erosita_ex_lx = np.loadtxt(data_dir / 'excised_lx.txt')
chandra_ex_lx = np.loadtxt(data_dir / 'chandra_excised_lx.txt')
print(erosita_ex_lx.shape)
print(chandra_ex_lx.shape)


# In[45]:


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


# In[49]:


loglogplot()
plt.plot(chandra_ex_lx[:,1],erosita_ex_lx[:,1],'.')

rel_err = (erosita_ex_lx[:,1] - chandra_ex_lx[:,1]) / chandra_ex_lx[:,1]

plot(semilogx=True)
plt.plot(chandra_ex_lx[:,1],rel_err,'.')

# this tells us that the chandra luminosities are basically always larger, and this should be no surprise
# since we have more pixels, we can throw out less light at the boundary

# however, i'm surprised at the percentage-level difference between the two


# In[54]:


# let's make sure the luminosities from the raw data files are the same though, just to confirm that the only effect
# is simply due to the core excision process
erosita_data = np.load(data_dir / 'clusterList.npy')
chandra_data = np.load(cluster_directory / 'clusterList.npy')

msk = np.isin(chandra_data['id'], erosita_data['id'])
chandra_data = chandra_data[msk]
msk = np.isin(erosita_data['id'], chandra_data['id'])
erosita_data = erosita_data[msk]
chandra_data = chandra_data[np.argsort(erosita_data['id'])]
erosita_data = erosita_data[np.argsort(erosita_data['id'])]

rel_err = (erosita_data['M500_msolh'] - chandra_data['M500_msolh']) / chandra_data['M500_msolh']
plot(semilogx=True)
plt.plot(chandra_data['M500_msolh'], rel_err)

# this is good, they're the exact same
# hence, the only cause for discrepancy in the values of the core-excised luminosities is the excision process


# In[ ]:





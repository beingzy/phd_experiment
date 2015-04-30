## ################### ##
## PYTHON LEARNING CODE ##
## #################### ##
 
## ##################################### ##
## EXPERIMENT: ##
## Two Clusters Separating ##
## ##
## AUTHOR: Yi Zhang <beingzy@gmail.com> ##
## Date: APR/10/2015 ##
## ##################################### ##
import os
import sys
import glob
import datetime
import time
 
import numpy as np
import scipy as sp
import pandas as pd
 
from scipy.spatial.distance import euclidean
from numpy.random import uniform
from numpy.random import binomial
from numpy.random import choice
 
from numpy import sin
from numpy import cos
 
from itertools import combinations
 
#/*------- Data Visualization ------ */
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
 
try:
    from learning_dist_metrics.ldm import LDM
    from learning_dist_metrics.datasets import load_sample_data
except:
    os.chdir("C:\\A-Playground\\experiment")
    from learning_dist_metrics.ldm import LDM
    from learning_dist_metrics.datasets import load_sample_data
 
## ########################## ##
## CREATE SIMULATE DATA SET ##
## ########################## ##
np.random.seed(20150408)
n_size = 100
c01 = [0, 0]
c02 = [1, 1]
 
## ####################### ##
## Generate group 01 data ##
## ####################### ##
theta = np.random.uniform(0, 2 * pi, n_size)
radius = np.random.uniform(0, 1, n_size)
err_x = np.random.normal(0, 0.2, n_size)
err_y = np.random.normal(0, 0.2, n_size)
 
x1 = []
x2 = []
for t, r, ex, ey in zip(theta, radius, err_x, err_y):
    x1.append( c01[0] + r * cos(t) + ex )
    x2.append( c01[1] + r * sin(t) + ey )
 
g1 = pd.DataFrame({
	"x1": x1,
    "x2": x2,
    "x3": uniform(0, 10, n_size),
    "label": [0] * n_size
    })
 
## ####################### ##
## Generate group 02 data ##
## ####################### ##
theta = np.random.uniform(0, 2 * pi, n_size)
radius = np.random.uniform(0, 1, n_size)
err_x = np.random.normal(0, 0.2, n_size)
err_y = np.random.normal(0, 0.2, n_size)
 
x1 = []
x2 = []
for t, r, ex, ey in zip(theta, radius, err_x, err_y):
    x1.append( c02[0] + r * cos(t) + ex )
    x2.append( c02[1] + r * sin(t) + ey )
 
g2 = pd.DataFrame({
    "x1": x1,
    "x2": x2,
    "x3": uniform(0, 10, n_size),
    "label": [1] * n_size
    })
## Concatenate two dataframes
db = g1.append(g2, ignore_index=True)
db.set_index(np.arange(db.shape[0]))
## Shuffle observations
db = db.loc[ np.random.choice(db.index, db.shape[0], replace=False) ]
## clean the environment
try:
    del g1, g2, t, r, ex, ey, theta, radius, err_x, err_y
except:
    pass
 
## ######################## ##
## Visualize the data ##
## ######################## ##
## def plot3d(df, **kwargs):
## fig = plt.figure()
## ax = fig.add_subplot(111, projection='3d')
## ax.scatter(df[:, 0], df[:, 1], df[:, 2], c = df[:,3])
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(db["x1"], db["x2"], db["x3"], c = db["label"])
 
## collect combinations of same class
idx_g1 = db.index[db.label == 0]
idx_g2 = db.index[db.label == 1]
 
all_com_g1 = [i for i in combinations(idx_g1, 2)]
all_com_g2 = [i for i in combinations(idx_g2, 2)]
 
sample_g1 = [ all_com_g1[i] for i in choice(range(len(all_com_g1)), 500) ]
sample_g2 = [ all_com_g2[i] for i in choice(range(len(all_com_g2)), 500) ]
sample_pair = sample_g1 + sample_g2
print "# of pairs in sample_pair: %d" % len(sample_pair)
 
## ############################# ##
## Training LDM ##
## ############################# ##
num_samples = [100, 500, 1000, 2000, 4000]
ratios = [1., .8, .6, .4, .2]
 
sim_pair_list_size = []
duration = []
fitted_trans_mtx = []
fitted_ratio = []
 
ldm = LDM()
 
for n_sample in num_samples:
    for ratio in ratios:
        # sample pair list prepration
        sample_g1 = [ all_com_g1[i] for i in choice(range(len(all_com_g1)), n_sample) ]
        sample_g2 = [ all_com_g2[i] for i in choice(range(len(all_com_g2)), n_sample * ratio) ]
        sample_pair = sample_g1 + sample_g2
 
        # model fitting
        start = time.time()
        ldm.fit(db.iloc[:, 1:].as_matrix(), sample_pair)
 
        # collect excution statistics and information
        duration.append(time.time() - start)
        sim_pair_list_size.append( len(sample_pair) )
        f        itted_trans_mtx.append( ldm.get_transform_matrix() )
        fitted_ratio.append( ldm.get_ratio )
 
print " *************************************** \n"
print " # of provided pair list of same class: %d \n" % len(sample_pair)
print " learned transformation matrix: %s \n" % ldm.get_transform_matrix()
print " Excuation time of fitting: %f s \n" % duration[-1]
print " **************************************** \n"

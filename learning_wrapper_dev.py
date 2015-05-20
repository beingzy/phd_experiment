# -*- coding: utf-8 -*-
"""
Algorithm Wrapper to implement the iteractive learning
process

Implement the initial iteration of learning by feeding
the entire data set and tried to learn the weighting vector



@author: beingzy
@create: 05/18/2015
@update: 05/19/2015
"""

import os
import sys
import glob

import numpy as np
import scipy as sp
import pandas as pd

from learning_dist_metrics.ldm import LDM
from learning_dist_metrics.dist_metrics import weighted_euclidean
from learning_dist_metrics.datasets import load_data
from UserBatch import SimScore, WeightedEuclidean, UserBatch

from matplotlib import pyplot

from scipy.stats import rayleigh
from scipy.stats import ks_2samp
from numpy import linspace

## ################## ##
## Setup envinronment ##
## ################## ##
ROOT_PATH = "/Users/beingzy/Documents/Projects/phd_experiment/"
DATA_PATH = ROOT_PATH + "data/sim_data_yi/"
IMG_PATH  = ROOT_PATH + "images/"
if os.getcwd() != ROOT_PATH:
    os.chdir(ROOT_PATH)
    print("Change the root directory successfully!")

## ################## ##
## Load data          ##
## ################## ##
##  simulate dataset with column names at 1st row
##  * user_profile.csv with x1 - x6, 6 variables, represent 2 group of users
##  * frienships.csv with isFriend, binary variable to indicate 1 for friends or 0 for
##    non-friends
##  * dist_mat.csv pre-calcualted pair-wise distance
users_df   = pd.read_csv(DATA_PATH + "users_profile.csv", header = 0, sep = ",")
friends_df = pd.read_csv(DATA_PATH + "friendships.csv", header = 0, sep = ",")
dist_df    = pd.read_csv(DATA_PATH + "dist_mat.csv", header = 0, sep = ",")

## ################### ##
## Data Processing     ##
## ################### ##
## friends_df is processed
## a. create a new column to denote the user pair
## b. exclude user-pair of non-friends
## c. drop the 'isFriend' columns
friends_df = friends_df[friends_df.isFriend == 1]
friends_df["pair"] = friends_df[["uid_a", "uid_b"]].apply(lambda x: (int(x[0]), int(x[1])), axis=1)
friends_df.drop("isFriend", axis=1, inplace=True)
friends_df = friends_df[["pair", "uid_a", "uid_b"]]
friends_df.head(3)

## ######################### ##
## Create Learning Container ##
## ######################### ##
##user_nx = UserBatch()
##user_nx.load_users(users_df)
##user_nx.load_friends(friends_df)

## Feed data into the Learning Distance Matrix Algorithm (LDM)
cols = ["x0", "x1", "x2", "x3", "x4", "x5"]

ldm = LDM()
ldm.fit(users_df[cols], friends_df.pair.as_matrix())

## ######################### ##
## Examine a single users's distance
## distribution of to-friends vs. to-non-friends
##
## step01: get the list of a user's friend
## step02: calculate distance of the user with friend-users (sim_dist_vec)
## step03: get the list of a user's non-friend (sample from all users but friends)
## step04: calculate distance of the user with non-friend-users (diff_dist_vec)
## step05: visualize the difference
## step06: fit with Rayleigh Distribution,
## step07: ks-test Rayleigh(friends) vs. Rayleigh(non-friends)
##
## data description:
## ----------------
## taste group 0 (%): ~ 70% (important features: x0 - x2)
## taste group 1 (%): ~ 30% (important features: x3 - x5)
##
## parameters:
## -----------
## a. threshold for significance
## b. minimum number of friends, if user having #friends LT min., is considered as majority
##
## issues:
## ------
## no implemented mechanism to handle users having too few friends
## solution:
## a. hold them off from learning
## b. consider them from the majority

## looping version
all_user_ids = list(set(users_df.ID))
the_user_id = 0
the_weights = ldm.get_transform_matrix()

ks_test_df = pd.DataFrame(columns = ["ID", "taste", "ks_pvalue"])

for counter, the_user_id in enumerate(all_user_ids):
	the_user_profile = users_df.ix[users_df.ID == the_user_id, cols].as_matrix()
	the_user_taste   = users_df.ix[users_df.ID == the_user_id, "decision_style"].as_matrix()[0]

	## step01
	## handle user_id present in both columns (uid_a, uid_b)
	#friends_id_ls = list(set(friends_df[friends_df.uid_a == the_user_id].uid_b.as_matrix()
	#                         friends_df[friends_df.uid_b == the_user_id].uid_a.as_matrix()
	#                         )
	#                   )
	friends_id_ls = friends_df[friends_df.uid_a == the_user_id].uid_b.as_matrix()
	friends_id_ls = list(set(friends_id_ls))
	## step02
	## calculate the distance with friend-user
	sim_dist_vec = []
	for f_id in friends_id_ls:
	    friend_profile = users_df.ix[users_df.ID == f_id, cols].as_matrix()
	    the_dist = weighted_euclidean(the_user_profile, friend_profile, the_weights)
	    sim_dist_vec.append(the_dist)

	## step03
	from numpy.random import choice
	non_friends_id_ls = [u for u in users_df.ID if u not in friends_id_ls + the_user_id]
	#non_friends_id_ls = choice()

	## step04
	diff_dist_vec = []
	for f_id in non_friends_id_ls:
	    friend_profile = users_df.ix[users_df.ID == f_id, cols].as_matrix()
	    the_dist = weighted_euclidean(the_user_profile, friend_profile, the_weights)
	    diff_dist_vec.append(the_dist)

	## step05
	##
	_max = max(sim_dist_vec + diff_dist_vec)
	_min = min(sim_dist_vec + diff_dist_vec)
	_nbins = 50
	bins = np.linspace(_min, _max, _nbins)

	pyplot.hist(sim_dist_vec, bins, alpha=0.5, label='friends')
	pyplot.hist(diff_dist_vec, bins, alpha=0.5, label='non-friends')
	pyplot.legend(loc='upper right')
	pyplot.title( "distance distrance of user (id: %d, taste: %d)" % (the_user_id, the_user_taste) )

 	## step06
	## processing dist data
	## exclude extreme small value
	sim_dist_vec = [i for i in sim_dist_vec if i > 0.0001]
	diff_dist_vec = [i for i in diff_dist_vec if i > 0.0001]

	friend_ray_param = rayleigh.fit(sim_dist_vec)
	nonfriend_ray_param = rayleigh.fit(diff_dist_vec)

	x = linspace(_min, _max, 100)
	# fitted distribution
	f_rayleigh_pdf = rayleigh.pdf(x, loc=friend_ray_param[0],scale=friend_ray_param[1])
	# original distribution
	nf_rayleigh_pdf = rayleigh.pdf(x, loc=nonfriend_ray_param[0], scale=nonfriend_ray_param[1])
	pyplot.plot(x, f_rayleigh_pdf ,'b-', x, nf_rayleigh_pdf ,'g-')

	#pyplot.show()
	file_name = "hist_id_%d.png" % the_user_id
	pyplot.savefig(IMG_PATH + file_name, format='png')
	pyplot.clf()

	## step07
	## ks-test
	res_ks_test = ks_2samp(sim_dist_vec, diff_dist_vec)
	ks_test_df.loc[counter] = [ the_user_id, the_user_taste, res_ks_test[1] ]


## ################################### ##
## 2nd round learning distance matrics ##
## ################################### ##
def hist_2d(x, y, title = None):
    """ Plot Histogram of two vectors, x and y
    """
    _max = max(sim_dist_vec + diff_dist_vec)
    _min = min(sim_dist_vec + diff_dist_vec)
    _nbins = 50
    bins = np.linspace(_min, _max, _nbins)

    pyplot.hist(x, bins, alpha=0.5, label='x')
    pyplot.hist(y, bins, alpha=0.5, label='y')
    pyplot.legend(loc='upper right')

    if title != None:
        pyplot.title(title)

    pyplot.show()
    pyplot.clf()

x = ks_test_df.ix[ks_test_df.taste == 0, "ks_pvalue"].as_matrix()
y = ks_test_df.ix[ks_test_df.taste == 1, "ks_pvalue"].as_matrix()
hist_2d(x, y)

## ########################
## split users into 2 groups
##
gamma = 0.20 ## threshod to split users
g0 = ks_test_df.ID[ ks_test_df.ks_pvalue <= gamma ]
g1 = ks_test_df.ID[ ks_test_df.ks_pvalue >  gamma ]
## g0: ~26 [0] vs. ~15 [1]
## g1: ~40 [0] vs. ~15 [1]

## process data for learning
##
## strategy 01
## friendship data having its either entity which is in the collection per pair
##
## strategy 02
## friendship data having its both entities which is in the collection per pair
friends_g0 = friends_df.ix[ friends_df.uid_a.isin(g0) & friends_df.uid_b.isin(g0) ]
friends_g1 = friends_df.ix[ friends_df.uid_a.isin(g1) & friends_df.uid_b.isin(g1) ]
## strategy 01 results:
## g0: [0.4, 0.54, 0.0, 0.06, 0.0, 0.0]
## g1: [0.37, 0.63, 0.0, 0.0, 0.0, 0.0]

## strategy 02 results:
## g0: [0.51, 0.32, 0.06, 0.08, 0.0, 0.02]
## g1: [0.4, 0.59, 0.0, 0.0, 0.0, 0.0]

## learning processing
ldm_g0 = LDM()
ldm_g1 = LDM()
## ldm.fit(users_df[cols], friends_df.pair.as_matrix())
ldm_g0.fit(users_df[cols], friends_g0.pair.as_matrix())
ldm_g1.fit(users_df[cols], friends_g1.pair.as_matrix())
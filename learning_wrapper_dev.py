# -*- coding: utf-8 -*-
"""
Algorithm Wrapper to implement the iteractive learning
process

Implement the initial iteration of learning by feeding
the entire data set and tried to learn the weighting vector

@dev_date: 05/24/2015
@description:
develop the iterative process following the initial learning:

[swaping learning]
a. pre-create two learning container to reprsent two
   separate groups
b. applying learned learning metrics to evaluate the
   goodness of the assumption that a user is a memember
   of taste group. if the evaluation test reject the
   assumption, the tested user will be assigned to the
   opposite group.
c. repeat the process until the a criteria had been met

@author: beingzy
@create: 05/18/2015
@update: 08/15/2015
"""

import os
import sys
import glob
import numpy as np
import scipy as sp
import pandas as pd
from matplotlib import pyplot
from scipy.stats import rayleigh
from scipy.stats import ks_2samp
from numpy import linspace
from numpy.random import choice
from numpy import floor

from learning_dist_metrics.ldm import LDM
from learning_dist_metrics.dist_metrics import weighted_euclidean

## ################## ##
## Setup envinronment ##
## ################## ##
# ROOT_PATH = "/Users/beingzy/Documents/Projects/phd_experiment/"
ROOT_PATH = "/home/beingzy/Documents/projects/phd_experiment/"
DATA_PATH = ROOT_PATH + "data/sim_data_yi/"
IMG_PATH  = ROOT_PATH + "images/"
if os.getcwd() != ROOT_PATH:
    os.chdir(ROOT_PATH)
    print("Change the root directory successfully!")

## ################## ##
## Load data          ##
## ################## ##
# simulate dataset with column names at 1st row
# * user_profile.csv with x1 - x6, 6 variables, represent 2 group of users
# * frienships.csv with isFriend, binary variable to indicate 1 for friends or
#   0 for non-friends
# * dist_mat.csv pre-calcualted pair-wise distance
users_df = pd.read_csv(DATA_PATH + "users_profile.csv", header=0, sep=",")
friends_df = pd.read_csv(DATA_PATH + "friendships.csv", header=0, sep=",")
dist_df = pd.read_csv(DATA_PATH + "dist_mat.csv", header=0, sep=",")

## ################### ##
## Data Processing     ##
## ################### ##
# friends_df is processed
# a. create a new column to denote the user pair
# b. exclude user-pair of non-friends
# c. drop the 'isFriend' columns
friends_df = friends_df[friends_df.isFriend == 1]
friends_df["pair"] = friends_df[["uid_a", "uid_b"]].apply(lambda x: (int(x[0]), int(x[1])), axis=1)
friends_df.drop("isFriend", axis=1, inplace=True)
friends_df = friends_df[["pair", "uid_a", "uid_b"]]

## ######################### ##
## Create Learning Container ##
## ######################### ##
##user_nx = UserBatch()
##user_nx.load_users(users_df)
##user_nx.load_friends(friends_df)
## Feed data into the Learning Distance Matrix Algorithm (LDM)
cols = ["x0", "x1", "x2", "x3", "x4", "x5"]

## subset users data to retain profile only
profile_df = users_df[["ID"] + cols]

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
# data description:
# ----------------
# taste group 0 (%): ~ 70% (important features: x0 - x2)
# taste group 1 (%): ~ 30% (important features: x3 - x5)
#
# parameters:
# -----------
# a. threshold for significance
# b. minimum number of friends, if user having #friends LT min., is considered as majority
#
# issues:
# ------
# no implemented mechanism to handle users having too few friends
# solution:
# a. hold them off from learning
#  b. consider them from the majority

# date: 05/23/2015
# task: wrap the logics into a function

# primary parameters:
all_user_ids = list(set(users_df.ID))
the_weights = ldm.get_transform_matrix()

# secondary parameters:
is_plot = False
image_name_prefix = "hist_id_"
out_image_dir = IMG_PATH

def user_grouped_dist(user_id, weights, profile_df, friends_networkx):
    """ Calculate distances between a user and whose friends
        and distance between a user and whose non-friends.
        The groupped distance vector will be output.

        Parameters:
        ----------
        * user_id: {integer}, the target user's ID
        * weights: {vector-like, float}, the vector of feature weights which
            is extracted by LDM().fit(x, y).get_transform_matrix()
        * profile_df: {matrix-like, pandas.DataFrame}, user profile dataframe
            with columns: ["ID", "x0" - "xn"]
        * friends_networkx: {networkx.Graph()}, Graph() object from Networkx
            to store the relationships information

        Returns:
        -------
        res: {list, list of integers}, a list of two lists, which store the dis
             -tances of either friends and non-friends separately.

        Examples:
        ---------
        weights = ldm().fit(df, friends_list).get_transform_matrix()
        profile_df = users_df[ ["ID"] + cols ]
        user_dist = user_grouped_dist(user_id = 0, weights = weights
            , profile_df, friends_df)
        print user_dist["friends"]
        print user_dist["nonfriends"]
    """
    cols = [col for col in profile_df.columns if col is not "ID"]
    # get the user profile information of the target users
    user_profile = profile_df.ix[profile_df.ID == user_id, cols].as_matrix()
    # get the user_id of friends of the target user
    friends_ls = friends_networkx.neighbors(user_id)
    all_ids = profile_df.ID
    non_friends_ls = [u for u in all_ids if u not in friends_ls + [user_id]]

    sim_dist_vec = []
    for f_id in friends_ls:
        friend_profile = profile_df.ix[profile_df.ID == f_id, cols].as_matrix()
        the_dist = weighted_euclidean(user_profile, friend_profile, weights)
        sim_dist_vec.append(the_dist)

    diff_dist_vec = []
    for nf_id in non_friends_ls:
        nonfriend_profile = profile_df.ix[profile_df.ID == nf_id, cols].as_matrix()
        the_dist = weighted_euclidean(user_profile, nonfriend_profile, weights)
        diff_dist_vec.append(the_dist)

    res = [sim_dist_vec, diff_dist_vec]
    return res


def user_dist_kstest(sim_dist_vec, diff_dist_vec,
                     fit_rayleigh=False, min_nobs=10, _n=100):
    """ Test the goodness of a given weights to defferentiate friend distance
        distributions and non-friend distance distributions of a given user.
        The distance distribution is considered to follow Rayleigh distribution.

    Parameters:
    ----------
    sim_dist_vec: {vector-like (list), float}, distances between friends
                  and the user
    diff_dist_vec: {vector-like (list), float}, distances between non-fri
                   -ends and the user
    fit_rayleigh: {boolean}, determine if fit data into Rayleigth distri
                  -bution
    min_nobs: {integer}, minmum number of observations required for compar
              -ing
    _n: {integer}, number of random samples generated from estimated
        distribution

    Returns:
    -------
    * res: {float}: p-value of ks-test with assumption that distances follow
            Rayleigh distribution.

    Examples:
    ---------
    pval = user_dist_kstest(sim_dist_vec, diff_dist_vec)
    """
    # is_valid = (len(sim_dist_vec) >= min_nobs) & \
    #           (len(diff_dist_vec) >= min_nobs) # not used yet
    if fit_rayleigh:
        friend_param = rayleigh.fit(sim_dist_vec)
        nonfriend_param = rayleigh.fit(diff_dist_vec)

        samp_friend = rayleigh.rvs(friend_param[0], friend_param[1], _n)
        samp_nonfriend = rayleigh.rvs(nonfriend_param[0], nonfriend_param[1],
                                      _n)

        # ouput p-value of ks-test
        res = ks_2samp(samp_friend, samp_nonfriend)[1]
    else:
        res = ks_2samp(sim_dist_vec, diff_dist_vec)[1]
    return res


def users_filter_by_weights(weights, profile_df, friends_networkx,
                            pval_threshold=1, mutate_rate=0.4,
                            min_friend_cnt=10, users_list=None):
    """ Split users into two groups, "keep" and "mutate", with respect to
        p-value of the ks-test on the null hypothesis that the distribution of
        friends' weighted distance is not significantly different from the
        couterpart for non-friends. Assume the weighted distances of each group
        follow Rayleigh distribution.

    Parameters:
    ----------
    weights: {vector-like, float}, the vector of feature weights which
        is extracted by LDM().fit(x, y).get_transform_matrix()
    users_list: {vector-like, integer}, the list of user id
    profile_df: {matrix-like, pandas.DataFrame}, user profile dataframe
        with columns: ["ID", "x0" - "xn"]
    friends_networkx: {networkx.Graph()}, Graph() object from Networkx to store
        the relationships information
    pval_threshold: {float}, the threshold for p-value to reject hypothesis
    min_friend_cnt: {integer}, drop users whose total of friends is less than
       this minimum count
    mutate_rate: {float}, a float value [0 - 1] determine the percentage of
       bad_fits member sent to mutation

    Returns:
    -------
    res: {list} grouped list of user ids
        res[0] stores all users whose null hypothesis does not holds;
        res[1] stores all users whose null hypothesis hold null hypothesis,
        given weights, distance distribution of all friends is significantly
        different from distance distribution of all non-friends

    Examples:
    --------
    weights = ldm().fit(df, friends_list).get_transform_matrix()
    profile_df = users_df[["ID"] + cols]
    grouped_users = users_filter_by_weights(weights,
                       profile_df, friends_df, pval_threshold = 0.10,
                       min_friend_cnt = 10)

    Notes:
    -----
    min_friend_cnt is not implemented
    """
    # all_users_ids = list(set(profile_df.ID))
    # users_list
    # container for users meeting different critiria
    pvals = []
    if users_list is None:
        users_list = list(profile_df.ix[:, 0])

    for uid in users_list:
        res_dists = user_grouped_dist(uid, weights, profile_df,
                                      friends_networkx=friends_networkx)
        pvals.append(user_dist_kstest(res_dists[0], res_dists[1]))

    sorted_id_pval = sorted(zip(users_list, pvals), key=lambda x: x[1])
    good_fits = [i for i, p in sorted_id_pval if p < pval_threshold]
    bad_fits = [i for i, p in sorted_id_pval if p >= pval_threshold]

    if len(bad_fits) > 0:
        mutate_size = floor(len(bad_fits) * mutate_rate)
        mutate_size = max(int(mutate_size), 1)

    id_mutate = bad_fits[:mutate_size]
    id_retain = good_fits + bad_fits[mutate_size:]
    res = [id_retain, id_mutate]
    return res


def ldm_train_with_list(users_list, profile_df, friends_df, retain_type=0):
    """ learning distance matrics with ldm() instance, provided with selected
        list of users.

    Parameters:
    -----------
    users_list: {vector-like, integer}, the list of user id
    profile_df: {matrix-like, pandas.DataFrame}, user profile dataframe
        with columns: ["ID", "x0" - "xn"]
    friends_df: {matrix-like, pandas.DataFrame}, pandas.DataFrame store pair of
        user ID(s) to represent connections with columns: ["uid_a", "uid_b"]
    friends_networkx: {networkx.Graph()}, Graph() object from Networkx to store
        the relationships information
    retain_type: {integer}, 0, adopting 'or' logic by keeping relationship in
        friends_df if either of entities is in user_list 1, adopting 'and'
        logic

    Returns:
    -------
    res: {vector-like, float}, output of ldm.get_transform_matrix()

    Examples:
    ---------
    new_dist_metrics = ldm_train_with_list(user_list, profile_df, friends_df)
    """
    ldm = LDM()
    if retain_type == 0:
        friends_df = friends_df.ix[friends_df.uid_a.isin(users_list) |
                                   friends_df.uid_b.isin(users_list)]
    else:
        friends_df = friends_df.ix[friends_df.uid_a.isin(users_list) &
                                   friends_df.uid_b.isin(users_list)]

    cols = profile_df.columns.drop("ID")
    ldm.fit(profile_df[cols], friends_df.pair.as_matrix())
    return ldm.get_transform_matrix()

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

## learning processing
ldm_g0 = LDM()
ldm_g1 = LDM()
## ldm.fit(users_df[cols], friends_df.pair.as_matrix())
ldm_g0.fit(users_df[cols], friends_g0.pair.as_matrix())
ldm_g1.fit(users_df[cols], friends_g1.pair.as_matrix())
## strategy 01 results:
## g0: [0.4, 0.54, 0.0, 0.06, 0.0, 0.0]
## g1: [0.37, 0.63, 0.0, 0.0, 0.0, 0.0]

## strategy 02 results:
## g0: [0.51, 0.32, 0.06, 0.08, 0.0, 0.02]
## g1: [0.4, 0.59, 0.0, 0.0, 0.0, 0.0]

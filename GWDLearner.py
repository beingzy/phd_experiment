import scipy as sp
import numpy as np
import pandas as pd

from scipy.stats import rayleigh
from scipy.stats import ks_2samp
from numpy import linspace
from numpy.random import choice
from networkx import Graph
from learning_dist_metrics.ldm import LDM
from learning_dist_metrics.dist_metrics import weighted_euclidean


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
        to store the relationships informat
    Returns:
    -------
    res: {list, list of integers}, a list of two lists, which store the distances
        of either friends and non-friends separately.

    Examples:
    ---------
    weights = ldm().fit(df, friends_list).get_transform_matrix()
    profile_df = users_df[ ["ID"] + cols ]
    user_dist = user_grouped_dist(user_id = 0, weights = weights
        , profile_df, friends_df)
    print user_dist["friends"]
    print user_dist["nonfriends"]"""
input info.:
----------
profile_df
friend_networkx

control parameters:
-------------------
t: fit score type

tuning parameter:
-----------------
threshold: cutoff value for kstest
c: regularization strength
min_delta_f: threshold for significant improvement
max_iter: maxmium number of trivial trial learning in a row
"""
# input info
k = 2
# user_profile

# tuing parameters
t = 2
c = 0.1
threshold = 0.5
min_delta_f = 0.02
max_iter = 10

# initiate the containers:
dist_metrics = init_dict_list(k) # distance metrics containers
fit_group = init_dict_list(k)    # members composition in fit groups
fit_pvals = init_dict_list(k)    # members' pvalue of KStest with their group distance metrics
unfit_group = init_dict_list(k)  # members is not considerd fit by its group distance metrics
unfit_pvals = init_dict_list(k)  # pvalues for members in unfit_group (maybe can be deleted)
buffer_group = []                # members are not considered having fit

# results value
fs_hist = []       # list of fit scores in sequence (lastest one is the last)
knowledge_pkg = [] # {index: {"dist_metrics", "fit_group", "buffer_group"}}

# calculate the the init distance metrics

# sampling is subset of users to calculate
# the distance metrics is good method

# dist_metrics: ldm() with subset of users
# fit_group: subsets of users
# buffer_group: useres are not sampled

# provide initial composition of fit_group
# and buffer_group for iterative learning
# procedure
# the even sampling strategy is implemeted
# here, however,
samp_size = len(all_uids) / k
samp_sizes = [samp_size] * k
all_uids_copy = [i for i in all_uids]

# generate k groups of sample user groups
for g, samp_size in zip(range(k), samp_sizes):
    # draw samples and assign them to fit_group
    samples = choice(all_uids_copy, samp_size, replace=False)
    fit_group[g] = list(samples)
    # remove samples from population pool
    for uid in samples:
        all_uids_copy.remove(uid)

# initiate fit user pvals
for g, uids in fit_group.iteritems():
    fit_pvals[g] = [0] * len(uids)

# if len(all_uids_copy) > 0:
#     buffer_group = all_uids_copy
# else:
#    buffer_group = []

_no_imp_counter = 0
_loop_counter = 0

while _no_imp_counter < max_iter:

    _loop_counter += 1
    print "%d iteration is in processing ..." % _loop_counter
    # step 01: learn distance metrics
    for g, uids in fit_group.iteritems():
        # learn distance metrics
        # here to update the computational mechanism
        dist = [np.random.uniform(0, 1, 1)[0] for i in range(4)]
        dist_metrics[g] = dist

    # step 02: update the member composite with updated group
    # distance metrics threshold is needed to be defined
    fit_group_copy = {k:[i for i in v] for k, v in fit_group.iteritems()}
    for g, uids in fit_group_copy.iteritems():
        target_dist = dist_metrics[g]
        for uid in uids:

            # calcualte the ks-pvalue with update distance metrics
            # target_dist
            pval = np.random.uniform(0, 1, 1)[0] #----- update needed ------- #

            if pval >= threshold:
                # remove the user and its information
                # from relevant container
                idx = [i for i, u in enumerate(fit_group[g]) if u == uid][0]
                fit_group[g].pop(idx)
                # fit_group[g].remove(uid)
                fit_pvals[g].pop(idx)

                # add the user to the unfit_group
                if g in unfit_group:
                    unfit_group[g].append(uid)
                else:
                    unfit_group[g] = [uid]

            else:
                idx = [i for i, u in enumerate(fit_group[g]) if u == uid][0]
                fit_pvals[g][idx] = pval

    tot_fit_group = np.sum([len(u) for g, u in fit_group.iteritems()])
    tot_unfit_group = np.sum([len(u) for g, u in unfit_group.iteritems()])
    tot_buffer_group = len(buffer_group)
    print "1) #fit: %d, #unfit: %d, #buffer: %d" % (tot_fit_group, tot_unfit_group, tot_buffer_group)

    # step 03: test members in unfit_group to see
    # if it has a good fit with other distmetrics
    # make a copy of the buffer group container
    buffer_group_copy = [i for i in buffer_group]
    if len(buffer_group_copy) > 0:
        for uid in buffer_group_copy:
            new_group, new_pval = find_fit_group(uid, dist_metrics, threshold)
            if new_group is not None:
                buffer_group.remove(uid)
                if new_group in fit_group:
                    fit_group[new_group].append(uid)
                    fit_pvals[new_group].append(new_pval)
                else:
                    fit_group[new_group] = [uid]
                    fit_pvals[new_group] = [new_pval]


    tot_fit_group = np.sum([len(u) for g, u in fit_group.iteritems()])
    tot_unfit_group = np.sum([len(u) for g, u in unfit_group.iteritems()])
    tot_buffer_group = len(buffer_group)
    print "2) #fit: %d, #unfit: %d, #buffer: %d" % (tot_fit_group, tot_unfit_group, tot_buffer_group)

    unfit_group_copy = {k:[i for i in v] for k, v in unfit_group.iteritems()}
    for g, uids in unfit_group_copy.iteritems():
        for uid in uids:
            new_group, new_pval = find_fit_group(uid, dist_metrics, threshold, g)
            unfit_group[g].remove(uid)

            if new_pval is None:
                buffer_group.append(uid)
            else:
                if new_group in fit_group:
                    fit_group[new_group].append(uid)
                    fit_pvals[new_group].append(new_pval)
                else:
                    fit_group[new_group] = [uid]
                    fit_pvals[new_group] = [new_pval]

    tot_fit_group = np.sum([len(u) for g, u in fit_group.iteritems()])
    tot_unfit_group = np.sum([len(u) for g, u in unfit_group.iteritems()])
    tot_buffer_group = len(buffer_group)
    print "3) #fit: %d, #unfit: %d, #buffer: %d" % (tot_fit_group, tot_unfit_group, tot_buffer_group)

    # step 04: calculate fit score
    fs = get_fit_score(fit_pvals, buffer_group, c=c, t=1)
    fs_hist.append(fs)

    # step 05: evaluate stop criteria
    package = {"dist_metrics": dist_metrics,
               "fit_group": fit_group,
               "buffer_group": buffer_group}

    knowledge_pkg.append(package)
    best_fs = min(fs_hist)

    if best_fs - fs <= min_delta_f:
        _no_imp_counter += _no_imp_counter
    else:
        _no_imp_counter = 0

    # print "fit score (type-%d): %.3f" % (t, fs)
    # print "best fit score: %.3f" % best_fs
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
                     fit_rayleigh=False, _n=100):

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
        samp_nonfriend = rayleigh.rvs(nonfriend_param[0], nonfriend_param[1], _n)

        # ouput p-value of ks-test
        res = ks_2samp(samp_friend, samp_nonfriend)[1]
    else:
        res = ks_2samp(sim_dist_vec, diff_dist_vec)[1]

    return res


def users_filter_by_weights(weights, profile_df, friends_networkx,
                            pval_threshold=0.5,
                            mutate_rate=0.4,
                            min_friend_cnt=10,
                            users_list=None,
                            fit_rayleigh=False,
                            _n=1000,
                            is_debug=False):

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
    fit_rayleigh: {boolean}, determine if fit data into Rayleigth distri
                  -bution
    _n: {integer}, number of random samples generated from estimated
        distribution
    is_debug: {boolean}, to control if it yeilds by-product information

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
        res_dists = user_grouped_dist(uid, weights, profile_df, friends_networkx)
        pval = user_dist_kstest(res_dists[0], res_dists[1], fit_rayleigh, _n)
        pvals.append(pval)

    sorted_id_pval = sorted(zip(users_list, pvals), key=lambda x: x[1])

    if is_debug:
        good_fits = [i for i, p in sorted_id_pval if p < pval_threshold]
        bad_fits = [i for i, p in sorted_id_pval if p >= pval_threshold]
        good_pvals = [p for i, p in sorted_id_pval if p < pval_threshold]
        bad_pvals = [p for i, p in sorted_id_pval if p >= pval_threshold]
    else:
        good_fits = [i for i, p in sorted_id_pval if p < pval_threshold]
        bad_fits = [i for i, p in sorted_id_pval if p >= pval_threshold]

    if len(bad_fits) > 0:
        mutate_size = np.ceil(len(bad_fits) * mutate_rate)
        mutate_size = max(int(mutate_size), 1)
        id_retain = good_fits + bad_fits[mutate_size:]
        id_mutate = bad_fits[:mutate_size]
        # split pval
        if is_debug:
            if len(good_pvals) > 0 or len(bad_pvals) > 0:
                pval_retain = good_pvals + bad_pvals[mutate_size:]
                pval_mutate = bad_pvals[mutate_size:]
    else:
        id_retain = good_fits
        id_mutate = bad_fits

        if is_debug:
            if len(good_pvals) > 0 or len(bad_pvals) > 0:
                pval_retain = pval_retain
                pval_mutate = bad_pvals

    if is_debug:
        res = [id_retain, id_mutate, pval_retain, pval_mutate]
    else:
        res = [id_retain, id_mutate]

    return res


def ldm_train_with_list(users_list, profile_df, friends, retain_type=1):
    """ learning distance matrics with ldm() instance, provided with selected
        list of users.

    Parameters:
    -----------
    users_list: {vector-like, integer}, the list of user id
    profile_df: {matrix-like, pandas.DataFrame}, user profile dataframe
        with columns: ["ID", "x0" - "xn"]
    friends: {list of tuple}, each tuple keeps a pair of user id
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
    if retain_type == 0:
        friends = [(a, b) for a, b in friends if \
            a in users_list or b in users_list]
    else:
        friends = [(a, b) for a, b in friends if \
            a in users_list and b in users_list]

    ldm = LDM()
    ldm.fit(profile_df, friends)
    weight_vec = ldm.get_transform_matrix()
    return weight_vec


def hyper_parameter_tester(weights_a, weights_b, fit_rayleigh, num):

    """
    """

    num_friends = []
    num_nonfriends = []
    ks_pvals_right = []
    ks_pvals_wrong = []

    for uid in tg0_ids:
        # Compare the distribution of a user's distances of all of his/her friends
        # against the distribuiton of a users's distances of all of his/her non-friends,
        # The collection of non-friends may include those users of two categories with
        # respect to their relationships to the target user:
        # a. the users who are not likened by the target users
        # b. the users who are likely to be befriended by the users however
        #    the users do not have a change to be exposed to her/him.
        sim_dists, diff_dists = user_grouped_dist(uid, weights_a, profile_df, fnx)
        pval = user_dist_kstest(sim_dists, diff_dists, fit_rayleigh=fit_rayleigh, _n = num)
        ks_pvals_right.append(pval)

        sim_dists, diff_dists = user_grouped_dist(uid, weights_b, profile_df, fnx)
        pval = user_dist_kstest(sim_dists, diff_dists, fit_rayleigh=fit_rayleigh, _n = num)
        ks_pvals_wrong.append(pval)

        num_friends.append(len(sim_dists))
        num_nonfriends.append(len(diff_dists))

    res_report = pd.DataFrame({"ID": tg0_ids,
    	                       "num_friends": num_friends,
                               "num_nonfriends": num_nonfriends,
                               "true_pval": ks_pvals_right,
                               "wrong_pval": ks_pvals_wrong})
    return res_report


def learning_wrapper():
"""
input info.:
----------
profile_df
friend_networkx

control parameters:
-------------------
t: fit score type

tuning parameter:
-----------------
threshold: cutoff value for kstest
c: regularization strength
min_delta_f: threshold for significant improvement
max_iter: maxmium number of trivial trial learning in a row
"""
# input info
k = 2
# user_profile

# tuing parameters
t = 2
c = 0.1
threshold = 0.5
n = 1000 # ks-test sample size for rayleigh
min_delta_f = 0.02
max_iter = 10

# initiate the containers:
dist_metrics = init_dict_list(k) # distance metrics containers
fit_group = init_dict_list(k)    # members composition in fit groups
fit_pvals = init_dict_list(k)    # members' pvalue of KStest with their group distance metrics
unfit_group = init_dict_list(k)  # members is not considerd fit by its group distance metrics
unfit_pvals = init_dict_list(k)  # pvalues for members in unfit_group (maybe can be deleted)
buffer_group = []                # members are not considered having fit

# results value
fs_hist = []       # list of fit scores in sequence (lastest one is the last)
knowledge_pkg = [] # {index: {"dist_metrics", "fit_group", "buffer_group"}}

# calculate the the init distance metrics

# sampling is subset of users to calculate
# the distance metrics is good method

# dist_metrics: ldm() with subset of users
# fit_group: subsets of users
# buffer_group: useres are not sampled

# provide initial composition of fit_group
# and buffer_group for iterative learning
# procedure
# the even sampling strategy is implemeted
# here, however,
samp_size = len(all_uids) / k
samp_sizes = [samp_size] * k
all_uids_copy = [i for i in all_uids]

# generate k groups of sample user groups
for g, samp_size in zip(range(k), samp_sizes):
    # draw samples and assign them to fit_group
    samples = choice(all_uids_copy, samp_size, replace=False)
    fit_group[g] = list(samples)
    # remove samples from population pool
    for uid in samples:
        all_uids_copy.remove(uid)

# initiate fit user pvals
for g, uids in fit_group.iteritems():
    fit_pvals[g] = [0] * len(uids)

# if len(all_uids_copy) > 0:
#     buffer_group = all_uids_copy
# else:
#    buffer_group = []

_no_imp_counter = 0
_loop_counter = 0

while _no_imp_counter < max_iter:

    _loop_counter += 1
    print "%d iteration is in processing ..." % _loop_counter

    # step 01: learn distance metrics
    for g, uids in fit_group.iteritems():
        # learn distance metrics
        # here to update the computational mechanism
        # dist = [np.random.uniform(0, 1, 1)[0] for i in range(4)]
        dist = ldm_train_with_list(uids, profile_df, friends_ls)
        dist_metrics[g] = dist

    # step 02: update the member composite with updated group
    # distance metrics threshold is needed to be defined
    fit_group_copy = {k:[i for i in v] for k, v in fit_group.iteritems()}
    for g, uids in fit_group_copy.iteritems():
        target_dist = dist_metrics[g]
        for uid in uids:

            # calcualte the ks-pvalue with update distance metrics
            # target_dist
            # pval = np.random.uniform(0, 1, 1)[0] #----- update needed ------- #
            sdist, ddist = user_grouped_dist(uid, dist_metrics, profile_df,
                                         friends_networkx)
            pval = user_dist_kstest(sdist, ddist, fit_rayleigh=True, _n=n)

            if pval >= threshold:
                # remove the user and its information
                # from relevant container
                idx = [i for i, u in enumerate(fit_group[g]) if u == uid][0]
                fit_group[g].pop(idx)
                # fit_group[g].remove(uid)
                fit_pvals[g].pop(idx)

                # add the user to the unfit_group
                if g in unfit_group:
                    unfit_group[g].append(uid)
                else:
                    unfit_group[g] = [uid]

            else:
                idx = [i for i, u in enumerate(fit_group[g]) if u == uid][0]
                fit_pvals[g][idx] = pval

    # tot_fit_group = np.sum([len(u) for g, u in fit_group.iteritems()])
    # tot_unfit_group = np.sum([len(u) for g, u in unfit_group.iteritems()])
    # tot_buffer_group = len(buffer_group)
    # print "1) #fit: %d, #unfit: %d, #buffer: %d" % (tot_fit_group, tot_unfit_group, tot_buffer_group)

    # step 03: test members in unfit_group to see
    # if it has a good fit with other distmetrics
    # make a copy of the buffer group container
    buffer_group_copy = [i for i in buffer_group]
    if len(buffer_group_copy) > 0:
        for uid in buffer_group_copy:
            new_group, new_pval = find_fit_group(uid, dist_metrics, threshold)
            if new_group is not None:
                buffer_group.remove(uid)
                if new_group in fit_group:
                    fit_group[new_group].append(uid)
                    fit_pvals[new_group].append(new_pval)
                else:
                    fit_group[new_group] = [uid]
                    fit_pvals[new_group] = [new_pval]

    unfit_group_copy = {k:[i for i in v] for k, v in unfit_group.iteritems()}
    for g, uids in unfit_group_copy.iteritems():
        for uid in uids:
            new_group, new_pval = find_fit_group(uid, dist_metrics, threshold, g)
            unfit_group[g].remove(uid)

            if new_pval is None:
                buffer_group.append(uid)
            else:
                if new_group in fit_group:
                    fit_group[new_group].append(uid)
                    fit_pvals[new_group].append(new_pval)
                else:
                    fit_group[new_group] = [uid]
                    fit_pvals[new_group] = [new_pval]

    # step 04: calculate fit score
    fs = get_fit_score(fit_pvals, buffer_group, c=c, t=1)
    fs_hist.append(fs)

    # step 05: evaluate stop criteria
    package = {"dist_metrics": dist_metrics,
               "fit_group": fit_group,
               "buffer_group": buffer_group}

    knowledge_pkg.append(package)
    best_fs = min(fs_hist)

    if best_fs - fs <= min_delta_f:
        _no_imp_counter += _no_imp_counter
    else:
        _no_imp_counter = 0

    # print "fit score (type-%d): %.3f" % (t, fs)
    # print "best fit score: %.3f" % best_fs
best_idx = [fs for fs in fs_hist if fs == best_fs]
best_knowledge = knowledge_pkg[best_idx]
return (best_knowledge, best_fs)

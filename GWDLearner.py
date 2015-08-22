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

from rpy2 import robjects
from rpy2.robjects.packages import importr
from rpy2.robjects.numpy2ri import numpy2ri

robjects.conversion.py2ri = numpy2ri
rstats = importr("stats")


def user_grouped_dist(user_id, weights, profile_df, friend_networkx):
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
    * friend_networkx: {networkx.Graph()}, Graph() object from Networkx
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
    print user_dist["nonfriends"]
    """
    cols = [col for col in profile_df.columns if col is not "ID"]
    # get the user profile information of the target users
    user_profile = profile_df.ix[profile_df.ID == user_id, cols].as_matrix()
    # user_row = profile_df.ix[profile_df.ID == user_id, cols]
    # user_profile = user_row.values.tolist()[0]
    # get the user_id of friends of the target user
    friends_ls = friend_networkx.neighbors(user_id)
    all_ids = profile_df.ID
    non_friends_ls = [u for u in all_ids if u not in friends_ls + [user_id]]

    sim_dist_vec = []
    for f_id in friends_ls:
        friend_profile = profile_df.ix[profile_df.ID == f_id, cols].as_matrix()
        # friend_row = profile_df.ix[profile_df.ID == f_id, cols]
        # friend_profile = friend_row.values.tolist()[0]
        the_dist = weighted_euclidean(user_profile, friend_profile, weights)
        sim_dist_vec.append(the_dist)

    diff_dist_vec = []
    for nf_id in non_friends_ls:
        nonfriend_profile = profile_df.ix[profile_df.ID == nf_id, cols].as_matrix()
        the_dist = weighted_euclidean(user_profile, nonfriend_profile, weights)
        diff_dist_vec.append(the_dist)

    res = [sim_dist_vec, diff_dist_vec]
    return res


def kstest_2samp_greater(x, y):
    """ Calcualte the test statistics and Pvalue for
        KS-test with two samples

        Hypothesis:
        H0: distr.(x) >= distr.(y) # not less than
        H1: distr.(x) < distr.(y)

        Pramaters:
        ----------
        x: {vector-like}
        y: {vector-like}

        Returns:
        -------
        cv, pval: {tuple}, test statistics, pvalue
    """

    greater = np.array(["less"], dtype="str")
    res = rstats.ks_test(x, y, alternative=greater)
    ts, pval = res[0][0], res[1][0]
    return ts, pval

def user_dist_kstest(sim_dist_vec, diff_dist_vec,
                     fit_rayleigh=False, _n=100):

    """ Test the goodness of a given weights to defferentiate friend distance
        distributions and non-friend distance distributions of a given user.
        The distance distribution can be assumed to follow Rayleigh distribution.

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
    # convert list to numpy.arrray, which can be
    # automatice transfer to R readable objects
    # for R-function, if the proper setting is
    # configured
    sim_dist_vec = np.array(sim_dist_vec)
    diff_dist_vec = np.array(diff_dist_vec)

    if fit_rayleigh:
        friend_param = rayleigh.fit(sim_dist_vec)
        nonfriend_param = rayleigh.fit(diff_dist_vec)

        samp_friend = rayleigh.rvs(friend_param[0], friend_param[1], _n)
        samp_nonfriend = rayleigh.rvs(nonfriend_param[0], nonfriend_param[1], _n)

        # ouput p-value of ks-test
        res = kstest_2samp_greater(samp_friend, samp_nonfriend)[1]
    else:
        res = kstest_2samp_greater(sim_dist_vec, diff_dist_vec)[1]

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

    pvals = []
    if users_list is None:
        users_list = list(profile_df.ix[:, 0])

    for uid in users_list:
        res_dists = user_grouped_dist(uid, weights, profile_df, friends_networkx)
        pval = user_dist_kstest(res_dists[0], res_dists[1], fit_rayleigh, _n)
        pvals.append(pval)

    sorted_id_pval = sorted(zip(users_list, pvals), key=lambda x: x[1])

    if is_debug:
        good_fits = [i for i, p in sorted_id_pval if p >= pval_threshold]
        bad_fits = [i for i, p in sorted_id_pval if p < pval_threshold]
        good_pvals = [p for i, p in sorted_id_pval if p >= pval_threshold]
        bad_pvals = [p for i, p in sorted_id_pval if p < pval_threshold]
    else:
        good_fits = [i for i, p in sorted_id_pval if p >= pval_threshold]
        bad_fits = [i for i, p in sorted_id_pval if p < pval_threshold]

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

def init_embed_list(n):
    """
    """
    ls = []
    for i in range(n):
        ls.append([])
    return ls

def init_dict_list(k):
    """ create dictionary with k items, each
        item is a empty list
    """
    res_dict = {}
    for i in range(k):
        res_dict[i] = []
    return res_dict

def find_fit_group(uid, dist_metrics, profile_df,
                   friend_networkx, threshold=0.5,
                   current_group=None, fit_rayleigh=False):
    """ calculate user p-value for the distance metrics of
        each group

    Parameters:
    ----------
    uid: {integer}, user id
    dist_metrics: {dictionary}, all {index: distance_metrics}
    profile_df: {DataFrame}, user profile includes "ID" column
    friend_networkx: {networkx.Graph}, user relationships
    threshold: {float}, threshold for qualifying pvalue of ks-test
    current_group: {integer}, group index
    fit_rayleigh: {boolean}

    Resutls:
    --------
    res: {list}, [group_idx, pvalue]
    """
    if current_group is None:
        other_group = dist_metrics.keys()
        other_dist_metrics = dist_metrics.values()
    else:
        other_group = [i for i in dist_metrics.keys() if i != current_group]
        other_dist_metrics = [d for g, d in dist_metrics.iteritems() if g != current_group]

    if len(other_dist_metrics) > 0:
        # only excute this is at least one alternative group
        pvals = []

        for d in other_dist_metrics:
            # loop through all distance metrics and calculate
            # p-value of ks-test by applying it to the user
            # relationships
            sdist, ddist = user_grouped_dist(user_id=uid, weights=d,
                        profile_df=profile_df, friend_networkx=friend_networkx)
            pval = user_dist_kstest(sim_dist_vec=sdist, diff_dist_vec=ddist,
                                fit_rayleigh=fit_rayleigh, _n=1000)
            pvals.append(pval)

        max_pval = max(pvals)
        max_index = [i for i, p in enumerate(pvals) if p == max_pval][0]
        best_group = other_group[max_index]

        if max_pval < threshold:
            # reject null hypothesis
            best_group = None
            max_pval = None

    else:
        best_group = None
        max_pval = None

    return (best_group, max_pval)

def get_fit_score(fit_pvals, buffer_group, c):
    """ calculate the fit score given the member composite
        and its pvalues with its group distance metrics, with
        c determinng the strength of penalty for keeping a
        larger number of users in buffer_group

    Parameters:
    -----------
    fit_pvals: {dict}, {index: [pvalues]}
    buffer_group: {list}, [userid, ...]
    c: {float},
    t: {integer} 1, 2 or 3, type of fit score

    Returns:
    --------
    fit_score: {float}, fit score, a smaller value indidcate
                a overall better fit

    Examples:
    ---------
    fit_group = fit_group
    fit_pvals = fit_pvals
    buffer_group = buffer_group
    c = 0.1
    fscore = get_fit_score(fit_group, fit_pvals, buffer_group, c)
    """

    # weighted sum of pvalues
    wsum_pval = 0
    num_users = 0
    for g, v in fit_pvals.iteritems():
        wsum_pval += sum(np.array(v) * 1.0) * (len(v) * len(v))
        num_users += len(v)
    wsum_pval = wsum_pval * 1.0 / num_users

    penalty = c * len(buffer_group)
    fit_score = wsum_pval - penalty # smaller value indicates a better overall fit

    return fit_score

def drawDropouts(users, pvals, dropout=0.1, desc=False):
    """ select a defined number of users from users
        list, based on dropout rate.

    Parameters:
    -----------
    * users, list
    * pvals, list
    * dropout, float, dropout rate
    * desc, boolean, True for sorting in descending order

    Returns:
    -------
    res: tuple, (users, pvals, user_dropout)
    """

    users_copy = [i for i in users]
    pvals_copy = [i for i in pvals]
    sort_idx = sorted(range(len(users_copy)), \
                      key=lambda k:pvals_copy[k],\
                      reverse=desc)
    # calculate number of users for dropout
    ndropout = int(np.ceil(len(users_copy) * dropout))

    if len(users_copy) > ndropout:
        if len(users_copy) >= 2:
            user_dropout = []
            pval_dropout = []
            remove_idx = sort_idx[:ndropout]
            for ridx in remove_idx:
                u = users_copy[ridx]
                p = pvals_copy[ridx]
                user_dropout.append(u)
                pval_dropout.append(p)
            # eleminate values for removed items
            users_copy = [v for i, v in enumerate(users_copy) \
                          if i not in remove_idx]
            pvals_copy = [v for i, v in enumerate(pvals_copy) \
                          if i not in remove_idx]

    return (users_copy, pvals_copy, user_dropout, pval_dropout)


def learning_wrapper(profile_df, friends_pair, k, c=0.1,
                     threshold_max=0.10, threshold_min=0.05,
                     min_size_group=10, min_delta_f=5,
                     dropout_rate=0.2,
                     max_iter=50, cum_iter=300, fit_rayleigh=False,
                     n=1000, verbose=False):
    """ learn the groupings and group-wise distance metrics

        1. "treshold" is fixed by treshold_max, the larger value will lead
        to more aggressive learning to make more member qualifying for
        group change. It should be considered an avenuae to gradaully
        reduce threshold for KS-test to mange the learning rate.
        2. The "dropout_rate" should be considered to vary over
        iteration as well. A decreasing dropout_rate promotes the converage
        of learning.

    Parameters:
    ----------
    profile_df: {pd.DataFrame}, with column ID and other attributes
    friends_pair: {list}, consisted of tuples of user id pairs
    k: {integer}, # of groups in the population
    c: {float}, strength of penalty for larger size of buffer group
    threshold_max: {float}, from 0 to 1, the initial threshold for ks-test
    threshold_min: {float}, form 0 to 1, the mimum possible threhsold for ks-test
    min_size_group: {integer}, minimal group size
    min_delta_f: {float}, minmal reduction considered substantial improvement
    max_iter: {integer}, maxmium number of sequential iterations with
        non-substantial improvement in fit score
    cum_iter: {integer}, overall maximum learning iteration
    fit_rayleigh: {boolean}, boolean fit rayleigh distribution to generate random
        data to compare two distance distributions
    n: {integer}, the samples of Rayleigh distribution for ks-test,
        it influence the sensitivity of KS-test
    verbose: {boolean}, display inprocess information

    Returns:
    -------
    res: {tuple}, (best_knowledge, best_fs)
    best_knowledge: {dictionary}, {"dist_metrics", "fit_group", "buffer_group"}
    best_fs: {float}, best fit score

    Examples:
    --------
    learning_wrapper(profile_df, friends_pair, k=2)
    """

    from networkx import Graph

    # convert pair-wise relationship data into Graph
    friend_networkx = Graph()
    friend_networkx.add_edges_from(friends_pair)

    # initiate the containers:
    # dist_metrics: ldm() with subset of users
    # fit_group: subsets of users
    # buffer_group: useres are not sampled
    dist_metrics = init_dict_list(k) # distance metrics containers
    fit_group    = init_dict_list(k) # members composition in fit groups
    fit_pvals    = init_dict_list(k) # members' pvalue of KStest with their group distance metrics
    unfit_group  = init_dict_list(k) # members is not considerd fit by its group distance metrics
    unfit_pvals  = init_dict_list(k) # pvalues for members in unfit_group (maybe can be deleted)
    buffer_group = []                # members are not considered having fit

    # results container
    fs_hist = []       # list of fit scores in sequence (lastest one is the last)
    knowledge_pkg = [] # {index: {"dist_metrics", "fit_group", "buffer_group"}}
    threshold = threshold_max

    # provide initial composition of fit_group  and buffer_group for iterative
    # learning procedure the even size sampling strategy is implemeted here,
    # however, benford's law can be used as alternative stratgey
    all_uids = list(set(profile_df.ID))
    samp_size = len(all_uids) / k
    samp_sizes = [samp_size] * k
    all_uids_copy = [i for i in all_uids]

    print "Initiating ..."

    # generate k groups of sample user groups
    for g, samp_size in zip(range(k), samp_sizes):
        # draw samples and assign them to fit_group
        samples = choice(all_uids_copy, samp_size, replace=False)
        fit_group[g] = list(samples)
        # remove sampled users from population pool
        for uid in samples:
            all_uids_copy.remove(uid)

    # initiate fit user pvals
    for g, uids in fit_group.iteritems():
        fit_pvals[g] = [0] * len(uids)

    # time counter()
    durations = []

    # iteration meters
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
            if len(uids) >= min_size_group:
                # ldm_train_with_list(users_list, profile_df, friends, retain_type=1)
                dist = ldm_train_with_list(users_list=uids,
                    profile_df=profile_df, friends=friends_pair)
                dist_metrics[g] = dist
            else:
                num_feat = profile_df.shape[1] - 1
                dist_metrics[g] = [1] * num_feat

        # step 02: update the member composite with updated group
        # distance metrics threshold is needed to be defined
        fit_group_copy = {k:[i for i in v] for k, v in fit_group.iteritems()}
        for g, uids in fit_group_copy.iteritems():
            target_dist = dist_metrics[g]

            for uid in uids:
                sdist, ddist = user_grouped_dist(uid, dist_metrics, profile_df,
                                          friend_networkx)
                pval = user_dist_kstest(sdist, ddist, fit_rayleigh=fit_rayleigh, _n=n)

                if pval < threshold:
                    # H0 does not hold, user should be re-assign to differnt
                    # group. remove from the current group and keep in the
                    # unfit group
                    idx = [i for i, u in enumerate(fit_group[g]) if u == uid][0]
                    fit_group[g].pop(idx)
                    fit_pvals[g].pop(idx)
                     # add the user to the unfit_group
                    if g in unfit_group:
                        unfit_group[g].append(uid)
                    else:
                        unfit_group[g] = [uid]

                else:
                    # H0 hold, update the pval
                    idx = [i for i, u in enumerate(fit_group[g]) if u == uid][0]
                    fit_pvals[g][idx] = pval

        if verbose:
            tot_fit_group = np.sum([len(u) for g, u in fit_group.iteritems()])
            tot_unfit_group = np.sum([len(u) for g, u in unfit_group.iteritems()])
            tot_buffer_group = len(buffer_group)
            print "1) #fit: %d, #unfit: %d, #buffer: %d" % (tot_fit_group,
            tot_unfit_group, tot_buffer_group)

        # step 03: test members in unfit_group to see
        # if it has a good fit with other dist metrics
        # make a copy of the buffer group container
        buffer_group_copy = [i for i in buffer_group]
        if len(buffer_group_copy) > 0:
            for uid in buffer_group_copy:
                new_group, new_pval = find_fit_group(uid, dist_metrics,
                    profile_df, friend_networkx, threshold, fit_rayleigh=fit_rayleigh)
                if new_group is not None:
                    buffer_group.remove(uid)
                    #gix = [i for i in fit_group.keys() if i==fit_group]
                    if new_group in fit_group:
                        fit_group[new_group].append(uid)
                        fit_pvals[new_group].append(new_pval)
                    else:
                        fit_group[new_group] = [uid]
                        fit_pvals[new_group] = [new_pval]

        if verbose:
            tot_fit_group = np.sum([len(u) for g, u in fit_group.iteritems()])
            tot_unfit_group = np.sum([len(u) for g, u in unfit_group.iteritems()])
            tot_buffer_group = len(buffer_group)
            print "1) #fit: %d, #unfit: %d, #buffer: %d" % (tot_fit_group,
            tot_unfit_group, tot_buffer_group)

        unfit_group_copy = {k:[i for i in v] for k, v in unfit_group.iteritems()}
        for g, uids in unfit_group_copy.iteritems():
            for uid in uids:
                new_group, new_pval = find_fit_group(uid, dist_metrics,
                    profile_df, friend_networkx, threshold, g, fit_rayleigh)
                unfit_group[g].remove(uid)

                if new_pval is None:
                    buffer_group.append(uid)
                else:
                    gix = [i for i in fit_group.keys() if i==fit_group]
                    if new_group in fit_group:
                        fit_group[new_group].append(uid)
                        fit_pvals[new_group].append(new_pval)
                    else:
                        fit_group[new_group] = [uid]
                        fit_pvals[new_group] = [new_pval]

        if verbose:
            tot_fit_group = np.sum([len(u) for g, u in fit_group.iteritems()])
            tot_unfit_group = np.sum([len(u) for g, u in unfit_group.iteritems()])
            tot_buffer_group = len(buffer_group)
            print "1) #fit: %d, #unfit: %d, #buffer: %d" % (tot_fit_group,
                tot_unfit_group, tot_buffer_group)

        # step 04: calculate fit score
        fs = get_fit_score(fit_pvals, buffer_group, c=c)
        try:
            prev_fs_best = max(fs_hist)
        except:
            prev_fs_best = 0
        fs_hist.append(fs)

        # step 05: evaluate stop criteria
        package = {"dist_metrics": dist_metrics,
                   "fit_group": fit_group,
                   "buffer_group": buffer_group}

        knowledge_pkg.append(package)
        #best_fs = max(fs_hist)

        if fs - prev_fs_best >= min_delta_f:
            # effective learning keep the momentum
            _no_imp_counter = 0
        else:
            # non-substantial improvement
            _no_imp_counter += 1
            user_dropout_dict = {}
            pval_dropout_dict = {}
            print "** dropout is activating ...\n"
            for g, uids in fit_group.iteritems():
                try:
                    pvals = fit_pvals[g]
                    new_uids, new_pvals, dropout_users, dropout_pvals = \
                        drawDropouts(uids, pvals, dropout_rate, desc=False)
                    fit_group[g] = list(new_uids)
                    fit_pvals[g] = list(new_pvals)
                    user_dropout_dict[g] = dropout_users
                    pval_dropout_dict[g] = dropout_pvals

                except:
                    print 'error pop up for dropouting! \n'

            # randomly reassign users to other group
            if len(user_dropout_dict) > 0:
                for g, uids in user_dropout_dict.iteritems():
                    pvals  = pval_dropout_dict[g]
                    groups = fit_group.keys()
                    if len(groups) > 1:
                        groups.remove(g)
                        tg = choice(groups, 1)[0]
                        fit_group[tg].extend(uids)
                        fit_pvals[tg].extend(pvals)

    # print "fit score (type-%d): %.3f" % (t, fs)
    # print "best fit score: %.3f" % best_fs
    best_fs = max(fs_hist)
    best_idx = [i for i, fs in enumerate(fs_hist) if fs == best_fs]
    best_knowledge = knowledge_pkg[best_idx[0]]

    if verbose:
        return (best_knowledge, best_fs, knowledge_pkg)
    else:
        return (best_knowledge, best_fs)

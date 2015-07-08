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
# samping is subset of users to calculate
# the distance metrics is good method

# dist_metrics: ldm() with subset of users
# fit_group: subsets of users
# buffer_group: useres are not sampled

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

    # step 02: update the member composite with updated group distance metrics
    # threshold is needed to be defined
    fit_group_copy = fit_group.copy()
    for g, uids in fit_group_copy.iteritems():
        target_dist = dist_metrics[g]
        for uid in uids:
            # calcualte the ks-pvalue with update distance metrics
            # target_dist
            pval = np.random.uniform(0, 1, 1)[0]
            if pval >= threshold:
                # remove the user and its information
                # from relevant container
                idx = [i for i, u in enumerate(fit_group[g]) if u == uid][0]
                fit_group[g].pop(idx)
                fit_pvals[g].pop(idx)
                # add the user to the unfit_group
                if g in unfit_group:
                    unfit_group[g].append(uid)
                else:
                    unfit_group[g] = [uid]

    # step 03: test members in unfit_group to see
    # if it has a good fit with other distmetrics
    # make a copy of the buffer group container
    buffer_group_copy = [i for i in buffer_group]
    if len(buffer_group_copy) > 0:
        for uid in buffer_group_copy:
            new_group, new_pval = find_fit_group(uid, dist_metrics, threshold)
            if not np.isnan(new_pval):
                buffer_group.remove(uid)
                if new_group in fit_group:
                    fit_group[new_group].append(uid)
                    fit_pvals[new_group].append(new_pval)
                else:
                    fit_group[new_group] = [uid]
                    fit_pvals[new_group] = [new_pval]

    unfit_group_copy = unfit_group.copy()
    for g, uids in unfit_group_copy.iteritems():
        for uid in uids:
            new_group, new_pval = find_fit_group(uid, dist_metrics, threshold, g)
            if np.isnan(new_pval):
                buffer_group.append(uid)
            else:
                unfit_group[g].remove(uid)
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

    print "fit score (type-%d): %.3f" % (t, fs)
    print "best fit score: %.3f" % best_fs

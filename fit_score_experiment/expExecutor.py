def expExectuor(param_grid, k_fold=100):
	""" Conduct simulation experiment based on given
	    experiment configuraiton.
	"""

    import time

    # input argument
    param_meshgrid_df = pd.DataFrame(list(param_grid))

    # calcualte job information
    total_iteration = param_meshgrid_df.shape[0]

    # output dataframe container
    exp_results = param_meshgrid_df.copy()
    # statistics container
    fs_mean, fs_median, fs_std, fs_min, fs_max = [], [], [], [], []
    exp_time_costs = []

    for i, param_row in param_meshgrid_df.iterrows():
        # load experiment parameter
        the_pop_size = param_row["pop_size"]
        the_group_prob = param_row["group_prob"]
        the_misclass_prob = param_row["misclass_prob"]

        print("{0}th (out of {1}) experiment is under conducting...".format(i, total_iteration))

        fit_scores = []
        start_time = time.time()
        for i_rep in range(k_fold):

            sim_result = genSimData(the_pop_size, the_group_prob, the_misclass_prob )
            transformed = simDataTransform(sim_result)
            the_fit_score = get_fit_score(transformed[0], transformed[1], c=1)
            fit_scores.append(the_fit_score)

        duration_time = (time.time() - start_time)
        print("Time cost: {0:.1f} seconds".format(duration_time))

        fs_mean.append(np.mean(fit_scores))
        fs_median.append(np.median(fit_scores))
        fs_std.append(np.std(fit_scores))
        fs_min.append(max(fit_scores))
        fs_max.append(min(fit_scores))

        exp_time_costs.append(duration_time)

    exp_results["pval_mean"] = fs_mean
    exp_results["pval_median"] = fs_median
    exp_results["pval_std"] = fs_std
    exp_results["pval_min"] = fs_min
    exp_results["pval_max"] = fs_max
    exp_results["time_cost"] = exp_time_costs
    return exp_results
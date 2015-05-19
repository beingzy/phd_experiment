	all_user_ids = list(set(users_df.ID))

	the_user_id = 0
	the_weights = ldm.get_transform_matrix()

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
	non_friends_id_ls = [u for u in users_df.ID if u not in friends_id_ls]
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
	_min = min([0] + sim_dist_vec + diff_dist_vec) # Include 0
	_nbins = 50
	bins = np.linspace(_min, _max, _nbins)

	pyplot.hist(sim_dist_vec, bins, alpha=0.5, label='friends')
	pyplot.hist(diff_dist_vec, bins, alpha=0.5, label='non-friends')
	pyplot.legend(loc='upper right')
	pyplot.title( "distance distrance of user (id: %d, taste: %d)" % (the_user_id, the_user_taste) )
	#pyplot.show()
	file_name = "hist_id_%d.png" % the_user_id
	pyplot.savefig(IMG_PATH + file_name, format='png')
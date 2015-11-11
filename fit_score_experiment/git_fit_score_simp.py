def get_fit_score(pvals, buffer_group, c):
	""" calculate the fit-score based on pvalues
	    of members with their in-group distance weights

		data model:
	    -----------
		pvals is a list of group-wise pvalue lists
	"""

	pop_size = sum([len(g) for g in pvals])

	aggregator = 0
	for group in pvals:
		pval_sum = sum(group)
		aggregator += pval_sum

	penalty = -1 * c * len(buffer_group)

	return aggregator + penalty

def get_fit_score(fit_pvals, buffer_group, c, t=2):
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
	
	import numpy as np
    
    # weighted sum of pvalues 
    if t not in [1, 2, 3]:
        raise NameError('Error: type (t) is not legal value (1 or 2)!')
    
    wsum_pval = 0
    if t == 1:
        for g, v in fit_pvals.iteritems():
            wsum_pval += sum(np.array(v) * 1.0 / len(v))
    if t == 2:
        for g, v in fit_pvals.iteritems():
            wsum_pval += sum(np.array(v)) * 1.0 / (len(v) * len(v))
    if t == 3:
        num_users = 0
        for g, v in fit_pvals.iteritems():
            wsum_pval += sum(np.array(v)) * 1.0 / (len(v) * len(v))
            num_users += len(v)
        wsum_pval = num_users * 1.0 * wsum_pval

    penalty = c * len(buffer_group)
    fit_score = wsum_pval + penalty # smaller value indicates a better overall fit
    
    return fit_score
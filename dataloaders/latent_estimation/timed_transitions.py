"""Calculates transitions with multiple steps apart"""
import numpy as np


def approx_time_transition(j, t, log_opts): 
    """P(xi_j | xi_t) = P(s_j, a_j, x_j| s_t, a_t, x_t)
    where we assume that policy and state transition is deterministic
    Output must be a matrix of |X| times |X|
    """
    slice_opts = log_opts[t+1:j]
    result = log_opts[t]
    for nexto in slice_opts:
        result = result@nexto
    return result


def approx_xsat_given_xsaprev_xj(j, t, log_acts, log_opts, j_value):
    """P(x_{t+1},s_{t+1},a_{t+1}|x_t, a_t, s_t, x_j)"""
    if t<0:
        import pdb; pdb.set_trace()
    assert j>t, "known timestep j must be larger than t"

    xj_given_xtplus1 = approx_time_transition(j, t+1, log_opts)[:, j_value] # states, option_model)[:, j_value]
    xj_given_xt = approx_time_transition(j, t, log_opts)[:, j_value]  #states, option_model)[:, j_value]
    
    log_opts = log_opts[t]
    log_acts = log_acts[None, t] 
    transition_matrix = log_opts * log_acts
    res = xj_given_xtplus1 * transition_matrix
    return (res.T / xj_given_xt).T


def time_transition(j, t, log_acts, log_opts): 
    """P(xi_j | xi_t) = P(s_j, a_j, x_j| s_t, a_t, x_t)
    Output must be a matrix of |X| times |X|
    """
    slice_acts, slice_opts = log_acts[t+1:j], log_opts[t+1:j]
    preva, prevo = log_acts[t], log_opts[t]
    result = preva * prevo
    for nexta, nexto in zip(slice_acts,slice_opts):
        next_trans = nexto * nexta
        result = result@next_trans
        # result /= result.sum(1)
    return result


def xsat_given_xsaprev_xj(j, t, log_acts, log_opts, j_value):
    """P(x_{t+1},s_{t+1},a_{t+1}|x_t, a_t, s_t, x_j)"""
    # NOTE: as j-->inf, result --> transition_matrix
    assert j>t, "known timestep j must be larger than t+1"
    # from dataloaders.latent_estimation.timed_transitions import time_transition
    xj_given_xtplus1 = time_transition(j, t+1, log_acts, log_opts)[:, j_value]
    xj_given_xt = time_transition(j, t, log_acts, log_opts)[:, j_value]
    
    log_opts = log_opts[t]
    log_acts = log_acts[None, t] 
    transition_matrix = log_opts * log_acts
    res = xj_given_xtplus1 * transition_matrix
    if (xj_given_xt==0).any():
        return transition_matrix
    res = (res.T / xj_given_xt).T
    if np.isnan(res).any() or res.sum()==0:
        res = transition_matrix
    else:
        res += 1e-45
    return res


def function(xj_given_xtplus1, xj_given_xt, transition_matrix,res):
    """Check if it is correct"""
    for j in range(7):
        for i in range(7):
            xt, xtp1 = i, j
            v1 = xj_given_xtplus1[xtp1]
            v2 = xj_given_xt[xt]
            v3 = transition_matrix[xt, xtp1]
            print((res[xt,xtp1]== v1*v3/v2).item())
import torch.multiprocessing as mp
import numpy as np
from dataloaders.latent_estimation.fb_algorithm_latent import (
    clean_forward_msg,
    clean_backward_msg,
    single_update_latent_viterbi)
from dataloaders.latent_estimation.log_probs import aux_probs
import torch

def single_prob_latent(args):
    """Get P(X|traj) of a single trajectory
    
    args (tuple) that contains the following:
        traj_num (int) - number of trajectory
        
    """
    (traj_num,
     prob_acts,
     prob_opts,
     last_step,
     list_queries,
     annotated_options,
     option_dim) = args

    fw = clean_forward_msg(
        prob_acts,
        prob_opts,
        last_step,
        traj_num,
        list_queries,
        annotated_options,
        option_dim=option_dim)[1:]

    bw = clean_backward_msg(
        prob_acts,
        prob_opts,
        last_step,
        traj_num,
        list_queries,
        annotated_options,
        option_dim=option_dim)[1:]
    res = np.nan_to_num(fw*bw)
    res = res / res.sum(1)[...,np.newaxis]
    return res


def paralellize_prob_latent(
        states,
        actions,
        masks,
        student=None,
        option_dim=7,
        idxs=None,
        dataset=None):

    log_acts_full, log_opts_full = [], []
    for sl in [slice(0, 200), slice(200, 400), slice(400, None)]:
        idx_slice = None
        if idxs is not None:
            idx_slice = idxs[sl]
        log_acts, log_opts = aux_probs(
            states[sl],
            actions[sl],
            student.state_prior,
            student.action_ae,
            option_dim,
            idx_slice,
            dataset
            )
        log_acts_full.append(log_acts)
        log_opts_full.append(log_opts)
    log_opts_full = torch.concatenate(log_opts_full)
    log_acts_full = torch.concatenate(log_acts_full)
    log_acts_full = log_acts_full.to('cpu').numpy()
    log_opts_full = log_opts_full.to('cpu').numpy()

    Ns = []
    for i, mm in enumerate(masks.to('cpu').numpy()):
        finished = np.where(mm==0)[0]
        if len(finished)>0:
            Ns.append(finished[0].item())
        else:
            Ns.append(len(mm))

    iter_data = zip(range(len(Ns)), log_acts_full, log_opts_full, Ns)
    iter_data = [(
        traj_num,
        prob_acts,
        prob_opts,
        last_step, 
        student.list_queries,
        student.annotated_options,
        option_dim
        ) for traj_num, prob_acts, prob_opts, last_step in iter_data]
    
    with mp.Pool() as pool:
        res = pool.map(single_prob_latent, iter_data)
    return res


def paralellize_update_latent_viterbi(
        states,
        actions,
        masks,
        student=None,
        option_dim=7,):
    log_acts_full, log_opts_full = [], []
    for sl in [slice(0, 200), slice(200, 400), slice(400, None)]:
        log_acts, log_opts = aux_probs(
            states[sl],
            actions[sl],
            student.state_prior,
            student.action_ae,
            option_dim
            )
        log_acts_full.append(log_acts)
        log_opts_full.append(log_opts)
    log_opts_full = torch.concatenate(log_opts_full)
    log_acts_full = torch.concatenate(log_acts_full)
    log_acts_full = log_acts_full.to('cpu').numpy()
    log_opts_full = log_opts_full.to('cpu').numpy()

    Ns = []
    for i, mm in enumerate(masks.to('cpu').numpy()):
        finished = np.where(mm==0)[0]
        if len(finished)>0:
            Ns.append(finished[0].item())
        else:
            Ns.append(len(mm))

    iter_data = zip(range(len(Ns)), log_acts_full, log_opts_full, Ns)
    iter_data = [(
        traj_num,
        prob_acts,
        prob_opts,
        last_step, 
        student.list_queries,
        student.annotated_options
        ) for traj_num, prob_acts, prob_opts, last_step in iter_data]

    with mp.Pool() as pool:
        res = pool.map(single_update_latent_viterbi, iter_data)
    
    return res

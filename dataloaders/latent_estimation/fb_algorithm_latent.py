
import numpy as np

from dataloaders.latent_estimation.log_probs import aux_probs
from dataloaders.latent_estimation.timed_transitions import xsat_given_xsaprev_xj


OPTION_DIM = 7
def single_update_latent_viterbi(args):
    """Apply Viterbi algorithm"""
    (traj_num,
     prob_acts,
     prob_opts,
     last_step,
     list_queries,
     annotated_options) = args
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    known_latents = list_queries.get(traj_num, set()) - set(range(last_step, 410))
    # with torch.no_grad():
        # forward
    max_path = np.empty((last_step, OPTION_DIM))
    accumulate_logp = np.zeros(OPTION_DIM)
    known_idxs = np.array(list(known_latents))
    for i in range(last_step):
        if i in known_idxs:
            accumulate_logp = - np.inf * np.ones([OPTION_DIM])
            query = annotated_options[traj_num][i]
            accumulate_logp[int(query)] = 0
            max_path[i, :] = annotated_options[traj_num][i] * np.ones([OPTION_DIM])
        else:
            h = min(known_idxs[known_idxs >i], default=None)
            if h is None:
                log_opts = prob_opts[i]
                log_acts = prob_acts[None, i] 
                log_acts = log_acts.reshape([-1, 1, OPTION_DIM])
                log_prob = log_opts * log_acts
                log_prob = np.log(log_prob)
            else:
                log_prob = xsat_given_xsaprev_xj(
                    h,
                    i,
                    log_acts=prob_acts,
                    log_opts=prob_opts,
                    j_value=int(annotated_options[traj_num][h]))
                # log_prob = approx_xsat_given_xsaprev_xj(
                #     h,
                #     i,
                #     states,
                #     actions,
                #     student,
                #     j_value=int(student.annotated_options[traj_num][h]))
                log_prob = log_prob[None]

                log_prob = np.log(log_prob+1e-10)
                # if i+1 in known_idxs:
                #     log_prob = -torch.inf * torch.ones((1, option_dim, option_dim), device=device)
                #     log_prob[:,:,int(student.annotated_options[traj_num][i+1])] = 0
            res = (accumulate_logp[:, None] + log_prob[0])
            accumulate_logp = res.max(-2)
            max_path[i, :] = res.argmax(-2)
    # backward
    c_array = -np.ones((last_step+1, 1), dtype=int)
    idx = accumulate_logp.argmax(-1)
    c_array[-1] = max_path[-1][idx]
    for i in range(last_step, 0, -1):
        c_array[i-1] = max_path[i-1][c_array[i]]
    return c_array[:-1]


def get_relevant_idxs(known_idxs, idx):
    lower = known_idxs[known_idxs <=idx]
    higher = known_idxs[known_idxs>idx]
    l = max(lower, default=None)
    h = min(higher, default=None)
    return l, h

### Single step message ###
def clean_forward_msg(
        log_acts_full,
        log_opts_full,
        N,
        traj_num,
        list_queries,
        annotated_options,
        option_dim=7):
    """
    states: T x S_dim
    """

    known_latents = list_queries.get(traj_num, set()) - set(range(N, 410))
    known_idxs = np.array(list(known_latents))
    forward_array = [np.ones(option_dim)/option_dim,] # TODO: use maybe init_state_probability model instead of uniform initialization
    for idx in range(N):
        # st = st[None]
        l, h = get_relevant_idxs(known_idxs, idx)

        if h is None:
            log_opts = log_opts_full[idx]
            log_acts = log_acts_full[None, idx] 
            transition = log_opts * log_acts
            # transition /= transition.sum(1)
            # transition = torch.log(log_prob)
        else:
            transition = xsat_given_xsaprev_xj(
                h,
                idx,
                log_acts=log_acts_full,
                log_opts=log_opts_full,
                j_value=int(annotated_options[traj_num][h]))
            # transition /= transition.sum(1)
            # transition = torch.log(transition)
        # transition = transition.detach().cpu().numpy()
        # Cases
        if l==idx:
            res = np.zeros(option_dim)
            res[int(annotated_options[traj_num][l])] = 1
        elif l==idx-1:
            res = transition[int(annotated_options[traj_num][l])]
        else:
            res = forward_array[-1] @ transition
            
        res = res/sum(res)
        forward_array.append(res)
    return np.array(forward_array)



# TODO: Unit tests
def clean_backward_msg(
        log_acts_full,
        log_opts_full,
        N,
        traj_num,
        list_queries,
        annotated_options,
        option_dim=7):
    # N = len(log_acts_full)
    known_latents = list_queries.get(traj_num, set()) - set(range(N, 410))
    known_idxs = np.array(list(known_latents))
    tmp = np.ones(option_dim)
    if N-1 in known_idxs:
        value_j = int(annotated_options[traj_num][N-1])
        mask = np.arange(option_dim)==value_j
        tmp[~mask] = 0

    backward_array = [tmp] # <- N-1
    for i in range(N):
        idx = N-i-1
        # h = min(known_idxs[known_idxs >=idx-1], default=None)
        j, h = get_relevant_idxs(known_idxs, idx)
        # Find P(e_{idx:N-1}|X_{idx-1})
        if h is None:
            log_opts = log_opts_full[idx-1]
            log_acts = log_acts_full[None, idx-1] 
            transition = log_opts * log_acts
        else:
            transition = xsat_given_xsaprev_xj(
                h,
                idx-1,
                log_acts=log_acts_full,
                log_opts=log_opts_full,
                j_value=int(annotated_options[traj_num][h]))
        
        # transition = transition.detach().cpu().numpy()

        if j==idx:
            res = np.zeros(option_dim)
            value_j = int(annotated_options[traj_num][idx])
            res[:] = backward_array[0][None,value_j] * transition.T[value_j]
            # res /= res.sum()
        else:
            res = backward_array[0] @ transition.T
        
        if idx-1 in known_idxs:
            value_j = int(annotated_options[traj_num][idx-1])
            mask = np.arange(option_dim)==value_j
            res[~mask] = 0
        # if res.sum()==0:
        #     import pdb; pdb.set_trace()
        res = res/sum(res)
        if np.isnan(res).any():
            import pdb; pdb.set_trace()
        backward_array.insert(0, res)
    return np.array(backward_array)

def prob_latent(
        states,
        actions,
        masks,
        student=None,
        option_dim=7):
    " Unparallelized"
    log_acts_full, log_opts_full = aux_probs(
        states,
        actions,
        student.state_prior,
        student.action_ae,
        option_dim
        )
    log_acts_full = log_acts_full.to('cpu').numpy()
    log_opts_full = log_opts_full.to('cpu').numpy()

    Ns = []
    for i, mm in enumerate(masks.cpu().numpy()):
        finished = np.where(mm==0)[0]
        if len(finished)>0:
            Ns.append(finished[0].item())
        else:
            Ns.append(len(mm))

    all_res = []
    for traj_num, log_acts in enumerate(log_acts_full):
        fw = clean_forward_msg(
            log_acts_full[traj_num],
            log_opts_full[traj_num],
            Ns[traj_num],
            traj_num,
            student.list_queries,
            student.annotated_options,
            option_dim)[1:]
        bw = clean_backward_msg(
            log_acts_full[traj_num],
            log_opts_full[traj_num],
            Ns[traj_num],
            traj_num,
            student.list_queries,
            student.annotated_options,
            option_dim)[1:]
        res = np.nan_to_num(fw*bw)
        
        res = res / res.sum(1)[...,np.newaxis]
        all_res.append(res)

    return all_res


def update_latent_viterbi(
        states,
        actions,
        masks,
        student=None,
        option_dim=7):

    log_acts_full, log_opts_full = aux_probs(
        states,
        actions,
        student.state_prior,
        student.action_ae,
        option_dim
        )
    log_acts_full = log_acts_full.to('cpu').numpy()
    log_opts_full = log_opts_full.to('cpu').numpy()
    Ns = []
    for i, mm in enumerate(masks.cpu().numpy()):
        finished = np.where(mm==0)[0]
        if len(finished)>0:
            Ns.append(finished[0].item())
        else:
            Ns.append(len(mm))

    array = []
    
    for traj_num in range(len(states)):
        args = (traj_num,
                log_acts_full[traj_num],
                log_opts_full[traj_num],
                Ns[traj_num],
                student.list_queries,
                student.annotated_options)
        array.append(single_update_latent_viterbi(args))
    return array
"""Utility functions."""

import torch
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix

import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from latent_active_learning.oracle import Oracle
from latent_active_learning.wrappers.latent_wrapper import FilterLatent

EPS = 1e-17
NEG_INF = -1e30
HOME_DIR = '/home/redacted/Documents/my-packages/latent_active_learning/'


def to_one_hot(indices, max_index):
    """Get one-hot encoding of index tensors."""
    zeros = torch.zeros(
        indices.size()[0], max_index, dtype=torch.float32,
        device=indices.device)
    return zeros.scatter_(1, indices.unsqueeze(1), 1)


def gumbel_sample(shape):
    """Sample Gumbel noise."""
    uniform = torch.rand(shape).float()
    return - torch.log(EPS - torch.log(uniform + EPS))


def gumbel_softmax_sample(logits, temp=1.):
    """Sample from the Gumbel softmax / concrete distribution."""
    gumbel_noise = gumbel_sample(logits.size())
    if logits.is_cuda:
        gumbel_noise = gumbel_noise.cuda()
    return F.softmax((logits + gumbel_noise) / temp, dim=-1)


def gaussian_sample(mu, log_var):
    """Sample from Gaussian distribution."""
    gaussian_noise = torch.randn(mu.size())
    if mu.is_cuda:
        gaussian_noise = gaussian_noise.cuda()
    return mu + torch.exp(log_var * 0.5) * gaussian_noise


def kl_gaussian(mu, log_var):
    """KL divergence between Gaussian posterior and standard normal prior."""
    return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)


def kl_categorical_uniform(preds):
    """KL divergence between categorical distribution and uniform prior."""
    kl_div = preds * torch.log(preds + EPS)  # Constant term omitted.
    return kl_div.sum(1)


def kl_categorical(preds, log_prior):
    """KL divergence between two categorical distributions."""
    kl_div = preds * (torch.log(preds + EPS) - log_prior)
    return kl_div.sum(1)


def poisson_categorical_log_prior(length, rate, device):
    """Categorical prior populated with log probabilities of Poisson dist."""
    rate = torch.tensor(rate, dtype=torch.float32, device=device)
    values = torch.arange(
        1, length + 1, dtype=torch.float32, device=device).unsqueeze(0)
    log_prob_unnormalized = torch.log(
        rate) * values - rate - (values + 1).lgamma()
    # TODO(tkipf): Length-sensitive normalization.
    return F.log_softmax(log_prob_unnormalized, dim=1)  # Normalize.


def log_cumsum(probs, dim=1):
    """Calculate log of inclusive cumsum."""
    return torch.log(torch.cumsum(probs, dim=dim) + EPS)


def generate_toy_data(num_symbols=5, num_segments=3, max_segment_len=5):
    """Generate toy data sample with repetition of symbols (EOS symbol: 0)."""
    seq = []
    symbols = np.random.choice(
        np.arange(1, num_symbols + 1), num_segments, replace=False)
    for seg_id in range(num_segments):
        segment_len = np.random.choice(np.arange(1, max_segment_len))
        seq += [symbols[seg_id]] * segment_len
    seq += [0]
    return torch.tensor(seq, dtype=torch.int64)


def get_lstm_initial_state(batch_size, hidden_dim, device):
    """Get empty (zero) initial states for LSTM."""
    hidden_state = torch.zeros(batch_size, hidden_dim, device=device)
    cell_state = torch.zeros(batch_size, hidden_dim, device=device)
    return hidden_state, cell_state


def get_segment_probs(all_b_samples, all_masks, segment_id):
    """Get segment probabilities for a particular segment ID."""
    neg_cumsum = 1 - torch.cumsum(all_b_samples[segment_id], dim=1)
    if segment_id > 0:
        return neg_cumsum * all_masks[segment_id - 1]
    else:
        return neg_cumsum


def get_losses(actions, outputs, args, beta_b=.1, beta_z=.1, prior_rate=3.,):
    """Get losses (NLL, KL divergences and neg. ELBO).

    Args:
        inputs: Padded input sequences.
        outputs: CompILE model output tuple.
        args: Argument dict from `ArgumentParser`.
        beta_b: Scaling factor for KL term of boundary variables (b).
        beta_z: Scaling factor for KL term of latents (z).
        prior_rate: Rate (lambda) for Poisson prior.
    """

    targets = actions.view(-1)
    all_encs, all_recs, all_masks, all_b, all_z = outputs
    input_dim = args.action_dim

    nll = 0.
    kl_z = 0.
    for seg_id in range(args.num_segments):
        seg_prob = get_segment_probs(
            all_b['samples'], all_masks, seg_id)
        preds = all_recs[seg_id].view(-1, input_dim)
        seg_loss = F.cross_entropy(
            preds, targets, reduction='none').view(-1, actions.size(1))

        # Ignore EOS token (last sequence element) in loss.
        nll += (seg_loss[:, :-1] * seg_prob[:, :-1]).sum(1).mean(0)

        # KL divergence on z.
        if args.latent_dist == 'gaussian':
            mu, log_var = torch.split(
                all_z['logits'][seg_id], args.latent_dim, dim=1)
            kl_z += kl_gaussian(mu, log_var).mean(0)
        elif args.latent_dist == 'concrete':
            kl_z += kl_categorical_uniform(
                F.softmax(all_z['logits'][seg_id], dim=-1)).mean(0)
        else:
            raise ValueError('Invalid argument for `latent_dist`.')

    # KL divergence on b (first segment only, ignore first time step).
    # TODO(tkipf): Implement alternative prior on soft segment length.
    probs_b = F.softmax(all_b['logits'][0], dim=-1)
    log_prior_b = poisson_categorical_log_prior(
        probs_b.size(1), prior_rate, device=actions.device)
    kl_b = args.num_segments * kl_categorical(
        probs_b[:, 1:], log_prior_b[:, 1:]).mean(0)

    loss = nll + beta_z * kl_z + beta_b * kl_b
    return loss, nll, kl_z, kl_b


def get_reconstruction_accuracy(actions, outputs, args):
    """Calculate reconstruction accuracy (averaged over sequence length)."""

    all_encs, all_recs, all_masks, all_b, all_z = outputs

    batch_size = actions.size(0)

    rec_seq = []
    latent_codes = []
    rec_acc = 0.
    for sample_idx in range(batch_size):
        # for each batch
        prev_boundary_pos = 0
        rec_seq_parts = []
        latent_codes_parts = []
        for seg_id in range(args.num_segments):
            # For each segment calculate the possible boundary
            # According to the segment number
            boundary_pos = torch.argmax(
                all_b['samples'][seg_id], dim=-1)[sample_idx] # scalar
            # scalar must be higher than previous scalar
            # otherwise empty segment is generated
            if prev_boundary_pos > boundary_pos: 
                boundary_pos = prev_boundary_pos
            
            # Now work with reconstructions, get reconstructions for that segment
            seg_rec_seq = torch.argmax(all_recs[seg_id], dim=-1)
            rec_seq_parts.append(
                seg_rec_seq[sample_idx, prev_boundary_pos:boundary_pos])
            
            tmp = torch.argmax(all_z['samples'][seg_id][sample_idx])
            latent_codes_parts.append(tmp.repeat(boundary_pos - prev_boundary_pos))
            prev_boundary_pos = boundary_pos
        rec_seq.append(torch.cat(rec_seq_parts))
        latent_codes.append(torch.cat(latent_codes_parts))
        cur_length = rec_seq[sample_idx].size(0)
        matches = rec_seq[sample_idx] == actions[sample_idx, :cur_length]
        rec_acc += matches.float().mean()
    rec_acc /= batch_size
    return rec_acc, rec_seq, latent_codes


def optimal_label_matching(seq1, seq2):
    # Create confusion matrix
    cm = confusion_matrix(seq1, seq2)
    
    # Convert to cost matrix (we want to maximize agreement, so minimize negative)
    cost_matrix = -cm

    # Solve the assignment problem
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Create label mapping: seq2_label -> seq1_label
    mapping = {col: row for row, col in zip(row_ind, col_ind)}
    
    # Apply mapping to seq2
    remapped_seq2 = np.array([mapping[label] for label in seq2])

    # Compute the number of mismatches (should be 0 if fully aligned)
    distance = np.sum(seq1 != remapped_seq2)/len(seq1)

    return distance, remapped_seq2, mapping

def hamming_latent_dist(z, true_opts):
    full_seq = torch.cat(z).cpu().numpy()
    true_full_seq = [true_opts[idx][:len(z[idx])] for idx in range(len(z))]
    true_full_seq = np.concatenate(true_full_seq)
    distance, _, optimal_mapping = optimal_label_matching(full_seq, true_full_seq)
    return distance, optimal_mapping

def get_boxworld_2targets(device):
    # Gets env and data
    env_name = "BoxWorld-v0"
    n_targets = 2
    filter_state_until = -1 - n_targets
    kwargs = {
        'size': 10,
        'n_targets': n_targets,
        'allow_variable_horizon': True,
        'fixed_targets': [ [0, 0], [9, 9] ],
        'latent_distribution': lambda x: 0,
        'render_mode': None
        }
    env = gym.make(env_name, **kwargs)

    env = Monitor(env)
    env = FilterLatent(env, list(range(filter_state_until, 0)))
    env.unwrapped._max_episode_steps = kwargs['size']**2 * n_targets
    path = HOME_DIR + 'expert_trajs/BoxWorld-size10-targets2'

    gini = Oracle.load(path)
    inputs, actions, lengths = get_data(
        gini.expert_trajectories, device)
    inputs_eval, actions_eval, lengths_eval = get_data(
    gini.expert_trajectories_test, device)

    info = {
        'data': (inputs, actions, lengths),
        'data_eval': (inputs_eval, actions_eval, lengths_eval),
        'env': env,
        'gini': gini
    }
    return info

def get_boxworld_6targets(device):
    env_name = "BoxWorld-v0"
    n_targets = 6
    filter_state_until = -1 - n_targets
    kwargs = {
        'size': 10,
        'n_targets': n_targets,
        'allow_variable_horizon': True,
        'fixed_targets': [
            [ 1, 1 ],
            [ 7, 1 ],
            [ 4, 6 ],
            [ 5, 0 ],
            [ 7, 3 ],
            [ 6, 8 ],
        ],
        'danger': [
            [ 2, 0 ],
            [ 1, 1 ],
            [ 8, 2 ],
            [ 7, 4 ],
            [ 3, 7 ],
            [ 4, 7 ],
            [ 6, 7 ],
        ],
        'obstacles': [
            [ 0, 4 ],
            [ 0, 5 ],
            [ 1, 2 ],
            [ 1, 5 ],
            [ 1, 8 ],
            [ 2, 3 ],
            [ 2, 8 ],
            [ 3, 1 ],
            [ 3, 5 ],
            [ 3, 6 ],
            [ 4, 1 ],
            [ 4, 2 ],
            [ 4, 5 ],
            [ 4, 8 ],
            [ 5, 5 ],
            [ 5, 8 ],
            [ 6, 1 ],
            [ 6, 2 ],
            [ 6, 3 ],
            [ 7, 0 ],
            [ 7, 2 ],
            [ 7, 6 ],
            [ 7, 8 ],
            [ 8, 8 ],
            [ 9, 4 ],
            [ 9, 5 ],
        ],
        'latent_distribution': lambda x: 0,
        'render_mode': None
        }
    env = gym.make(env_name, **kwargs)

    env = Monitor(env)
    env = FilterLatent(env, list(range(filter_state_until, 0)))
    env.unwrapped._max_episode_steps = kwargs['size']**2 * n_targets

    path = HOME_DIR + 'expert_trajs/BoxWorld-size10-targets6'
    # Add path for LIBERO
    # Add path for Franka Kitchen
    # Replace LSTM with the transformer for those two.
    # Implement termination and rollout policy
    # Organize code
    gini = Oracle.load(path)
    inputs, actions, lengths = get_data(
        gini.expert_trajectories, device)
    inputs_eval, actions_eval, lengths_eval = get_data(
    gini.expert_trajectories_test, device)

    info = {
        'data': (inputs, actions, lengths),
        'data_eval': (inputs_eval, actions_eval, lengths_eval),
        'env': env,
        'gini': gini
    }
    return info


def get_data(trajectories, device):
    data = []
    actions = []
    for dd in trajectories:
        data.append(torch.from_numpy(dd.obs[:-1]))
        actions.append(torch.from_numpy(dd.acts))

    lengths = torch.tensor(list(map(len, data)))
    lengths = lengths.to(device)
    inputs = torch.nn.utils.rnn.pad_sequence(data, batch_first=True).to(device, torch.float32)
    actions = torch.nn.utils.rnn.pad_sequence(actions, batch_first=True).to(device, torch.int64)
    return inputs, actions, lengths
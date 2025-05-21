"""
Torch functions to calculate probability of observed acts and opts
"""
import torch


@torch.no_grad()
def prob_action(observations, actions=None, state_prior=None, option_dim=7,
                vocab_size=64, idxs=None, dataset=None):
    """
    Return probability P(a|s, o) N x option_dim

    Inputs:
        - observations (B, H, W) tensor, where B is the number of
            trajectories, H is the length, W is thestate dimension
        - actions (B, H) tensor
        - policy: a mingpt.skip_gpt.GPT policy
        - option_dim: the dimension, or number, of the options
    """
    policy = state_prior.model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    B, H, W = observations.shape
    # Flatten
    # states = observations.reshape(-1, 1, W)
    observations = observations.reshape(-1, 1, W)
    acts = actions.reshape(-1)
    results = []
    if idxs is None:
        states = observations
    else:
        states = torch.ones(observations.size(0), 1, 137, device=device)
        states[:,:, :9] = observations
        start = idxs[0][0].item()
        available_idxs = []
        imgs1, imgs2 = [], []
        for i in range(idxs[0][0], idxs[-1][-1]):
            if i in dataset.idx2trj_dict:
                img1, img2 = dataset.get_images(i)
                imgs1.append(img1)
                imgs2.append(img2)
                available_idxs.append(i)

        chunks1 = [imgs1[x:x+500] for x in range(0, len(imgs1), 500)]
        chunks2 = [imgs2[x:x+500] for x in range(0, len(imgs2), 500)]
        k = []
        for i in range(len(chunks1)):
            res = state_prior.encode_images(chunks1[i], chunks2[i])
            k.append(res)
        res = torch.concat(k)
        for num, i in enumerate(available_idxs):
            states[i-start,:,9:] = res[num]
    # For each possible option find out probability of observed actions
    for o in range(option_dim):
        opts = torch.ones([B*H, 1], dtype=int, device=device) * o
        logits, _ = policy((states, opts))
        logits = logits[:,:,:vocab_size]
        logits = torch.nn.Softmax(2)(logits)
        logits = logits[range(B*H), 0, acts]
        results.append(logits)
    results = torch.stack(results, axis=1)
    return results.reshape((B, H, option_dim))


@torch.no_grad()
def prob_option(observations, option_model, option_dim=7):
    """
    Return probability P(o|s, o_); a  B x H x option_dim x option_dim
    size N x (option_dim (a.k.a. o_)) x option_dim (a.k.a. o)
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    B, H, W = observations.shape
    states = observations.reshape(-1,1,W)

    results = []
    # For each possible option find out option prob distribution
    for o in range(option_dim):
        opts = torch.ones([B*H, 1], dtype=int, device=device) * o
        logits, _ = option_model((states, opts))
        logits = torch.nn.Softmax(dim=2)(logits)
        results.append(logits.squeeze(1))
    results = torch.stack(results, axis=1)
    results = results.reshape((B, H, option_dim, option_dim))

    return results


@torch.no_grad()
def aux_probs(
        states,
        actions,
        state_prior,
        action_ae,
        option_dim=7,
        idxs=None,
        dataset=None):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    option_model = state_prior.option_model
    policy = state_prior.model
    option_model.eval()
    policy.eval()

    acts = action_ae.encode_into_latent(actions.to(device).contiguous())[0]
    log_opts_full = prob_option(states, option_model, option_dim=option_dim)
    log_acts_full = prob_action(states,
                            acts,
                            option_dim=option_dim,
                            state_prior=state_prior,
                            vocab_size=state_prior.vocab_size,
                            idxs=idxs,
                            dataset=dataset
                            )
    return log_acts_full, log_opts_full
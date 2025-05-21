import argparse
import torch
import wandb
from datetime import datetime
import numpy as np
import torch.nn.functional as F
from stable_baselines3.common.evaluation import evaluate_policy
import random

import utils
import modules
import modules_termination
from new import add_termination_prediction_training


parser = argparse.ArgumentParser()
parser.add_argument('--iterations', type=int, default=100,
                    help='Number of training iterations.')
parser.add_argument('--learning-rate', type=float, default=1e-2,
                    help='Learning rate.')
parser.add_argument('--hidden-dim', type=int, default=32,
                    help='Number of hidden units.')
parser.add_argument('--latent-dim', type=int, default=2,
                    help='Dimensionality of latent variables.')
parser.add_argument('--latent-dist', type=str, default='concrete',
                    help='Choose: "gaussian" or "concrete" latent variables.')
parser.add_argument('--batch-size', type=int, default=32,
                    help='Mini-batch size (for averaging gradients).')

parser.add_argument('--input-dim', type=int, default=6, # input dimension 6, 14
                    help='Number of distinct symbols in data generation.')
parser.add_argument('--num-segments', type=int, default=6,
                    help='Number of segments in data generation.')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disable CUDA training.')
parser.add_argument('--log-interval', type=int, default=10,
                    help='Logging interval.')
parser.add_argument('--action-dim', type=int, default=4,
                    help='Logging interval.')
parser.add_argument('--seed', type=int, default=0,
                    help='Logging interval.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

device = torch.device('cuda' if args.cuda else 'cpu')
timestamp = lambda: datetime.now().strftime('%m-%d-%Y_%H-%M-%S')
seed = args.seed
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)


def get_rollout(model, t_model, env, latent_dim):
    model.eval()
    t_model.eval()
    import numpy as np
    observations = []
    observation = env.reset()
    observations.append(observation[0][None])
    state = 0
    last_b = 0
    returns = 0
    step = 0
    while state <latent_dim and step < 50:
        step +=1
        # Create initial mask.
        obs = np.concatenate(observations)[last_b:]
        act = model.get_action(obs, state)
        obs = torch.Tensor(obs).to('cuda')
        obs = obs.reshape(1, -1, obs.size(-1))
        term = t_model(obs)[-1]['samples'][0][0][-1]
        if term > 0.5:
            state +=1
            last_b = len(obs[0])
        next_obs, rew, done, terminate, _ = env.step(act[0][-1].item())
        observations.append(next_obs[None])
        returns +=rew
        if done or terminate:
            break
        if rew > 0:
            print(rew)
    return returns


def log_metrics(run, epoch_num, hamming_loss, hamming_loss_test,
                rollout_mean, rollout_std, prob_true_action):
    metrics = {
        "env/epoch": epoch_num,
        "env/hamming_loss": hamming_loss,
        "env/hamming_loss_test": hamming_loss_test,
        "env/rollout_mean": rollout_mean,
        "env/rollout_std": rollout_std,
        "bc_lo/prob_true_action": prob_true_action
    }
    run.log(metrics, step=epoch_num)
timestamp = lambda: datetime.now().strftime('%m-%d-%Y_%H-%M-%S')


run = wandb.init(
    entity='vinson-liuba-experiments',
    project='BoxWorld-size10-targets2',
    name='CompILE' + str(timestamp()),
    tags=['compile'],
    config=args,
    save_code=True
)


model = modules.CompILE(
    input_dim=args.input_dim,  # +1 for EOS/Padding symbol.
    action_dim=args.action_dim,
    hidden_dim=args.hidden_dim,
    latent_dim=args.latent_dim,
    max_num_segments=args.num_segments,
    latent_dist=args.latent_dist).to(device)
t_model = modules_termination.TermCompILE(
    input_dim=args.input_dim,  # +1 for EOS/Padding symbol.
    action_dim=args.action_dim,
    hidden_dim=args.hidden_dim,
    latent_dim=args.latent_dim,
    max_num_segments=args.num_segments,
    latent_dist=args.latent_dist).to(device)

model, utils = add_termination_prediction_training(model)

optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
optimizer_term = torch.optim.Adam(t_model.parameters(), lr=args.learning_rate)

info = utils.get_boxworld_2targets(device)
inputs, actions, lengths = info['data']
inputs_eval, actions_eval, lengths_eval = info['data_eval']
env = info['env']
gini = info['gini']
# acts = actions
# actions = F.one_hot(actions).to(torch.float32)


print('Training model...')
# # Train model.
for step in range(1000):
    optimizer.zero_grad()
    optimizer_term.zero_grad()
    # Run forward pass.
    model.train()
    t_model.train()

    outputs = model.forward(inputs, actions, lengths)
    loss, nll, kl_z, kl_b = utils.get_losses(actions, outputs, args)#[:-1]
    all_encs, all_recs, all_masks, all_b, all_z = outputs[:-1]
    
    ground_truth = sum(all_b['samples']).clamp(max=1).detach()
    _, _, all_b = t_model(inputs)
    all_b = all_b['samples'][0]
    term_loss = F.binary_cross_entropy(all_b, ground_truth)
    
    term_loss.backward()
    optimizer_term.step()
    
    loss.backward()
    optimizer.step()

    if step % args.log_interval == 0:
        # Run evaluation.
        rec = None
        batch_loss = 0
        batch_acc = 0
        model.eval()
        t_model.eval()

        outputs = model.forward(inputs, actions, lengths)[:-1]
        all_encs, all_recs, all_masks, all_b, all_z = outputs
        acc, rec, z = utils.get_reconstruction_accuracy(actions, outputs, args)

        # Accumulate metrics.
        batch_acc += acc.item()
        batch_loss += nll.item()
        outputs_eval = model.forward(inputs_eval, actions_eval, lengths_eval)[:-1]
        acc_eval, _, z_eval = utils.get_reconstruction_accuracy(actions_eval, outputs_eval, args)

        dist, optimal_mapping = utils.hamming_latent_dist(z, gini.true_options)
        dist_test, optimal_mapping = utils.hamming_latent_dist(z_eval, gini.true_options_test)
        k = [get_rollout(model, t_model, env, args.latent_dim) for _ in range(5)]
        returns = np.mean(k)
        std = np.std(k)
        print('step: {}, nll_train: {:.6f}, rec_acc_train: {:.3f}, rec_acc_eval: {:.3f},' \
        'hamming_distance: {}, hamming_distance_test: {}, returns: {}'.format(
            step, batch_loss, batch_acc, acc_eval.item(), dist, dist_test, returns))
        
        print('input sample: {}'.format(actions[-1, :lengths[-1] - 1]))
        print('reconstruction: {}'.format(rec[-1]))
        log_metrics(run=run,
                    epoch_num=step,
                    hamming_loss=dist,
                    hamming_loss_test=dist_test,
                    rollout_mean=returns,
                    rollout_std=std,
                    prob_true_action=acc)

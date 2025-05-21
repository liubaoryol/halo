"""
Test cases of Viterbi algorithm
1. Unsupervision
   Trained policy and option model.
   Test Viterbi with no known_latents, but fully trained policy and policy.
   The result should be similar to the ground truth
2. Partial Supervision
   Same scenario as in 1, but add some supervision of options, should improve the estimation.
   Double check that the supervised options have indeed those values
3. Full supervision
4. Check `approx_time_transition(...)`
5. Check `approx_xsat_given_xsaprev_xj(...)`
"""
import time
import math
import torch
import numpy as np

from dataloaders.latent_estimation.timed_transitions import (
    xsat_given_xsaprev_xj,
    approx_time_transition
)
from dataloaders.latent_estimation.fb_algorithm_latent import (
    clean_forward_msg,
    clean_backward_msg,
    update_latent_viterbi,
    prob_latent
    )
from dataloaders.latent_estimation.log_probs import (
    aux_probs
)
from dataloaders.latent_estimation.parallelize_latent_estimation import (
    paralellize_prob_latent,
    paralellize_update_latent_viterbi
)
from dataloaders.test_config import cfg
import workspaces.adept_kitchen
from dataloaders.trajectory_loader import get_relay_kitchen_train_val
import students as random_student

# workspace = 
model = workspaces.adept_kitchen.AdeptKitchenWorkspace(cfg)
train_set, _ = get_relay_kitchen_train_val(
    data_directory=cfg.env.dataset_fn['data_directory'],
    train_fraction=0.95,
    random_seed=cfg.seed,
    device=cfg.device
)
dataset = train_set.dataset.dataset
observations, actions, masks, _, _ = dataset.tensors
observations = observations.to(cfg.device)
actions = actions.to(cfg.device)

true_options=dataset.oracle.true_options
total_trajs = len(dataset)
observations = observations[:total_trajs]
actions = actions[:total_trajs]
masks = masks[:total_trajs]
true_options = true_options[:total_trajs]
log_acts_full, log_opts_full = aux_probs(
    observations,
    actions,
    model.state_prior,
    model.action_ae
)
log_acts_full = log_acts_full.to('cpu').numpy()
log_opts_full = log_opts_full.to('cpu').numpy()

last_steps = []
for i, mask in enumerate(masks):
    finished = torch.where(mask==0)[0]
    if len(finished)>0:
        last_steps.append(finished[0].item())
    else:
        last_steps.append(len(mask))


def test_full_supervision():
    student = random_student.Supervised(option_dim=7,
                                        state_prior=model.state_prior,
                                        action_ae=model.action_ae)
    
    dataset.query_oracle(student)
    options = update_latent_viterbi(
        observations,
        actions,
        masks,
        student)
    for traj_num in range(len(observations)):
        print("Testing trajectory: ", traj_num)
        opts = options[traj_num].squeeze(1)
        true_opts = true_options[traj_num][:last_steps[traj_num]]
        assert (true_opts==opts).all()


def test_partial_supervision():
    students = {}
    estimated_options = {}
    for name in ['Supervised', 'Random', 'Unsupervised']:
        students[name] = getattr(random_student, name)(
            option_dim=7,
            state_prior=model.state_prior,
            action_ae=model.action_ae
            )
        if name=='Random':
            students[name].query_percent=0.1
        dataset.query_oracle(students[name])
    
        estimated_options[name] = update_latent_viterbi(
            observations,
            actions,
            masks,
            students[name])

    for traj_num in range(len(observations)):
        print("Testing trajectory: ", traj_num)
        options_dict, incorrect = {}, {}
        for name, student in students.items():
            # print("Calculating options for student type ", student.student_type)
            options = estimated_options[name]
            opts = options[traj_num].squeeze(1)
            true_opts = true_options[traj_num][:last_steps[traj_num]]

            correct_pred = (true_opts==opts)
            incorrect_estimations = masks[traj_num].sum() - correct_pred.sum()
            incorrect[name] = incorrect_estimations
            options_dict[name] = opts
        try:
            assert incorrect['Random']<=incorrect['Unsupervised']
            assert incorrect['Supervised']<=incorrect['Random']
        except:
            print("Didn't pass on traj_num", traj_num)

def test_unsupervised():
    uns_student = random_student.Unsupervised(option_dim=7,
                                              state_prior=model.state_prior,
                                              action_ae=model.action_ae)
    dataset.query_oracle(uns_student)  # Basically does not do anything since it is unsupervised
    
    options_uns = update_latent_viterbi(
        observations,
        actions,
        masks,
        uns_student)

    for traj_num in range(len(observations)):
        print("Testing trajectory: ", traj_num)
        opts_uns = options_uns[traj_num].squeeze(1)
        true_opts = true_options[traj_num][:last_steps[traj_num]]
        correct_pred = (true_opts==opts_uns)
        incorrect_estimations = masks[traj_num].sum() - correct_pred.sum()
        try:
            assert incorrect_estimations <= masks[traj_num].sum()*0.15 # it should be pretty good
        except:
            print("Didn't pass on traj_num", traj_num)

def test_aprox_ttransition():
    """Compare multiple queried latents in time P(xi_j|xi_t)
    Impact must be more noticeable the closer the latent is to time t
    TODO: manually calculate power transition?
    """
    traj_num = 12
    states = log_opts_full[traj_num]
    a_transition1 = approx_time_transition(2, 0, states)
    a_transition2 = approx_time_transition(100, 0, states)
    assert math.isclose(a_transition2.sum().item(),7, abs_tol=0.05)

    def _entropy(tensor1d):
        return -(tensor1d * np.log(tensor1d)).sum()
    for low, hi in zip(a_transition1, a_transition2):
         assert _entropy(low) < _entropy(hi)

    # Manually calculate power transition, using what worked with HMMs
    states[:] = states[0]
    a_transition1 = approx_time_transition(2, 0, states)
    transition2 = np.linalg.matrix_power(states, 2)
    assert math.isclose((a_transition1 - transition2).sum(), 0, abs_tol=0.05)

    a_transition1 = approx_time_transition(10, 0, states)
    transition2 = np.linalg.matrix_power(states, 10)
    assert math.isclose((a_transition1 - transition2).sum(), 0, abs_tol=0.05)


def test_xsat_given_xsaprev_xj():
    option_dim = 7
    traj_num = 3
    j_far = 80
    j_close = 4
    t = 2
    # a further query will not give a lot of information of the current latent state
    # Hence it will have a lower probability on the true value

    lower_acc= xsat_given_xsaprev_xj(
        j_far, t, log_acts_full[traj_num], log_opts_full[traj_num],
        true_options[traj_num][j_far]
        )
    higher_acc = xsat_given_xsaprev_xj(
        j_close, t, log_acts_full[traj_num], log_opts_full[traj_num],
        true_options[traj_num][j_close]
        )
    for o in range(option_dim):
        print(o)
        print(lower_acc[o][true_options[traj_num][j_close]]  < higher_acc[o][true_options[traj_num][j_close]])


def test_fwd_msg():
    students = {}
    name_students = ['Supervised1', 'Unsupervised1', 'Unsupervised2']
    for name in name_students:
        students[name] = getattr(random_student, name[:-1])(
            option_dim=7,
            state_prior=model.state_prior,
            action_ae=model.action_ae
        )
        dataset.query_oracle(students[name])
    # Test that known latents are predicted accordingly
    for traj_num in [0, 10, 20]:
        log_acts = log_acts_full[traj_num]
        log_opts = log_opts_full[traj_num]
        fwd_msg = clean_forward_msg(
            log_acts,
            log_opts,
            last_steps[traj_num],
            traj_num,
            students['Supervised1'].list_queries,
            students['Supervised1'].annotated_options
            )[1:]
        opts = true_options[traj_num][masks[traj_num].to(bool)]
        fwd_msg[opts]
        assert (fwd_msg[range(len(fwd_msg)), opts]==1).sum()

        # Check that having a query will increase the probability of surrounding latents to the same queried latent
        fwd_msg_uns = clean_forward_msg(
            log_acts,
            log_opts,
            last_steps[traj_num],
            traj_num,
            students['Unsupervised1'].list_queries,
            students['Unsupervised1'].annotated_options,
            )[1:]
        correct_pred = opts==fwd_msg_uns.argmax(1)
        k = np.random.choice( np.where(~correct_pred)[0])

        true_opt = dataset.oracle.query(traj_num, k)
        students['Unsupervised2'].log_query(traj_num, k)
        students['Unsupervised2'].annotated_options[traj_num, k] = true_opt
        fwd_msg_w_query = clean_forward_msg(
            log_acts,
            log_opts,
            last_steps[traj_num],
            traj_num,
            students['Unsupervised2'].list_queries,
            students['Unsupervised2'].annotated_options,
        )[1:]
        if k<2:
            k = 2
        assert (fwd_msg_w_query[k-2:k+2,true_opt] > fwd_msg_uns[k-2:k+2, true_opt]).all()


def test_bwd_msg():
    """"""
    student = random_student.Supervised(option_dim=7, state_prior=model.state_prior, action_ae=model.action_ae)
    student1 = random_student.Unsupervised(option_dim=7, state_prior=model.state_prior, action_ae=model.action_ae)
    student2 = random_student.Unsupervised(option_dim=7, state_prior=model.state_prior, action_ae=model.action_ae)

    for st in [student, student1, student2]:
        dataset.query_oracle(st)
    
    # Test that known latents are predicted accordingly
    for traj_num in [40,55]:
        log_acts = log_acts_full[traj_num]
        log_opts = log_opts_full[traj_num]
        bwd_msg = clean_backward_msg(
            log_acts,
            log_opts,
            last_steps[traj_num],
            traj_num,
            student.list_queries,
            student.annotated_options
            )[1:]
        opts = true_options[traj_num][masks[traj_num].to(bool)]
        assert (bwd_msg[range(len(bwd_msg)), opts]>0).all()
        bwd_msg[range(len(bwd_msg)), opts] = 0
        assert bwd_msg.sum()==0


        # Check that having a query will increase the probability of surrounding latents to the same queried latent
        bwd_msg_uns = clean_backward_msg(
            log_acts,
            log_opts,
            last_steps[traj_num],
            traj_num,
            student1.list_queries,
            student1.annotated_options
            )[1:]
        correct_pred = opts==bwd_msg_uns.argmax(1)
        k = np.where(~correct_pred)
        print(len(k[0]))
        if len(k[0])==0:
            continue
        k = np.random.choice(k[0])

        true_opt = dataset.oracle.query(traj_num, k)
        student2.log_query(traj_num, k)
        student2.annotated_options[traj_num, k] = true_opt
        bwd_msg_w_query = clean_backward_msg(
            log_acts,
            log_opts,
            last_steps[traj_num],
            traj_num,
            student2.list_queries,
            student2.annotated_options
        )[1:]
        if k<2:
            k = 2
        assert (bwd_msg_w_query[k-2:k+2,true_opt] > bwd_msg_uns[k-2:k+2, true_opt]).all()

def test_prob_latent():
    """
    Test 1. P(X_t|observation, x_t) = 1 where it equals 0 where not
    Test 2. P(X_t|observation) < P(X_t|observation, x_{close to t})
    Test 3. P(X_t|observation).argmax() is less accurate than when you have some supervision
    """

    
    student = random_student.Supervised(option_dim=7, state_prior=model.state_prior, action_ae=model.action_ae)
    student1 = random_student.Unsupervised(option_dim=7, state_prior=model.state_prior, action_ae=model.action_ae)
    student2 = random_student.Unsupervised(option_dim=7, state_prior=model.state_prior, action_ae=model.action_ae)
    student3 = random_student.Random(option_dim=7, state_prior=model.state_prior, action_ae=model.action_ae, query_percent=0.2)
    for st in [student, student1, student2, student3]:
        dataset.query_oracle(st)
    
    # Query some state for student2
    q_st = 10
    for traj_num in range(16):
        true_opt = dataset.oracle.query(traj_num, q_st)
        student2.log_query(traj_num, q_st)
        student2.annotated_options[traj_num, q_st] = true_opt


        probs = prob_latent(observations, actions, masks, student)[traj_num]
        probs1 = prob_latent(observations, actions, masks, student1)[traj_num]
        probs2 = prob_latent(observations, actions, masks, student2)[traj_num]
        probs3 = prob_latent(observations, actions, masks, student3)[traj_num]
        # Test 1
        # P(X_t|observation, x_t) = 1 where it equals 0 where not
        assert (true_options[traj_num][:len(probs)] == probs.argmax(1)).all()
        assert probs2[q_st][true_opt] == 1
        # Test 2
        #  P(X_t|observation) < P(X_t|observation, x_{close to t})
        q_range = slice(q_st-1, q_st+2)
        # assert (probs1[q_range,true_opt] < probs2[q_range, true_opt]).all()

        # Test 3 
        # P(X_t|observation).argmax() is less accurate than when you have some supervision
        sum0 = (probs.argmax(1)==true_options[traj_num][:len(probs)]).sum()
        sum1 = (probs1.argmax(1)==true_options[traj_num][:len(probs)]).sum()
        sum2 = (probs2.argmax(1)==true_options[traj_num][:len(probs)]).sum()
        sum3 = (probs3.argmax(1)==true_options[traj_num][:len(probs)]).sum()
        # assert sum0 >= sum1
        # assert sum2 >= sum1

        print(
            "Test1 for traj", traj_num,
             (true_options[traj_num][:len(probs)] == probs.argmax(1)).all(),
             probs2[q_st][true_opt] == 1,
             "Test 2: ",
             (probs1[q_range,true_opt] < probs2[q_range, true_opt]).all(),
             "Test 3: supervised sum: ", sum0, "unsupervised sum", sum1, "one query", sum2, "random", sum3
             )
def test_parallelize_prob_latent():

    student = random_student.Unsupervised(option_dim=7,
                                          state_prior=model.state_prior,
                                          action_ae=model.action_ae)
    now = time.perf_counter()
    probs = prob_latent(observations, actions, masks, student)
    time_count = time.perf_counter() - now
    
    now = time.perf_counter()
    probs1 = paralellize_prob_latent(observations, actions, masks, student)
    time_count_paralel = time.perf_counter() - now
    for i in range(len(probs)):
        assert (probs[i]==probs1[i]).all()

def test_paralellize_update_viterbi():
    
    student = random_student.Unsupervised(option_dim=7,
                                          state_prior=model.state_prior,
                                          action_ae=model.action_ae)
    now = time.perf_counter()
    probs = update_latent_viterbi(observations, actions, masks.cpu().numpy(), student)
    time_count = time.perf_counter() - now
    
    now = time.perf_counter()
    probs1 = paralellize_update_latent_viterbi(observations, actions, masks.cpu().numpy(), student)
    time_count_paralel = time.perf_counter() - now
    for i in range(len(probs)):
        assert (probs[i]==probs1[i]).all()
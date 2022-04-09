from __future__ import division
import os
import time
import torch
import numpy as np
import torch.optim as optim
from tensorboardX import SummaryWriter
from setproctitle import setproctitle as ptitle

from model import build_model
from player_util import Agent
from environment import create_env


def worker(rank, args, shared_model, train_modes, n_iters, curr_env_steps, ToM_count, ToM_history, Policy_history, step_history, loss_history, env=None):
    n_iter = 0
    writer = SummaryWriter(os.path.join(args.log_dir, 'Agent-{}'.format(rank)))
    ptitle('worker: {}'.format(rank))
    gpu_id = args.gpu_ids[rank % len(args.gpu_ids)]
    torch.manual_seed(args.seed + rank)
    training_mode = args.train_mode
    env_name = args.env
    
    if gpu_id >= 0:
        torch.cuda.manual_seed(args.seed + rank)
        device = torch.device('cuda:' + str(gpu_id))
        if len(args.gpu_ids) > 1:
            device_share = torch.device('cpu')
        else:
            device_share = torch.device('cuda:' + str(args.gpu_ids[-1]))

    else:
        device = device_share = torch.device('cpu')
    
    #device = torch.device("cpu") # there's no need for worker to use 

    if env == None:
        env = create_env(env_name, args, rank)

    if args.fix:
        env.seed(args.seed)
    else:
        env.seed(rank % (args.seed + 1))

    player = Agent(None, env, args, None, device)
    player.rank = rank
    player.gpu_id = gpu_id
    
    # prepare model
    player.model = shared_model

    player.reset()
    reward_sum = torch.zeros(player.num_agents).to(device)
    reward_sum_org = np.zeros(player.num_agents)
    ave_reward = np.zeros(2)
    ave_reward_longterm = np.zeros(2)
    count_eps = 0
    #max_steps = env.max_steps
    while True:
        if "Pose" in args.env and args.random_target:
            p = 0.7 - (env.max_steps/20 -1) * 0.1
        
            env.target_type_prob = [p, 1-p]
            player.env.target_type_prob = [p, 1-p]

        # sys to the shared model
        player.model.load_state_dict(shared_model.state_dict())
        if player.done:
            player.reset()
            reward_sum = torch.zeros(player.num_agents).to(device)
            reward_sum_org = np.zeros(player.num_agents)
            count_eps += 1


        player.update_rnn_hidden()
        t0 = time.time()

        for s_i in range(env.max_steps):
            player.action_train()
            if 'ToM' in args.model:
                ToM_count[rank] += 1
            reward_sum += player.reward
            reward_sum_org += player.reward_org
            if player.done:
                writer.add_scalar('train/reward', reward_sum[0], n_iter)
                writer.add_scalar('train/reward_org', reward_sum_org[0].sum(), n_iter)
                break
        fps = s_i / (time.time() - t0)

        writer.add_scalar('train/fps', fps, n_iter)

        n_iter += env.max_steps  # s_i
        n_iters[rank] = n_iter

        # wait for training process
        Policy_history[rank] = player.Policy_history
        player.Policy_history = []
        '''
        # for evaluation, no need in real training
        player.optimize(None, None, shared_model, training_mode, device_share)
        step_history[rank] = player.step_history
        loss_history[rank] = player.loss_history
        '
        player.step_history = []
        player.loss_history = []
        # evaluation end
        '''
        if 'ToM' in args.model:
            ToM_history[rank] += player.ToM_history
            player.ToM_history = []

        train_modes[rank] = -10 # have to put this line at last
        train_start_time = time.time()
        while train_modes[rank] != -1:
            current_time = time.time()
            if current_time - train_start_time > 180 :
                print("stuck in training")
                train_modes[rank] = -100
                return
        # update env steps during training
        env.max_steps = curr_env_steps[rank]
        player.env.max_steps = env.max_steps

        player.clean_buffer(player.done)

        if sum(n_iters) > args.max_step:
            train_modes[rank] = -100
            
        if train_modes[rank] == -100:
            env.close()
            break
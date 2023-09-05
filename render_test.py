from __future__ import division
from setproctitle import setproctitle as ptitle

import os
import time
import torch
import logging
import numpy as np
import argparse
from tensorboardX import SummaryWriter

from model import build_model
from utils import setup_logger
from player_util import Agent
from environment import create_env


parser = argparse.ArgumentParser(description='render')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--test-eps', type=int, default=5, metavar='M', help='testing episode length')
parser.add_argument('--env', default='simple', metavar='Pose-v0', help='environment to train on (default: Pose-v0|Pose-v1)')
parser.add_argument('--load-coordinator-dir', default=None, help='folder to load trained high-level models from')
parser.add_argument('--load-executor-dir', default=None, help='folder to load trained low-level models from')
parser.add_argument('--env-steps', type=int, default=20, help='env steps')
parser.add_argument('--model', default='single', metavar='M', help='multi-shapleyV|')
parser.add_argument('--lstm-out', type=int, default=32, metavar='LO', help='lstm output size')
parser.add_argument('--mask', dest='mask', action='store_true', help='mask ToM and communication to those out of range')
parser.add_argument('--mask-actions', dest='mask_actions', action='store_true', help='mask unavailable actions to boost training')
parser.add_argument('--gpu-id', type=int, default=-1, nargs='+', help='GPUs to use [-1 CPU only] (default: -1)')
parser.add_argument('--render', dest='render', action='store_true', help='render test')
parser.add_argument('--render_save', dest='render_save', action='store_true', help='render save')

parser.add_argument('--num-agents', type=int, default=-1)   # if -1, then the env will load the default setting
parser.add_argument('--num-targets', type=int, default=-1)  # else, you can assign the number of agents and targets yourself


def render_test(args):
    gpu_id = args.gpu_id
    
    torch.manual_seed(args.seed)
    if gpu_id >= 0:
        torch.cuda.manual_seed(args.seed)
        device = torch.device('cuda:' + str(gpu_id))
    else:
        device = torch.device('cpu')

    env = create_env(args.env, args)

    env.seed(args.seed)

    player = Agent(None, env, args, None, device)
    player.gpu_id = gpu_id
    player.model = build_model(player.env, args, device).to(device)
    player.model.eval()
    
    saved_state = torch.load(args.load_coordinator_dir)
    player.model.load_state_dict(saved_state['model'],strict=False)

    for i_episode in range(args.test_eps):
        player.reset()
        print(f"Episode:{i_episode}")
        for i_step in range(args.env_steps):
            player.action_test()

if __name__ == '__main__':
    args = parser.parse_args()
    render_test(args)
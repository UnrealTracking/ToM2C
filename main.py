from __future__ import print_function, division
import os
import time
import torch
import argparse
from datetime import datetime
import torch.multiprocessing as mp

from test import test
from train import train
from worker import worker
#from train_new import Policy_train
from model import build_model
from environment import create_env
from shared_optim import SharedRMSprop, SharedAdam

os.environ["OMP_NUM_THREADS"] = "1"

parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 0.0001)')
parser.add_argument('--gamma', type=float, default=0.1, metavar='G', help='discount factor for rewards (default: 0.99)')
parser.add_argument('--gamma-rate', type=float, default=0.002, metavar='G', help='the increase rate of gamma')
parser.add_argument('--gamma-final', type=float, default=0.9, metavar='G', help='the increase rate of gamma')
parser.add_argument('--tau', type=float, default=1.00, metavar='T', help='parameter for GAE (default: 1.00)')
parser.add_argument('--entropy', type=float, default=0.005, metavar='T', help='parameter for entropy (default: 0.01)')
parser.add_argument('--grad-entropy', type=float, default=1.0, metavar='T', help='parameter for entropy (default: 0.01)')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--workers', type=int, default=1, metavar='W', help='how many training processes to use (default: 32)')
#parser.add_argument('--punish-rate', type=float, default=0.0, metavar='W', help='punish tracking same target')
parser.add_argument('--A2C-steps', type=int, default=20, metavar='NS', help='number of forward steps in A2C (default: 300)')
parser.add_argument('--env-steps', type=int, default=20, metavar='NS', help='number of steps in one env episode')
parser.add_argument('--start-eps', type=int, default=2000, metavar='NS', help='number of episodes before increasing gamma and env steps')
parser.add_argument('--ToM-train-loops', type=int, default=1, metavar='NS', help='ToM training loops num')
parser.add_argument('--policy-train-loops', type=int, default=1, metavar='NS', help='Policy training loops num')
parser.add_argument('--test-eps', type=int, default=20, metavar='M', help='testing episodes')
parser.add_argument('--ToM-frozen', type=int, default=5, metavar='M', help='episode length of freezing ToM in training')
parser.add_argument('--env', default='MSMTC-v3', help='environment to train on')
parser.add_argument('--optimizer', default='Adam', metavar='OPT', help='shares optimizer choice of Adam or RMSprop')
parser.add_argument('--amsgrad', default=True, metavar='AM', help='Adam optimizer amsgrad parameter')
parser.add_argument('--load-model-dir', default=None, metavar='LMD', help='folder to load trained models from')
parser.add_argument('--load-executor-dir', default=None, metavar='LMD', help='folder to load trained low-level policy models from')
parser.add_argument('--log-dir', default='logs/', metavar='LG', help='folder to save logs')
parser.add_argument('--model', default='ToM2C', metavar='M', help='ToM2C')
parser.add_argument('--gpu-id', type=int, default=-1, nargs='+', help='GPU to use [-1 CPU only] (default: -1)')
parser.add_argument('--norm-reward', dest='norm_reward', action='store_true', default='True', help='normalize reward')
parser.add_argument('--train-comm', dest='train_comm', action='store_true', help='train comm')
parser.add_argument('--random-target', dest='random_target', action='store_true', default='True', help='random target')
parser.add_argument('--mask-actions', dest='mask_actions', action='store_true', help='mask unavailable actions to boost training')
parser.add_argument('--mask', dest='mask', action='store_true', help='mask ToM and communication to those out of range')
parser.add_argument('--render', dest='render', action='store_true', help='render test')
parser.add_argument('--fix', dest='fix', action='store_true', help='fix random seed')
parser.add_argument('--shared-optimizer', dest='shared_optimizer', action='store_true', help='use an optimizer without shared statistics.')
parser.add_argument('--train-mode', type=int, default=-1, metavar='TM', help='his')
parser.add_argument('--input-size', type=int, default=80, metavar='IS', help='input image size')
parser.add_argument('--lstm-out', type=int, default=32, metavar='LO', help='lstm output size')
parser.add_argument('--sleep-time', type=int, default=0, metavar='LO', help='seconds')
parser.add_argument('--max-step', type=int, default=3000000, metavar='LO', help='max learning steps')
parser.add_argument('--render_save', dest='render_save', action='store_true', help='render save')

parser.add_argument('--agent', type=int, default=5)
parser.add_argument('--target', type=int, default=5)
# num_step: 20
# max_step: 500000
# env_max_step: 100
# low-level step: 10
# training mode: -1 for worker collecting trajectories, -10 for workers waiting for training process, -20 for training, -100 for all processes end

def start():
    args = parser.parse_args()
    args.shared_optimizer = True
    if args.gamma_rate == 0:
        args.gamma = 0.9
        args.env_steps *= 5
    if args.gpu_id == -1:
        torch.manual_seed(args.seed)
        args.gpu_id = [-1]
        device_share = torch.device('cpu')
        mp.set_start_method('spawn')
    else:
        torch.cuda.manual_seed(args.seed)
        mp.set_start_method('spawn', force=True)
        if len(args.gpu_id) > 1:
            raise AssertionError("Do not support multi-gpu training")
            #device_share = torch.device('cpu')
        else:
            device_share = torch.device('cuda:' + str(args.gpu_id[-1]))
    #device_share = torch.device('cuda:0')
    env = create_env(args.env, args)
    assert env.max_steps % args.A2C_steps == 0
    shared_model = build_model(env, args, device_share).to(device_share)
    shared_model.share_memory()
    shared_model.train()
    env.close()
    del env

    if args.load_model_dir is not None:
        saved_state = torch.load(
            args.load_model_dir,
            map_location=lambda storage, loc: storage)
        if args.load_model_dir[-3:] == 'pth':
            shared_model.load_state_dict(saved_state['model'], strict=False)
        else:
            shared_model.load_state_dict(saved_state)

    #params = shared_model.parameters()
    params = []
    params_ToM = []
    for name, param in shared_model.named_parameters():
        if 'ToM' in name or 'other' in name:
            #print("ToM: ",name)
            params_ToM.append(param)
        else:
            #print("Not ToM: ",name)
            params.append(param)
    
    if args.shared_optimizer:
        print('share memory')
        if args.optimizer == 'RMSprop':
            optimizer_Policy = SharedRMSprop(params, lr=args.lr)
            if 'ToM' in args.model:
                optimizer_ToM = SharedRMSprop(params_ToM, lr=args.lr)
            else:
                optimizer_ToM = None
        if args.optimizer == 'Adam':
            optimizer_Policy = SharedAdam(params, lr=args.lr, amsgrad=args.amsgrad)
            if 'ToM' in args.model:
                print("ToM optimizer lr * 10")
                optimizer_ToM = SharedAdam(params_ToM, lr=args.lr*10, amsgrad=args.amsgrad)
            else:
                optimizer_ToM = None
        optimizer_Policy.share_memory()
        if optimizer_ToM is not None:
            optimizer_ToM.share_memory()
    else:
        optimizer_Policy = None
        optimizer_ToM = None

    current_time = datetime.now().strftime('%b%d_%H-%M')
    args.log_dir = os.path.join(args.log_dir, args.env, current_time)

    processes = []
    manager = mp.Manager()
    train_modes = manager.list()
    n_iters = manager.list()
    curr_env_steps = manager.list()
    ToM_count = manager.list()
    ToM_history = manager.list()
    Policy_history = manager.list()
    step_history = manager.list()
    loss_history = manager.list()

    #if 'ToM' in args.model:
    #    args.ToM_file = os.path.join(args.log_dir, 'ToM.json')

    for rank in range(0, args.workers):
        p = mp.Process(target=worker, args=(rank, args, shared_model, train_modes, n_iters, curr_env_steps, ToM_count, ToM_history, Policy_history, step_history, loss_history))

        train_modes.append(args.train_mode)
        n_iters.append(0)
        curr_env_steps.append(args.env_steps)
        ToM_count.append(0)
        ToM_history.append([])
        Policy_history.append([])
        step_history.append([])
        loss_history.append([])

        p.start()
        processes.append(p)
        time.sleep(args.sleep_time)

    p = mp.Process(target=test, args=(args, shared_model, optimizer_Policy, optimizer_ToM, train_modes, n_iters))
    p.start()
    processes.append(p)
    time.sleep(args.sleep_time)

    if args.workers > 0:
        # not only test
        p = mp.Process(target=train, args=(args, shared_model, optimizer_Policy, optimizer_ToM, train_modes, n_iters, curr_env_steps, ToM_count, ToM_history, Policy_history, step_history, loss_history))
        p.start()
        processes.append(p)
        time.sleep(args.sleep_time)

    for p in processes:
        time.sleep(args.sleep_time)
        p.join()


if __name__=='__main__':
    start()
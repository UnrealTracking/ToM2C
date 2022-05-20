from __future__ import division
from setproctitle import setproctitle as ptitle

import os
import time
import torch
import logging
import numpy as np
from tensorboardX import SummaryWriter

from model import build_model
from utils import setup_logger
from player_util import Agent
from environment import create_env


def test(args, shared_model, optimizer, optimizer_ToM, train_modes, n_iters):
    ptitle('Test Agent')
    n_iter = 0
    writer = SummaryWriter(os.path.join(args.log_dir, 'Test'))
    gpu_id = args.gpu_ids[-1]
    log = {}
    print(os.path.isdir(args.log_dir))
    setup_logger('{}_log'.format(args.env),
                 r'{0}/logger'.format(args.log_dir))
    log['{}_log'.format(args.env)] = logging.getLogger(
        '{}_log'.format(args.env))
    d_args = vars(args)
    for k in d_args.keys():
        log['{}_log'.format(args.env)].info('{0}: {1}'.format(k, d_args[k]))

    torch.manual_seed(args.seed)
    if gpu_id >= 0:
        torch.cuda.manual_seed(args.seed)
        device = torch.device('cuda:' + str(gpu_id))
    else:
        device = torch.device('cpu')

    env = create_env(args.env, args)
    #env.seed(args.seed)
    if "MSMTC" in args.env:
        # freeze env max steps to 100
        env.max_steps = 100
    
    start_time = time.time()
    count_eps = 0

    player = Agent(None, env, args, None, device)
    player.gpu_id = gpu_id
    player.model = build_model(player.env, args, device).to(device)
    player.model.eval()
    max_score = -100

    ave_reward_list = []
    comm_cnt_list = []
    comm_bit_list = []
    tmp_list_1 = []
    tmp_list_2 = []
    while True:
        AG = 0
        reward_sum = np.zeros(player.num_agents)
        reward_sum_list = []
        len_sum = 0

        for i_episode in range(args.test_eps):
            player.model.load_state_dict(shared_model.state_dict())
            player.reset()
            reward_sum_ep = np.zeros(player.num_agents)
            rotation_sum_ep = 0

            fps_counter = 0
            t0 = time.time()
            count_eps += 1
            fps_all = []

            comm_cnt = 0
            comm_bit = 0
            ToM_acc = 0
            ToM_target_acc = 0
            while True:
                player.action_test()
                fps_counter += 1
                reward_sum_ep += player.reward
                
                #ToM_acc += player.random_ToM_acc
                #ToM_target_acc += player.random_ToM_target_acc
                # comm_ToM_loss += player.comm_ToM_loss
                # no_comm_ToM_loss +=player.no_comm_ToM_loss
                # ToM_loss +=player.ToM_loss
                if 'comm' in args.model or 'ToM-v5' in args.model:
                    comm_cnt += player.comm_cnt
                    comm_bit += player.comm_bit
                if player.done:
                    # print(ToM_acc/fps_counter)
                    # print(ToM_target_acc/fps_counter)
                    tmp_list_1.append(ToM_acc/fps_counter)
                    tmp_list_2.append(ToM_target_acc/fps_counter)

                    # if len(tmp_list_1) == 3:
                    #     print(np.mean(tmp_list_1),np.std(tmp_list_1))
                    #     print(np.mean(tmp_list_2),np.std(tmp_list_2))

                    #print("steps:{}".format(fps_counter))
                    #print("comm:{}, no comm:{}, Total:{}".format(comm_ToM_loss.item()/fps_counter,no_comm_ToM_loss.item()/fps_counter,\
                    #    ToM_loss.item()/fps_counter))
                    #print("reward:{}".format(reward_sum_ep[0]))
                    #AG += reward_sum_ep[0]/rotation_sum_ep*player.num_agents
                    reward_sum += reward_sum_ep
                    reward_sum_list.append(reward_sum_ep[0])
                    len_sum += player.eps_len
                    fps = fps_counter / (time.time()-t0)
                    #n_iter = n_iters[0] if len(n_iters) > 0 else count_eps
            
                    #for n in n_iters:
                    #    n_iter += n
                    new_n_iter = sum(n_iters)
                    if new_n_iter > n_iter:
                        n_iter = new_n_iter
                    # for i, r_i in enumerate(reward_sum_ep):
                    #     writer.add_scalar('test/reward'+str(i), r_i, n_iter)
                        writer.add_scalar('test/reward', reward_sum_ep[0], n_iter)
                        writer.add_scalar('test/fps', fps, n_iter)
                    fps_all.append(fps)
                    player.clean_buffer(player.done)
                        
                    #writer.add_scalar('test/eps_len', player.eps_len, n_iter)
                    break
        '''
            comm_cnt_list.append(comm_cnt/env.max_steps)
            comm_bit_list.append(comm_bit/env.max_steps)
        print("cnt: ",np.mean(comm_cnt_list),np.std(comm_cnt_list))
        print("bit: ",np.mean(comm_bit_list),np.std(comm_bit_list))
        comm_bit_list=[]
        comm_cnt_list=[]        
        
        comm_cnt_avg = comm_cnt/(args.test_eps * 100)
        comm_bit_avg = comm_bit/(args.test_eps * 100)
        print("comm_cnt",comm_cnt_avg)
        print("comm_bandwidth",comm_bit_avg)
        comm_cnt_list.append(comm_cnt_avg)
        comm_bit_list.append(comm_bit_avg)
        if len(comm_cnt_list)==5:
            print(np.mean(comm_cnt_list),np.std(comm_cnt_list))
            print(np.mean(comm_bit_list),np.std(comm_bit_list))
            comm_bit_list=[]
            comm_cnt_list=[]
        '''
        # player.max_length:
        ave_AG = AG/args.test_eps
        ave_reward_sum = reward_sum/args.test_eps
        len_mean = len_sum/args.test_eps
        reward_step = reward_sum / len_sum
        mean_reward = np.mean(reward_sum_list)
        std_reward = np.std(reward_sum_list)

        if args.workers == 0:
            # pure test, so compute reward mean and std
            ave_reward_list.append(mean_reward)
            if len(ave_reward_list) == 5:
                reward_mean = np.mean(ave_reward_list)
                reward_std = np.std(ave_reward_list)
                ave_reward_list = []
                log['{}_log'.format(args.env)].info("mean reward {0}, std reward {1}".format(reward_mean, reward_std))
                print("---------------")
        #n_iter = sum(n_iters)
        #writer.add_scalar('test/reward', ave_reward_sum[0], n_iter)

        log['{}_log'.format(args.env)].info(
            "Time {0}, ave eps reward {1}, ave eps length {2}, reward step {3}, FPS {4}, "
            "mean reward {5}, std reward {6}, AG {7}".
            format(
                time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time)),
                np.around(ave_reward_sum, decimals=2), np.around(len_mean, decimals=2),
                np.around(reward_step, decimals=2), np.around(np.mean(fps_all), decimals=2),
                mean_reward, std_reward, np.around(ave_AG, decimals=2)
            ))

        # save model
        if ave_reward_sum[0] > max_score:
            print('save best!')
            max_score = ave_reward_sum[0]
            model_dir = os.path.join(args.log_dir, 'best.pth')
        elif n_iter % 100000 == 0:
            model_dir = os.path.join(args.log_dir, ('new_'+str(n_iter)+'.pth').format(args.env))
        else:
            model_dir = os.path.join(args.log_dir, 'new.pth'.format(args.env))
        state_to_save = {"model": player.model.state_dict(),
                         "optimizer": optimizer.state_dict()}
        torch.save(state_to_save, model_dir)

        time.sleep(args.sleep_time)

        for rank in range(args.workers):
            if train_modes[rank] == -100:
                print("test process ended due to train process collapse")
                return

        if n_iter > args.max_step:
            env.close()
            for id in range(0, args.workers):
                train_modes[id] = -100
            break

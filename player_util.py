from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import json
from utils import ensure_shared_grads

class Agent(object):
    def __init__(self, model, env, args, state, device):
        self.model = model
        self.env = env
        self.num_agents = env.n
        #self.num_targets = env.observation_space.shape[1]
        self.num_targets = env.num_target
        self.state_dim = env.observation_space.shape[2]
        self.model_name = args.model 
        self.prior = torch.FloatTensor(np.array([0.7, 0.3]))  # communication edge prior

        self.model_name = args.model
        self.eps_len = 0
        self.eps_num = 0
        self.args = args
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        self.rewards_eps = []
        self.done = True
        self.info = None
        self.reward = 0
        self.device = device
        self.lstm_out = args.lstm_out
        self.reward_mean = None
        self.reward_std = 1
        self.num_steps = 0
        self.env_step = 0
        self.vk = 0
        self.state = state
        self.rank = 0
        # evaluation for ToM & Comm
        self.comm_ToM_loss = torch.zeros(1)
        self.no_comm_ToM_loss = torch.zeros(1)
        self.ToM_loss = torch.zeros(1)

        self.hself = torch.zeros(self.num_agents, self.lstm_out).to(device)
        self.hToM = torch.zeros(self.num_agents, self.num_agents, self.lstm_out).to(device)
        
        self.poses = None # cam_dim=3 ndarray
        self.ToM_history = []
        self.Policy_history = []
        #self.step_history = []
        #self.loss_history = []


    def get_other_poses(self):
        # ToM2C requires the poses of each agent, so you need to declare how to get the poses for each env
        if "MSMTC" in self.args.env:
            cam_states = self.env.get_cam_states()
            cam_states = torch.from_numpy(np.array(cam_states)).float().to(self.device)

            # compute relative camera poses in self coordinate
            cam_dim = cam_states.size()[-1] # cam_dim=3
            cam_states_duplicate = cam_states.unsqueeze(0).expand(self.num_agents, self.num_agents, cam_dim)
            cam_states_relative = cam_states_duplicate - cam_states.unsqueeze(1).expand(self.num_agents, self.num_agents, cam_dim)
            cam_state_theta = ((cam_states_relative[:,:,-1]/180) * np.pi).reshape(self.num_agents, self.num_agents, 1)
            poses = torch.cat((cam_states_relative[:,:,:2], torch.cos(cam_state_theta), torch.sin(cam_state_theta)),-1)
            return poses
        elif "CN" in self.args.env:
            return torch.zeros(self.num_agents, self.num_agents, 1)
    
    def get_mask(self):
        if not self.args.mask:
            return torch.ones(self.num_agents, self.num_agents, 1)
        # ToM2C provides the option to mask the ToM inference and communication to agents out of ranges(include self)
        if "MSMTC" in self.args.env:
            mask = self.env.get_mask()
            mask = torch.from_numpy(mask).unsqueeze(-1).bool()
            mask = mask.to(self.device)
            return mask
        else:
            return torch.ones(self.num_agents, self.num_agents, 1)

    def get_available_actions(self):
        available_actions = self.env.get_available_actions()
        available_actions = torch.from_numpy(available_actions).to(self.device)
        return available_actions

    def action_train(self):
        if self.args.mask_actions:
            available_actions = self.get_available_actions()
            available_actions_data = available_actions.cpu().numpy()
        else:
            available_actions = None
            available_actions_data = 0

        self.poses = self.get_other_poses()
        self.mask = self.get_mask()
        value_multi, actions, entropy, log_prob, hn_self, hn_ToM, ToM_goals, edge_logits, comm_edges, probs, real_cover, ToM_target_cover =\
            self.model(self.state, self.hself, self.hToM, self.poses, self.mask, available_actions = available_actions)
        
        actions_env = actions.cpu().numpy() # only ndarrays can be processed by the environment
        state_multi, reward, self.done, self.info = self.env.step(actions_env)#,obstacle=True)
        reward_multi = reward.repeat(self.num_agents) # all agents share the same reward

        self.reward_org = reward_multi.copy()

        if self.args.norm_reward:
            reward_multi = self.reward_normalizer(reward_multi)

        # save state for training
        Policy_data = {"state":self.state.detach().cpu().numpy(), "poses": self.poses.detach().cpu().numpy(),"actions": actions_env, "reward": reward_multi,\
            "mask":self.mask.detach().cpu().numpy(),"available_actions": available_actions_data}
        real_goals = torch.cat((1-actions,actions),-1)
        ToM_data = {"state":self.state.detach().cpu().numpy(), "poses":self.poses.detach().cpu().numpy(), "mask":self.mask.detach().cpu().numpy(),\
            "real":real_goals.detach().cpu().numpy(), "available_actions": available_actions_data}
        self.Policy_history.append(Policy_data)
        self.ToM_history.append(ToM_data)

        if isinstance(self.done, list): self.done = np.sum(self.done)
        self.state = torch.from_numpy(np.array(state_multi)).float().to(self.device)
            
        self.reward = torch.tensor(reward_multi).float().to(self.device)
        self.eps_len += 1

        self.hself=hn_self
        self.hToM=hn_ToM

        self.env_step += 1
        if self.env_step >= self.env.max_steps:
            self.done = True

    def action_test(self):
        if self.args.mask_actions:
            available_actions = self.get_available_actions()
        else:
            available_actions = None
        
        with torch.no_grad():
            self.poses = self.get_other_poses()
            self.mask = self.get_mask()
            value_multi, actions, entropy, log_prob, hn_self, hn_ToM, ToM_goals, edge_logits, comm_edges, probs, real_cover, ToM_target_cover=\
                self.model(self.state, self.hself, self.hToM, self.poses, self.mask, True, available_actions = available_actions)
            
            self.comm_cnt = torch.sum(comm_edges)
            self.comm_bit = self.comm_cnt * self.num_targets
            self.env.comm_edges = comm_edges

            '''
            # compute ToM prediction accuracy
            ToM_goal = (ToM_goals[:,:,:,-1]>=0.1).unsqueeze(-1) # n * n-1 * m * 1
            random_ToM_goal = torch.randint(2,(self.num_agents,self.num_agents-1,self.num_targets,1))
            real_goal = torch.from_numpy(actions)
            real_goal = real_goal.unsqueeze(0).repeat(self.num_agents,1,1,1)
            idx= (torch.ones(self.num_agents, self.num_agents) - torch.diag(torch.ones(self.num_agents))).bool()
            real_goal = real_goal[idx].reshape(self.num_agents, self.num_agents-1, self.num_targets, -1)
            ToM_cover = (ToM_target_cover >= 0.1)
            random_ToM_cover = torch.randint(2,(self.num_agents,self.num_agents-1,self.num_targets,1))
            self.ToM_acc = (ToM_goal==real_goal)[real_cover].float()
            self.ToM_acc = torch.mean(self.ToM_acc)
            self.ToM_target_acc = torch.mean((real_cover==ToM_cover)[real_cover].float())
            self.random_ToM_acc = torch.mean((random_ToM_goal==real_goal)[real_cover].float())
            self.random_ToM_target_acc = torch.mean((real_cover==random_ToM_cover)[real_cover].float())
            #print(torch.mean(ToM_goal.float()))
            '''
        state_multi, self.reward, self.done, self.info = self.env.step(actions)#, obstacle=True)
        if isinstance(self.done, list): self.done = np.sum(self.done)
        self.state = torch.from_numpy(np.array(state_multi)).float().to(self.device)
        self.eps_len += 1

        self.hself=hn_self
        self.hToM=hn_ToM

        self.env_step += 1
        if self.env_step >= self.env.max_steps:
            self.done = True

    def reset(self):
        obs = self.env.reset()
        self.state = torch.from_numpy(np.array(obs)).float().to(self.device)

        self.eps_len = 0
        self.eps_num += 1
        self.reset_rnn_hidden()
        
        self.model.sample_noise()

    def clean_buffer(self, done):
        self.env_step = 0
        # outputs
        self.values = []
        self.log_probs = []
        self.entropies = []
        # gt
        self.rewards = []
        if done:
            # clean
            self.rewards_eps = []

        return self

    def reward_normalizer(self, reward):
        reward = np.array(reward)
        self.num_steps += 1
        if self.num_steps == 1:
            self.reward_mean = reward
            self.vk = 0
            self.reward_std = 1
        else:
            delt = reward - self.reward_mean
            self.reward_mean = self.reward_mean + delt/self.num_steps
            self.vk = self.vk + delt * (reward-self.reward_mean)
            self.reward_std = np.sqrt(self.vk/(self.num_steps - 1))
        reward = (reward - self.reward_mean) / (self.reward_std + 1e-8)
        return reward

    def reset_rnn_hidden(self):
        self.hself = torch.zeros(self.num_agents, self.lstm_out).to(self.device)
        self.hToM = torch.zeros(self.num_agents, self.num_agents, self.lstm_out).to(self.device)

    def update_rnn_hidden(self):
        self.hself = Variable(self.hself.data)
        self.hToM = Variable(self.hToM.data)


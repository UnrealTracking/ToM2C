from __future__ import division
import torch
import numpy as np
import torch.nn as nn
from gym import spaces
import torch.nn.functional as F
from torch.autograd import Variable

from utils import norm_col_init, weights_init
from perception import NoisyLinear, RNN, AttentionLayer

def build_model(env, args, device):
    name = args.model

    if "ToM2C" in name and "MSMTC" in args.env:
        model = ToM2C_multi(env.observation_space,env.action_space,args,device)
        model.num_agents = env.n
        model.num_targets = env.num_target
    elif "ToM2C" in name and "CN" in args.env:
        model = ToM2C_single(env.observation_space,env.action_space,args,device)
        model.num_agents = env.n
        model.num_targets = env.num_target
    else:
        raise NotImplementedError

    model.train()
    return model

def wrap_action(self, action):
    action = np.squeeze(action)
    out = action * (self.action_high - self.action_low) / 2 + (self.action_high + self.action_low) / 2.0
    return out

def sample_action(mu_multi, sigma_multi, device, test=False):
    # discrete
    logit = mu_multi
    prob = F.softmax(logit, dim=-1)
    log_prob = F.log_softmax(logit, dim=-1)
    entropy = -(log_prob * prob).sum(-1, keepdim=True)
    if test: # test
        # not always sample action instead of choosing the best action, where this branch should never be visited
        #print("test")
        action = prob.max(-1)[1].data
        action_env = action.cpu().numpy()  # np.squeeze(action.cpu().numpy(), axis=0)
    else:
        action = prob.multinomial(1).data
        log_prob = log_prob.gather(-1, Variable(action))  # [num_agent, 1] # comment for sl slave
        action_env = action.squeeze(0)
    
    return action_env, entropy, log_prob, prob

class ValueNet(nn.Module):
    def __init__(self, input_dim, head_name, num=1):
        super(ValueNet, self).__init__()
        if 'ns' in head_name:
            self.noise = True
            self.critic_linear = NoisyLinear(input_dim, num, sigma_init=0.017)
        else:
            self.noise = False
            self.critic_linear = nn.Linear(input_dim, num)
            self.critic_linear.weight.data = norm_col_init(self.critic_linear.weight.data, 0.1)
            self.critic_linear.bias.data.fill_(0)

    def forward(self, x):
        value = self.critic_linear(x)
        return value

    def sample_noise(self):
        if self.noise:
            self.critic_linear.sample_noise()

    def remove_noise(self):
        if self.noise:
            self.critic_linear.sample_noise()

class PolicyNet(nn.Module):
    def __init__(self, input_dim, action_space, head_name, device):
        super(PolicyNet, self).__init__()
        self.head_name = head_name
        self.device = device
        num_outputs = action_space.n

        if 'ns' in head_name:
            self.noise = True
            self.actor_linear = NoisyLinear(input_dim, num_outputs, sigma_init=0.017)
        else:
            self.noise = False
            self.actor_linear = nn.Linear(input_dim, num_outputs)

            # init layers
            self.actor_linear.weight.data = norm_col_init(self.actor_linear.weight.data, 1)
            self.actor_linear.bias.data.fill_(0)

    def forward(self, x, test=False, available_actions = None):
        mu = F.relu(self.actor_linear(x))
        if available_actions is not None:
            # mask unavailable actions
            # size of mu&available actions: b*n*m, 2
            idx = (available_actions == 0)
            mu[idx] = 1e-10
        sigma = torch.ones_like(mu)
        action, entropy, log_prob, prob = sample_action(mu, sigma, self.device, test)
        return action, entropy, log_prob, prob

    def sample_noise(self):
        if self.noise:
            self.actor_linear.sample_noise()
            self.actor_linear2.sample_noise()

    def remove_noise(self):
        if self.noise:
            self.actor_linear.sample_noise()
            self.actor_linear2.sample_noise()

class PolicyNet_Single(nn.Module):
    def __init__(self, input_dim, action_space, head_name, device):
        super(PolicyNet_Single, self).__init__()
        self.head_name = head_name
        self.device = device
        num_outputs = 1

        if 'ns' in head_name:
            self.noise = True
            self.actor_linear = NoisyLinear(input_dim, num_outputs, sigma_init=0.017)
        else:
            self.noise = False
            self.actor_linear = nn.Linear(input_dim, num_outputs)

            # init layers
            self.actor_linear.weight.data = norm_col_init(self.actor_linear.weight.data, 0.1)
            self.actor_linear.bias.data.fill_(0)

    def forward(self, x, test=False, available_actions = None):
        mu = F.leaky_relu(self.actor_linear(x))
        if available_actions is not None:
            # mask unavailable actions
            # size of mu&available actions: b*n*m, 2
            idx = (available_actions == 0)
            mu[idx] = -1e10
        sigma = torch.ones_like(mu)
        prob = F.softmax(mu, dim=-2)
        log_prob = (F.log_softmax(mu, dim=-2))
        entropy = -(log_prob * prob)#.sum(-2, keepdim=True)
        mask = (available_actions != 0)
        entropy = entropy[mask]

        if test: # test
            # not always sample action instead of choosing the best action
            action = prob.max(-2)[1].data
            action = action.cpu().numpy()  # np.squeeze(action.cpu().numpy(), axis=0)
        else:
            prob = prob.squeeze(-1)
            log_prob = log_prob.squeeze(-1)
            action = prob.multinomial(1).data
            log_prob = log_prob.gather(-1, Variable(action))  # [num_agent, 1] # comment for sl slave

        return action, entropy, log_prob, prob

    def sample_noise(self):
        if self.noise:
            self.actor_linear.sample_noise()
            self.actor_linear2.sample_noise()

    def remove_noise(self):
        if self.noise:
            self.actor_linear.sample_noise()
            self.actor_linear2.sample_noise()

class EncodeLinear(torch.nn.Module):
    def __init__(self, dim_in, dim_out=32, head_name='lstm', device=None):
        super(EncodeLinear, self).__init__()

        self.features = nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.ReLU(),
            nn.Linear(dim_out, dim_out),
            nn.ReLU()
        )

        self.head_name = head_name
        self.feature_dim = dim_out
        self.train()

    def forward(self, inputs):
        x = inputs
        feature = self.features(x)
        return feature

class GoalLayer(nn.Module):
    def __init__(self,in_dim, hidden_dim=32, device=torch.device('cpu')):
        super(GoalLayer,self).__init__()
        self.feature_dim=in_dim
        self.net=nn.Sequential(
            nn.Linear(in_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,2),
            nn.Softmax(dim=-1)
        )
        self.train()
        self.device=device
    
    def forward(self,inputs):
        x=inputs
        assign_prob=self.net(x) # The probability of assigning this goal to the agent
        return assign_prob

class GoalLayer_Single(nn.Module):
    def __init__(self,in_dim, hidden_dim=32, device=torch.device('cpu')):
        super(GoalLayer_Single,self).__init__()
        self.feature_dim=in_dim
        self.net=nn.Sequential(
            nn.Linear(in_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,1),
            nn.Softmax(dim=-2)
        )
        self.train()
        self.device=device

    def forward(self,inputs):
        x=inputs
        assign_prob=self.net(x) # The probability of assigning this goal to the agent
        return assign_prob

class PropNet(nn.Module):
    def __init__(self, node_dim_in, edge_dim_in, hidden_dim, node_dim_out, edge_dim_out, batch_norm=0, pstep=2):

        super(PropNet, self).__init__()

        self.node_dim_in = node_dim_in
        self.edge_dim_in = edge_dim_in
        self.hidden_dim = hidden_dim

        self.node_dim_out = node_dim_out
        self.edge_dim_out = edge_dim_out

        self.pstep = pstep

        # node encoder
        modules = [
            nn.Linear(node_dim_in, hidden_dim),
            nn.ReLU()]
        if batch_norm == 1:
            modules.append(nn.BatchNorm1d(hidden_dim))
        self.node_encoder = nn.Sequential(*modules)

        # edge encoder
        modules = [
            nn.Linear(node_dim_in * 2 + edge_dim_in, hidden_dim),
            nn.ReLU()]
        if batch_norm == 1:
            modules.append(nn.BatchNorm1d(hidden_dim))
        self.edge_encoder = nn.Sequential(*modules)

        # node propagator
        modules = [
            # input: node_enc, node_rep, edge_agg
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()]
        if batch_norm == 1:
            modules.append(nn.BatchNorm1d(hidden_dim))
        self.node_propagator = nn.Sequential(*modules)

        # edge propagator
        self.edge_propagators = nn.ModuleList()
        for i in range(pstep):
            modules = [
                # input: node_rep * 2,  edge_rep
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()]
            if batch_norm == 1:
                modules.append(nn.BatchNorm1d(hidden_dim))
            edge_propagator = nn.Sequential(*modules)
            self.edge_propagators.append(edge_propagator)

        # commented due to None grad problem
        # node predictor
        # modules = [
        #     nn.Linear(hidden_dim * 2, hidden_dim),
        #     nn.ReLU()]
        # if batch_norm == 1:
        #     modules.append(nn.BatchNorm1d(hidden_dim))
        # modules.append(nn.Linear(hidden_dim, node_dim_out))
        # self.node_predictor = nn.Sequential(*modules)

        # edge predictor
        modules = [
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU()]
        if batch_norm == 1:
            modules.append(nn.BatchNorm1d(hidden_dim))
        modules.append(nn.Linear(hidden_dim, edge_dim_out))
        self.edge_predictor = nn.Sequential(*modules)

    def forward(self, node_rep, edge_rep=None, edge_type=None,
                ignore_node=False, ignore_edge=False):
        # node_rep: batch x N x node_dim_in
        # edge_rep: N x N x edge_dim_in
        # edge_type: N x N x edge_type_num
        B, N, _ = node_rep.size()

        # node_enc
        node_enc = self.node_encoder(node_rep.view(-1, self.node_dim_in)).view(B, N, self.hidden_dim)
        # edge_enc
        node_rep_r = node_rep[:, :, None, :].repeat(1, 1, N, 1)
        node_rep_s = node_rep[:, None, :, :].repeat(1, N, 1, 1)
        if edge_rep is not None:
            tmp = torch.cat([node_rep_r, node_rep_s, edge_rep], -1)
        else:
            tmp = torch.cat([node_rep_r, node_rep_s], -1)

        edge_enc = self.edge_encoder(tmp.view(B * N * N, -1)).view(B, N, N, self.hidden_dim)

        if edge_type is not None:
            edge_enc = edge_enc * edge_type.view(B, N, N, self.edge_type_num, 1)[:, :, :, -1, :]

        for i in range(self.pstep):
            if i == 0:
                node_effect = node_enc
                edge_effect = edge_enc

            # calculate edge_effect
            node_effect_r = node_effect[:, :, None, :].repeat(1, 1, N, 1)
            node_effect_s = node_effect[:, None, :, :].repeat(1, N, 1, 1)
            tmp = torch.cat([node_effect_r, node_effect_s, edge_effect], -1)

            edge_effect = self.edge_propagators[i](tmp.view(B * N * N, -1)).view(B, N, N, -1)
            # edge effect: N * N * hidden_dim

            if edge_type is not None:
                edge_effect = edge_effect * edge_type.view(B, N, N, self.edge_type_num, 1)[:, :, :, -1, :]

            # calculate node_effect
            edge_effect_agg = edge_effect.sum(2)
            tmp = torch.cat([node_enc, node_effect, edge_effect_agg], -1)
            node_effect = self.node_propagator(tmp.view(B * N, -1)).view(B, N, self.hidden_dim)

        node_effect = torch.cat([node_effect, node_enc], -1).view(B * N, -1)
        edge_effect = torch.cat([edge_effect, edge_enc], -1).view(B * N * N, -1)

        # node_pred: B x N x node_dim_out
        # edge_pred: B x N x N x edge_dim_out
        if ignore_node:
            edge_pred = self.edge_predictor(edge_effect)
            return edge_pred.view(B, N, N, -1)
        if ignore_edge:
            node_pred = self.node_predictor(node_effect)
            return node_pred.view(B, N, -1)

        node_pred = self.node_predictor(node_effect).view(B, N, -1)
        edge_pred = self.edge_predictor(edge_effect).view(B, N, N, -1)
        return node_pred, edge_pred

class Graph_Infer(nn.Module):
    def __init__(self, dim_in, device=torch.device('cpu')):
        super(Graph_Infer, self).__init__()
        #self.N = num_agents
        self.propnet_selfloop = False
        #self.mask_remove_self_loop = torch.FloatTensor(
        #    np.ones((num_agents, num_agents)) - np.eye(num_agents)).view(1, num_agents, num_agents, 1)
        #self.mask_remove_self_loop = self.mask_remove_self_loop.to(device)
        self.device = device
        self.edge_type_num = 2
        # edge type: 2
        # 0 stands for null edge, and 1 stands for real edge

        self.graph_infer = PropNet(
            node_dim_in= dim_in, 
            edge_dim_in=0,
            hidden_dim= dim_in * 3,
            node_dim_out=0,
            edge_dim_out=self.edge_type_num,
            pstep=2,
            batch_norm=0)

    def forward(self, node_features, hard=True):
        # node_features: batch * N * global_feature_dim
        B, N, _ = node_features.size()

        edge_type_logits = self.graph_infer(node_features, ignore_node=True)
        
        edge_type = F.gumbel_softmax(edge_type_logits.view(B * N * N, self.edge_type_num), tau=0.5, hard=hard)
        edge_type = edge_type.view(B, N, N, self.edge_type_num)[:,:,:,-1].view(B, N, N, 1)
        
        if self.propnet_selfloop == False:
            mask_remove_self_loop = torch.FloatTensor(np.ones((N, N)) - np.eye(N)).view(1, N, N, 1)
            mask_remove_self_loop = mask_remove_self_loop.to(self.device)
            edge_type = edge_type * mask_remove_self_loop

        return edge_type_logits, edge_type

class ToM2C_multi(torch.nn.Module):
    # partial obs + communication + obstacle(MSMTC observation includes obstacles)
    # each agent can choose multiple targets at the same time
    def __init__(self, obs_space, action_spaces, args, device=torch.device('cpu')):
        super(ToM2C_multi, self).__init__()
        self.num_agents, num_entity, self.pose_dim = obs_space.shape  # entity = target + obstacles
    
        lstm_out = args.lstm_out
        head_name = args.model
        self.head_name = head_name
        self.mask = args.mask

        self.encoder = EncodeLinear(self.pose_dim, lstm_out, head_name, device)
        feature_dim = self.encoder.feature_dim

        self.attention = AttentionLayer(feature_dim, lstm_out * 2, device)
        feature_dim = self.attention.feature_dim

        self.GRU=RNN(feature_dim ,lstm_out ,1,device,'GRU')
        
        # create ToM, including camera ToM & target ToM
        cam_state_size = 4  # [x, y, cos, sin]

        # comment this model because it's useless, and if exists, ensure_share_grad will collapse due to none grad cited
        self.ToM_target = nn.Sequential(
            nn.Linear(lstm_out + self.attention.feature_dim*2, lstm_out),
            nn.ReLU(),
            nn.Linear(lstm_out,1),
            nn.Sigmoid()
        )
        self.ToM_GRU=RNN(cam_state_size, lstm_out, 1, device, 'GRU')
        self.other_goal=GoalLayer(lstm_out + self.attention.feature_dim + self.GRU.feature_dim, device=device)
        
        # GNN based communication inference
        self.graph_infer = Graph_Infer(self.attention.feature_dim + self.ToM_GRU.feature_dim, device=device)

        feature_dim=self.GRU.feature_dim + self.attention.feature_dim#+ 1 + 2# GRU_output + attention_feature + max_prob + msgs

        self.reduce_dim = nn.Sequential(
            nn.Linear(feature_dim , lstm_out),
            nn.ReLU(),
        )
        feature_dim = lstm_out + 3
        # create actor & critic
        self.actor = PolicyNet(feature_dim, spaces.Discrete(2), head_name, device)

        # centralized training
        self.critic_encoder = AttentionLayer(feature_dim, 6 * feature_dim, device)
        self.critic = ValueNet(6 * feature_dim, head_name, 1)

        self.train()
        self.device = device

    def forward(self, multi_obs, self_hiddens, ToM_hiddens, poses, mask, test=False, available_actions = None, train_comm = False):
        num_agents = self.num_agents
        num_targets = self.num_targets
        if len(multi_obs.size()) != 4:
            # not batch data, so turn them into a size-1 batch
            batch = False
            multi_obs = multi_obs.unsqueeze(0)
            self_hiddens = self_hiddens.unsqueeze(0)
            ToM_hiddens = ToM_hiddens.unsqueeze(0)
            poses = poses.unsqueeze(0)
            mask = mask.unsqueeze(0)
        else:
            batch = True

        batch_size = multi_obs.size()[0]
        obs_dim = multi_obs.size()[-1] # [batch,n,m,obs_dim]
        num_both = multi_obs.size()[2]

        idx = (torch.ones(num_agents, num_agents) - torch.diag(torch.ones(num_agents))).bool()
        # compute the mask of ToM inference and comm_domain
        #self_cam_loc = poses[:,:,:2].unsqueeze(2).repeat(1,1,num_agents,1)
        #other_cam_loc = poses[:,:,:2].unsqueeze(1).repeat(1,num_agents,1,1)
        #cam_dist = (self_cam_loc - other_cam_loc).float()
        #cam_dist = torch.norm(cam_dist,p=2,dim=-1).reshape(batch_size,num_agents,num_agents,1)
        #cam_mask = (cam_dist <= 2 * 800)
        if self.mask:
            comm_domain = mask
            ToM_goal_mask = mask[:,idx].reshape(batch_size,num_agents,num_agents-1,1,1)
        else:
            comm_domain = torch.zeros(batch_size, num_agents, num_agents, 1).to(self.device)
            comm_domain[:,idx] = 1
            comm_domain = comm_domain.bool()
            ToM_goal_mask = None

        # compute real cover: whether target covered by an agent is coverd by another agent.
        # real target coverage, 0.4 = 800(radius)/2000(area size)
        #real_target_cover = (multi_obs[:,:,-1] <= 0.4).reshape(num_agents, num_targets, 1).detach()
        real_target_cover = (multi_obs[:,:,:self.num_targets,-1] != 0).reshape(batch_size, num_agents, num_targets, 1).detach()

        real_self_cover = real_target_cover.unsqueeze(2).repeat(1, 1, num_agents - 1, 1, 1)
        real_others_cover = real_target_cover.unsqueeze(1).repeat(1, num_agents, 1, 1, 1)
        real_others_cover = real_others_cover[:,idx].reshape(batch_size, num_agents, num_agents-1, num_targets, 1)
        real_cover = real_others_cover # for ToM target training

        feature_target = Variable(multi_obs, requires_grad=True)

        feature_target = self.encoder(feature_target)  # [batch_size, num_agents, num_both, feature_dim]

        feature_target = feature_target.reshape(batch_size * num_agents, num_both, -1)
        att_features, global_feature = self.attention(feature_target) 
        att_features = att_features.reshape(batch_size, num_agents, num_both, -1)
        global_feature = global_feature.reshape(batch_size, num_agents, -1)
        #features: [batch, num_agents, num_targets, feature_dim]
        
        # split features into targets & obstacles
        ############ begin ###########
        obstacle_features = torch.mean(att_features[:,:,self.num_targets:], 2) if num_both > num_targets else torch.zeros(batch_size, num_agents, att_features.size()[-1]).to(self.device)
        att_features = att_features[:,:,:self.num_targets]
        ############# end ############

        h_self = self_hiddens.reshape(1, num_agents * batch_size, -1) # [1, num_agents * batch, hidden_size]
        global_features = global_feature.reshape(num_agents * batch_size, 1, -1) # [num_agents * batch, 1, feature_dim]
        GRU_outputs, hn_self = self.GRU(global_features, h_self)
        #GRU_outputs=GRU_outputs.squeeze(1) 
        hn_self=hn_self.reshape(batch_size, num_agents, -1)

        # ToM_input
        # cam_states = poses  #[batch, num_agents, cam_dim]
        # print(poses)
        # cam_dim = cam_states.size()[-1] # cam_dim=3

        # # transform to relative cam_states 
        # cam_states_duplicate = cam_states.unsqueeze(1).expand(batch_size, num_agents, num_agents, cam_dim)
        # cam_states_relative = cam_states_duplicate - cam_states.unsqueeze(2).expand(batch_size, num_agents, num_agents, cam_dim)
        # cam_state_theta = ((cam_states_relative[:,:,:,-1]/180) * np.pi).reshape(batch_size, num_agents, num_agents, 1)
        # camera_states = torch.cat((cam_states_relative[:,:,:,:2], torch.cos(cam_state_theta), torch.sin(cam_state_theta)),-1)
        poses = poses.reshape(batch_size*num_agents*(num_agents),1,-1) # [batch*n*n,1,4]
        h_ToM = ToM_hiddens.reshape(1,-1,self.ToM_GRU.feature_dim) #[1,batch*n*(n),dim]

        # model others' hidden state
        ToM_output,hn_ToM = self.ToM_GRU(poses,h_ToM)

        # GoalLayer input concat
        hn_ToM = hn_ToM.reshape(batch_size, num_agents, num_agents, -1)
        ToM_output = ToM_output.reshape(batch_size, num_agents, num_agents, -1)
        ToM_output_other = ToM_output[:,idx].reshape(batch_size, num_agents, num_agents-1, 1, -1)

        GRU_dim=GRU_outputs.size()[-1]
        ToM_dim=ToM_output.size()[-1]
        att_dim=att_features.size()[-1]
        GRU_output_expand = GRU_outputs.reshape(batch_size, num_agents, 1, 1, -1)
        GRU_output_expand = GRU_output_expand.expand(batch_size, num_agents, num_agents-1, num_targets, GRU_dim)
        ToM_output_expand = ToM_output_other.expand(batch_size, num_agents, num_agents-1, num_targets, ToM_dim)
        att_feature_expand = att_features.unsqueeze(2).expand(batch_size,num_agents,num_agents-1,num_targets,att_dim)

        # ToM_target: predicted version
        #camera_states_others = camera_states.reshape(batch_size, num_agents, num_agents, -1)[:,idx]
        #camera_states = (camera_states_others.reshape(batch_size,num_agents,num_agents-1,1,-1)).repeat(1, 1, 1, num_targets, 1)
        obstacle_features = obstacle_features.reshape(batch_size, num_agents, 1, 1, -1).repeat(1, 1, num_agents - 1, num_targets, 1)
        ToM_target_feature = torch.cat((att_feature_expand.detach(), obstacle_features.detach(), ToM_output_expand),-1)
        ToM_target_cover = self.ToM_target(ToM_target_feature) #[b, n, n-1, m, 1]
        #print(ToM_target_cover)
        # ToM_target: Groud Truth Version
        #ToM_target_cover = real_target_cover.unsqueeze(1).expand(batch_size, num_agents, num_agents, num_targets, 1)
        #idx = (torch.ones(num_agents, num_agents) - torch.diag(torch.ones(num_agents))).bool()
        #ToM_target_cover = ToM_target_cover[:,idx].reshape(batch_size, num_agents, num_agents-1, num_targets, 1)
        

        # other goals
        goal_feature = torch.cat((att_feature_expand.detach(), GRU_output_expand.detach(), ToM_output_expand),-1) # detach ToM_target here
        other_goals = self.other_goal(goal_feature) # [batch, n, n-1, m, 2]
        if self.mask:
            other_goals = other_goals * ToM_goal_mask

        # prepare masks
        mask = torch.ones([num_agents, num_agents-1]).to(self.device)
        mask_u = torch.triu(mask,0).reshape(1, num_agents, num_agents-1, 1).repeat(batch_size, 1, 1, num_targets)
        mask_l = torch.tril(mask,-1).reshape(1, num_agents, num_agents-1, 1).repeat(batch_size, 1, 1, num_targets)
        zeros = torch.zeros([batch_size, num_agents, 1, num_targets]).to(self.device)

        ############ GNN based communication ################
        ToM_goals = (other_goals[:,:,:,:,1] > 0.5).detach()
        tri_u_ToM = torch.cat((zeros, ToM_goals * mask_u), 2)
        tri_l_ToM = torch.cat((ToM_goals * mask_l, zeros), 2)
        ToM_goals = tri_u_ToM + tri_l_ToM
        diag_idx = torch.diag(torch.ones(num_agents)).bool()
        ToM_goals[:,diag_idx] += 1
        ToM_goals = ToM_goals.unsqueeze(-1) #[b, n, n, m, 1]
        node_features = att_features.unsqueeze(2).repeat(1,1,num_agents,1,1).detach()
        # goal-filter 
        node_features = torch.sum(node_features * ToM_goals, 3) # sum all targets 
        node_features = torch.cat((node_features, ToM_output.detach()),-1).reshape(batch_size * num_agents, num_agents, -1)

        #edge_logits, comm_edges = self.graph_infer(global_feature.reshape(batch_size,num_agents,-1))
        edge_logits, comm_edges = self.graph_infer(node_features)
        edge_logits = edge_logits.reshape(batch_size, num_agents, num_agents, num_agents, -1)[:,diag_idx] #[b, n, n, 2]
        comm_edges = comm_edges.reshape(batch_size, num_agents, num_agents, num_agents, -1)[:,diag_idx]  # [b, n, n, 1]
        comm_edges = comm_edges * comm_domain # only communicate with agents in self domain
        comm_domain_reshape = comm_domain.reshape(-1,1).repeat(1,2).bool() #[b*n*n, 2]
        # only for ablation test
        #comm_edges = torch.zeros(batch_size, num_agents, num_agents, 1).to(self.device)

        edge_logits = edge_logits.reshape(-1,2)[comm_domain_reshape]
        edge_logits = edge_logits.reshape(-1,2) #[k,2] only logits of those edges in comm domain can be saved for training
        edge_logits = F.softmax(edge_logits, dim=-1)

        comm_target_edges = comm_edges.reshape(batch_size, num_agents, num_agents, 1).repeat(1, 1, 1, num_targets)
        # although we use real target cover here, but it is only self real cover duplicate, so it still keeps the decentralized execution mode
        comm_target_edges = comm_target_edges * (real_target_cover.reshape(batch_size, num_agents, 1, num_targets).repeat(1, 1, num_agents, 1)) 
        comm_cnt = torch.sum(comm_target_edges,1).reshape(batch_size, num_agents, num_targets, 1) 

        # selected msg
        comm_msg = other_goals[:,:,:,:,1].detach()     # [batch, n, n-1, m]  comm_edge:[n, n, 1]
        # transfer [n,n-1,m] to [n,n,m], where the added part is the diagnoal
        # e.g. [[1,2],[3,4],[5,6]] -> [[0,1,2],[3,0,4],[5,6,0]] 
        tri_u = torch.cat((zeros, comm_msg * mask_u), 2)
        tri_l = torch.cat((comm_msg * mask_l, zeros), 2)
        comm_msg = tri_u + tri_l    # [batch, n, n, m] 
        # evaluation
        #comm_edges = 1 - comm_edges
        #comm_cnt = self.num_agents - 1 - comm_cnt
        #comm_edges = torch.zeros(self.num_agents,self.num_agents,1)
        #comm_cnt  = torch.zeros(self.num_agents,self.num_targets,1)
        #end of evaluation
        msgs = comm_msg * comm_target_edges
        msgs = torch.sum(msgs, 1).reshape(batch_size, num_agents, num_targets, 1)
        msgs = torch.cat((msgs,comm_cnt),-1)

        ######### end of GNN based communication ###############

        # decide self goals
        max_prob , _ = torch.max(other_goals[:,:,:,:,1],2) 
        max_prob = max_prob.reshape(batch_size, num_agents, num_targets, 1).detach() #[batch,n,m,1]
        GRU_outputs = GRU_outputs.reshape(batch_size, num_agents, 1, -1).expand(batch_size, num_agents, num_targets, self.GRU.feature_dim)
        # new version of actor feature, reduce self feature dim
        self_feature = torch.cat((att_features, GRU_outputs), -1) 
        self_feature = self.reduce_dim(self_feature)
        ToM_msgs = torch.cat((max_prob, msgs),-1)
        #ToM_msgs = self.raise_dim(ToM_msgs)
        # actor_feature = actor_feature + ToM_msgs
        actor_feature = torch.cat((self_feature,ToM_msgs),-1)

        #actor_feature=torch.cat((GRU_outputs, att_features, max_prob, msgs), -1) #[batch,n,m,dim]
        actor_dim = actor_feature.size()[-1]
        critic_feature = torch.sum(actor_feature,2)#.reshape(batch_size, 1, -1).repeat(1, num_agents, 1) #expand(num_agents, num_agents*actor_dim) #[b,n,dim]
        actor_feature = actor_feature.reshape(-1,actor_dim) #[batch*n*m,dim]

        if available_actions is not None:
            available_actions = available_actions.reshape(batch_size * num_agents * num_targets, -1)

        actions, entropies, log_probs, probs = self.actor(actor_feature, test, available_actions)

        
        if train_comm:
            zero_msgs = torch.zeros(batch_size, num_agents, num_targets, 3).to(self.device)
            zero_actor_feature = torch.cat((self_feature, zero_msgs),-1).reshape(-1,actor_dim)
            _,_,_,zero_probs = self.actor(zero_actor_feature, test, available_actions)
            probs = probs.reshape(batch_size, num_agents, num_targets, 2)
            zero_probs = zero_probs.reshape(batch_size, num_agents, num_targets,2)
            a = probs.max(-1)[1]
            b = zero_probs.max(-1)[1]
            real_edges = torch.sum(a ^ b,-1)
            real_edges = 1 - (real_edges == 0).float()
            real_edges = real_edges.unsqueeze(1).repeat(1, num_agents, 1)
            edges_label = real_edges.reshape(-1,1)[comm_domain]
            
            return hn_self, hn_ToM, edge_logits, comm_edges.squeeze(-1), real_edges, edges_label
        
        probs = probs.reshape(batch_size, num_agents, num_targets, -1)
        actions = actions.reshape(batch_size, num_agents, num_targets, -1)
        entropies = entropies.reshape(batch_size, num_agents, num_targets, -1)
        log_probs = log_probs.reshape(batch_size,num_agents, num_targets, -1)

        _, global_critic_feature = self.critic_encoder(critic_feature)
        values = self.critic(global_critic_feature).unsqueeze(1).repeat(1, num_agents, 1)

        if not batch:
            # squeeze all the tensor for env
            values = values.squeeze(0)
            actions = actions.squeeze(0)
            entropies = entropies.squeeze(0)
            log_probs = log_probs.squeeze(0)
            hn_self = hn_self.squeeze(0)
            hn_ToM = hn_ToM.squeeze(0)
            other_goals = other_goals.squeeze(0)
            #edge_logits = edge_logits.squeeze(0)
            comm_edges = comm_edges.squeeze(0)
            probs = probs.squeeze(0)
            real_cover =real_cover.squeeze(0)
            ToM_target_cover = ToM_target_cover.squeeze(0)

        return values, actions, entropies, log_probs, hn_self, hn_ToM, other_goals, edge_logits, comm_edges, probs, real_cover, ToM_target_cover

    def sample_noise(self):
        self.actor.sample_noise()
        self.actor.sample_noise()

    def remove_noise(self):
        self.actor.remove_noise()
        self.actor.remove_noise()

class ToM2C_single(torch.nn.Module):
    # partial obs + communication
    # each agent can only choose one target at the same time
    def __init__(self, obs_space, action_spaces, args, device=torch.device('cpu')):
        super(ToM2C_single, self).__init__()
        self.num_agents, num_entity, self.pose_dim = obs_space.shape  # num_targets = target + obstacles
        #self.num_obstacles = num_entity - self.num_targets
        lstm_out = args.lstm_out
        head_name = args.model
        self.head_name = head_name

        self.encoder = EncodeLinear(self.pose_dim, lstm_out, head_name, device)
        feature_dim = self.encoder.feature_dim

        self.attention = AttentionLayer(feature_dim, lstm_out, device)
        feature_dim = self.attention.feature_dim

        self.GRU=RNN(feature_dim ,lstm_out ,1,device,'GRU')

        # create ToM, including camera ToM & target ToM
        cam_state_size = 4  # [vx, vy, x, y]

        # comment this model because it's useless, and if exists, ensure_share_grad will collapse due to none grad cited
        self.ToM_target = nn.Sequential(
            nn.Linear(lstm_out + self.attention.feature_dim*2, lstm_out),
            nn.ReLU(),
            nn.Linear(lstm_out,1),
            nn.Sigmoid()
        )
        self.ToM_GRU=RNN(cam_state_size, lstm_out, 1, device, 'GRU')
        self.other_goal=GoalLayer_Single(lstm_out + self.attention.feature_dim, device=device)#+ self.GRU.feature_dim

        # GNN based communication inference
        self.graph_infer = Graph_Infer(self.attention.feature_dim + self.ToM_GRU.feature_dim, device=device)

        feature_dim=self.attention.feature_dim #+self.GRU.feature_dim #+ 1 + 2# GRU_output + attention_feature + max_prob + msgs


        self.reduce_dim = nn.Sequential(
            #nn.Linear(feature_dim, lstm_out * 2),
            #nn.ReLU(),
            nn.Linear(feature_dim , lstm_out),
            nn.LeakyReLU(),
        )
        feature_dim = self.num_agents*self.pose_dim + 3 + lstm_out
        # create actor & critic
        self.actor = PolicyNet_Single(feature_dim, spaces.Discrete(2), head_name, device)

        # centralized training
        self.critic_encoder = AttentionLayer(feature_dim, 6 * feature_dim, device)
        self.critic = ValueNet(6 * feature_dim, head_name, 1)

        self.train()
        self.device = device

    def forward(self, multi_obs, self_hiddens, ToM_hiddens, poses, masks, test=False, available_actions = None, train_comm=False):
        num_agents = self.num_agents
        num_targets = self.num_targets

        if len(multi_obs.size()) != 3:
            # not batch data, so turn them into a size-1 batch
            batch = False
            multi_obs = multi_obs.unsqueeze(0)
            self_hiddens = self_hiddens.unsqueeze(0)
            ToM_hiddens = ToM_hiddens.unsqueeze(0)
            #poses = poses.unsqueeze(0)
            #mask = mask.unsqueeze(0)
        else:
            batch = True

        batch_size = multi_obs.size()[0]
        comm_domain = (torch.ones(num_agents, num_agents) - torch.diag(torch.ones(num_agents))).bool().to(self.device)
        comm_domain = comm_domain.reshape(1,num_agents,num_agents,1).repeat(batch_size,1,1,1)

        self_vel = Variable(multi_obs[:,:,:2], requires_grad= True)
        self_pos = Variable(multi_obs[:,:,2:4], requires_grad= True)
        multi_obs = Variable(multi_obs[:,:,4+(num_agents-1)*2:].reshape(batch_size, num_agents, num_targets, 2),requires_grad=True)
        obs_dim = multi_obs.size()[-1]
        num_both = multi_obs.size()[2]

        # compute
        target_dis = torch.norm(multi_obs,2,dim=-1) # b*n*m
        min_dis,_ = torch.min(target_dis, dim=-1)
        min_dis = min_dis.unsqueeze(-1).repeat(1, 1, num_both)
        # compute real cover: whether target covered by an agent is coverd by another agent.
        # real target coverage, 0.4 = 800(radius)/2000(area size)
        #real_target_cover = (multi_obs[:,:,-1] <= 0.4).reshape(num_agents, num_targets, 1).detach()
        idx = (torch.ones(num_agents, num_agents) - torch.diag(torch.ones(num_agents))).bool()
        #real_target_cover = (multi_obs[:,:,:self.num_targets,-1] != 100).reshape(batch_size, num_agents, num_targets, 1).detach()
        real_target_cover = (min_dis == target_dis).reshape(batch_size, num_agents, num_targets, 1).detach()

        #real_self_cover = real_target_cover.unsqueeze(2).repeat(1, 1, num_agents - 1, 1, 1)
        real_others_cover = real_target_cover.unsqueeze(1).repeat(1, num_agents, 1, 1, 1)
        real_others_cover = real_others_cover[:,idx].reshape(batch_size, num_agents, num_agents-1, num_targets, 1)
        real_cover = real_others_cover # for ToM target training
        #multi_obs = Variable(multi_obs, requires_grad=True)

        feature_target = self.encoder(multi_obs)  # [batch_size, num_agents, num_both, feature_dim]

        feature_target = feature_target.reshape(batch_size * num_agents, num_both, -1)

        att_features, global_feature = self.attention(feature_target)
        att_features = att_features.reshape(batch_size, num_agents, num_both, -1)
        global_feature = global_feature.reshape(batch_size, num_agents, -1)
        #features: [batch, num_agents, num_targets, feature_dim]

        # split features into targets & obstacles
        ############ begin ###########
        #obstacle_features = torch.mean(att_features[:,:,self.num_targets:], 2) if num_both > num_targets else torch.zeros(batch_size, num_agents, att_features.size()[-1]).to(self.device)
        att_features = att_features[:,:,:self.num_targets]
        ############# end ############

        h_self = self_hiddens.reshape(1, num_agents * batch_size, -1) # [1, num_agents * batch, hidden_size]
        global_features = global_feature.reshape(num_agents * batch_size, 1, -1) # [num_agents * batch, 1, feature_dim]
        GRU_outputs, hn_self = self.GRU(global_features, h_self)
        #GRU_outputs=GRU_outputs.squeeze(1)
        hn_self=hn_self.reshape(batch_size, num_agents, -1)
        # ToM_input
        cam_states = self_pos  #[batch, num_agents, cam_dim]
        cam_dim = cam_states.size()[-1] # cam_dim=2

        # transform to relative cam_states
        cam_states_duplicate = cam_states.unsqueeze(1).expand(batch_size, num_agents, num_agents, cam_dim)
        self_vel_duplicate = self_vel.unsqueeze(1).expand(batch_size, num_agents, num_agents, 2)
        #idx = (torch.ones(num_agents,num_agents) - torch.diag(torch.ones(num_agents))).bool()
        #cam_states_duplicate = cam_states_duplicate[:,idx].reshape(batch_size, num_agents, num_agents-1, cam_dim) # [batch,n,n-1,3]
        cam_states_relative = cam_states_duplicate - cam_states.unsqueeze(2).expand(batch_size, num_agents, num_agents, cam_dim)
        other_pos = cam_states_relative[:,idx].reshape(batch_size, num_agents,(num_agents-1)*cam_dim)
        other_pos = other_pos.unsqueeze(2).repeat(1,1,num_targets,1) # used for final goal decision
        camera_states = torch.cat((cam_states_relative, self_vel_duplicate), -1)

        # pose mask
        mask = False
        cam_dis = torch.norm(cam_states_relative,p=2,dim=-1)
        _,sort_id = torch.sort(cam_dis,dim=2)
        sort_id = (sort_id == 1) # can only see nearest 4 poses
        near_dist = (cam_dis[sort_id]).reshape(batch_size,num_agents,1).repeat(1,1,num_agents)
        pose_mask = (cam_dis <= near_dist).unsqueeze(-1)
        if mask:
            ToM_goal_mask = pose_mask[:,idx].reshape(batch_size,num_agents,num_agents-1,1,1)
            comm_domain = pose_mask * comm_domain

        #cam_state_theta = ((cam_states_relative[:,:,:,-1]/180) * np.pi).reshape(batch_size, num_agents, num_agents, 1)
        #camera_states = torch.cat((cam_states_relative[:,:,:,:2], torch.cos(cam_state_theta), torch.sin(cam_state_theta)),-1)
        camera_states = camera_states.reshape(batch_size*num_agents*(num_agents),1,-1) # [batch*n*n,1,4]
        h_ToM = ToM_hiddens.reshape(1,-1,self.ToM_GRU.feature_dim) #[1,batch*n*(n),dim]

        # ToM_camera prediction
        ToM_output,hn_ToM = self.ToM_GRU(camera_states,h_ToM)

        # GoalLayer input concat
        hn_ToM = hn_ToM.reshape(batch_size, num_agents, num_agents, -1)
        ToM_output = ToM_output.reshape(batch_size, num_agents, num_agents, -1)
        ToM_output_other = ToM_output[:,idx].reshape(batch_size, num_agents, num_agents-1, 1, -1)

        GRU_dim=GRU_outputs.size()[-1]
        ToM_dim=ToM_output.size()[-1]
        att_dim=att_features.size()[-1]
        GRU_output_expand = GRU_outputs.reshape(batch_size, num_agents, 1, 1, -1)
        GRU_output_expand = GRU_output_expand.expand(batch_size, num_agents, num_agents-1, num_targets, GRU_dim)
        ToM_output_expand = ToM_output_other.expand(batch_size, num_agents, num_agents-1, num_targets, ToM_dim)
        att_feature_expand = att_features.unsqueeze(2).expand(batch_size,num_agents,num_agents-1,num_targets,att_dim)
        global_features_expand = global_features.reshape(batch_size, num_agents, 1, 1, -1).repeat(1,1,num_agents-1,num_targets,1)
        # ToM_target: predicted version
        #camera_states_others = camera_states.reshape(batch_size, num_agents, num_agents, -1)[:,idx]
        #camera_states = (camera_states_others.reshape(batch_size,num_agents,num_agents-1,1,-1)).repeat(1, 1, 1, num_targets, 1)
        #obstacle_features = obstacle_features.reshape(batch_size, num_agents, 1, 1, -1).repeat(1, 1, num_agents - 1, num_targets, 1)
        ToM_target_feature = torch.cat((att_feature_expand.detach(), global_features_expand.detach(), ToM_output_expand),-1)
        ToM_target_cover = self.ToM_target(ToM_target_feature) #[b, n, n-1, m, 1]

        # ToM_target: Groud Truth Version
        #ToM_target_cover = real_target_cover.unsqueeze(1).expand(batch_size, num_agents, num_agents, num_targets, 1)
        #idx = (torch.ones(num_agents, num_agents) - torch.diag(torch.ones(num_agents))).bool()
        #ToM_target_cover = ToM_target_cover[:,idx].reshape(batch_size, num_agents, num_agents-1, num_targets, 1)


        # other goals
        goal_feature = torch.cat((att_feature_expand.detach(), ToM_output_expand),-1)#GRU_output_expand.detach(), # detach ToM_target here
        other_goals = self.other_goal(goal_feature) # [batch, n, n-1, m, 1]
        if mask:
            other_goals = other_goals * ToM_goal_mask
        # prepare masks
        mask = torch.ones([num_agents, num_agents-1]).to(self.device)
        mask_u = torch.triu(mask,0).reshape(1, num_agents, num_agents-1, 1).repeat(batch_size, 1, 1, num_targets)
        mask_l = torch.tril(mask,-1).reshape(1, num_agents, num_agents-1, 1).repeat(batch_size, 1, 1, num_targets)
        zeros = torch.zeros([batch_size, num_agents, 1, num_targets]).to(self.device)

        ############ GNN based communication ################
        other_goals = other_goals.squeeze(-1) #[b,n,n-1,m]
        ToM_goals = other_goals.max(dim=-1)[0]
        ToM_goals = ToM_goals.reshape(batch_size, num_agents, num_agents-1, 1).repeat(1,1,1,num_targets)
        ToM_goals = (other_goals >= ToM_goals).detach()
        tri_u_ToM = torch.cat((zeros, ToM_goals * mask_u), 2)
        tri_l_ToM = torch.cat((ToM_goals * mask_l, zeros), 2)
        ToM_goals = tri_u_ToM + tri_l_ToM
        diag_idx = torch.diag(torch.ones(num_agents)).bool()
        ToM_goals[:,diag_idx] += 1
        ToM_goals = ToM_goals.unsqueeze(-1) #[b, n, n, m, 1]
        node_features = att_features.unsqueeze(2).repeat(1,1,num_agents,1,1).detach()
        node_features = torch.sum(node_features * ToM_goals, 3) # sum all targets
        node_features = torch.cat((node_features, ToM_output.detach()),-1).reshape(batch_size * num_agents, num_agents, -1)

        #edge_logits, comm_edges = self.graph_infer(global_feature.reshape(batch_size,num_agents,-1))
        edge_logits, comm_edges = self.graph_infer(node_features)
        edge_logits = edge_logits.reshape(batch_size, num_agents, num_agents, num_agents, -1)[:,diag_idx] #[b, n, n, 2]
        comm_edges = comm_edges.reshape(batch_size, num_agents, num_agents, num_agents, -1)[:,diag_idx]  # [b, n, n, 1]
        comm_edges = comm_edges * comm_domain # only communicate with agents in self domain
        comm_domain_reshape = comm_domain.reshape(-1,1).repeat(1,2) #[b*n*n, 2]

        # only for ablation test
        #comm_edges = torch.zeros(batch_size, num_agents, num_agents, 1).to(self.device)

        edge_logits = edge_logits.reshape(-1,2)[comm_domain_reshape]
        edge_logits = edge_logits.reshape(-1,2) #[k,2] only logits of those edges in comm domain can be saved for training
        edge_logits = F.softmax(edge_logits, dim=-1)

        comm_target_edges = comm_edges.reshape(batch_size, num_agents, num_agents, 1).repeat(1, 1, 1, num_targets)
        # although we use real target cover here, it is only self real cover duplicate, so it still keeps the decentralized execution mode
        comm_target_edges = comm_target_edges * (real_target_cover.reshape(batch_size, num_agents, 1, num_targets).repeat(1, 1, num_agents, 1))
        comm_cnt = torch.sum(comm_target_edges,1).reshape(batch_size, num_agents, num_targets, 1)
        # selected msg
        comm_msg = other_goals.detach()     # [batch, n, n-1, m]  comm_edge:[n, n, 1]
        # transfer [n,n-1,m] to [n,n,m], where the added part is the diagnoal
        # e.g. [[1,2],[3,4],[5,6]] -> [[0,1,2],[3,0,4],[5,6,0]]
        tri_u = torch.cat((zeros, comm_msg * mask_u), 2)
        tri_l = torch.cat((comm_msg * mask_l, zeros), 2)
        comm_msg = tri_u + tri_l    # [batch, n, n, m]
        # evaluation
        #comm_edges = 1 - comm_edges
        #comm_cnt = self.num_agents - 1 - comm_cnt
        #comm_edges = torch.zeros(self.num_agents,self.num_agents,1)
        #comm_cnt  = torch.zeros(self.num_agents,self.num_targets,1)
        #end of evaluation
        msgs = comm_msg * comm_target_edges
        msgs = torch.sum(msgs, 1).reshape(batch_size, num_agents, num_targets, 1)
        msgs = torch.cat((msgs,comm_cnt),-1)

        ######### end of GNN based communication ###############

        # decide self goals
        max_prob , _ = torch.max(other_goals,2)
        max_prob = max_prob.reshape(batch_size, num_agents, num_targets, 1).detach() #[batch,n,m,1]
        GRU_outputs = GRU_outputs.reshape(batch_size, num_agents, 1, -1).expand(batch_size, num_agents, num_targets, self.GRU.feature_dim)
        # new version of actor feature, reduce self feature dim
        #self_feature = torch.cat((att_features, GRU_outputs), -1)
        self_feature = self.reduce_dim(att_features)
        ToM_msgs = torch.cat((max_prob, msgs),-1)
        #ToM_msgs = self.raise_dim(ToM_msgs)
        # actor_feature = actor_feature + ToM_msgs
        #feature_target = feature_target.reshape(batch_size,num_agents,num_targets,-1)
        actor_feature = torch.cat((multi_obs,self_feature,other_pos,ToM_msgs),-1)
        #actor_feature=torch.cat((GRU_outputs, att_features, max_prob, msgs), -1) #[batch,n,m,dim]
        actor_dim = actor_feature.size()[-1]
        critic_feature = torch.sum(actor_feature,2)#.reshape(batch_size, 1, -1).repeat(1, num_agents, 1) #expand(num_agents, num_agents*actor_dim) #[b,n,dim]
        actor_feature = actor_feature.reshape(batch_size * num_agents, num_targets, actor_dim) #[batch*n*m,dim]
        # only select target in one's view or received communication
        #print(real_target_cover)
        if available_actions is None:
            available_actions = (real_target_cover + comm_cnt)>0
            available_actions = available_actions.reshape(batch_size * num_agents, num_targets, -1)
        else:
            available_actions = available_actions.reshape(batch_size * num_agents, num_targets, -1)

        actions, entropies, log_probs, probs = self.actor(actor_feature, test, available_actions=None)
      
        if train_comm:
            zero_msgs = torch.zeros(batch_size, num_agents, num_targets, 3).to(self.device)
            zero_actor_feature = torch.cat((multi_obs,self_feature,other_pos, zero_msgs),-1).reshape(batch_size * num_agents, num_targets, actor_dim)
            _,_,_,zero_probs = self.actor(zero_actor_feature, test, None)
            probs = probs.reshape(batch_size, num_agents, -1)
            zero_probs = zero_probs.reshape(batch_size, num_agents,-1)
            a = probs.max(-2)[1].unsqueeze(-1)
            b = zero_probs.max(-2)[1].unsqueeze(-1)
            #print(a.squeeze())
            #print(b.squeeze())
            
            real_edges = torch.sum(a == b,-1)
            real_edges = 1 - (real_edges == 1).float()
            real_edges = real_edges.unsqueeze(1).repeat(1, num_agents, 1)
            edges_label = real_edges.reshape(-1,1)[comm_domain.reshape(-1,1)]
            #print(edges_label)
            #print("----------")
            
            return hn_self, hn_ToM, edge_logits, comm_edges.squeeze(-1), real_edges, edges_label

        probs = probs.reshape(batch_size, num_agents, -1)
        actions = actions.reshape(batch_size, num_agents, -1)
        #entropies = entropies.reshape(batch_size, num_agents, -1)
        log_probs = log_probs.reshape(batch_size,num_agents, -1)

        _, global_critic_feature = self.critic_encoder(critic_feature)
        values = self.critic(global_critic_feature).unsqueeze(1).repeat(1, num_agents, 1)

        if not batch:
            # squeeze all the tensor for env
            values = values.squeeze(0)
            actions = actions.squeeze(0)
            #entropies = entropies.squeeze(0)
            log_probs = log_probs.squeeze(0)
            hn_self = hn_self.squeeze(0)
            hn_ToM = hn_ToM.squeeze(0)
            other_goals = other_goals.squeeze(0)
            #edge_logits = edge_logits.squeeze(0)
            comm_edges = comm_edges.squeeze(0)
            probs = probs.squeeze(0)
            real_cover =real_cover.squeeze(0)
            ToM_target_cover = ToM_target_cover.squeeze(0)
            available_actions = available_actions.reshape(num_agents, num_targets, 1)

        return values, actions, entropies, log_probs, hn_self, hn_ToM, other_goals, edge_logits, comm_edges, probs, real_cover, ToM_target_cover

    def sample_noise(self):
        self.actor.sample_noise()
        self.actor.sample_noise()

    def remove_noise(self):
        self.actor.remove_noise()
        self.actor.remove_noise()

class A3C_Single(torch.nn.Module):  # single vision Tracking
    def __init__(self, obs_space, action_spaces, args, device=torch.device('cpu')):
        super(A3C_Single, self).__init__()
        self.n = len(obs_space)
        obs_dim = obs_space[0].shape[0]
        lstm_out = args.lstm_out
        head_name = args.model

        self.head_name = head_name

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim,lstm_out),
            nn.ReLU(),
            nn.Linear(lstm_out, 16),
            nn.ReLU()
        )
        self.encoder[0].weight.data = norm_col_init(self.encoder[0].weight.data, 0.1)
        self.encoder[0].bias.data.fill_(0)
        self.encoder[2].weight.data = norm_col_init(self.encoder[2].weight.data, 0.1)
        self.encoder[2].bias.data.fill_(0)

        self.critic = ValueNet(16, head_name, 1)
        self.actor = PolicyNet(16, action_spaces[0], head_name, device)

        self.train()
        self.device = device

    def forward(self, inputs, test=False):
        data = Variable(inputs, requires_grad=True)
        if len(data.size()) != 3:
            # not batch data, so turn them into a size-1 batch
            batch = False
            data = data.unsqueeze(0)
        else:
            batch = True
        batch_size = data.size()[0]
        feature = self.encoder(data)
        feature = feature.reshape(batch_size*self.n , -1)

        actions, entropies, log_probs, probs = self.actor(feature, test)
        values = self.critic(feature)

        actions = actions.reshape(batch_size, self.n,-1)
        entropies = entropies.reshape(batch_size,self.n,-1)
        log_probs = log_probs.reshape(batch_size,self.n,-1)
        probs = probs.reshape(batch_size, self.n, -1)
        values = values.reshape(batch_size,self.n,-1)
        if not batch:
            actions = actions.squeeze(0)
            entropies = entropies.squeeze(0)
            log_probs = log_probs.squeeze(0)
            probs = probs.squeeze(0)
            values = values.squeeze(0)
        else:
            actions_env = None
        return values, actions, entropies, log_probs, probs

    def sample_noise(self):
        self.actor.sample_noise()
        self.actor.sample_noise()

    def remove_noise(self):
        self.actor.remove_noise()
        self.actor.remove_noise()

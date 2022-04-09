import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn import Parameter
from torch.autograd import Variable


class NoisyLinear(nn.Linear):
    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
        super(NoisyLinear, self).__init__(in_features, out_features, bias=True)
        self.sigma_init = sigma_init
        self.sigma_weight = Parameter(torch.Tensor(out_features, in_features))
        self.sigma_bias = Parameter(torch.Tensor(out_features))
        self.register_buffer('epsilon_weight', torch.zeros(out_features, in_features))
        self.register_buffer('epsilon_bias', torch.zeros(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, 'sigma_weight'):
            init.uniform(self.weight, -math.sqrt(3 / self.in_features), math.sqrt(3 / self.in_features))
            init.uniform(self.bias, -math.sqrt(3 / self.in_features), math.sqrt(3 / self.in_features))
            init.constant(self.sigma_weight, self.sigma_init)
            init.constant(self.sigma_bias, self.sigma_init)

    def forward(self, input):
        return F.linear(input, self.weight + self.sigma_weight * Variable(self.epsilon_weight),
                        self.bias + self.sigma_bias * Variable(self.epsilon_bias))

    def sample_noise(self):
        self.epsilon_weight = torch.randn(self.out_features, self.in_features)
        self.epsilon_bias = torch.randn(self.out_features)

    def remove_noise(self):
        self.epsilon_weight = torch.zeros(self.out_features, self.in_features)
        self.epsilon_bias = torch.zeros(self.out_features)


class BiRNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, device, head_name):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        if 'lstm' in head_name:
            self.lstm = True
        else:
            self.lstm = False
        if self.lstm:
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True).to(device)
        else:
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True).to(device)
        self.feature_dim = hidden_size * 2
        self.device = device

    def forward(self, x, state=None):
        # Set initial states

        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(self.device)  # 2 for bidirection
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(self.device)

        # Forward propagate LSTM
        if self.lstm:
            out, (_, hn) = self.rnn(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        else:
            out, hn = self.rnn(x, h0)  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        return out, hn

class RNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, device, head_name):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        if 'lstm' in head_name:
            self.lstm = True
        else:
            self.lstm = False
        if self.lstm:
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True).to(device)
        else:
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True).to(device)
        
        self.feature_dim = hidden_size
        # add layer normalization to stable training
        self.LayerNorm = nn.LayerNorm([hidden_size])
        self.device = device

    def forward(self, x, h0, c0=None, state=None):  
        # x: [batch_size, seq_length, input_size] h:[num_layers, batch_size, hidden_size]
        # Forward propagate LSTM
        h0 = self.LayerNorm(h0)
        if self.lstm:
            out, (_, hn) = self.rnn(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        else:
            out, hn = self.rnn(x, h0)  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        hn = self.LayerNorm(hn)

        return out, hn


def xavier_init(layer):
    torch.nn.init.xavier_uniform_(layer.weight)
    torch.nn.init.constant_(layer.bias, 0)
    return layer


class AttentionLayer(torch.nn.Module):
    def __init__(self, feature_dim, weight_dim, device=torch.device('cpu')):
        super(AttentionLayer, self).__init__()
        self.in_dim = feature_dim
        self.device = device

        self.Q = xavier_init(nn.Linear(self.in_dim, weight_dim))
        self.K = xavier_init(nn.Linear(self.in_dim, weight_dim))
        self.V = xavier_init(nn.Linear(self.in_dim, weight_dim))

        self.feature_dim = weight_dim

    def forward(self, x):
        '''
        inference
        :param x: [num_agent, num_target, feature_dim]
        :return z: [num_agent, num_target, weight_dim]
        '''
        # z = softmax(Q,K)*V
        q = torch.tanh(self.Q(x))  # [batch_size, sequence_len, weight_dim]
        k = torch.tanh(self.K(x))  # [batch_size, sequence_len, weight_dim]
        v = torch.tanh(self.V(x))  # [batch_size, sequence_len, weight_dim]

        z = torch.bmm(F.softmax(torch.bmm(q, k.permute(0, 2, 1)), dim=2), v)  # [batch_size, sequence_len, weight_dim]

        global_feature = z.sum(dim=1)
        return z, global_feature

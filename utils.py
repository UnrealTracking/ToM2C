from __future__ import division
import math
import json
import torch
import logging
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

def setup_logger(logger_name, log_file, level=logging.INFO):
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    l.setLevel(level)
    l.addHandler(fileHandler)
    l.addHandler(streamHandler)


def read_config(file_path):
    """Read JSON config."""
    json_object = json.load(open(file_path, 'r'))
    return json_object


def norm_col_init(weights, std=1.0):
    x = torch.randn(weights.size())
    x *= std / torch.sqrt((x ** 2).sum(1, keepdim=True))
    return x


def ensure_shared_grads(model, shared_model, device, device_share):
    diff_device = device != device_share
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if param.grad is None:
            continue
        if shared_param.grad is not None and not diff_device:
            return
        elif not diff_device:
            shared_param._grad = param.grad
        else:
            shared_param._grad = param.grad.to(device_share)


def ensure_shared_grads_param(params, shared_params, gpu=False):
    for param, shared_param in zip(params, shared_params):
        # print (shared_param)
        if shared_param.grad is not None and not gpu:
            return

        if not gpu:
            shared_param._grad = param.grad
        else:
            shared_param._grad = param.grad.clone().cpu()


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)


def weights_init_mlp(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / \
                         torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


def normal(x, mu, sigma, device):
    pi = np.array([math.pi])
    pi = torch.from_numpy(pi).float()
    pi = Variable(pi).to(device)
    a = (-1 * (x - mu).pow(2) / (2 * sigma)).exp()
    b = 1 / (2 * sigma * pi.expand_as(sigma)).sqrt()
    return a * b


def check_path(path):
    import os
    if not os.path.exists(path):
        os.mkdir(path)


def goal_id_filter(goals):
    return np.where(goals > 0.5)[0]


def norm(x, scale):
    assert len(x.shape) <= 2
    x = scale * (x - x.mean(0)) / (x.std(0) + 1e-6)  # normalize with batch mean and std; plus a small number to prevent numerical problem
    return x


class ToTensor(object):
    def __call__(self, sample):
        sample = sample.transpose(0, 3, 1, 2)
        return torch.from_numpy(sample.astype(np.float32))

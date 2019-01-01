import torch.multiprocessing as mp
import os, time, random
from othello import Othello
from othello_utils import *
from MCTS import MCTS
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
import warnings

warnings.filterwarnings('ignore')

def weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d):
        nn.init.normal(m.weight.data, 0, 2)
        # nn.init.normal(m.bias.data, 0, 2) 

def loss_fn(my_value, labels, my_probs, rollout_prob):
    # print(my_value[0], labels[0], my_probs[0].reshape(8,8), rollout_prob[0].reshape(8,8))
    return torch.mean(((my_value - torch.Tensor(labels.astype(float)).reshape(-1,1).cuda())**2) - torch.log(my_probs+1e-7).mm(torch.t(torch.Tensor(rollout_prob).cuda())).gather(1, torch.range(0, 127).reshape(-1,1).long().cuda())).cuda()

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, learning_rate=0.01, weight_decay=0):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=1)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=1)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=1)
        self.conv_policy = nn.Conv2d(512, 2, 1)
        self.bn_policy = nn.BatchNorm2d(2)
        self.fc_policy =  nn.Linear(2*8*8, 64)

        self.conv_value = nn.Conv2d(512, 1, 1)
        self.bn_value = nn.BatchNorm2d(1)
        self.fc_value_1 = nn.Linear(1*8*8, 32)
        self.fc_value_2 = nn.Linear(32, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        if len(x.shape) ==2:
            x = torch.Tensor(x[np.newaxis, np.newaxis, :, :])
        else:
            x = torch.Tensor(x)
        # print(x.shape)
        x = x.cuda()
        # print(type(x))
        out = F.relu(self.bn1(self.conv1(x)))
        # print(x.shape)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        x = self.layer4(out)

        policy = F.relu(self.bn_policy(self.conv_policy(x)))
        policy = policy.view(-1, 8*8*2)
        policy = F.softmax(self.fc_policy(policy))
        v = F.relu(self.bn_value(self.conv_value(x)))
        v = v.view(-1, 8*8*1)
        v = F.relu(self.fc_value_1(v))
        v = F.tanh(self.fc_value_2(v))
        return policy, v


def CreateResNet(layer_size=[2,2,2,2], learning_rate=0.01, weight_decay=0):
    return ResNet(BasicBlock, layer_size, learning_rate, weight_decay)
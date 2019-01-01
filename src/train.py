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

N_HIDDEN = 5

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
    def __init__(self, block, num_blocks):
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
        self.optimizer = optim.Adam(self.parameters(), lr=0.002)

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


def ResNetNet():
    return ResNet(BasicBlock, [2,2,2,2])


class Net(nn.Module):
    """docstring for Net"""
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(32)

        self.conv_policy = nn.Conv2d(32, 2, 1)
        self.bn_policy = nn.BatchNorm2d(2)
        self.fc_policy =  nn.Linear(2*8*8, 64)

        self.conv_value = nn.Conv2d(32, 1, 1)
        self.bn_value = nn.BatchNorm2d(1)
        self.fc_value_1 = nn.Linear(1*8*8, 32)
        self.fc_value_2 = nn.Linear(32, 1)
        self.c = 0.0
        self.optimizer = optim.Adam(self.parameters(), lr=0.01)


        
   
    

    def forward(self, x):
#         print(x.type())
        if len(x.shape) ==2:
            x = torch.Tensor(x[np.newaxis, np.newaxis, :, :])
        else:
            x = torch.Tensor(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        policy = F.relu(self.bn_policy(self.conv_policy(x)))
        policy = policy.view(-1, 8*8*2)
        policy = F.softmax(self.fc_policy(policy))
        # print('AAAAAAAAAAA\n',policy, policy.shape)
        v = F.relu(self.bn_value(self.conv_value(x)))
        v = v.view(-1, 8*8*1)
        # print('BBBBBBBBBBB\n',v, v.shape)
        v = F.relu(self.fc_value_1(v))
        v = F.tanh(self.fc_value_2(v))
        # print(policy, v)
        return policy, v


def expand_func(s0, prob, w, m):
    '''
    Input:
        s0: game board
        prob: prob from simulation
        w: winner
        m: expand method
    Output: 
        s0_, prob_: change from s0 and prob
    '''
    s0_ = np.rot90(s0)
    prob_ = np.rot90(prob.reshape(8, 8))
    for _ in range(m//2):
        s0_ = np.rot90(s0_)
        prob_ = np.rot90(prob_)
    if m%2 == 0:
        s0_ = s0_.T
        prob_ = prob_.T
    prob_ = prob_.reshape((64,))
    return s0_, prob_, w


def self_play(i, net):
    # print("Begin %d process..." % i)
    st = time.time()
    net.optimizer.zero_grad()
    
    batch_size = 128
    state_data = []
    game = Othello()
    mctsTest = MCTS(net, 1000)
    mctsTest.virtualLoss(game)
    side = -1
    Tau = 1
    while not game.game_over():
        # print(i)
        # game.print_board(side)
        game.board *= -side
        probs = mctsTest.search(game, Tau)
        # Tau *= 0.9
        state_data.append([game.board.copy(), probs, side])
        # print(probs)
        if np.sum(probs) > 0:
            action = np.sum(np.random.rand()>np.cumsum(probs))
#             action = np.argmax(probs)
            game.board *= -side
            game.play_move(*convert_mv_ind_to_tuple(action), side)
        else:
            game.play_move(-1,-1,-1)

        side = -side
        
    
    # print("finish search ", i)
    winner = game.get_winner()
#     print(winner)
    for state, _ in enumerate(state_data):
        state_data[state][2] *= -winner
    
    
    expand_data = []
    for s in state_data:
        # print("------------------------")
        # print('board: ')
        # print(s[0], type(s[0]), s[0].shape)
        # print('probs: ')
        # print(s[1], type(s[1]), s[1].shape)
        # print('side: ')
        # print(s[2])
        for func_index in np.random.permutation(7)[:2]:
            expand_data.append(expand_func(s[0], s[1], s[2], func_index))
            # print("=======================")
            # print(s[0], s[1], s[2])
            # print(expand_data[-1])
    
    # print('s',i)
    np.random.shuffle(expand_data)
    batch_data = np.concatenate([state_data, expand_data[:batch_size - len(state_data)]], axis=0)
    inputs = np.concatenate(batch_data[:, 0]).reshape(-1, 8, 8)[:, np.newaxis, :, :]
    rollout_prob = np.concatenate(batch_data[:,1]).reshape(-1, 64)
    labels = batch_data[:, 2]
    # print('b',i)
# for kkk in range(1000):
    my_probs, my_value = net(inputs)
    # print('aa',i)
#     print(my_value)
    loss = loss_fn(my_value, labels, my_probs, rollout_prob)
    net.optimizer.zero_grad()                # clear gradients for next train
    loss.backward(retain_graph=True)
    net.optimizer.step()
    # print('lllllllllll.lllllllllllllllllllll',kkk, float(loss))
    # print('kk',i)
    ed = time.time()
    print("%6d game, time=%4.4fs, loss = %5.5f" % (i, ed-st, float(loss)))
    return inputs, rollout_prob, labels


def play_game():
    game = Othello()
    mctsTest = MCTS(net, 1000)





# def NetworkRefresh(net, q):
#     game_count = 0
#     buffer_count = 0

#     buffer_size = 1000
#     batch_size = 200

#     board_buffer = np.zeros((buffer_size, 1,8,8))
#     probs_buffer = np.zeros((buffer_size, 64))
#     v_buffer = np.zeros((buffer_size))
    
#     while True:
#         game_states = q.get(True)
#         game_count += 1
#         for state in game_states:
#             board_buffer[buffer_count, :, :, :] = state[0][np.newaxis, :, :] 
#             probs_buffer[buffer_count, :] = state[1]
#             v_buffer[buffer_count] = state[2]
#             buffer_count = (buffer_count+1) % buffer_size
        


import argparse
import sys

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter", type=int, default=100)
    parser.add_argument("--start_iter", type=int, default=-1)
    parser.add_argument("--log_dir",type=str,default="./log.txt")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]='0,5'
    # num_processes = 10
    # model = Net().float()
    # model.apply(model.weights_init)
    # model.share_memory()
    # pool = mp.Pool()
    # for rank in range(num_processes):
    #     pool.apply_async(self_play, args=(rank,model))
    # pool.close()
    # pool.join()
    log = open(args.log_dir,'a')
    sys.stdout = log
    model = ResNetNet().cuda()
    if args.start_iter != -1:
        print("loading model...")
        model=torch.load("./model/model"+str(args.start_iter)+".pkl").cuda()
    else:
        model.apply(weights_init)
    print("start training")
    print(model)
    # model = Net().float()
    # model.share_memory()
    # pool = mp.Pool()
    # for rank in range(num_processes):
    #     pool.apply_async(self_play, args=(rank,model))
    # pool.close()
    # pool.join()
    for rank in range(args.start_iter+1,args.start_iter+args.iter):
        self_play(rank, model)
        log.flush()
        if rank % 20 == 0:
            torch.save(model, './model/model'+str(rank)+'.pkl')
    print("Finish all the train!")

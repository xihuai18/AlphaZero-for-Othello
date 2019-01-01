from othello import Othello
from DQN import DQN, MEMORY_CAPACITY
from player import RandomPlayer
from tqdm import tqdm
from othello_utils import *
import torch.nn as nn
import argparse

SAVER_ITER = 100
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
parser = argparse.ArgumentParser()
parser.add_argument("--iter", type=int, default=100)
parser.add_argument("--start_iter", type=int)
parser.add_argument("--log_dir", type=str, default="./logDQN.txt")
args = parser.parse_args()

log = open(args.log_dir,'a+',encoding='utf8')

dqn = DQN()
if args.start_iter:
    dqn.load("./model",args.start_iter)
else:
    args.start_iter = -1
side = -1
for i_episode in range(args.start_iter+1,args.iter):
    game = Othello()
    while not game.game_over():
        s = convert_board_to_feature(game.board, side)
        a = dqn.choose_action(s, side)
        game.play_move(a[0],a[1],side)
        s_p = convert_board_to_feature(game.board, side)
        # MCTS
        subGame = game.copy()
        subSide = side
        rp1 = RandomPlayer(side)
        rp2 = RandomPlayer(-side)
        while not subGame.game_over():
            subGame.play_move(*rp2.pick_move(subGame),-side)
            subGame.play_move(*rp1.pick_move(subGame),side)
        winner = subGame.get_winner()
        # 注意，这个实现不区分执子方，只要记录s和a对应的r和s_p即可
        if winner == side:
            r = 10
        elif winner == -side:
            r = -10
        else:
            r = 0
        if not (a[0] == -1 and a[1] == -1):
            dqn.store_transition(s,convert_mv_tuple_to_ind(a),r,s_p)
        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn()
        side *= -1
    loss = dqn.get_loss()
    print("iteration %d" % i_episode, "loss =", loss,file=log)
    print("iteration %d" % i_episode, "loss =", loss)
    log.flush()
    if i_episode % SAVER_ITER == 0:
        dqn.save("./model",i_episode)
log.close()

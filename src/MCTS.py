from multiprocessing import Pool
from copy import deepcopy
from othello_utils import *
from numba import jit
import numpy as np
import multiprocessing
import math

B = -1

class MCTS(object):
    """
    MC Tree Search used in AlphaZero Algorithm
    """
    def __init__(self, evalNet, MCTSTimes, Cpuct=1):
        '''
        input:
            Cpuct: hyperparameter for the UCT, actually, I don't know what it should be
        '''
        super(MCTS, self).__init__()
        self.evalNet = evalNet
        self.MCTSTimes = MCTSTimes
        self.Cpuct = Cpuct
        # 4 parameters mentioned in MCTS algorithm
        # (s,a) is the key, a is the index
        self.Nsa = {}
        self.Wsa = {}
        self.Qsa = {}
        # this one is different, s is key, the value is a numpy.ndarray
        self.Psa = {} # records the probabilities of moves of the states, and used to compute the self.Qsa
        # record the encountered States
        self.Vs = {} # valid moves of a state
        self.Ns = {} # the times encountered the states, unnecessary, but convenient
        
    def virtualLoss(self, game):
        LOSS = 10
        actions = list(map(convert_mv_tuple_to_ind,game.possible_moves(B)))
        selectAction = np.random.choice(actions)
        print(selectAction)
        sDump = game.dump()
        for action in actions:
            self.Nsa[(sDump,action)] = 1
            if action == selectAction:
                self.Qsa[(sDump,action)] = 0
                self.Wsa[(sDump,action)] = 0
            else:
                self.Qsa[(sDump,action)] = -LOSS
                self.Wsa[(sDump,action)] = -LOSS

    #####################################
    ##### unified to in Black's view ####
    def search(self, game, Tau):
        '''
        expand the tree to a leaf node for self.MCTSTimes times
        input:
            game: Othello(unified to in Black's view)
            Tau: the temperature using in Annealing
        output:
            Pi: probabilities of the actions in the currenct state of the game
        '''
        # pay attention that the game should not be modified    
        for _ in range(self.MCTSTimes):
            newGame = deepcopy(game)
            self.expand(newGame)

        sDump = game.dump()
        counts = [self.Nsa[(sDump, a)] if (sDump,a) in self.Nsa else 0 for a in range(game.getActionSize())]
        if Tau == 0:
            bestAction = np.argmax(counts)
            probs = np.zeros((len(counts)))
            probs[bestAction] = 1
            return probs

        counts = [x**(1./Tau) for x in counts]
        if np.sum(counts) > 0:
            probs = [x*1.0/(np.sum(counts)) for x in counts]
        else:
            probs = np.zeros((len(counts)))
        return probs



    #####################################    #####################################
    ##### unified to in Black's view ####
    #####################################
    # @jit
    def expand(self, game):
        '''
        expand the tree from the current state to a leaf node, using UCT(UCB) and updating all parameters
        leaf node: a node that is met firstly

        input:
            game: Othello(unified to in Black's view)
        '''
        sDump = game.dump()

        if game.game_over():
            return -game.get_winner()

        if sDump not in self.Psa: # leaf state, use the evalNet to predict the value and return
            ######################################
            # NOTE THE SHAPE OF THE OUTPUT OF THE self.evalNet!!!!!!!!!
            ######################################
            self.Psa[sDump], v = self.evalNet(game.board)
            self.Psa[sDump] = self.Psa[sDump].detach().numpy()[0]
            # -1 is the best, while it is designed to choose the maximum of Q + U
            v = v.detach().numpy()[0][0]
            validMoves = game.possible_moves(B)
            validMask = np.zeros((game.getActionSize()))
            if len(validMoves) > 0:
                for mv in validMoves:
                    validMask[convert_mv_tuple_to_ind(mv)] = 1
                self.Psa[sDump] = self.Psa[sDump] * validMask
                sumPs = np.sum(self.Psa[sDump])

                if sumPs > 0:
                    self.Psa[sDump] /= sumPs
                else:
                    print("Pay attention!!!! No valid moves hove positive probability!!!! Check your code!!!!")
                    self.Psa[sDump] = self.Psa[sDump] + validMask
                    self.Psa[sDump] /= np.sum(self.Psa[sDump])

            self.Vs[sDump] = validMask
            self.Ns[sDump] = 0
            return -v

        validMask = self.Vs[sDump]
        curBest = -float('inf')
        bestAction = -1

        # pick the action with the highest upper confidence bound
        for action in range(game.getActionSize()):
            if validMask[action]:
                if (sDump,action) in self.Qsa:
                    uct = self.Qsa[(sDump,action)] + self.Cpuct*self.Psa[sDump][action]*math.sqrt(self.Ns[sDump])/(1+self.Nsa[(sDump,action)])
                else:
                    # not sure!!!!!
                    uct = self.Cpuct*self.Psa[sDump][action]*math.sqrt(self.Ns[sDump])
                if uct > curBest:
                    curBest = uct
                    bestAction = action

        game.play_move(*convert_mv_ind_to_tuple(bestAction), B)
        #########################################
        # flip the board to the view of Black!!!
        #########################################
        game.board = game.board*-1
        # it doesn't matter whether the game.board is changed after expand this node
        v = self.expand(game)

        if (sDump, bestAction) in self.Qsa:
            self.Nsa[(sDump,bestAction)] += 1
            self.Wsa[(sDump,bestAction)] -= v
            self.Qsa[(sDump,bestAction)] = self.Wsa[(sDump,bestAction)]*1.0/self.Nsa[(sDump,bestAction)]
        else:
            self.Nsa[(sDump,bestAction)] = 1
            self.Wsa[(sDump,bestAction)] = -v
            self.Qsa[(sDump,bestAction)] = self.Wsa[(sDump,bestAction)]*1.0/self.Nsa[(sDump,bestAction)]
        self.Ns[sDump] += 1

        return -v
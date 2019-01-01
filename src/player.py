from abc import ABC, abstractmethod
from othello import Othello
import random

class Player(ABC):
 
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def pick_move(self, game):
        pass
    
class RandomPlayer(Player):
    def __init__(self, side):
        self.side = side

    def pick_move(self, game):
        t = game.possible_moves(self.side)
        if len(t) == 0:
            return (-1, -1)
        r = random.randint(0, len(t)-1)
        return (t[r][0], t[r][1])
    
class HumanPlayer(Player): 
    def __init__(self, side):
        self.side = side

    def pick_move(self, game):
        print("You are playing", Othello.piece_map(self.side))
        t = game.possible_moves(self.side)
        if len(t) == 0:
            game.print_board(self.side)
            print("No moves availible. Turn skipped.")
            return (-1, -1)
        move = (-1, -1)
        while move not in t:
            try:
                row = int(input("Please input row: "))
                col = int(input("Please input col: "))
                move = (row, col)
                if move not in t:
                    game.print_board(self.side)
                    print("Please input a valid move")
            except Exception:
                game.print_board(self.side)
                print("Please input a valid move")
        print()
        return move

class SearchPlayer(Player):
    """docstring for SearchPlayer"""
    def __init__(self, heuristic, side, depth=5):
        super(SearchPlayer, self).__init__()
        self.heuristic = heuristic
        self.side = side
        self.depth = depth
        self.history = {}
        self.opphistory = {}


    def getMovable(self, game, side):
        return game.possible_moves(side)

    def cutOffTest(self, game, curDepth):
        return self.depth >= curDepth
    
    def pick_move(self, game):
        mv = self.AlphaBetaSearch(game)
        print("SearchPlayer put in (%d,%d)" % mv)
        return mv

    def AlphaBetaSearch(self, game):
        game = game.copy()
        movable = self.getMovable(game, self.side)
        if len(movable) <= 0:
            return (-1,-1)
        alpha = -float('inf')
        beta = float('inf')
        maxVal = -float('inf')
        maxMove = None
        for mv in movable:
            val = self.AlphaMax(game, mv[0], mv[1], 1, alpha, beta)
            if val > maxVal:
                maxVal = val
                maxMove = mv
        return maxMove

    def AlphaMax(self, game, x,y, depth, alpha, beta):
        if self.cutOffTest(game, self.depth):
            return self.heuristic(game, self.side)
        game = game.copy()
        game.play_move(x,y,self.side)
        dumpstr = game.dump()
        if dumpstr in self.history:
            return self.history[dumpstr]
        ''' 对方行动 '''
        if(game.game_over()):
            return 10000 if game.get_winner() == self.side else -10000
        movable = self.getMovable(game, -self.side)
        if (len(movable) <= 0):
            movable2 = self.getMovable(game, self.side)
            for mv in movable2:
                v = self.AlphaMax(game, mv[0], mv[1], depth+1, alpha, beta)
                if v >= beta: return v
                if v > alpha: alpha = v
        for mv in movable:
            v = self.BetaMin(game, mv[0], mv[1], depth+1, alpha, beta)
            if v >= beta: return v
            if v > alpha: alpha = v
        self.history[dumpstr] = v
        return v


    def BetaMin(self, game, x,y, depth, alpha, beta):
        if self.cutOffTest(game, self.depth):
            return self.heuristic(game, self.side)
        game = game.copy()
        game.play_move(x,y,1-self.side)
        dumpstr = game.dump()
        if dumpstr in self.opphistory:
            return self.opphistory[dumpstr]
        ''' 对方行动 '''
        if(game.game_over()):
            return 10000 if game.get_winner() == self.side else -10000
        movable = self.getMovable(game, self.side)
        if (len(movable) <= 0):
            movable2 = self.getMovable(game, self.side)
            for mv in movable2:
                v = self.BetaMin(game, mv[0], mv[1], depth+1, alpha, beta)
                if v <= alpha: return v
                if v < beta: alpha = v
        for mv in movable:
            v = self.AlphaMax(game, mv[0], mv[1], depth+1, alpha, beta)
            if v <= alpha: return v
            if v < beta: alpha = v
        self.opphistory[dumpstr] = v
        return v


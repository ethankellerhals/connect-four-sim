import sys
import numpy as np
from collections import deque
from queue import PriorityQueue
import copy as cp
from tqdm import tqdm

class State:
    def __init__(self, moves):
        self.to_move = 'R' 
        self.utility = 0
        self.board = {}
        self.moves = moves

class ConnectFour:
    def __init__(self, nrow, ncol, nwin):
        self.nrow = nrow
        self.ncol = ncol
        self.nwin = nwin
        moves = [(row,col) for row in range(1, nrow + 1) for col in range(1, ncol + 1)]
        self.state = State(moves)
        self.expanded_states = 0

    def actions(self, state):
        return [(row, col) for (row, col) in state.moves 
                if row == self.nrow or (row + 1 , col ) in state.board]
    def result(self, move, state):

        if move not in state.moves:
            return state

        new_state = cp.deepcopy(state)
        
        new_state.utility = self.compute_utility(move, state)
        
        new_state.board[move] = state.to_move
        new_state.moves.remove(move)
 
        new_state.to_move = ('B' if state.to_move == 'R' else 'R')
        
        return new_state 

    def compute_utility(self, move, state):
    
        row, col = move
        player = state.to_move

        board = cp.deepcopy(state.board)
        board[move] = player

        #check for row win
        row_count = 0

        for c in range(1,self.ncol+1):
            if self.nwin == 3:
                if board.get((row,c))==player and board.get((row,c+1))==player and board.get((row,c+2))==player:
                    row_count = 3
            if self.nwin == 4:
                if board.get((row,c))==player and board.get((row,c+1))==player and board.get((row,c+2))==player and board.get((row,c+3))==player:
                    row_count = 4
            if self.nwin == 6:
                if board.get((row,c))==player and board.get((row,c+1))==player and board.get((row,c+2))==player and board.get((row,c+3))==player and board.get((row,c+4))==player and board.get((row,c+5))==player:
                    row_count = 6
                

        #check for col win
        col_count = 0

        for r in range(1, self.nrow+1):
            if self.nwin == 3:
                if board.get((r,col))==player and board.get((r+1,col))==player and board.get((r+2,col))==player:
                    col_count = 3
            if self.nwin == 4:
                if board.get((r,col))==player and board.get((r+1,col))==player and board.get((r+2,col))==player and board.get((r+3,col))==player:
                    col_count = 4
            if self.nwin == 6:
                if board.get((r,col))==player and board.get((r+1,col))==player and board.get((r+2,col))==player and board.get((r+3,col))==player and board.get((r+4,col))==player and board.get((r+5,col))==player:
                    col_count = 6
        
        
        #check for diag win
        #top left to bot right
        diag_top_to_bot_count = 0

        for c in range((self.ncol+1)-self.nwin):
            for r in range(self.nwin,(self.nrow+1)):
                if self.nwin == 3:
                    if board.get((r,c))==player and board.get((r-1,c+1))==player and board.get((r-2,c+2))==player:
                        diag_top_to_bot_count = 3
                if self.nwin == 4:
                    if board.get((r,c))==player and board.get((r-1,c+1))==player and board.get((r-2,c+2))==player and board.get((r-3,c+3))==player:
                        diag_top_to_bot_count = 4
                if self.nwin == 6:
                    if board.get((r,c))==player and board.get((r-1,c+1))==player and board.get((r-2,c+2))==player and board.get((r-3,c+3))==player and board.get((r-4,c+4))==player and board.get((r-5,c+5))==player:
                        diag_top_to_bot_count = 6

        #check bot right to top right
        diag_bot_to_top_count = 0

        for c in range((self.ncol+1) - self.nwin):
            for r in range((self.nrow+1) - self.nwin):
                if self.nwin == 3:
                    if board.get((r,c))==player and board.get((r+1,c+1))==player and board.get((r+2,c+2))==player:
                        diag_bot_to_top_count = 3
                if self.nwin == 4:
                    if board.get((r,c))==player and board.get((r+1,c+1))==player and board.get((r+2,c+2))==player and board.get((r+3,c+3))==player:
                        diag_bot_to_top_count = 4
                if self.nwin == 6:
                    if board.get((r,c))==player and board.get((r+1,c+1))==player and board.get((r+2,c+2))==player and board.get((r+3,c+3))==player and board.get((r+4,c+4))==player and board.get((r+5,c+5))==player:
                        diag_bot_to_top_count = 6

        #check for win

        if self.nwin in [row_count, col_count, diag_top_to_bot_count, diag_bot_to_top_count]:
            return 1 if player=='R' else -1
        else:
            return 0

    def game_over(self, state):
    
        return state.utility!=0 or len(state.moves)==0

    def utility(self, state, player):
        
        return state.utility if player == 'R' else -state.utility

    def display(self):

        board = self.state.board

        for row in range(1, self.nrow + 1):
            for col in range(1, self.ncol + 1):
                print(board.get((row, col), '.'), end=' ')
            print()

    def play_game(self, player1, player2):
    
        turn = 0
        variable_board_size = self.nrow * self.ncol
        
        while turn<=variable_board_size:
            for player in [player1, player2]:
                turn += 1
                move = player(self)
                self.state = self.result(move, self.state)

                if self.game_over(self.state):
                    #self.display()
                    return self.state.utility
    

def random_player(game):

    all_moves = game.state.moves
    random_col = all_moves[np.random.randint(low=0, high=len(game.actions(game.state)))]
   
    for row in reversed(range(game.nrow+1)):
        if (row, random_col[1]) in game.actions(game.state):
            return (row, random_col[1])


def rand_rand_test():

    niter = 1000
    wins, draws, losses = 0, 0, 0
    progress_bar = tqdm(total=niter, desc="running", unit="games")
    for i in range(niter):
        c4 = ConnectFour(nrow=6, ncol=7, nwin=4)
        out = c4.play_game(random_player, random_player)
        # c4.display()
        if out == 1:
            wins += 1
        elif out == -1:
            losses += 1
        else:
            draws += 1
        progress_bar.update(1)
    progress_bar.close()
    return (wins/niter) * 100, (draws/niter) * 100, (losses/niter) * 100


#possible: if desired results are hard, up nwin=6
def diff_size_test():
    niter = 1000
    wins, draws, losses = 0, 0, 0
    progress_bar = tqdm(total=niter, desc="running", unit="games")
    for i in range(niter):
        c4 = ConnectFour(nrow=10, ncol=10, nwin=6)
        out = c4.play_game(random_player, random_player)
        
        if out == 1:
            wins += 1
        elif out == -1:
            losses += 1
        else:
            draws += 1
        progress_bar.update(1)
    progress_bar.close()
    return (wins/niter) * 100, (draws/niter) * 100, (losses/niter) * 100


def alphabeta_player(game):
    alpha = -np.inf
    beta = +np.inf
    player = game.state.to_move
    
    def max_value(state, alpha, beta):
        if (game.game_over(state)):
            return game.utility(state, player)
        value = -np.inf
        for action in game.actions(state):
            game.expanded_states += 1
            value = max(value, min_value(game.result(action, state), alpha, beta))
            if value >= beta: #comment this if statement out and/or alpha=max line to see number of expanded nodes without pruning
                return value
            alpha = max(alpha, value)
        return value

    def min_value(state, alpha, beta):
        if (game.game_over(state)):
            return game.utility(state, player)
        value = +np.inf
        for action in game.actions(state):
            game.expanded_states += 1
            value = min(value, max_value(game.result(action, state), alpha, beta))
            if value <= alpha: #comment this if statement and/or beta=min line out to see number of expanded nodes without pruning
                return value
            beta = min(beta, value)
        return value
    #need best_action

    
    beta = +np.inf
    best_action = None
    best_score = -np.inf

    for action in game.actions(game.state):
        value = min_value(game.result(action, game.state), best_score, beta)
        if value > best_score:
            best_score = value
            best_action = action
    return best_action
    

def minimax(game):
    player = game.state.to_move
    
    def max_value(state):
        if game.game_over(state):
            return game.utility(state, player)
        value = -np.inf
        for action in game.actions(state):
            game.expanded_states += 1
            value = max(value, min_value(game.result(action, state)))
        return value
    def min_value(state):
        if game.game_over(state):
            return game.utility(state, player)
        value = +np.inf
        for action in game.actions(state):
            game.expanded_states += 1
            value = min(value, max_value(game.result(action, state)))
        return value

    return max(game.actions(game.state), key=lambda action: min_value(game.result(action, game.state)))



def ab_rand_test():
    niter = 10
    wins, draws, losses = 0, 0, 0
    progress_bar = tqdm(total=niter, desc="running", unit="games")
    for i in range(niter):
        c4 = ConnectFour(nrow=3, ncol=4, nwin=3)
        out = c4.play_game(alphabeta_player, random_player)
        #c4.display()
        if out == 1:
            wins += 1
        elif out == -1:
            losses += 1
        else:
            draws += 1 
        progress_bar.update(1)
    progress_bar.close()
    return (wins/niter) * 100, (draws/niter) * 100, (losses/niter) * 100

def rand_ab_test():
    niter = 10
    wins, draws, losses = 0, 0, 0
    progress_bar = tqdm(total=niter, desc="running", unit="games")
    for i in range(niter):
        c4 = ConnectFour(nrow=3, ncol=3, nwin=3)
        out = c4.play_game(random_player, alphabeta_player)
        if out == 1:
            losses += 1
        elif out == -1:
            wins += 1
        else:
            draws += 1
        progress_bar.update(1)
    progress_bar.close()
    return (wins/niter) * 100, (draws/niter) * 100, (losses/niter) * 100


# ncol = 4 leads to first player winning, ncol=3 leads to draw
def ab_ab_test():
    niter = 4
    wins, draws, losses = 0, 0, 0
    progress_bar = tqdm(total=niter, desc="running", unit="games")
    for i in range(niter):
        c4 = ConnectFour(nrow=3, ncol=3, nwin=3)
        out = c4.play_game(alphabeta_player, alphabeta_player)
        #c4.display()
        if out == 1:
            wins += 1
        elif out == -1:
            losses += 1
        else:
            draws += 1
        progress_bar.update(1)
    progress_bar.close()
    return (wins/niter) * 100, (draws/niter) * 100, (losses/niter) * 100

#4 in a row = 1000, 3 in a row = 100, 2 in a row = 10, 1 

def astar(game):
    player = 'R'
    
    def heuristic_cost(state):
        (x1, y1) = game.result((state), player)
        (x2, y2) = game.utility(state, player)
        return (x2,y2) - (x1,y1)

    frontier = PriorityQueue(game.state)
    explored = set()
    previous = {game.state:None}

    while not frontier.empty():
        (cost, s) = frontier.get()
        explored.add(s)
        if game.game_over(s, player):
            return game.utility(s, player)
        for s2 in game.actions(s):
            if s2 not in explored:

                updated_cost = cost + heuristic_cost(s2) - heuristic_cost(s)
                if s2 not in frontier.states:
                    previous[(s2)] = s
                    frontier.put((cost + heuristic_cost(s), updated_cost))
                elif frontier.states[(s2)] > updated_cost:
                    previous[(s2)] = s
                    frontier.replace(s2, updated_cost)

    return None



def astar_rand_test():
    niter = 10
    wins, draws, losses = 0, 0, 0
    progress_bar = tqdm(total=niter, desc="running", unit="games")
    for i in range(niter):
        c4 = ConnectFour(nrow=3,ncol=4,nwin=3)
        out = c4.play_game(astar, random_player)
        if out == 1:
            wins += 1
        elif out == -1:
            losses += 1
        else:
            draws += 1
        progress_bar.update(1)
    progress_bar.close()
    return (wins/niter) * 100, (draws/niter) * 100, (losses/niter) * 100


def astar_ab_test():
    niter = 10
    wins, draws, losses = 0, 0, 0
    progress_bar = tqdm(total=niter, desc="running", unit="games")
    for i in range(niter):
        c4 = ConnectFour(nrow=3,ncol=4,nwin=3)
        out = c4.play_game(astar, alphabeta_player)
        if out == 1:
            wins += 1
        elif out == -1:
            losses += 1
        else:
            draws += 1
        progress_bar.update(1)
    progress_bar.close()
    return (wins/niter) * 100, (draws/niter) * 100, (losses/niter) * 100


if __name__ == '__main__':
    print("-" * 50)

    print(" " * 12, "Intro to A.I. Project:")  
    print(" " * 12, "\033[1;3;40mConnect Four Simulator\033[m")

    print("-" * 50)

    print("random (first player) vs random (second player)")
    rand_win, rand_draw, rand_loss = rand_rand_test()
    print("first player winning: {}%".format(rand_win))
    print("first player drawing: {}%".format(rand_draw))
    print("first player losing: {}%".format(rand_loss))

    print("-" * 50)

    print("win percentage on a 10x10 board with a 6 sequence")
    diff_win, diff_draw, diff_loss = diff_size_test()
    print("first player winning: {}%".format(diff_win))
    print("first player drawing: {}%".format(diff_draw))
    print("first player losing: {}%".format(diff_loss))

    print("-" * 50)

    print("alphabeta (first player) vs random (second player)")
    ab_rand_win, ab_rand_draw, ab_rand_loss = ab_rand_test()
    print("alphabeta winning: {}%".format(ab_rand_win))
    print("alphabeta drawing: {}%".format(ab_rand_draw))
    print("alphabeta losing: {}%".format(ab_rand_loss))

    print("-" * 50)

    print("random (first player) vs alphabeta (second player)")
    rand_ab_win, rand_ab_draw, rand_ab_loss = rand_ab_test()
    print("alphabeta winning: {}%".format(rand_ab_win))
    print("alphabeta drawing: {}%".format(rand_ab_draw))
    print("alphabeta losing: {}%".format(rand_ab_loss))

    print("-" * 50)

    print("alphabeta (first player) vs alphabeta (second player)")
    ab_ab_win, ab_ab_draw, ab_ab_loss = ab_ab_test()
    print("alphabeta winning: {}%".format(ab_ab_win))
    print("alphabeta drawing: {}%".format(ab_ab_draw))
    print("alphabeta losing: {}%".format(ab_ab_loss))

    print("-" * 50)

    print("astar (first player) vs random (second player)")
    astar_rand_win, astar_rand_draw, astar_rand_loss = astar_rand_test()
    print("astar winning: {}%".format(astar_rand_win))
    print("astar drawing: {}%".format(astar_rand_draw))
    print("astar losing: {}%".format(astar_rand_loss))

    print("-" * 50)

    print("astar (first player) vs alphabeta (second player)")
    astar_ab_win, astar_ab_draw, astar_ab_loss = astar_ab_test()
    print("astar winning: {}%".format(astar_ab_win))
    print("astar drawing: {}%".format(astar_ab_draw))
    print("astar losing: {}%".format(astar_ab_loss))

    user_input = input("Would you like to see the number of states expanded by the minimax algorithm with and without pruning? (Y/N): ").upper()
    if user_input == 'Y':
        prune = ConnectFour(3,4,3)
        alphabeta_player(prune)
        print('states expanded with pruning:', prune.expanded_states)


        no_prune = ConnectFour(3,4,3)
        minimax(no_prune)
        print('states expanded without pruning:', no_prune.expanded_states)

        print('result:', (prune.expanded_states / no_prune.expanded_states))
        print('Percent savings using AB pruning: {}%'.format((prune.expanded_states / no_prune.expanded_states) * 100))
    
    sys.exit(0)
"""
Tic Tac Toe Player
"""

import math
import copy
from queue import Empty

X = "X"
XXX=X+X+X
O = "O"
OOO=O+O+O
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    
    Xcount, Ocount = 0, 0
    for line in board:
        for item in line:
            if item == X:
                Xcount += 1
            elif item == O:
                Ocount += 1
    if Xcount == Ocount:
        return X
    return O


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    
    possible_actions = set()
    for i, line in enumerate(board):
        for j, item in enumerate(line):
            if item == EMPTY:
                possible_actions.add((i, j))
    return possible_actions



def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    i, j = action

    if not (0 <= i < 3 and 0 <= j < 3):
        raise Exception("invalid action!")
    
    nb = copy.deepcopy(board)
    if nb[i][j] != EMPTY:
        raise Exception("invalid action!")
    nb[i][j] = player(nb)
    return nb


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """

    for line in board:
        try:
            word = line[0] + line[1] + line[2]
            if word == XXX:
                return X
            elif word == OOO:
                return O
        except TypeError:
            continue
        finally:
            pass
    
    try:
        word = board[0][0] + board[1][1] + board[2][2]
        if word == XXX:
            return X
        elif word == OOO:
            return O
    except TypeError:
        pass
    finally:
        pass
    try:
        word = board[2][0] + board[1][1] + board[0][2]
        if word == XXX:
            return X
        elif word == OOO:
            return O
    except TypeError:
        pass
    finally:
        pass
    try:
        word = board[0][0] + board[1][0] + board[2][0]
        if word == XXX:
            return X
        elif word == OOO:
            return O
    except TypeError:
        pass
    finally:
            pass
    try:
        word = board[0][1] + board[1][1] + board[2][1]
        if word == XXX:
            return X
        elif word == OOO:
            return O
    except TypeError:
        pass
    finally:
        pass
    try:
        word = board[0][2] + board[1][2] + board[2][2]
        if word == XXX:
            return X
        elif word == OOO:
            return O
    except TypeError:
        pass
    finally:
            pass

    return None

def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    if winner(board) is not None:
        return True
    
    for row in board:
        for cell in row:
            if cell == EMPTY:
                return False
    
    return True


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    
    return 1 if winner(board) == X else -1 if winner(board) == O else 0


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """

    if terminal(board):
        return None
    
    curr_player = player(board)

    if curr_player == X:
        mm = float('-inf')
        mv = None

        for action in actions(board):
            res = result(board, action)
            if min_(res) > mm:
                mm = min_(res)
                mv = action
        return mv
    
    if curr_player == O:
        mn = float('+inf')
        mv = None

        for action in actions(board):
            res = result(board, action)
            if max_(res) < mn:
                mn = max_(res)
                mv = action
        return mv

def max_(board):
    if terminal(board):
        return utility(board)

    mm = float('-inf')
    for action in actions(board):
        res = result(board, action)

        mn = min_(res)
        if mn > mm:
            mm = mn
    return mm

def min_(board):
    if terminal(board):
        return utility(board)

    mn = float('inf')
    for action in actions(board):
        res = result(board, action)

        mm = max_(res)
        if mm < mn:
            mn = mm
    return mn
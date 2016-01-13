#!/usr/bin/env python2

NORTH = 0
EAST = 1
SOUTH = 2
WEST = 3

class PickupGame(object):
    def __init__(self, board, agent):
        self.board = board
        self.agent = agent

    def next(self, action):
        ar, ac = self.agent
        rows, cols = board.shape
        if action == NORTH:
            ar = max(ar - 1, 0)
        elif action == EAST:
            ac = min(ac + 1, cols - 1)
        elif action == SOUTH:
            ar = min(ar + 1, rows - 1)
        elif action == WEST:
            ac = max(ac - 1, 0)

        new_board = self.board.copy()
        reward = new_board[ar,ac]
        new_board[ar,ac] = 0

        return reward, PickupGame(self.board, (ar, ac))

    @classmethod
    def sample(cls):
        ROWS=10
        COLS=10
        TREATS=10

        board = np.zeros((ROWS, COLS))
        planted_treats = 0
        while planted_treats < TREATS:
            r = np.random.randint(ROWS)
            c = np.random.randint(COLS)
            if (r, c) == (0, 0):
                continue
            if board[r,c] > 0:
                continue
            board[r,c] = 1
            planted_treats += 1

class Agent(object):
    def __init__(self):
        self.net = ApolloNet()

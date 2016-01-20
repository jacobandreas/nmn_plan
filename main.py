#!/usr/bin/env python2

from layers import Index

from apollocaffe import ApolloNet
from apollocaffe.layers import *
import curses
import numpy as np
import time

BATCH_SIZE = 50

NORTH = 0
EAST = 1
SOUTH = 2
WEST = 3

class TSPGame(object):
    def __init__(self, board, agent_pos):
        self.board = board
        self.agent_pos = agent_pos

        rows, cols = board.shape
        board_container = np.zeros((rows * 3, cols * 3))
        board_container[rows:2*rows,cols:2*cols] = board
        ar, ac = agent_pos
        agent_slice = board_container[rows+ar-rows:rows+ar+rows,cols+ac-cols:cols+ac+cols]
        self.features = agent_slice.ravel()

        self.n_actions = 4

    #def __eq__(self, other):
    #    return other.board == board and other.agent_pos == agent_pos

    #def __hash__(self):
    #    return hash(self.board) + 3 * hash(self.agent_pos)

    def step(self, action):
        ar, ac = self.agent_pos
        rows, cols = self.board.shape
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

        if np.sum(new_board) > 0:
            reward -= 0.1

        return reward, TSPGame(new_board, (ar, ac))

    @classmethod
    def sample(cls):
        ROWS=10
        COLS=10
        TREATS=3

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

        return TSPGame(board, (0,0))

class MetricAgent(object):
    def __init__(self):
        self.net = ApolloNet()
        self.experiences = []

    def choose(self, state):
        self.current_state = state

        dirs = [NORTH, EAST, SOUTH, WEST]
        interesting_dirs = \
                [d for d in dirs 
                 if state.step(d)[1].agent_pos != state.agent_pos]
        return np.random.choice(interesting_dirs)

    def forward(self, prefix, feats_from, feats_to, cost, n_actions):
        net = self.net

        l_data_from = prefix + "data_from"
        l_ip1_from = prefix + "ip1_from"
        l_relu1_from = prefix + "relu1_from"
        l_ip2_from = prefix + "ip2_from"

        l_data_to = prefix + "data_to"
        l_ip1_to = prefix + "ip1_to"
        l_relu1_to = prefix + "relu1_to"
        l_ip2_to = prefix + "ip2_to"

        l_inv = "inv"
        l_diff = "diff"
        l_sq = "sq"
        l_reduce = "reduce"
        l_target = "target"
        l_loss = "loss"

        p_ip1 = [prefix + "ip1_weight", prefix + "ip1_bias"]
        p_ip2 = [prefix + "ip2_weight", prefix + "ip2_bias"]
        p_reduce = ["reduce_weight", "reduce_bias"]

        net.f(NumpyData(l_data_from, feats_from))
        net.f(InnerProduct(l_ip1_from, 50, bottoms=[l_data_from], param_names=p_ip1))
        net.f(ReLU(l_relu1_from, bottoms=[l_ip1_from]))
        net.f(InnerProduct(l_ip2_from, 50, bottoms=[l_relu1_from], param_names=p_ip2))

        net.f(NumpyData(l_data_to, feats_to))
        net.f(InnerProduct(l_ip1_to, 50, bottoms=[l_data_to], param_names=p_ip1))
        net.f(ReLU(l_relu1_to, bottoms=[l_ip1_to]))
        net.f(InnerProduct(l_ip2_to, 50, bottoms=[l_relu1_to], param_names=p_ip2))

        net.f(Power(l_inv, scale=-1, bottoms=[l_ip2_from]))
        net.f(Eltwise(l_diff, "SUM", bottoms=[l_ip2_to, l_inv]))
        net.f(Power(l_sq, power=2, bottoms=[l_diff]))

        net.f(InnerProduct(
            l_reduce, 
            1, 
            bottoms=[l_sq],
            param_names=p_reduce,
            param_lr_mults=[0, 0], 
            weight_filler=Filler("constant", 1), 
            bias_filler=Filler("constant", 0)))

        net.f(NumpyData(l_target, np.ones((BATCH_SIZE, 1))))
        loss = net.f(EuclideanLoss(l_loss, bottoms=[l_reduce, l_target]))

        return loss

    def update(self, reward, new_state):
        self.experiences.append((self.current_state, new_state))
        if len(self.experiences) < BATCH_SIZE:
            return 0

        replay_choices = np.random.choice(len(self.experiences), BATCH_SIZE)
        replay_transitions = [self.experiences[i] for i in replay_choices]

        from_features = np.asarray([t[0].features for t in replay_transitions])
        to_features = np.asarray([t[1].features for t in replay_transitions])

        self.net.clear_forward()
        loss = self.forward("", from_features, to_features, 1, new_state.n_actions)
        self.net.backward()
        self.net.update(lr=0.01)

        return loss

    def update_target(self):
        pass


class Agent(object):
    def __init__(self):
        self.net = ApolloNet()

        self.transitions = []

    def forward(self, features, n_actions, prefix=""):
        net = self.net

        l_data = prefix + "data"
        l_ip1 = prefix + "ip1"
        l_relu1 = prefix + "relu1"
        l_ip2 = prefix + "ip2"
        l_relu2 = prefix + "relu2"
        l_ip3 = prefix + "ip3"

        p_ip1 = [prefix + "ip1_weight", prefix + "ip1_bias"]
        p_ip2 = [prefix + "ip2_weight", prefix + "ip2_bias"]

        net.f(NumpyData(l_data, features))
        net.f(InnerProduct(l_ip1, 50, bottoms=[l_data], param_names=p_ip1))
        net.f(ReLU(l_relu1, bottoms=[l_ip1]))
        net.f(InnerProduct(l_ip2, n_actions, bottoms=[l_relu1], param_names=p_ip2))
        #net.f(ReLU(relu2, bottoms=[ip2]))
        #net.f(InnerProduct(ip3, state.n_actions, bottoms=[relu2]))

        return l_ip2


    def choose(self, state):
        self.net.clear_forward()
        q_layer = self.forward(np.asarray([state.features]), state.n_actions, "now_")
        q_data = self.net.blobs[q_layer].data.ravel()

        if len(self.transitions) < BATCH_SIZE or np.random.random() < 0.1:
        #if True:
            action = np.random.choice(state.n_actions)
        else:
            action = np.argmax(q_data)

        self.current_state = state
        self.current_action = action

        return action

    def update(self, reward, new_state):
        current_transition = \
                (self.current_state, self.current_action, new_state, reward)
        self.transitions.append(current_transition)

        if len(self.transitions) < BATCH_SIZE:
            return

        replay_choices = np.random.choice(len(self.transitions), BATCH_SIZE)
        replay_transitions = [self.transitions[i] for i in replay_choices]

        replay_before_data = np.asarray([r[0].features for r in replay_transitions])
        replay_action_data = np.asarray([r[1] for r in replay_transitions])
        replay_after_data = np.asarray([r[2].features for r in replay_transitions])
        replay_reward_data = np.asarray([r[3] for r in replay_transitions])

        net = self.net
        
        self.net.clear_forward()
        l_q_now = self.forward(replay_before_data, new_state.n_actions, "now_")
        l_q_fut = self.forward(replay_after_data, new_state.n_actions, "fut_")

        pred_value = 0.9 * np.max(net.blobs[l_q_fut].data, axis=1) + replay_reward_data

        l_action_now = "action_now"
        l_index = "index"
        l_pred_value = "pred_value"
        l_loss = "loss"

        net.f(NumpyData(l_action_now, replay_action_data))
        net.f(Index(l_index, {}, bottoms=[l_q_now, l_action_now]))
        net.f(NumpyData(l_pred_value, pred_value))
        net.f(EuclideanLoss(l_loss, bottoms=[l_index, l_pred_value]))

        net.backward()
        net.update(lr=0.01)

    def update_target(self):
        net = self.net
        if "fut_ip1_weight" not in net.params.keys():
            return
        net.params["fut_ip1_weight"].data[...] = net.params["now_ip1_weight"].data
        net.params["fut_ip1_bias"].data[...] = net.params["now_ip1_bias"].data
        net.params["fut_ip2_weight"].data[...] = net.params["now_ip2_weight"].data
        net.params["fut_ip2_bias"].data[...] = net.params["now_ip2_bias"].data

def visualize(states):
    s0 = states[0]
    stdscr = curses.initscr()
    win = stdscr
    for state in states:
        rows, cols = state.board.shape
        for r in range(rows):
            for c in range(cols):
                if state.agent_pos == (r, c):
                    win.addch(r, c, "O")
                elif state.board[r,c] > 0:
                    win.addch(r, c, "*")
                else:
                    win.addch(r, c, " ")
        time.sleep(0.25)
        win.refresh()
    win.refresh()
    curses.endwin()

def main():
    agent = MetricAgent() # Agent()
    total_reward = 0
    total_loss = 0
    for i_epoch in range(100000):
        state = TSPGame.sample()
        history = [state]
        for i_step in range(50):
            action = agent.choose(state)
            reward, state = state.step(action)
            loss = agent.update(reward, state)
            agent.update_target()
            total_reward += reward
            total_loss += loss
            history.append(state)

        if i_epoch % 100 == 0:
            print "%3.4f\t%3.4f" % (total_reward, total_loss)
            total_reward = 0
            total_loss = 0
            #visualize(history)

if __name__ == "__main__":
    main()

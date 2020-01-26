from environment.grid_env import Environment
import numpy as np

# global variables
BOARD_ROWS = 3
BOARD_COLS = 4

# Actions
UP      = 'up'
DOWN    = 'down'
LEFT    = 'left'
RIGHT   = 'right'

class Agent:
    def __init__(self):
        self.states = []
        self.actions = [UP, DOWN, LEFT, RIGHT]
        self.State = Environment()
        self.isEnd = self.State.isEnd
        self.lr = 0.2
        self.exp_rate = 0.3
        self.decay_gamma = 0.9
        
        # initial Q values
        self.Q_values = {}
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                self.Q_values[(i, j)] = {}
                for a in self.actions:
                    self.Q_values[(i, j)][a] = 0  # Q value is a dict of dict

    def chooseAction(self):
        # choose action with most expected value
        mx_nxt_reward = 0
        action = ""

        if np.random.uniform(0, 1) <= self.exp_rate:
            action = np.random.choice(self.actions)
        else:
            # greedy action
            current_position = self.State.state
            for a in self.actions:
                nxt_reward = self.Q_values[current_position][a]
                if nxt_reward >= mx_nxt_reward:
                    action = a
                    mx_nxt_reward = nxt_reward
            # print("current pos: {}, greedy aciton: {}".format(self.State.state, action))
        return action

    def takeAction(self, action):
        position = self.State.nextPosition(action)
        # update State
        self.State.state=position
        return self.State

    def reset(self):
        self.states = []
        self.State = Environment()
        self.isEnd = self.State.isEnd

    def play(self, rounds=10):
        i = 0
        while i < rounds:
            # to the end of game back propagate reward
            if self.State.isEnd:
                # back propagate
                reward = self.State.giveReward()
                for a in self.actions:
                    self.Q_values[self.State.state][a] = reward
                print("Game End Reward", reward)
                for s in reversed(self.states):
                    current_q_value = self.Q_values[s[0]][s[1]]
                    reward = current_q_value + self.lr * (self.decay_gamma * reward - current_q_value)
                    self.Q_values[s[0]][s[1]] = round(reward, 3)
                self.reset()
                i += 1
            else:
                # self.State.showBoard()
                action = self.chooseAction()
                # append trace
                self.states.append([(self.State.state), action])
                print("current position {} action {}".format(self.State.state, action))
                # by taking the action, it reaches the next state
                self.State = self.takeAction(action)
                # mark is end
                self.State.isEndPosition()
                print("new state", self.State.state)
                print("---------------------")
                self.isEnd = self.State.isEnd
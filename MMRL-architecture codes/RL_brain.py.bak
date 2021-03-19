# -*- coding: UTF-8 -*-
"""



"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.95):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64) # actions: 0,1,2,3,4

    def choose_action(self, observation):
        self.check_state_exist(observation)
        if np.random.uniform() < self.epsilon:
            state_action = self.q_table.loc[observation, :]
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a] # 之前q_table 当中保存的 v(s,a)的值
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # 实际得到的 v(s,a)的值，这样的值与之前预期的值之间存在一个误差，然
        else:
            q_target = r  # next state is terminal
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )
            #print(self.q_table)

    def plot_cost(self):
        plt.plot(np.arange(20), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()
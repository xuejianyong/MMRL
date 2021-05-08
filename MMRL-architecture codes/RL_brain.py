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

        self.memory = []

        self.p_elements = ['s', 'a', 's_', 'c', 'p', 'r','d']  # 7: state, action, next_state, count, probability, reward, done
        self.p_table = pd.DataFrame(columns=self.p_elements,dtype=np.float64)  # 主要保存的经验当中，observation, action, observation_, 在 module_i 的经验当中 observation_ == s_的概率

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
        q_predict = self.q_table.loc[s, a] 
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()  
        else:
            q_target = r 
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)

        self.update_p_table(s, a, s_, r)

        memory_i = []
        memory_i.append(s)
        memory_i.append(a)
        memory_i.append(s_)
        memory_i.append(r)
        self.memory.append(memory_i)

    def update_p_table(self, s, a, s_, r):  # (s, a, s_, r, d)
        # p_table: ['s', 'a', 's_', 'c', 'p', 'r', 'd']
        if s_ == 'terminal':
            is_done = 1
        else:
            is_done = 0
        s_a_s_record = self.p_table.loc[(self.p_table['s'] == s) & (self.p_table['a'] == a) & (self.p_table['s_'] == s_)]
        if s_a_s_record.shape[0] == 0:
            new_s_a_s_record = pd.Series([s, a, s_, 1, 0.0, r, is_done], index=self.p_table.columns, name=s)
            self.p_table = self.p_table.append(new_s_a_s_record)
        else:
            self.p_table.loc[(self.p_table['s'] == s) & (self.p_table['a'] == a) & (self.p_table['s_'] == s_), ['c']] += 1

        # 完成数据更新之后，compute the probabilities
        s_a_records = self.p_table.loc[(self.p_table['s'] == s) & (self.p_table['a'] == a)]
        sum_counts = sum(s_a_records['c'])
        next_state_list = s_a_records['s_'].values
        for next_state_i in next_state_list:
            count_i = self.p_table.loc[(self.p_table['s'] == s) & (self.p_table['a'] == a) & (self.p_table['s_'] == next_state_i), ['c']].values[0]
            self.p_table.loc[(self.p_table['s'] == s) & (self.p_table['a'] == a) & (self.p_table['s_'] == next_state_i), ['p']] = float(count_i) / float(sum_counts)


    def check_state_exist(self, state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )


    def plot_cost(self):
        plt.plot(np.arange(20), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()
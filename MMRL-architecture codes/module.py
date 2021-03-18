# -*- coding: UTF-8 -*-

import pandas as pd
import numpy as np
import math

class MultipleModel:
    def __init__(self, actions, n_modules, learning_rate, reward_decay, e_greedy):  # 这里的actions其实是一个list
        self.actions = actions
        self.n_modules = n_modules
        self.learning_rate = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy

        self.alpha = 0.1  # 用来计算 ^lamda_i(t)
        self.beta = 0.5  # 在action selection 当中会用到stochastic version of the greedy action selection 论文第12页

        # 初始化 MM 架构
        self.modules = []  # 初始化设定的几个modules,将其放置于这个list当中
        for i in range(self.n_modules):
            module_i = Module(i, self.actions, self.learning_rate, self.gamma, self.n_modules)
            self.modules.append(module_i)

        # 存储信息
        # self.resSignal = pd.DataFrame(columns=self.n_modules,dtype=np.float64)

        # 根据论文的内容，在MM当中不应该有这个q_table，而是应该放在module当中
        # self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)  # actions: 0,1,2,3,4 用来做动作选择

    # 因为每个module的responsibility signal计算涉及到所有的module，因此这个计算过程应该放在MM模型当中  testici
    # 之后将这些responsibility signal 分配到各个模块当中
    # 计算 responsibility signal 是这项工作当中比较核心的内容
    # 计算需要考虑 responsibility predictor 的内容，即为prior knowledge or belief about the module selection
    # 这里需要考虑另一个问题是，先选择module然后在该module当中选择动作呢，还是直接选择动作，
    # A：对于prediction error， 通过softmax function 得到responsibility signal
    def responsibility_signal(self):
        sum_resp_signal_up = 0.0
        resp_signal_uplist = []  # 存放每个 module 的分子值
        """
        # 普通计算responsibility signal 的办法
        for module_i in self.modules:
            # 求出每个单一模块的 responsibility signal 的值, 由于在刚开始的时候，每一个module的 prior_ probability 都是 1/n，所以每个module的分子值为
            resp_signal_up_i = module_i.prior_probability*module_i.getPyt_i()
            resp_signal_uplist.append(resp_signal_up_i)
            sum_resp_signal_up += resp_signal_up_i
        all_responisibility_signal = np.array(resp_signal_uplist)/sum_resp_signal_up  # 得到了每个模块的 responsibility signal 值
        """
        # 第一种 responsibility prediction 来计算 responsibility signal的方法
        for module_i in self.modules:
            # 融入了 temporal continuity 来计算 responsibility_signal的办法：
            # resp_signal_i_proportional = module_i.respPredictor() * module_i.getPyt_i()  # 实现responsibility predictor 来计算 responsibility signal 的值
            resp_signal_up_i = module_i.responsibility_predictor() * module_i.getPyt_i()
            resp_signal_uplist.append(resp_signal_up_i)
            sum_resp_signal_up += resp_signal_up_i
        all_responisibility_signal = np.array(resp_signal_uplist)/sum_resp_signal_up  # 得到了每个模块的 responsibility signal 值

        index = 0
        for module_i in self.modules:
            module_i.responsibility_signal = all_responisibility_signal[index]
            index += 1
        """
        # 另外一种计算 responsibility signal 的方式
        for module_i in self.modules:
            resp_signal_up_list = []
            resp_signal_up = 1.0
            sum_resp_signal_up = 0.0
            k = t #  step
            for pyt_k in module_i.pyt_i_list:
                resp_signal_up = resp_signal_up * pow(pyt_k, pow(self.alpha, k))
            resp_signal_up_list.append(resp_signal_up)
            sum_resp_signal_up +=resp_signal_up
        self.all_resp_signal_propotional = np.array(resp_signal_up_list) / sum_resp_signal_up
        """

    def getOutput_precition_model(self):
        # the prediction of the next state
        return sum(self.all_responisibility_signal*self.pyt_list)

    # 这里的动作该如何进行选择
    def choose_action(self, observation):  # testici
        # action selection
        # 在论文当中，采用一种softmax的方式来选择动作，和这里方式不同 testici
        """
        # softmax function，用在动作的选择方面
        z = [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0]
        z_exp = [math.exp(i) for i in z]
        print(z_exp)  # Result: [2.72, 7.39, 20.09, 54.6, 2.72, 7.39, 20.09]
        sum_z_exp = sum(z_exp)
        print(sum_z_exp)  # Result: 114.98
        # Result: [0.024, 0.064, 0.175, 0.475, 0.024, 0.064, 0.175]
        softmax = [round(i / sum_z_exp, 3) for i in z_exp]
        print(softmax)
        """
        # q_predict =
        # 选择一个动作
        v_modules_actions = pd.Series([0]*len(self.actions), index=self.actions, name='modules_actions')
        for module_i in self.modules:
            module_i.check_state_exist(observation)
            v_state_actions_i = module_i.q_table.loc[observation, :]
            weighted_v_state_actions_i = v_state_actions_i * module_i.responsibility_signal
            v_modules_actions += weighted_v_state_actions_i


            if np.random.uniform() < self.epsilon:
                state_action = module_i.q_table.loc[observation, :]
                action = np.random.choice(state_action[state_action == np.max(state_action)].index)
            else:
                action = np.random.choice(self.actions)
            action_output_i = module_i.q_table.loc[observation, action]  # 之前q_table 当中保存的 v(s,a)的值
            weighted_action_output_i = action_output_i*module_i.responsibility_signal





        # 这里epsilon 的greedy action selection，在论文中作者采用了一种 stochastic version of the greey action selection testici
        if np.random.uniform() < self.epsilon:
            state_action = self.q_table.loc[observation, :]
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            action = np.random.choice(self.actions)
        return action



    def learn(self, s, a, r, s_):  # learning from the feedback of the interaction
        for module_i in self.modules:
            module_i.update(s, a, r, s_)


class Module:
    def __init__(self, serial,actions,learning_rate,reward_decay,n_modules):  # 这里可能还需要添加一些参数
        self.serial = serial
        self.actions = actions
        self.learning_rate = learning_rate
        self.gamma = reward_decay
        self.n_modules = n_modules
        self.alpha = 0.5  # 参考第5页的 controls the strength of the memory effect
        self.prior_probability = 1 / n_modules  # the p(i)
        self.responsibility_signal = 0.0

        # 考虑到在 module 选择当中的 prior knowledge 或者 belief，作者添加了一个 responsibility predictor
        # temporal continuity 计算  ^lamda_i(t)
        self.responsibility_signal_pre = 0.0  # 上一步的 prior probability，在计算responsibility signal的时候会用到

        # 由于历史的 pyt_i 是和具体的module相关联的，所以这个属性应该放在module当中
        self.pyt_i_list = []

        # 为了解决 module_i 中得到s_的可能性，这个要结合 module_i 以往的交互经验得到，所以需要保存一些这方面的信息，
        # 主要保存的经验当中，observation, action, observation_,
        # 在 module_i 的经验当中 observation_ == s_的概率
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)  # actions: 0,1,2,3,4 用来做动作选择，用于保存value
        self.t_table = pd.DataFrame(columns=['s', 'a', 's_'])  # 需要保存 observation，action， observation_的信息
        self.memory = []  # 保存 每次交互信息： observation, action, reward, observation_

    # 在 module_i 当中得到 p(y(t))
    def getPyt_i(self, s, a, s_):  # testici
        # 按照第一种解释的实现
        # the purpose of the prediction model is to predict the next state based on the observation of the state and the action
        # 在本module当中，在状态s 和动作a上，取得状态s_的概率
        # 在module当中 得到 s_ 的概率，需要把所有的t_table当中得到 s_个数/ 所有s_的个数
        # retrieve the memory, then have the p(y(t)|i)
        # p(y(t) | i) = fi(y(t), x(t), u(t))
        s_table = self.t_table['s_']
        return s_table[s_table == s_].size/s_table.size

        # 另外一种计算p(y(t))的方法
        #s_table = self.t_table
        #one = s_table[s_table['s'] == s]
        #two = one[one['a'] == a]
        #three = two[two['s_'] == s_]
        #return three.size/two.size

        # 第三种计算 p(y(t))的方法
        #s_table = self.t_table
        #one = s_table[s_table['s'] == s]
        #three = one[one['s_'] == s_]
        # return one.size/three.size

    def responsibility_predictor(self):  # 计算得到 ^lamda_i(t)
        # temporal continuity
        responsibility_prediction = pow(self.responsibility_signal_pre, self.alpha)
        return responsibility_prediction

    def update(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]  # 之前q_table 当中保存的 v(s,a)的值
        if s_ != 'terminal':
            # 实际得到的 v(s,a)的值，这样的值与之前预期的值之间存在一个误差，使用误差来调节q_table
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()
        else:
            q_target = r
        td_error = self.learning_rate * (q_target - q_predict)  # 得到 TD error
        # 得到 TD error weighted by the responsibility signal
        td_error_resp_signal = self.responsibility_signal * td_error
        self.q_table.loc[s, a] += self.learning_rate * (q_target - q_predict)  # 更新 q_table 当中 (s,a)的值

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )


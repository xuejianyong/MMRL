# -*- coding: UTF-8 -*-
"""
The implementation of Multiple model-based reinforcement learning

"""
from maze_env import Maze
from RL_brain import QLearningTable
import matplotlib.pyplot as plt
import numpy as np
#from module import MultipleModel

N_MODULE = 6
EPISODES = 20
MAX_STEP = 100

step_list = []
def update():
    print("--- Interaction starts ---")

    for episode in range(EPISODES):  # 总共进行100次的完整迭代
        step = 0
        #print("--- Episode: %d ---" % episode)
        observation = env.reset()  # initial observation
        while True:  # agent开始每一步的与环境的交互
            action = RL.choose_action(str(observation))  # RL choose action based on observation
            #action = MM.choose_action(str(observation))
            #print(env.action_space[action])
            # 这里需要给出agent预计到达的状态，以及可能获得的reward
            observation_, reward, done = env.step(action)  # RL take action and get next observation and reward
            env.render()  # fresh env 每一步之间停留的时间的设置也在这块，先更新显示，然后等待一下
            RL.learn(str(observation), action, reward, str(observation_))  # RL learn from this transition
            observation = observation_  # swap observation
            step += 1
            if done or (step >= MAX_STEP):  # break while when the hunter catches a prey, or within 100 steps
                break
        print('The episode: %d,  with %d steps.' % (episode, step))
        step_list.append(step)
    print('over')  # end of game
    env.destroy()




def plot_step():
    plt.plot(list(np.arange(EPISODES) + 1), step_list)
    plt.xlabel('Episodes')
    plt.ylabel('Step')
    plt.show()


if __name__ == "__main__":
    env = Maze()  # 创建一个交互的环境
    RL = QLearningTable(actions=list(range(env.n_actions)))  # 建立一个学习模型
    #MM = MultipleModel(actions=list(range(env.n_actions)), n_modules=list(range(N_MODULE)))
    env.after(500, update)  # 继续进行后续的交互
    env.mainloop()
    plot_step()

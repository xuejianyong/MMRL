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

    for episode in range(EPISODES):  
        step = 0
        #print("--- Episode: %d ---" % episode)
        observation = env.reset() 
        while True: 
            action = RL.choose_action(str(observation)) 
            
            observation_, reward, done = env.step(action)  
            env.render() 
            RL.learn(str(observation), action, reward, str(observation_))
            observation = observation_  
            step += 1
            if done or (step >= MAX_STEP): 
                break
        print('The episode: %d,  with %d steps.' % (episode, step))
        step_list.append(step)
    # print(RL.memory)
    memory_standby = []
    for memory_i in RL.memory:
        if not memory_i[0] in memory_standby:
            memory_standby.append(memory_i[0])
    print(memory_standby)
    for memory_standby_i in memory_standby:
        for memory_i in RL.memory:
            if memory_i[0] == memory_standby_i:
                print(memory_i)
        print()

    print('over')  # end of game
    env.destroy()




def plot_step():
    plt.plot(list(np.arange(EPISODES) + 1), step_list)
    plt.xlabel('Episodes')
    plt.ylabel('Step')
    plt.show()


if __name__ == "__main__":
    env = Maze()  
    RL = QLearningTable(actions=list(range(env.n_actions)))  
    #MM = MultipleModel(actions=list(range(env.n_actions)), n_modules=list(range(N_MODULE)))
    env.after(500, update)  
    env.mainloop()
    plot_step()

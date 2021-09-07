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

    # print(RL.p_table)
    states_list = []
    records = RL.p_table['s'].values
    for record_i in records:
        if record_i not in states_list:
            states_list.append(record_i)
    for state_i in states_list:
        print(RL.p_table.loc[RL.p_table['s'] == state_i])
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

import gym
import gym_snake
import numpy as np
import numpy
from gym import wrappers
import itertools
import io
import random
# Construct Environment
env = gym.make('snake-v0')
#bservation_n = env.reset() 
#print('observation' , observation_n)
#env.render()
#l = []

#dt = np.dtype(np.uint32).newbyteorder('>')
#v = np.frombuffer(io.TextIOWrapper.read(4), dtype=dt)[0]
#print('===v==' , v)
#Reading snake coords and food coords from the files.

s = open('snakeCoords.txt' , 'r')
rl = s.readlines()
#print("===rl===" , rl)
#x = str(rl[0][1])
#y = str(rl[0][3])

#xCoords = int(x)
#yCoords = int(y)
#print("x",xCoords,"y" , yCoords)

#print('test' ,  np.zeros(150))
#return 
'''a = np.zeros([env.observation_n, env.action_space])  
#np.zeros([env.observation_n, env.action_space]) 
#snakeCoords = str(s.readlines()[0])
s.close()

f = open('foodCoords.txt' , 'r')
print('foodCoords' , f.readlines()[-1])
#foodCoords = str(f.readlines()[0])
f.close()
'''

f = open('readRewards.txt' , 'a+')
#dt = numpy.dtype(numpy.uint32).newbyteorder('>')
#return np.frombuffer(bytestream.read(4), dtype=dt)[0]

#rs = observation_n.reshape((1, -1))
#print('pata nahi' , np.frombuffer(bytestream.read(observation_n), dtype=int)[0])
rewards=[]
epsilon = 1.0       
max_steps = 99            # Exploration rate
max_epsilon = 1.0             # Exploration probability at start
min_epsilon = 0.01 
decay_rate = 0.002 
total_episodes = 150
#total_rewards = 0
def q_learning(env, total_episodes, learning_rate=0.95, discount_factor=0.99):
    #alpha = 0.1
    gamma = 0.6
#    lstRewards = []
    #epsilon = 0.1

    state = env.reset()
    step = 0
    done = False
    total_rewards = 0

    #np.frombuffer(bytestream.read(observation_n), dtype=dt)[0]
        
    # decaying epsilon, i.e we will divide num of episodes passed
    epsilon = 1.0
    # create a numpy array filled with zeros 
    # rows = number of observations & cols = possible actions
    qtable = np.zeros((67500,4))

    for i_episode in range(total_episodes):
            # reset the env
            epochs, penalties, reward, = 0, 0, 0
            done = False
            state=env.reset()
            while not done:
                if random.uniform(0, 1) < epsilon:
                    action = env.action_space.sample() # Explore action space
                    #print(action)
                else:
                    action = np.argmax(qtable[state]) # Exploit learned values
                    
                #next_state, reward, done, info = env.step(action) 

                new_state, reward, done, info = env.step(action)
                f.write('Reward' +str(reward))
                print('Reward' +str(reward))
                
                reward+=1
        # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        # qtable[new_state,:] : all the actions we can take from new state
                qtable[state, action] = qtable[state, action] + learning_rate * (reward + gamma * np.max(qtable[new_state, :]) - qtable[state, action])
        
                total_rewards += reward
        
        # Our new state is state
                state = new_state
        
        # If done (if we're dead) : finish episode
                if done == True: 
                    break
        
    # Reduce epsilon (because we need less and less exploration)
    epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*i_episode) 
    rewards.append(total_rewards)
    
    return qtable
    
print ("Score over time: " +  str(sum(rewards)/total_episodes))
qtable = q_learning(env , total_episodes)
print(qtable)
print('Rewards , ' , rewards )
                

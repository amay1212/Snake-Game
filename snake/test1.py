import gym
import gym_snake
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import time
import itertools
from gym.envs.classic_control import rendering
from gym_snake.envs.snake.snake import Snake as snke
W = 10  # 20
H = 5  # 11
L0 = 4  # 9

sn = snke()
env = gym.make('snake-v0')

env.reset()
action_chain = [2,3,2,3,2,1,1,1,1,0,1,2,2,3,3,3,3,2,3,2,3,2,3,2,3,3]
#action_chain = [3]*(W - L0)+[2]+([2]*(H - 2)+[1]+[0]*(H - 2)+[1])*(int(W / 2) - 1)+[2]*(H - 2)+[1]+[0]*(H - 2)+[0]+[3]*(L0 - 1)
# for action in action_chain:
done = False
i = 0
while not done:
    env.render()
    #state = env.reset()
    if False:
        import msvcrt
        key = ord(msvcrt.getch())
        if key == 224: #Special keys (arrows, f keys, ins, del, etc.)
            key = ord(msvcrt.getch())
            if key == 80: #Down arrow
                action = 0
            elif key == 72: #Up arrow
                action = 2
            elif key == 75: #left arrow
                action = 1
            elif key == 77: #right arrow
                action = 3
        else:
            pass
    else:
        action = action_chain[i]
        i += 1
        if i == len(action_chain):
            i = 0
        # time.sleep(0.01)
    state, reward, done, info = env.step(action)

    print(sn.getHead())
    
    if done:
        state = env.reset()
        break

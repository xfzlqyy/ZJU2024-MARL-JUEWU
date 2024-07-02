import gym
import numpy as np
from collections import deque

import time


env = gym.make('CliffWalking-v0')
print('observation space:', env.observation_space)
print('action space:', env.action_space)
threshold = env.spec.reward_threshold
print('threshold: ', threshold)

env.seed(0)
np.random.seed(0)

class SARSA:
    def __init__(self, env, alpha=0.1, gamma=0.95, epsilon=0.1):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = np.zeros((env.observation_space.n, env.action_space.n))
        
    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.Q[state, :])
        
    def learn(self, state, action, reward, next_state):
        next_action = np.argmax(self.Q[next_state, :])
        predict = self.Q[state, action]
        target = reward + self.gamma * self.Q[next_state, next_action]
        self.Q[state, action] += self.alpha * (target - predict)

    def train(self,epoch_num,max_steps=1000):
        state = self.env.reset()
        done = False
        total_reward = 0
        step = 0
        while True: 
            action = self.choose_action(state)
            next_state, reward, done, _ = self.env.step(action)
            self.learn(state, action, reward, next_state)
            state = next_state
            total_reward += reward
            step += 1
            if step >= max_steps or done:
                return total_reward, step
    def test(self):
        state = self.env.reset()
        total_reward = 0
        step = 0
        while True:
            time.sleep(0.01)
            env.render(mode='human')
            action = np.argmax(self.Q[state, :])
            state, reward, done, _ = self.env.step(action)
            total_reward += reward
            step += 1
            if done:
                return total_reward, step         


sarsa = SARSA(env)
train_epochs = 5000  
for i in range(1,train_epochs+1):
    total_reward,step=sarsa.train(epoch_num=i,max_steps=100)
    print(f'Epoch {i}: total_reward={total_reward}, steps={step}',end='\n')

total_reward,step=sarsa.test()
print(f'Test: total_reward={total_reward}, steps={step}',end='\n')
    

    
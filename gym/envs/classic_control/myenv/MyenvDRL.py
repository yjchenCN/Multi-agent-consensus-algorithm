import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt
from typing import Optional, Union
import random
from gym.error import DependencyNotInstalled
import itertools
import math
from gym import spaces, logger
from gym.utils import seeding

class Agent:
    def __init__(self, initial_position, index):
        self.position = initial_position
        self.index = index
        self.neighbors = []
        self.last_broadcast_position = self.position  #存储了该智能体最近一次广播的位置
        self.trigger_points = []
        self.u_i = 0
        self.c_0 = 0.0001
        self.c_1 = 0.2499
        self.alpha = 0.4669

    def add_neighbor(self, neighbor):
        if neighbor not in self.neighbors:
            self.neighbors.append(neighbor)
            neighbor.neighbors.append(self)

    def is_neighbor(self, agent):
        return agent in self.neighbors
    
    def update_position(self, t, dt):
        e_i = self.last_broadcast_position - self.position
        trigger_condition = np.abs(e_i) - (self.c_0 + self.c_1 * np.exp(- self.alpha * t))
        
        # 如果事件触发函数大于等于0，则更新位置
        if trigger_condition >= 0 or t == 0:
            self.u_i = - sum(self.is_neighbor(neighbor) * (self.last_broadcast_position - neighbor.last_broadcast_position) for neighbor in self.neighbors)
            self.position += self.u_i * dt
            self.last_broadcast_position = self.position
            #记录触发的相关信息
            self.trigger_points.append((t, self.position))
        else:
            self.position += self.u_i * dt


class Consensus(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        super(Consensus, self).__init__()
        self.num_agents = num_agents
        self.num_iterations = num_iterations
        self.dt = dt
        self.agents = [self.Agent(np.random.rand() * 10, i) for i in range(self.num_agents)]
        self.define_neighbors()

        # 定义观测空间和动作空间
        # 观测空间为所有智能体的位置
        self.observation_space = spaces.Box(low=np.array([0.0, 0.0, 0.0], dtype=np.float32), high=np.array([0.01, 10, 8,], dtype=np.float32), shape=(3,1))
        # 动作空间为智能体的位置更新，这里简化为每个智能体的位移
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.num_agents,), dtype=np.float32)
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # 应用动作更新智能体位置
        for i, agent in enumerate(self.agents):
            agent.update_position_based_on_action(action[i], self.dt)

        # 计算新的观测和奖励
        new_obs = np.array([agent.position for agent in self.agents])
        reward = self.calculate_reward(new_obs)  # 定义奖励计算方法
        done = self.is_done()  # 定义终止条件

        return new_obs, reward, done, {}


    def reset(self):
        # 重置环境
        self.agents = [self.Agent(np.random.rand() * 10, i) for i in range(self.num_agents)]
        self.define_neighbors()
        return np.array([agent.position for agent in self.agents])

    def render(self, mode='human'):
        # 渲染环境
        pass



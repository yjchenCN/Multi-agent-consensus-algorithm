from turtle import pen
import numpy as np
from pyparsing import original_text_for
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

#v0
class Consensus(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }
    def __init__(self, num_agents=5, num_iterations=200, dt=0.1):
        self.c0_range = np.arange(0, 0.001, 0.0001)
        self.c1_range = np.arange(0, 2, 0.1)
        self.alpha_range = np.arange(0, 2, 0.1)
        # 动作空间定义为c0, c1, alpha组合的索引
        self.action_space = spaces.Discrete(len(self.c0_range) * len(self.c1_range) * len(self.alpha_range))
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(num_agents,), dtype=np.float32)
        self.action_matrix = np.array(np.meshgrid(self.c0_range, self.c1_range, self.alpha_range)).T.reshape(-1,3) #所有动作的可能性
        self.num_agents = num_agents
        self.num_iterations = num_iterations
        self.dt = dt
        self.current_iteration = 0
        self.agents = [self.Agent(np.random.uniform(-1, 1), i) for i in range(self.num_agents)]
        self.time_step = 0
        self.init_neighbors()
        self.epsilon = 0.005
        
    def init_neighbors(self):
        self.agents[0].add_neighbor(self.agents[1])
        self.agents[0].add_neighbor(self.agents[2])
        self.agents[1].add_neighbor(self.agents[2])
        self.agents[2].add_neighbor(self.agents[3])
        self.agents[3].add_neighbor(self.agents[4])
        
    def reset(self):
        #initial_positions = np.linspace(-1, 1, self.num_agents)
        initial_positions = [0.55, 0.4, -0.05, -0.1, -0.7]
        self.agents = [self.Agent(pos, i) for i, pos in enumerate(initial_positions)]  #固定智能体的位置
        #self.agents = [self.Agent(np.random.uniform(-1, 1), i) for i in range(self.num_agents)]  #随机智能体的位置
        self.init_neighbors()
        self.time_step = 0
        return self.get_state()
    
    def get_state(self):
        # 可以根据需要设计状态表示
        positions = np.array([agent.position for agent in self.agents])
        return positions
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
    
        # 从action_matrix中检索c_0, c_1, 和 alpha的值
        c_0, c_1, alpha = self.action_matrix[action]  # 使用单一索引从action_matrix中检索
        self.current_iteration = 0  # 重置当前迭代计数器
        time_to_reach_epsilon = None  # 达到epsilon条件的时间
        epsilon_violated = True  # 标记是否存在智能体位置差大于epsilon的情况

        while self.current_iteration < self.num_iterations:
            for agent in self.agents:
                agent.update_position(self.current_iteration, self.dt, c_0 , c_1, alpha)

            # 检查所有智能体与其邻居之间的位置差是否都小于epsilon
            all_within_epsilon = all(all(abs(agent.position - neighbor.position) < self.epsilon for neighbor in agent.neighbors) for agent in self.agents)

            if all_within_epsilon:
                if epsilon_violated:
                    # 如果之前的状态是大于epsilon的，现在变为小于epsilon，更新时间为当前迭代次数
                    time_to_reach_epsilon = self.current_iteration
                    epsilon_violated = False  # 更新状态为没有违反epsilon条件
            else:
                epsilon_violated = True  # 标记存在位置差大于epsilon的情况
                time_to_reach_epsilon = None  # 由于存在违反条件的情况，所以设置为None

            self.current_iteration += 1

        # 计算奖励
        if time_to_reach_epsilon is not None:
            # 计算0到time_to_reach_epsilon时间段内的触发次数
            trigger_counts = sum(len([point for point in agent.trigger_points if point[0] <= time_to_reach_epsilon]) for agent in self.agents)
            reward = - time_to_reach_epsilon - trigger_counts
        else:
            trigger_counts = -1000
            reward = trigger_counts
        
        done = True  # 因为我们运行了完整的迭代，所以这一步总是结束的
        return self.get_state(), reward, done, {"time_to_reach_epsilon": time_to_reach_epsilon, "trigger_counts": trigger_counts}



    class Agent:
        def __init__(self, initial_position, index):
            self.position = initial_position
            self.index = index
            self.neighbors = []
            self.last_broadcast_position = self.position
            self.trigger_points = []
            self.u_i = 0
             
        def add_neighbor(self, neighbor):
            if neighbor not in self.neighbors:
                self.neighbors.append(neighbor)
                neighbor.neighbors.append(self)

        def is_neighbor(self, agent):
            return agent in self.neighbors
        
        def update_position(self, t, dt, c0, c1, alpha):
            e_i = self.last_broadcast_position - self.position
            trigger_condition = np.abs(e_i) - (c0 + c1 * np.exp(-alpha * t))
            
            if trigger_condition >= 0 or t == 0:
                self.u_i = -sum((self.last_broadcast_position - neighbor.last_broadcast_position) for neighbor in self.neighbors if self.is_neighbor(neighbor))
                self.position += self.u_i * dt
                self.last_broadcast_position = self.position
                self.trigger_points.append((t, self.position))
            else:
                self.position += self.u_i * dt

#env = Consensus()

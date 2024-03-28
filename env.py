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

class Consensus(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }
    def __init__(self, num_agents=5, num_iterations=800, dt=0.1):
        self.c0_range = np.arange(0, 0.001, 0.00001)
        self.c1_range = np.arange(0, 10, 0.1)
        self.alpha_range = np.arange(0, 8, 0.1)
        # 动作空间定义为c0, c1, alpha组合的索引
        self.action_space = spaces.Discrete(len(self.c0_range) * len(self.c1_range) * len(self.alpha_range))
        
        # 观测空间定义为c0, c1, alpha的当前值
        self.observation_space = spaces.Box(low=np.array([0, 0, 0], dtype=np.float32), 
                                            high=np.array([0.001, 10, 8], dtype=np.float32), 
                                            shape=(3,))
        self.num_agents = num_agents
        self.num_iterations = num_iterations
        self.dt = dt
        self.current_iteration = 0
        self.agents = [self.Agent(np.random.uniform(-1, 1), i) for i in range(self.num_agents)]
        self.time_step = 0
        self.init_neighbors()

    def reset(self):
        # 随机选择初始状态
        c0 = np.random.choice(self.c0_range)
        c1 = np.random.choice(self.c1_range)
        alpha = np.random.choice(self.alpha_range)
        self.state = np.array([c0, c1, alpha])
        return self.state
        
    def init_neighbors(self):
        self.agents[0].add_neighbor(self.agents[1])
        self.agents[0].add_neighbor(self.agents[2])
        self.agents[1].add_neighbor(self.agents[2])
        self.agents[2].add_neighbor(self.agents[3])
        self.agents[3].add_neighbor(self.agents[4])
        
    def reset(self):
        self.agents = [self.Agent(np.random.uniform(-1, 1), i) for i in range(self.num_agents)]
        self.init_neighbors()
        self.time_step = 0
        return self.get_state()
    
    def get_state(self):
        # 可以根据需要设计状态表示
        positions = np.array([agent.position for agent in self.agents])
        return positions
    
    '''def step(self, action):
        c0, c1, alpha = action
        # 确保动作值在指定的范围内
        c0 = np.clip(c0, 0, 0.001)
        c1 = np.clip(c1, 0, 10)
        alpha = np.clip(alpha, 0, 8)
        
        # 更新智能体的c0, c1, alpha值
        for agent in self.agents:
            agent.c_0 = c0
            agent.c_1 = c1
            agent.alpha = alpha
            
        trigger_counts_before_t = 0
        min_distance_time = np.inf
        
        # 进行一次模拟
        for t in range(self.num_iterations):
            for agent in self.agents:
                agent.update_position(t, self.dt)
            
            # 检查是否所有智能体之间的位置差距都小于epsilon
            if t < min_distance_time and all(np.abs(agent.position - self.agents[0].position) < self.epsilon for agent in self.agents[1:]):
                min_distance_time = t
                
            self.time_step += 1
        
        # 计算触发次数和奖励
        trigger_counts = sum(len(agent.trigger_points) for agent in self.agents)
        reward = - (trigger_counts + min_distance_time)  # 奖励设计为负的触发次数加时间，您可以根据需要调整
        
        done = True if self.time_step >= self.num_iterations else False
        
        return self.get_state(), reward, done, {"trigger_counts": trigger_counts, "time_to_reach_epsilon": min_distance_time}'''
    
    def step(self, action):
        # 检查动作的有效性
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        
        for agent in self.agents:
            agent.update_position(self.current_iteration, self.dt)
            self.positions[self.current_iteration, agent.index] = agent.position
            
        self.current_iteration += 1
        # 计算c0, c1, alpha对应的索引，这里简化为直接选择
        c0, c1, alpha = self.sample_action_by_index(action)
        
        # 更新状态
        self.state = np.array([c0, c1, alpha])
        
        # 计算奖励和是否完成
        done = self.is_done()  # 这里需要根据您的逻辑定义is_done方法
        reward = self.calculate_reward()  # 根据新的状态计算奖励
        
        return self.state, reward, done, {}

    def sample_action_by_index(self, index):
        # 根据给定的索引从动作空间中选择动作，这里简化处理
        # 实际上，您需要根据index计算c0, c1, alpha的具体值
        c0 = np.random.choice(self.c0_range)
        c1 = np.random.choice(self.c1_range)
        alpha = np.random.choice(self.alpha_range)
        return c0, c1, alpha

    def is_done(self):
        # 检查是否完成了预设的模拟轮数
        return self.current_iteration >= self.num_iterations

    def calculate_reward(self):
        # 根据当前状态计算奖励
        # 示例：使用标准差来计算奖励，您可以根据需要调整
        standard_deviation = np.std(self.state)
        return -standard_deviation  # 假设奖励是负的标准差


    class Agent:
        def __init__(self, initial_position, index):
            self.position = initial_position
            self.index = index
            self.neighbors = []
            self.last_broadcast_position = self.position
            self.trigger_points = []
            self.u_i = 0
            self.c_0 = 0.0001
            self.c_1 = 6
            self.alpha = 0.12
             
        def add_neighbor(self, neighbor):
            if neighbor not in self.neighbors:
                self.neighbors.append(neighbor)
                neighbor.neighbors.append(self)

        def is_neighbor(self, agent):
            return agent in self.neighbors
        
        def update_position(self, t, dt):
            e_i = self.last_broadcast_position - self.position
            trigger_condition = np.abs(e_i) - (self.c_0 + self.c_1 * np.exp(- self.alpha * t))
            
            if trigger_condition >= 0 or t == 0:
                self.u_i = - sum(self.is_neighbor(neighbor) * (self.last_broadcast_position - neighbor.last_broadcast_position) for neighbor in self.neighbors)
                self.position += self.u_i * dt
                self.last_broadcast_position = self.position
                self.trigger_points.append((t, self.position))
            else:
                self.position += self.u_i * dt

# 初始化环境
env = Consensus()

# 重置环境，开始新的一轮模拟
state = env.reset()

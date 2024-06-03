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


class Consensus1(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }
    def __init__(self, num_agents = 5, num_iterations = 200, dt = 0.1, L = np.array([[ 2, -1, -1,  0,  0],
                                                                                    [-1,  2, -1,  0,  0],
                                                                                    [-1, -1,  3, -1,  0],
                                                                                    [ 0,  0, -1,  2, -1],
                                                                                    [ 0,  0,  0, -1,  1]])):
        self.c0_range = np.arange(0, 0.001, 0.00001)
        self.c1_range = np.arange(0, 10, 0.1)
        self.alpha_range = np.arange(0, 3, 0.1)
        # 动作空间定义为c0, c1, alpha组合的索引
        self.num_agents = num_agents
        self.action_space = spaces.Discrete(len(self.c0_range) * len(self.c1_range) * len(self.alpha_range))
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(num_agents,), dtype=np.float32)
        self.action_matrix = np.array(np.meshgrid(self.c0_range, self.c1_range, self.alpha_range)).T.reshape(-1,3) #所有动作的可能性
        self.num_iterations = num_iterations
        self.dt = dt
        self.current_iteration = 0
        self.initial_positions = [0.5, 0.51, 0.52, 0.53, 0.54]
        self.agents = [self.Agent(pos, i) for i, pos in enumerate(self.initial_positions)] 
        #self.agents = [self.Agent(np.random.uniform(-1, 1), i) for i in range(self.num_agents)]
        self.time_step = 0
        self.L = L
        self.init_neighbors()
        self.epsilon = 0.001
      

    def init_neighbors(self):
        # 根据L矩阵初始化邻居关系
        for i in range(self.num_agents):
            for j in range(i+1, self.num_agents):
                if self.L[i, j] < 0:  # 如果L[i, j]小于0，说明i和j是邻居
                    self.agents[i].add_neighbor(self.agents[j])
        
    def reset(self):
        initial_positions = [0.5, 0.51, 0.52, 0.53, 0.54]
        self.agents = [self.Agent(pos, i) for i, pos in enumerate(initial_positions)] 
        #initial_positions = np.linspace(-1, 1, self.num_agents)
        #self.agents = [self.Agent(pos, i) for i, pos in enumerate(initial_positions)]  #固定智能体的位置
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
                if not epsilon_violated:  # 如果之前没有位置差大于epsilon的情况
                    time_to_reach_epsilon = self.current_iteration
                else:  # 如果之前位置差大于epsilon，现在又小于epsilon，更新时间
                    epsilon_violated = False
            else:
                epsilon_violated = True  # 标记存在位置差大于epsilon的情况

            self.current_iteration += 1

        # 循环结束后，如果time_to_reach_epsilon不为None，则计算在此之前的触发次数
        if all_within_epsilon:
            trigger_counts = sum(len(agent.trigger_points) for agent in self.agents)
        else:
            trigger_counts = 1000  #设为很大数

        # 计算奖励，可以根据需要进一步调整
        reward = (self.num_iterations - (time_to_reach_epsilon if time_to_reach_epsilon is not None else self.num_iterations)) - trigger_counts
        
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

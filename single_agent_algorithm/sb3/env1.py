from matplotlib.pylab import get_state
import numpy as np
import gymnasium as gym  # 使用 gymnasium 代替 gym
from gymnasium import spaces, logger
import matplotlib.pyplot as plt
from typing import Optional, Union
import random
from gymnasium.error import DependencyNotInstalled
import itertools
import math
from gymnasium.utils import seeding

#v4
class Consensus_D_F(gym.Env):
    metadata = {
        'render_modes': ['human', 'rgb_array'],  # 将 render.modes 修改为 render_modes
        'video.frames_per_second': 50
    }
    
    def __init__(self, num_agents=5, num_iterations=200, dt=0.1):
        self.num_agents = num_agents
        self.action_space = spaces.Discrete(2**num_agents)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(num_agents,), dtype=np.float32)
        self.action_matrix = self.calculate_action_matrix(num_agents)  # 所有动作的可能性
        self.num_iterations = num_iterations
        self.dt = dt
        self.current_iteration = 0
        self.initial_positions = [0.55, 0.4, -0.05, -0.1, -0.7]
        self.agents = [self.Agent(pos, i) for i, pos in enumerate(self.initial_positions)]
        self.init_neighbors()
        self.epsilon = 0.005
        self.time_to_reach_epsilon = None  # 达到epsilon条件的时间
        self.epsilon_violated = True  # 标记是否存在智能体位置差大于epsilon的情况
        self.all_within_epsilon = False
        self.total_trigger_count = 0

        self.total = 0  # 新增的属性
        self.time = None  # 新增的属性

    def calculate_action_matrix(self, num_agents):
        # 生成所有可能动作的矩阵，每行是一个可能的动作组合
        num_actions = 2**num_agents
        action_matrix = np.zeros((num_actions, num_agents), dtype=int)
        for i in range(num_actions):
            binary_string = format(i, f'0{num_agents}b')
            action_matrix[i] = np.array([int(bit) for bit in binary_string])
        return action_matrix

    def init_neighbors(self):
        self.agents[0].add_neighbor(self.agents[1])
        self.agents[0].add_neighbor(self.agents[2])
        self.agents[1].add_neighbor(self.agents[2])
        self.agents[2].add_neighbor(self.agents[3])
        self.agents[3].add_neighbor(self.agents[4])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agents = [self.Agent(pos, i) for i, pos in enumerate(self.initial_positions)]  # 固定智能体的位置
        self.init_neighbors()
        self.current_iteration = 0
        self.total_trigger_count = 0
        self.u_i = 0
        return self.get_state(), {}  # 返回观测值和空信息字典

    def get_state(self):
        # 可以根据需要设计状态表示
        positions = np.array([agent.position for agent in self.agents], dtype=np.float32)  # 确保返回的状态为 float32
        return positions

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def compute_average_position_difference(self):
        total_difference = 0
        count = 0
        for i, agent_i in enumerate(self.agents):
            for j, agent_j in enumerate(self.agents):
                if i < j:  # 避免重复计算和自己与自己的比较
                    total_difference += abs(agent_i.position - agent_j.position)
                    count += 1
        if count > 0:
            return total_difference / count
        else:
            return 0

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        triggers = self.action_matrix[action]
        trigger_count = np.sum(triggers)
        self.total_trigger_count += trigger_count
        

        for i, agent in enumerate(self.agents):
            agent.update_position(self.current_iteration, self.dt, triggers[i])

        self.all_within_epsilon = all(all(abs(agent.position - neighbor.position) < self.epsilon for neighbor in agent.neighbors) for agent in self.agents)

        if self.all_within_epsilon:
            if self.epsilon_violated:
                self.time_to_reach_epsilon = self.current_iteration
                self.epsilon_violated = False
        else:
            self.epsilon_violated = True
            self.time_to_reach_epsilon = None

        self.current_iteration += 1
        done = self.current_iteration >= self.num_iterations

        if not done:
            average_position_difference = self.compute_average_position_difference()
            if self.time_to_reach_epsilon is not None:
                reward = 5 + 3 * (5 - trigger_count)
            else:
                reward = - 5 * average_position_difference
        else:
            if self.time_to_reach_epsilon is not None:
                reward = 2100 - 2 * self.total_trigger_count
            else:
                reward = 0
            self.total = self.total_trigger_count
            self.time = self.time_to_reach_epsilon
            
        return self.get_state(), reward, done, False, {}  # 返回 (状态, 奖励, done, truncated, 信息字典)

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

        def update_position(self, t, dt, trigger):
            if trigger == 1:
                self.u_i = -sum((self.last_broadcast_position - neighbor.last_broadcast_position) for neighbor in self.neighbors if self.is_neighbor(neighbor))
                self.position += self.u_i * dt
                self.last_broadcast_position = self.position
                self.trigger_points.append((t, self.position))
            else:
                self.position += self.u_i * dt
from matplotlib.pylab import get_state
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

class Consensus_F(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }
    def __init__(self, num_agents = 5, num_iterations = 200, dt = 0.1):
        self.num_agents = num_agents
        self.action_space = spaces.Discrete(2**num_agents)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(num_agents,), dtype=np.float32)
        self.action_matrix = self.calculate_action_matrix(num_agents) #所有动作的可能性
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
        #print(self.action_matrix[16])
        #print(self.action_matrix[10])
        #print(self.action_matrix[4])
        #print(self.action_matrix[26])

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
        
    def reset(self):
        #initial_positions = np.linspace(-1, 1, self.num_agents)
        #initial_positions = [0.55, 0.4, -0.05, -0.1, -0.7]
        self.agents = [self.Agent(pos, i) for i, pos in enumerate(self.initial_positions)]  #固定智能体的位置
        #self.agents = [self.Agent(np.random.uniform(-1, 1), i) for i in range(self.num_agents)]  #随机智能体的位置
        self.init_neighbors()
        self.current_iteration = 0
        return self.get_state()
    
    def get_state(self):
        # 可以根据需要设计状态表示
        positions = np.array([agent.position for agent in self.agents])
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
        #print(np.array([agent.position for agent in self.agents]))
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        triggers = self.action_matrix[action]
        
        trigger_count = np.sum(triggers)
        self.total_trigger_count += trigger_count

        for i, agent in enumerate(self.agents):
            agent.update_position(self.current_iteration, self.dt, triggers[i])

        # 检查所有智能体与其邻居之间的位置差是否都小于epsilon
        self.all_within_epsilon = all(all(abs(agent.position - neighbor.position) < self.epsilon for neighbor in agent.neighbors) for agent in self.agents)

        if self.all_within_epsilon:
            if self.epsilon_violated:
                # 如果之前的状态是大于epsilon的，现在变为小于epsilon，更新时间为当前迭代次数
                self.time_to_reach_epsilon = self.current_iteration
                self.epsilon_violated = False  # 更新状态为没有违反epsilon条件
        else:
            self.epsilon_violated = True  # 标记存在位置差大于epsilon的情况
            self.time_to_reach_epsilon = None  # 由于存在违反条件的情况，所以设置为None

        self.current_iteration += 1

        done = self.current_iteration >= self.num_iterations

        if not done:
            average_position_difference = self.compute_average_position_difference()
            if self.time_to_reach_epsilon is not None:
                reward = 5 - trigger_count
                #print("t",reward)
            else:
                #print(average_position_difference)
                # 将平均位置差的负值作为奖励，差值越小（智能体越接近），奖励越高
                reward = - 20 * np.abs(average_position_difference)
                #reward = 0
                #print("a",reward)
        else:
            # 计算奖励
            if self.time_to_reach_epsilon is not None:
                # 计算0到time_to_reach_epsilon时间段内的触发次数
                trigger_counts = sum(len([point for point in agent.trigger_points if point[0] <= self.time_to_reach_epsilon]) for agent in self.agents)
                reward = 1200 - self.time_to_reach_epsilon - 2 * trigger_counts - self.total_trigger_count
                #print(self.time_to_reach_epsilon)
                #print(2 * trigger_counts)
                #print(self.total_trigger_count)
                #print("1")
                #print()
            else:
                trigger_counts = 200
                reward = -2000
            '''a = np.random.uniform(-100,1)
            if a > 0:
                print(trigger_counts)
                print(self.time_to_reach_epsilon)
                print(self.total_trigger_count)'''
            
            self.total_trigger_count = 0
        
        return self.get_state(), reward, done, {}



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

#env = Consensus_F()
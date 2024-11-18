import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
import networkx as nx
from stable_baselines3 import PPO

class DistributedConsensusEnv(gym.Env):
    metadata = {
        'render_modes': ['human']
    }

    def __init__(self, num_iterations=20000, dt=0.001, max_neighbors=4):
        self.num_iterations = num_iterations
        self.dt = dt
        self.current_iteration = 0
        self.max_neighbors = max_neighbors

        # 动作空间和观测空间
        self.action_space = spaces.Discrete(2)
        #self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(max_neighbors + 1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_neighbors + 2,), dtype=np.float32)

        # 创建初始主智能体
        self.main_agent = self.Agent(0, initial_position=np.random.uniform(-1, 1))
        self.agents = [self.main_agent]

        # 参数设置
        self.c_0 = 0.00068
        self.c_1 = 1.2
        self.alpha = 1.4

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_iteration = 0
        self.total_trigger_count = 0

        # 创建五个智能体
        self.agents = [self.Agent(i, initial_position=round(np.random.uniform(-1, 1), 2)) for i in range(5)]
        self.main_agent = random.choice(self.agents)

        # 确保没有孤立节点
        for agent in self.agents:
            potential_neighbors = [other_agent for other_agent in self.agents if other_agent != agent]
            if not agent.neighbors:
                chosen_neighbor = random.choice(potential_neighbors)
                agent.neighbors.append(chosen_neighbor)
                chosen_neighbor.neighbors.append(agent)

            num_neighbors = random.randint(0, min(self.max_neighbors // 2, len(potential_neighbors)))
            additional_neighbors = random.sample(potential_neighbors, num_neighbors)
            for neighbor in additional_neighbors:
                if neighbor not in agent.neighbors:
                    agent.neighbors.append(neighbor)
                    neighbor.neighbors.append(agent)

        return self.get_observation(0), {}

    def get_observation(self, agent_index=0):
        agent = self.main_agent
        obs = [agent.position]  # 主智能体的位置
        if agent.neighbors:
            neighbor_differences = [(neighbor.position - agent.position) for neighbor in agent.neighbors]
            obs.extend(neighbor_differences)
        
        # 确保返回的长度是 max_neighbors + 1，填充差异值为 0
        while len(obs) < self.max_neighbors + 1:
            obs.append(0.0)
        
        # 添加邻居数量作为额外的观测值
        obs.append(len(agent.neighbors))  # 在观测中加入邻居数量

        return np.array(obs, dtype=np.float32)

    def step(self, action):
        main_agent = self.main_agent
        if self.current_iteration == 0:
            main_agent.update_position_1(self.dt, trigger=True)
            for agent in self.agents:
                if agent != main_agent:
                    agent.update_position_formula_with_hold_1(self.dt, self.c_0, self.c_1, self.alpha, self.current_iteration)
            self.total_trigger_count += 1
            reward = 0
        else:
            position_difference = abs(main_agent.position - main_agent.last_broadcast_position)
            time_scaling = np.exp(-self.alpha * self.current_iteration * self.dt)
            threshold = self.c_0 + self.c_1 * time_scaling

            if action == 1:
                self.total_trigger_count += 1
                main_agent.update_position(self.dt, trigger=True)
                reward = 2 if position_difference >= threshold else 0
            else:
                main_agent.update_position(self.dt, trigger=False)
                reward = 2 if position_difference < threshold else 0

            # if action == 1:
            #     self.total_trigger_count += 1
            #     main_agent.update_position(self.dt, trigger=True)
            # else:
            #     main_agent.update_position(self.dt, trigger=False)

            # if position_difference >= threshold:
            #     reward = -5 * (position_difference - threshold)
            # else:
            #     reward = 2  # 正常奖励值



            for agent in self.agents:
                if agent != main_agent:
                    agent.update_position_formula_with_hold(self.dt, self.c_0, self.c_1, self.alpha, self.current_iteration)

        self.current_iteration += 1
        done = self.current_iteration >= self.num_iterations
        if done:
            self.t = self.total_trigger_count

        return self.get_observation(0), reward, done, False, {}
    
    def render(self, action):
        """根据动作更新环境，仅更新当前主智能体，其他智能体在模拟中被遍历决策"""
        main_agent = self.main_agent
        if self.current_iteration == 0:
            main_agent.update_position_1(self.dt, trigger=True)
            # for agent in self.agents:
            #     if agent != main_agent:
            #         agent.update_position_formula_with_hold_1(self.dt, self.c_0, self.c_1, self.alpha, self.current_iteration)
            self.total_trigger_count += 1
            reward = 0
        else:
            position_difference = abs(main_agent.position - main_agent.last_broadcast_position)
            time_scaling = np.exp(-self.alpha * self.current_iteration * self.dt)
            threshold = self.c_0 + self.c_1 * time_scaling

            if action == 1:
                self.total_trigger_count += 1
                main_agent.update_position(self.dt, trigger=True)
                reward = 1 if position_difference >= threshold else -1
            else:
                main_agent.update_position(self.dt, trigger=False)
                reward = 1 if position_difference < threshold else -1

        self.current_iteration += 1
        done = self.current_iteration >= self.num_iterations

        if done:
            self.t = self.total_trigger_count

        return self.get_observation(0), reward, done, False, {}

    class Agent:
        def __init__(self, index, initial_position=0.0):
            self.position = initial_position
            self.index = index
            self.neighbors = []
            self.last_broadcast_position = self.position
            self.u_i = 0.0

        def add_neighbor(self, neighbor):
            if neighbor not in self.neighbors:
                self.neighbors.append(neighbor)

        def update_position(self, dt, trigger):
            if trigger:
                self.u_i = -sum((self.last_broadcast_position - neighbor.last_broadcast_position) for neighbor in self.neighbors)
                self.position += self.u_i * dt
                self.last_broadcast_position = self.position
            else:
                self.position += self.u_i * dt

        def update_position_1(self, dt, trigger):
            if trigger:
                self.u_i = -sum((self.last_broadcast_position - neighbor.last_broadcast_position) for neighbor in self.neighbors)
                self.position += self.u_i * dt
            else:
                self.position += self.u_i * dt

        def update_position_formula_with_hold(self, dt, c_0, c_1, alpha, current_iteration):
            e_i = self.position - self.last_broadcast_position
            f_i = abs(e_i) - (c_0 + c_1 * np.exp(-alpha * current_iteration * dt))
            if f_i >= 0:
                self.u_i = -sum((self.last_broadcast_position - neighbor.last_broadcast_position) for neighbor in self.neighbors)
                self.position += self.u_i * dt
                self.last_broadcast_position = self.position
            else:
                self.position += self.u_i * dt

        def update_position_formula_with_hold_1(self, dt, c_0, c_1, alpha, current_iteration):
            e_i = self.position - self.last_broadcast_position
            f_i = abs(e_i) - (c_0 + c_1 * np.exp(-alpha * current_iteration * dt))
            if f_i >= 0:
                self.u_i = -sum((self.last_broadcast_position - neighbor.last_broadcast_position) for neighbor in self.neighbors)
                self.position += self.u_i * dt
            else:
                self.position += self.u_i * dt


import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, List
import matplotlib.pyplot as plt

class Consensus_D_F(gym.Env):
    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }
    
    def __init__(self, num_agents=5, num_iterations=200, dt=0.1, max_neighbors=4):
        self.num_agents = num_agents
        self.num_iterations = num_iterations
        self.dt = dt
        self.current_iteration = 0
        self.max_neighbors = max_neighbors
        
        # 动作空间：每个智能体的触发决策是0或1
        self.action_space = spaces.MultiBinary(num_agents)
        
        # 观测空间：包含每个智能体与邻居平均位置差
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(num_agents, max_neighbors + 1), dtype=np.float32)
        
        # 固定初始位置和拓扑矩阵
        self.laplacian_matrix = np.array([
            [ 3, -1,  0, -1, -1],
            [-1,  2, -1,  0,  0],
            [ 0, -1,  2,  0, -1],
            [-1,  0,  0,  1,  0],
            [-1,  0, -1,  0,  2],
        ])
        self.initial_positions = [0.2, -0.7, 0.1, -0.4, 0.8]
        
        # 初始化智能体
        self.agents = [self.Agent(i) for i in range(self.num_agents)]
        self.set_fixed_neighbors()

        # 其他参数
        self.epsilon = 0.005
        self.total_trigger_count = 0
        self.time_to_reach_epsilon = None
        self.epsilon_violated = True
        self.all_within_epsilon = False
        self.success = 0

    def set_fixed_neighbors(self):
        """根据固定的拉普拉斯矩阵设置邻居关系"""
        for i, agent in enumerate(self.agents):
            agent.neighbors = []  # 清空当前邻居
            for j in range(self.num_agents):
                if i != j and self.laplacian_matrix[i, j] == -1:  # -1 表示邻接
                    agent.neighbors.append(self.agents[j])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_iteration = 0
        self.total_trigger_count = 0
        self.time_to_reach_epsilon = None
        self.epsilon_violated = True
        self.all_within_epsilon = False
        self.u_i = 0
        self.success = 0
        
        # 使用固定的初始位置
        for i, agent in enumerate(self.agents):
            agent.position = self.initial_positions[i]
            agent.last_broadcast_position = agent.position

        return self.get_observation(), {"initial_positions": self.initial_positions}

    def get_observation(self):
        """生成观测值，观测值为每个智能体自己的位置和所有邻居的信息"""
        observations = []
        for agent in self.agents:
            # 观测包括智能体自身位置
            obs = [agent.position]

            # 获取所有邻居的位置信息
            if agent.neighbors:
                neighbor_positions = [neighbor.position for neighbor in agent.neighbors]
                # 填充到最大邻居数量，未填充部分使用0.0
                obs.extend(neighbor_positions)
            obs.extend([0.0] * (self.max_neighbors - len(agent.neighbors)))  # 确保填充到最大邻居数

            # 确保观测长度一致
            obs = obs[:self.max_neighbors + 1]
            obs = np.clip(obs, -1.0, 1.0)  # 将观测值限制在[-1, 1]范围内
            observations.append(obs)

        return np.array(observations, dtype=np.float32)

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        triggers = action
        self.total_trigger_count += np.sum(triggers)
        
        for i, agent in enumerate(self.agents):
            agent.update_position(self.current_iteration, self.dt, triggers[i])

        self.all_within_epsilon = all(
            all(abs(agent.position - neighbor.position) < self.epsilon for neighbor in agent.neighbors)
            for agent in self.agents
        )

        # 在每一步迭代中检查是否达成一致性
        if self.all_within_epsilon:
            if self.epsilon_violated:
                self.time_to_reach_epsilon = self.current_iteration
                self.epsilon_violated = False
            self.success += 1
        else:
            self.epsilon_violated = True
            self.time_to_reach_epsilon = None

        self.current_iteration += 1
        done = self.current_iteration >= self.num_iterations

        if not done:
            self.average_difference = self.compute_average_position_difference()
            if self.time_to_reach_epsilon is not None:
                reward = 5 + 5 * (5 - np.sum(triggers)) 
                self.success += 1
            else:
                reward = -2 -2 * np.clip(np.exp(self.average_difference), 0, 10)
        else:
            if self.all_within_epsilon:
                self.success += 1000
            reward = 0
            self.s = self.success
            self.total = self.total_trigger_count
            self.time = self.time_to_reach_epsilon

        return self.get_observation(), reward, done, False, {}

    def compute_average_position_difference(self):
        total_difference = 0
        count = 0
        for agent in self.agents:
            if agent.neighbors:
                total_difference += sum(abs(agent.position - neighbor.position) for neighbor in agent.neighbors)
                count += len(agent.neighbors)
        return total_difference / count if count > 0 else 0

    def render(self, model, laplacian_matrix: np.ndarray, initial_positions: List[float], num_steps=200):
        """基于给定的拉普拉斯矩阵和初始位置，输出每一步的动作矩阵并绘制图像"""
        actions_over_time = []
        positions_over_time = []

        for i, agent in enumerate(self.agents):
            agent.neighbors = []  # 清空当前邻居
            for j in range(len(laplacian_matrix)):
                if i != j and laplacian_matrix[i, j] == -1:  # -1 表示邻接
                    agent.add_neighbor(self.agents[j])

        for i, pos in enumerate(initial_positions):
            self.agents[i].position = pos

        for step in range(num_steps):
            obs = self.get_observation()
            action = model.predict(obs, deterministic=True)[0]
            actions_over_time.append(action)
            positions_over_time.append([agent.position for agent in self.agents])
            self.step(action)

        times = range(num_steps)
        positions_over_time = np.array(positions_over_time)
        plt.figure(figsize=(10, 4))
        for i in range(self.num_agents):
            plt.plot(times, positions_over_time[:, i], label=f'Agent {i + 1}')
        plt.xlabel("Times")
        plt.ylabel("Positions")
        plt.legend(loc="upper right")
        plt.grid()
        plt.show()

        plt.figure(figsize=(10, 3))
        for i in range(self.num_agents):
            trigger_times = [t for t in times if actions_over_time[t][i] == 1]
            plt.scatter(trigger_times, [i + 1] * len(trigger_times), label=f'Agent {i + 1}', s=20)
        plt.xlabel("Times")
        plt.ylabel("Agents")
        plt.yticks(range(1, self.num_agents + 1), [f'Agent {i + 1}' for i in range(self.num_agents)])
        plt.grid()
        plt.show()

        return actions_over_time

    class Agent:
        def __init__(self, index, initial_position=0.0):
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
                self.u_i = -sum((self.last_broadcast_position - neighbor.last_broadcast_position)
                                for neighbor in self.neighbors if self.is_neighbor(neighbor))
                self.position += self.u_i * dt
                self.last_broadcast_position = self.position
                self.trigger_points.append((t, self.position))
            else:
                self.position += self.u_i * dt
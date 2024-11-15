import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, List
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle

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

        # 动作空间：将动作空间转换为离散类型，大小为 2^num_agents
        self.action_space = spaces.Discrete(2 ** self.num_agents)

        # 观测空间：包含每个智能体与邻居平均位置差
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(num_agents, max_neighbors + 1), dtype=np.float32)

        # 初始化智能体和固定的拉普拉斯矩阵
        self.agents = [self.Agent(i) for i in range(self.num_agents)]
        self.laplacian_matrix = np.array([
            [ 2, -1, -1,  0,  0],
            [-1,  2, -1,  0,  0],
            [-1, -1,  3, -1,  0],
            [ 0,  0, -1,  2, -1],
            [ 0,  0,  0, -1,  1]
        ])
        self.init_neighbors_fixed()

        # 其他参数
        self.epsilon = 0.005
        self.total_trigger_count = 0
        self.time_to_reach_epsilon = None
        self.epsilon_violated = True
        self.all_within_epsilon = False

    def init_neighbors_fixed(self):
        """使用固定的拉普拉斯矩阵来设置邻居关系"""
        for i, agent in enumerate(self.agents):
            agent.neighbors = []
            for j in range(len(self.laplacian_matrix)):
                if i != j and self.laplacian_matrix[i, j] == -1:
                    agent.add_neighbor(self.agents[j])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_iteration = 0
        self.total_trigger_count = 0
        self.time_to_reach_epsilon = None
        self.epsilon_violated = True
        self.all_within_epsilon = False
        self.u_i = 0
        self.success = 0

        # 随机初始化智能体位置并保留一位小数
        self.agents = [self.Agent(i, initial_position=round(np.random.uniform(-1, 1), 1)) for i in range(self.num_agents)]
        self.init_neighbors_fixed()
        initial_positions = [agent.position for agent in self.agents]

        return self.get_observation(), {"initial_positions": initial_positions}

    def get_observation(self):
        """生成观测值，观测值为每个智能体自己的位置和所有邻居的信息"""
        observations = []
        for agent in self.agents:
            obs = [agent.position]
            if agent.neighbors:
                neighbor_positions = [neighbor.position for neighbor in agent.neighbors]
                obs.extend(neighbor_positions)
            obs.extend([0.0] * (self.max_neighbors - len(agent.neighbors)))
            obs = obs[:self.max_neighbors + 1]
            obs = np.clip(obs, -1.0, 1.0)
            observations.append(obs)
        return np.array(observations, dtype=np.float32)

    def step(self, action):
        binary_action = [int(x) for x in format(action, f'0{self.num_agents}b')]
        assert len(binary_action) == self.num_agents, "解码后的动作长度与智能体数量不匹配"
        triggers = np.array(binary_action)
        self.total_trigger_count += np.sum(triggers)

        for i, agent in enumerate(self.agents):
            agent.update_position(self.current_iteration, self.dt, triggers[i])

        positions = [agent.position for agent in self.agents]
        self.all_within_epsilon = (max(positions) - min(positions)) <= self.epsilon

        if self.all_within_epsilon:
            if self.epsilon_violated:
                self.time_to_reach_epsilon = self.current_iteration
                self.epsilon_violated = False
        else:
            self.epsilon_violated = True
            self.time_to_reach_epsilon = None

        self.current_iteration += 1
        done = self.current_iteration >= self.num_iterations

        # if not done:
        #     self.average_difference = self.compute_average_position_difference()
        #     if self.all_within_epsilon:
        #         reward = 5 + 30 * (5 - np.sum(triggers))
        #         self.success += 1
        #     else:
        #         reward = reward = - np.clip((10 * np.exp(self.average_difference)), 0, 50)
        # else:
        #     if self.all_within_epsilon:
        #         reward = - 5 * self.total_trigger_count
        #         self.success += 1000
        #     else:
        #         self.success += 0
        #         reward = -10000 - 5 * self.total_trigger_count
        #     self.s = self.success
        #     self.total = self.total_trigger_count
        #     self.time = self.time_to_reach_epsilon

        # 确定是否处于早期阶段或后期阶段
        phase_threshold = self.num_iterations // 4  # 将阶段划分为前50%和后50%
        self.average_difference = self.compute_average_position_difference()
        if not done:
            if self.current_iteration <= phase_threshold:
                # 早期阶段：注重探索和实现一致性
                # self.average_difference = self.compute_average_position_difference()
                if self.all_within_epsilon:
                    reward = 50 + (5 - np.sum(triggers))  # 较大的固定奖励，鼓励实现一致性
                    self.success += 1
                else:
                    reward = -np.clip((10 * np.exp(self.average_difference)), 0, 50)  # 惩罚差异，但不是太严重
            else:
                
                if self.all_within_epsilon:
                    reward = 5 + 30 * (5 - np.sum(triggers))  # 奖励与触发次数减少相关
                    self.success += 1
                else:
                    reward = -np.clip((20 * np.exp(self.average_difference)), 0, 80)
        else:
            if self.all_within_epsilon:
                reward = -5 * self.total_trigger_count
                self.success += 1000
            else:
                reward = -5 * self.total_trigger_count
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

    def render(self, model, num_steps=200):
        """基于固定的拉普拉斯矩阵和随机生成的初始位置，输出每一步的动作矩阵并绘制图像"""
        actions_over_time = []
        positions_over_time = []

        # 使用固定的拉普拉斯矩阵来设置邻居关系
        laplacian_matrix = np.array([[ 2, -1, -1,  0,  0],
                                    [-1,  2, -1,  0,  0],
                                    [-1, -1,  3, -1,  0],
                                    [ 0,  0, -1,  2, -1],
                                    [ 0,  0,  0, -1,  1]])

        for i, agent in enumerate(self.agents):
            agent.neighbors = []  # 清空当前邻居
            for j in range(len(laplacian_matrix)):
                if i != j and laplacian_matrix[i, j] == -1:  # -1 表示邻接
                    agent.add_neighbor(self.agents[j])

        # 随机生成初始位置并保留一位小数
        initial_positions = [round(np.random.uniform(-1, 1), 1) for _ in range(self.num_agents)]
        for i, pos in enumerate(initial_positions):
            self.agents[i].position = pos

        # 使用预训练模型预测每一步动作
        for step in range(num_steps):
            obs = self.get_observation()
            action = model.predict(obs, deterministic=True)[0]
            
            # 将动作转换为二进制表示
            binary_action = [int(x) for x in format(action, f'0{self.num_agents}b')]
            assert len(binary_action) == self.num_agents, "解码后的动作长度与智能体数量不匹配"
            actions_over_time.append(binary_action)

            # 记录当前所有智能体的位置
            positions_over_time.append([agent.position for agent in self.agents])

            # 执行动作以更新环境
            self.step(action)

        # 输出随机生成的初始位置
        print(f"随机生成的初始位置: {initial_positions}")

        # 输出总触发次数
        print(f"总共的触发次数: {self.total_trigger_count}")

        # 输出实现一致性的时间或未实现的消息
        if self.all_within_epsilon:
            print(f"实现一致性的时间: {self.time_to_reach_epsilon}")
        else:
            print("未实现一致性")
        
        # 绘制智能体位置随时间变化的图像
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

        # 绘制触发状态的图像
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
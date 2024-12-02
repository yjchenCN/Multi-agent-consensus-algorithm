import numpy as np
import gymnasium as gym
from gymnasium import spaces
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

        # 动作空间：主智能体的动作，只有0和1
        self.action_space = spaces.Discrete(2)

        # 观测空间：主智能体的位置及邻居的位置（加掩码）
        self.observation_space = spaces.Dict({
            "positions": spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32),
            "mask": spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32)
        })

        # 初始化智能体
        self.agents = [self.Agent(i) for i in range(self.num_agents)]
        self.adjacency_matrix = None  # 将邻接矩阵作为实例变量
        self.init_neighbors_random()

        # 主智能体参数
        self.epsilon = 0.005
        self.total_trigger_count = 0
        self.time_to_reach_epsilon = None
        self.epsilon_violated = True
        self.all_within_epsilon = False

    def init_neighbors_random(self):
        """随机生成邻接矩阵，确保没有孤立节点"""
        while True:
            # 生成随机对称矩阵
            random_matrix = np.random.randint(0, 2, size=(self.num_agents, self.num_agents))
            adjacency_matrix = (random_matrix + random_matrix.T) > 0  # 确保对称性
            np.fill_diagonal(adjacency_matrix, 0)  # 对角线置为0，避免自环

            # 检查是否存在孤立节点
            degrees = np.sum(adjacency_matrix, axis=1)
            if np.all(degrees > 0):  # 确保没有孤立节点
                break

        # 转换为拉普拉斯矩阵
        degree_matrix = np.diag(degrees)
        self.adjacency_matrix = degree_matrix - adjacency_matrix  # 保存拉普拉斯矩阵
        self.neighbor_matrix = adjacency_matrix  # 保存邻接矩阵

        # 更新每个智能体的邻居
        for i, agent in enumerate(self.agents):
            agent.neighbors.clear()  # 清空之前的邻居关系
            for j in range(self.num_agents):
                if adjacency_matrix[i, j] == 1:
                    agent.add_neighbor(self.agents[j])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_iteration = 0
        self.total_trigger_count = 0
        self.time_to_reach_epsilon = None
        self.epsilon_violated = True
        self.all_within_epsilon = False

        # 随机初始化智能体位置并保留一位小数
        self.agents = [
            self.Agent(i, initial_position=round(np.random.uniform(-1, 1), 2))
            for i in range(self.num_agents)
        ]
        # 随机初始化邻居关系
        self.init_neighbors_random()
        return self.get_observation(), {}

    def get_observation(self):
        """生成主智能体的观测值及对应的掩码"""
        agent = self.agents[0]  # 主智能体
        positions = [agent.position]  # 主智能体位置
        mask = [1]  # 主智能体的观测有效

        for neighbor in agent.neighbors:
            positions.append(neighbor.position)
            mask.append(1)

        # 填充无效位置和掩码
        while len(positions) < self.max_neighbors + 1:
            positions.append(0.0)  # 填充值
            mask.append(0)  # 无效位置

        # 确保返回的长度固定
        positions = positions[:self.max_neighbors + 1]
        mask = mask[:self.max_neighbors + 1]

        return {
            "positions": np.array(positions, dtype=np.float32),
            "mask": np.array(mask, dtype=np.float32)
        }

    def step(self, action):
        assert action in [0, 1], "动作必须是0或1"

        # 更新主智能体
        self.total_trigger_count += action
        self.agents[0].update_position(self.current_iteration, self.dt, action)

        # 更新其他智能体
        for agent in self.agents[1:]:
            if self.current_iteration == 0:
                # 强制触发并更新位置
                agent.force_update_position(self.dt)
            else:
                agent.update_position_formula_with_hold(
                    dt=self.dt, c_0=0.00068, c_1=1.2, alpha=1.4, current_iteration=self.current_iteration
                )

        # 计算全体智能体的位置
        positions = [agent.position for agent in self.agents]
        max_position = max(positions)
        min_position = min(positions)

        # 检查是否实现一致性
        self.all_within_epsilon = (max_position - min_position) <= self.epsilon

        # 奖励计算逻辑
        if self.all_within_epsilon:
            if action == 0:
                reward = 10
            else:
                reward = 2
        else:
            distances = [
                abs(self.agents[0].position - neighbor.position)
                for neighbor in self.agents[0].neighbors
            ]
            reward = -10 * sum(distances)

        self.current_iteration += 1
        done = self.current_iteration >= self.num_iterations

        if done:
            final_avg_position = np.mean([agent.position for agent in self.agents])
            initial_avg_position = np.mean([agent.initial_position for agent in self.agents])
            position_shift_penalty = abs(final_avg_position - initial_avg_position)
            if self.all_within_epsilon:
                reward = -200 * position_shift_penalty
            else:
                reward = -500 -200 * position_shift_penalty

        self.t = self.total_trigger_count
        return self.get_observation(), reward, done, False, {}

    class Agent:
        def __init__(self, index, initial_position=0.0):
            self.position = initial_position
            self.index = index
            self.neighbors = []
            self.last_broadcast_position = self.position
            self.initial_position = initial_position 
            self.u_i = 0

        def add_neighbor(self, neighbor):
            if neighbor not in self.neighbors:
                self.neighbors.append(neighbor)
                neighbor.neighbors.append(self)

        def update_position(self, t, dt, trigger):
            if trigger == 1:
                self.u_i = -sum((self.last_broadcast_position - neighbor.last_broadcast_position)
                                for neighbor in self.neighbors)
                self.position += self.u_i * dt
                self.last_broadcast_position = self.position
            else:
                self.position += self.u_i * dt

        def update_position_formula_with_hold(self, dt, c_0, c_1, alpha, current_iteration):
            e_i = self.position - self.last_broadcast_position
            f_i = abs(e_i) - (c_0 + c_1 * np.exp(-alpha * current_iteration * dt))
            if f_i >= 0:
                self.u_i = -sum((self.last_broadcast_position - neighbor.last_broadcast_position)
                                for neighbor in self.neighbors)
                self.position += self.u_i * dt
                self.last_broadcast_position = self.position
            else:
                self.position += self.u_i * dt

        def force_update_position(self, dt):
            """强制触发并更新位置"""
            self.u_i = -sum((self.last_broadcast_position - neighbor.last_broadcast_position)
                            for neighbor in self.neighbors)
            self.position += self.u_i * dt
            self.last_broadcast_position = self.position
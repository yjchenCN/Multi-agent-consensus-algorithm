import numpy as np
from pettingzoo import ParallelEnv


class Agent:
    def __init__(self, num_dimensions):
        self.num_dimensions = num_dimensions
        self.position = np.zeros(num_dimensions)
        self.velocity = np.zeros(num_dimensions)
        self.alpha_weights = np.zeros(num_dimensions)

    def set_position(self, new_position):
        self.position = new_position

    def set_velocity(self, new_velocity):
        self.velocity = new_velocity

    def get_velocity(self):
        return self.velocity

    def are_velocities_close(self, other_agent, threshold=0.01):
        return np.all(np.abs(self.velocity - other_agent.velocity) < threshold)


class CustomEnvironment(ParallelEnv):
    metadata = {
        "name": "custom_environment_v0",
    }

    def __init__(self, num_agents=3, num_dimensions=1):
        self.num_agents = num_agents
        self.num_dimensions = num_dimensions

        # 创建智能体列表
        self.agents = [Agent(num_dimensions) for _ in range(num_agents)]

        # 创建拉普拉斯矩阵
        self.laplacian_matrix = self.generate_laplacian_matrix(num_agents)

        self.observation_spaces = {f"agent_{i}": num_dimensions for i in range(num_agents)}
        self.action_spaces = {f"agent_{i}": num_dimensions for i in range(num_agents)}

    def generate_laplacian_matrix(self, num_agents):
        # 创建邻接矩阵（这里假设所有智能体都互相通信）
        adjacency_matrix = np.ones((num_agents, num_agents)) - np.eye(num_agents)

        # 创建度矩阵
        degree_matrix = np.diag(np.sum(adjacency_matrix, axis=1))

        # 计算拉普拉斯矩阵
        laplacian_matrix = degree_matrix - adjacency_matrix

        return laplacian_matrix

    def reset(self, seed=None, options=None):  #随机化
        for agent in self.agents:
            agent.position = np.zeros(self.num_dimensions)
            agent.velocity = np.zeros(self.num_dimensions)
            agent.alpha_weights = np.zeros(self.num_dimensions)
        return {f"agent_{i}": agent.position.copy() for i, agent in enumerate(self.agents)}

    def step(self, actions):
        done = False

        while not done:
            reward = 0

            for i, agent in enumerate(self.agents):
                agent.set_velocity(-np.dot(self.laplacian_matrix[i], (agent.position - np.array([a.position for a in self.agents]))))
                system_velocities = np.array([agent.velocity for agent in self.agents])
                system_std = np.std(system_velocities)
                reward += 1.0 / (1.0 + system_std)

                agent.set_position(agent.velocity)

            # 检查是否所有智能体的速度都足够接近
            done = all(agent.are_velocities_close(self.agents[0]) for agent in self.agents)

        reward += 2 * (1.0 / (1.0 + system_std))

        rewards = {f"agent_{i}": reward for i in range(self.num_agents)}

        return {f"agent_{i}": agent.position.copy() for i, agent in enumerate(self.agents)}, rewards, done, {}

    def render(self):
        pass

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]
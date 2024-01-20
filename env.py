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

        # 定义通信矩阵，可以在每个步骤中随时间变化
        self.alpha_matrix = np.random.rand(num_agents, num_agents)

        self.observation_spaces = {f"agent_{i}": num_dimensions for i in range(num_agents)}
        self.action_spaces = {f"agent_{i}": num_dimensions for i in range(num_agents)}

    def reset(self, seed=None, options=None):
        for agent in self.agents:
            agent.position = np.zeros(self.num_dimensions)
            agent.velocity = np.zeros(self.num_dimensions)
            agent.alpha_weights = np.zeros(self.num_dimensions)
        return {f"agent_{i}": agent.position.copy() for i, agent in enumerate(self.agents)}

    def step(self, actions):
        done = False

        while not done:
            for i, agent in enumerate(self.agents):
                agent.velocity = -np.sum(self.alpha_matrix[i] * (agent.position - np.array([a.position for a in self.agents])), axis=0)
                agent.position += agent.velocity

            # 检查是否所有智能体的速度都足够接近
            done = all(agent.are_velocities_close(self.agents[0]) for agent in self.agents)

        rewards = {f"agent_{i}": 0.0 for i in range(self.num_agents)}

        return {f"agent_{i}": agent.position.copy() for i, agent in enumerate(self.agents)}, rewards, done, {}

    def render(self):
        pass

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]
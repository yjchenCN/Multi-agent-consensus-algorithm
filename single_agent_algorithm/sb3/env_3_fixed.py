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

    def __init__(self, num_agents=3, num_iterations=10000, dt=0.001, max_neighbors=2):
        self.num_agents = num_agents
        self.num_iterations = num_iterations
        self.dt = dt
        self.current_iteration = 0
        self.max_neighbors = max_neighbors

        self.action_space = spaces.Discrete(2 ** self.num_agents)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(num_agents, max_neighbors + 1), dtype=np.float32)

        self.agents = [self.Agent(i) for i in range(self.num_agents)]
        self.laplacian_matrix = np.array([
            [ 2, -1, -1],
            [-1,  2, -1],
            [-1, -1,  2]
        ])
        self.init_neighbors_fixed()

        self.epsilon = 0.005
        self.total_trigger_count = 0
        self.time_to_reach_epsilon = None
        self.epsilon_violated = True
        self.all_within_epsilon = False

    def init_neighbors_fixed(self):
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

        # 使用固定位置 [0.9, -0.2, -0.5]
        initial_positions = [0.9, -0.2, -0.5]
        self.agents = [self.Agent(i, initial_position=initial_positions[i]) for i in range(self.num_agents)]
        self.init_neighbors_fixed()

        return self.get_observation(), {"initial_positions": initial_positions}

    def get_observation(self):
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

        phase_threshold = self.num_iterations // 4
        self.average_difference = self.compute_average_position_difference()
        if not done:
            if self.current_iteration <= phase_threshold:
                if self.all_within_epsilon:
                    reward = 5 + (5 - np.sum(triggers))
                    self.success += 1
                else:
                    reward = -np.clip((np.exp(self.average_difference)), 0, 10)
            else:
                if self.all_within_epsilon:
                    reward = 5 + 5 * (5 - np.sum(triggers))
                    self.success += 1
                else:
                    reward = -np.clip((3 * np.exp(self.average_difference)), 0, 30)
        else:
            if self.all_within_epsilon:
                reward = - self.total_trigger_count
                self.success += 100000
            else:
                reward = - self.total_trigger_count
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

    def render(self, model, num_steps=10000):
        actions_over_time = []
        positions_over_time = []

        laplacian_matrix = np.array([
            [ 2, -1, -1],
            [-1,  2, -1],
            [-1, -1,  2]
        ])

        for i, agent in enumerate(self.agents):
            agent.neighbors = []
            for j in range(len(laplacian_matrix)):
                if i != j and laplacian_matrix[i, j] == -1:
                    agent.add_neighbor(self.agents[j])

        initial_positions = [0.9, -0.2, -0.5]
        for i, pos in enumerate(initial_positions):
            self.agents[i].position = pos

        for step in range(num_steps):
            obs = self.get_observation()
            action = model.predict(obs, deterministic=True)[0]
            binary_action = [int(x) for x in format(action, f'0{self.num_agents}b')]
            assert len(binary_action) == self.num_agents, "解码后的动作长度与智能体数量不匹配"
            actions_over_time.append(binary_action)
            positions_over_time.append([agent.position for agent in self.agents])
            self.step(action)

        print(f"固定的初始位置: {initial_positions}")
        print(f"总共的触发次数: {self.total_trigger_count}")

        if self.all_within_epsilon:
            print(f"实现一致性的时间: {self.time_to_reach_epsilon}")
        else:
            print("未实现一致性")

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
from sympy import AlgebraicField
from pettingzoo import ParallelEnv
import numpy as np
from gym import spaces
import numpy as np

class CustomMAEnvironment1(ParallelEnv):
    metadata = {
        'render.modes': ['human'],
        'name': 'custom_environment_demo'
    }
    
    def __init__(self, num_agents=5, num_iterations=200, dt=0.1):
        self.agents = ["agent_" + str(i) for i in range(num_agents)]
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = dict(zip(self.agents, list(range(num_agents))))
        
        self.num_iterations = num_iterations
        self.dt = dt
        self.current_iteration = 0
        
        initial_positions = [0.55, 0.4, -0.05, -0.1, -0.7]
        self.agent_objs = [self.Agent(pos, i) for i, pos in enumerate(initial_positions)]
        self.init_neighbors()
        
        self.epsilon = 0.005
        self.time_to_reach_epsilon = None
        self.epsilon_violated = True
        self.all_within_epsilon = False
        self.total_trigger_count = 0
        self.time_to_reach_epsilon_changes = 0
        self.max_obs_size = self.compute_max_obs_size()

        # 计算最大邻居数量
        self.max_neighbors = max(len(agent.neighbors) for agent in self.agent_objs)
        self.state_dim = 1 + self.max_neighbors  # 每个智能体的状态维度（包括自己和最大邻居数）
    
    def compute_max_obs_size(self):
        max_neighbors = max(len(agent.neighbors) for agent in self.agent_objs)
        return 1 + max_neighbors

    def init_neighbors(self):
        self.agent_objs[0].add_neighbor(self.agent_objs[1])
        self.agent_objs[0].add_neighbor(self.agent_objs[2])
        self.agent_objs[0].add_neighbor(self.agent_objs[3])
        self.agent_objs[0].add_neighbor(self.agent_objs[4])
        self.agent_objs[1].add_neighbor(self.agent_objs[2])
        self.agent_objs[1].add_neighbor(self.agent_objs[3])
        self.agent_objs[1].add_neighbor(self.agent_objs[4])
        self.agent_objs[2].add_neighbor(self.agent_objs[3])
        self.agent_objs[2].add_neighbor(self.agent_objs[4])
        self.agent_objs[3].add_neighbor(self.agent_objs[4])
        
    
        
    def reset(self, seed=None, options=None):
        initial_positions = [0.55, 0.4, -0.05, -0.1, -0.7]
        self.agent_objs = [self.Agent(pos, i) for i, pos in enumerate(initial_positions)]
        self.init_neighbors()
        self.current_iteration = 0
        self.epsilon_violated = True
        self.all_within_epsilon = False
        self.total_trigger_count = 0
        self.time_to_reach_epsilon_changes = 0
        self.time_to_reach_epsilon = None
        
        observations = {agent: self.get_observation(agent) for agent in self.agents}
        return observations
    
    def get_state(self):
        state = []
        for agent in self.agent_objs:
            state.append(agent.position)
            for neighbor in agent.neighbors:
                state.append(neighbor.position)
        return np.array(state)

    
    def get_observation(self, agent):
        agent_index = self.agent_name_mapping[agent]
        agent_obj = self.agent_objs[agent_index]
        neighbors_positions = [neighbor.position for neighbor in agent_obj.neighbors]
        obs = np.array([agent_obj.position] + neighbors_positions, dtype=np.float32)
        return obs
    
    def compute_average_position_difference(self):
        total_difference = 0
        count = 0
        for i, agent_i in enumerate(self.agent_objs):
            for j, agent_j in enumerate(self.agent_objs):
                if i < j:
                    total_difference += abs(agent_i.position - agent_j.position)
                    count += 1
        if count > 0:
            return total_difference / count
        else:
            return 0
    
    def step(self, actions):
        triggers = np.array([actions[agent] for agent in self.agents])
        trigger_count = np.sum(triggers)
        self.total_trigger_count += trigger_count

        for i, agent in enumerate(self.agent_objs):
            agent.update_position(self.current_iteration, self.dt, triggers[i])

        self.all_within_epsilon = all(all(abs(agent.position - neighbor.position) < self.epsilon for neighbor in agent.neighbors) for agent in self.agent_objs)

        if self.all_within_epsilon:
            if self.epsilon_violated:
                self.time_to_reach_epsilon = self.current_iteration
                self.epsilon_violated = False
                self.time_to_reach_epsilon_changes += 1
        else:
            self.epsilon_violated = True
            self.time_to_reach_epsilon = None
        
        self.current_iteration += 1

        # 检查是否有智能体位置超出范围
        out_of_bounds = any(abs(agent.position) > 1.5 for agent in self.agent_objs)
        done = self.current_iteration >= self.num_iterations or out_of_bounds
        out_of_bounds_done = out_of_bounds

        rewards = {}
        if not done and not out_of_bounds_done:
            average_position_difference = self.compute_average_position_difference()
            for agent in self.agents:
                if self.time_to_reach_epsilon is not None:
                    if actions[agent] == 1:
                        rewards[agent] = 0
                    else:
                        rewards[agent] = 10
                else:
                    agent_index = self.agent_name_mapping[agent]
                    agent_position = self.agent_objs[agent_index].position
                    other_positions = [other_agent.position for other_agent in self.agent_objs if other_agent != self.agent_objs[agent_index]]
                    average_position = np.mean(other_positions)
                    position_difference = abs(agent_position - average_position)
                    rewards[agent] = - 20 * np.abs(position_difference)
        else:
            if out_of_bounds_done:
                global_reward = -200  # 惩罚因超出边界导致的done
            else:
                if self.time_to_reach_epsilon is not None:
                    global_reward = 1200 - self.time_to_reach_epsilon - self.total_trigger_count
                else:
                    global_reward = -200
            for agent in self.agents:
                rewards[agent] = global_reward

        observations = {agent: self.get_observation(agent) for agent in self.agents}
        dones = {agent: done for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        
        return observations, rewards, dones, infos

    
    def render(self, mode='human'):
        positions = [agent.position for agent in self.agent_objs]
        print(f"Positions: {positions}")

    def observation_space(self, agent):
        agent_index = self.agent_name_mapping[agent]
        agent_obj = self.agent_objs[agent_index]
        obs_size = 1 + len(agent_obj.neighbors)
        return spaces.Box(low=-100, high=100, shape=(obs_size,), dtype=np.float32)
    
    def action_space(self, agent):
        return spaces.Discrete(2)

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
            if trigger == 1 or t == 0:
                self.u_i = -sum((self.last_broadcast_position - neighbor.last_broadcast_position) for neighbor in self.neighbors if self.is_neighbor(neighbor))
                self.position += self.u_i * dt
                self.last_broadcast_position = self.position
                self.trigger_points.append((t, self.position))
            else:
                self.position += self.u_i * dt


env = CustomMAEnvironment1()

# 打印每个智能体的观测空间形状
for agent in env.agents:
    obs_space = env.observation_space(agent)
    print(f"Agent {agent}: Observation space shape = {obs_space.shape}")

print(env.agents)

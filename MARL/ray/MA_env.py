import numpy as np
from gym import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv

class MAEnvironment(MultiAgentEnv):
    def __init__(self, num_agents=5, num_iterations=200, dt=0.1):
        self.num_agents = num_agents
        self.agents = ["agent_" + str(i) for i in range(num_agents)]
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

    def reset(self):
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

    def get_observation(self, agent):
        agent_index = self.agent_name_mapping[agent]
        agent_obj = self.agent_objs[agent_index]
        neighbors_positions = [neighbor.position for neighbor in agent_obj.neighbors]
        obs = np.array([agent_obj.position] + neighbors_positions, dtype=np.float32)
        
        # 填充观测到最大观测大小
        if len(obs) < self.max_obs_size:
            padding = np.zeros(self.max_obs_size - len(obs))
            obs = np.concatenate([obs, padding])
        
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
    
    def step(self, action_dict):
        triggers = np.array([action_dict[agent] for agent in self.agents])
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
        done = self.current_iteration >= self.num_iterations

        rewards = {}
        if not done:
            average_position_difference = self.compute_average_position_difference()
            for agent in self.agents:
                if self.time_to_reach_epsilon is not None:
                    rewards[agent] = 20 if action_dict[agent] == 0 else -5  # 动作为0奖励，1惩罚
                else:
                    rewards[agent] = -20 - 5 * np.abs(average_position_difference)
        else:
            if self.time_to_reach_epsilon is not None:
                global_reward = 5000 - self.time_to_reach_epsilon - self.total_trigger_count
            else:
                global_reward = -5000
            for agent in self.agents:
                rewards[agent] = global_reward

        observations = {agent: self.get_observation(agent) for agent in self.agents}
        dones = {agent: done for agent in self.agents}
        dones["__all__"] = done
        infos = {agent: {} for agent in self.agents}
        
        return observations, rewards, dones, infos
    
    def render(self, mode='human'):
        positions = [agent.position for agent in self.agent_objs]
        print(f"Positions: {positions}")
    
    def observation_space(self, agent):
        return spaces.Box(low=-1.0, high=1.0, shape=(self.max_obs_size,), dtype=np.float32)
    
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

        def update_position(self, t, dt, trigger):
            if trigger == 1 or t == 0:
                self.u_i = -sum((self.last_broadcast_position - neighbor.last_broadcast_position) for neighbor in self.neighbors)
                self.position += self.u_i * dt
                self.last_broadcast_position = self.position
                self.trigger_points.append((t, self.position))
            else:
                self.position += self.u_i * dt
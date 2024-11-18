from sympy import AlgebraicField
from pettingzoo import ParallelEnv
import numpy as np
from gym import spaces


class CustomEnvironment(ParallelEnv):
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
    
    def get_observation(self, agent):
        agent_index = self.agent_name_mapping[agent]
        agent_obj = self.agent_objs[agent_index]
        # 获取自身和邻居的位置作为观测
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

        self.all_within_epsilon = all(
            abs(agent_i.position - agent_j.position) < self.epsilon
            for i, agent_i in enumerate(self.agent_objs)
            for j, agent_j in enumerate(self.agent_objs)
            if i < j
        )

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
        phase_cutoff = self.num_iterations * 0.25  # 前25%的迭代用于鼓励实现一致性

        for agent in self.agents:
            agent_index = self.agent_name_mapping[agent]
            agent_obj = self.agent_objs[agent_index]
            agent_action = actions[agent]

            # 计算与邻居的平均位置差距
            if agent_obj.neighbors:
                neighbor_positions = [neighbor.position for neighbor in agent_obj.neighbors]
                avg_neighbor_position = np.mean(neighbor_positions)
                position_difference = abs(agent_obj.position - avg_neighbor_position)
            else:
                position_difference = 0  # 如果没有邻居，则没有位置差

            # 奖励逻辑
            if not done:
                if self.current_iteration <= phase_cutoff:
                    # 前25%阶段主要鼓励一致性
                    if not self.all_within_epsilon:
                        # 没有实现一致性时，惩罚距离邻居较远的情况
                        rewards[agent] = -10 * position_difference
                    else:
                        # 达到一致性时，根据触发状态给予奖励或惩罚
                        if agent_action == 0:
                            rewards[agent] = 20  # 奖励未触发
                        else:
                            rewards[agent] = -5  # 轻微惩罚触发
                else:
                    # 后75%阶段鼓励减少触发
                    if not self.all_within_epsilon:
                        rewards[agent] = -5 * position_difference
                    else:
                        if agent_action == 0:
                            rewards[agent] = 20  # 奖励未触发
                        else:
                            rewards[agent] = -20  # 严重惩罚触发
            else:
                # 迭代结束时的奖励或惩罚
                if self.all_within_epsilon:
                    # 实现一致性
                    rewards[agent] = 200  # 给予较高的奖励
                else:
                    # 没有实现一致性
                    rewards[agent] = -1000  # 给予较大的惩罚

        observations = {agent: self.get_observation(agent) for agent in self.agents}
        dones = {agent: done for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        
        return observations, rewards, dones, infos

    
    def render(self, mode='human'):
        positions = [agent.position for agent in self.agent_objs]
        print(f"Positions: {positions}")
    
    def observation_space(self, agent):
        agent_index = self.agent_name_mapping[agent]
        num_neighbors = len(self.agent_objs[agent_index].neighbors)
        return spaces.Box(low=-1.0, high=1.0, shape=(1 + num_neighbors,), dtype=np.float32)
    
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
from sympy import AlgebraicField
from pettingzoo import ParallelEnv
import numpy as np
from gym import spaces
import numpy as np



class CustomMAEnvironment(ParallelEnv):
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
        
    # def init_neighbors(self):
    #     self.agent_objs[0].add_neighbor(self.agent_objs[1])
    #     self.agent_objs[0].add_neighbor(self.agent_objs[2])
    #     self.agent_objs[1].add_neighbor(self.agent_objs[2])
    #     self.agent_objs[2].add_neighbor(self.agent_objs[3])
    #     self.agent_objs[3].add_neighbor(self.agent_objs[4])

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
        # 获取每个智能体的状态，按其邻居数动态调整
        state = []
        for agent in self.agent_objs:
            state.append(agent.position)  # 当前智能体的位置
            for neighbor in agent.neighbors:
                state.append(neighbor.position)  # 邻居的位置信息
        return np.array(state)

    
    def get_observation(self, agent):
        """
        获取指定代理的动态观测，仅包含自身位置和邻居位置。
        """
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
        #print("env", actions)
        triggers = np.array([actions[agent] for agent in self.agents])
        trigger_count = np.sum(triggers)
        #print(triggers)
        #print(trigger_count)
        self.total_trigger_count += trigger_count
        #print(self.total_trigger_count)

        for i, agent in enumerate(self.agent_objs):
            agent.update_position(self.current_iteration, self.dt, triggers[i])

        #print("done")

        
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
        #print(self.current_iteration)
        done = self.current_iteration >= self.num_iterations
        #print(done)
        rewards = {}
        '''if not done:
            average_position_difference = self.compute_average_position_difference()
            for agent in self.agents:
                if self.all_within_epsilon and trigger_count == 0:
                    rewards[agent] = 10
                elif self.all_within_epsilon:
                    rewards[agent] = 5 - trigger_count
                else:
                    rewards[agent] = -20 * np.abs(average_position_difference) - trigger_count * 0.1 * self.num_iterations
        else:
            for agent in self.agents:
                if self.time_to_reach_epsilon is not None:
                    trigger_counts = sum(len([point for point in agent_obj.trigger_points if point[0] <= self.time_to_reach_epsilon]) for agent_obj in self.agent_objs)
                    rewards[agent] = 200 - self.time_to_reach_epsilon - trigger_counts
                else:
                    rewards[agent] = -2000'''
        
        if not done:
            average_position_difference = self.compute_average_position_difference()
            for agent in self.agents:
                if self.time_to_reach_epsilon is not None:
                    if actions[agent] == 1:
                        rewards[agent] = 0  # 动作为1，给予惩罚
                    else:
                        rewards[agent] = 10  # 动作为0，给予奖励
                    #rewards[agent] = 30 - 5 * trigger_count
                    #print("1", 50 - 5 * trigger_count)

                else:
                    # 计算当前智能体与所有其他智能体的平均位置之差
                    agent_index = self.agent_name_mapping[agent]
                    #print(agent_index)
                    agent_position = self.agent_objs[agent_index].position
                    other_positions = [other_agent.position for other_agent in self.agent_objs if other_agent != self.agent_objs[agent_index]]
                    average_position = np.mean(other_positions)
                    position_difference = abs(agent_position - average_position)
                    rewards[agent] = -5 -  20 * np.abs(position_difference) # 当前智能体与所有其他智能体的平均位置之差作为惩罚
                    #rewards[agent] =  - 5 - 5 * position_difference
                    #print(position_difference)
                '''else:
                    #print(average_position_difference)
                    # 将平均位置差的负值作为奖励，差值越小（智能体越接近），奖励越高
                    rewards[agent] = -8 - 5 * np.abs(average_position_difference)
                    #rewards[agent] = 0
                    #print("2", - 10 * np.abs(average_position_difference))'''
        else:
            #print("!!!!!!")
            if self.time_to_reach_epsilon is not None:
                trigger_counts = sum(len([point for point in agent.trigger_points if point[0] <= self.time_to_reach_epsilon]) for agent in self.agent_objs)
                global_reward = 1500 - self.time_to_reach_epsilon - self.total_trigger_count 
                #- self.total_trigger_count
                #global_reward = 1000
                #print(self.time_to_reach_epsilon)
                #print(2 * trigger_counts)
                #print(self.total_trigger_count)
                #print("1")
            else:
                global_reward = -2000
                #print("2")
            #self.total_trigger_count = 0
            
            for agent in self.agents:
                rewards[agent] = 0
                #rewards[agent] = 0

        observations = {agent: self.get_observation(agent) for agent in self.agents}
        dones = {agent: done for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        
        return observations, rewards, dones, infos

    
    def render(self, mode='human'):
        positions = [agent.position for agent in self.agent_objs]
        print(f"Positions: {positions}")
    
    # def observation_space(self, agent):
    #     """
    #     定义观测空间为动态范围（只要值在 -1.0 到 1.0 之间即可）。
    #     """
    #     return spaces.Box(low=-np.inf, high=np.inf, shape=(None,), dtype=np.float32)
    # def observation_space(self, agent):
    #     """
    #     返回最大可能的观测空间，适配固定 Box。
    #     """
    #     max_possible_neighbors = len(self.agent_objs) - 1  # 所有其他代理
    #     max_obs_size = max_possible_neighbors + 1  # 包括自身位置
    #     return spaces.Box(low=-100, high=100, shape=(max_obs_size,), dtype=np.float32)

    def observation_space(self, agent):
        """
        为指定的智能体动态生成观测空间，适配其实际邻居数量。
        """
        agent_index = self.agent_name_mapping[agent]
        agent_obj = self.agent_objs[agent_index]
        obs_size = 1 + len(agent_obj.neighbors)  # 包括自身位置 + 邻居位置
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
                #print("trigger")
                self.u_i = -sum((self.last_broadcast_position - neighbor.last_broadcast_position) for neighbor in self.neighbors if self.is_neighbor(neighbor))
                self.position += self.u_i * dt
                self.last_broadcast_position = self.position
                self.trigger_points.append((t, self.position))
            else:
                #print("not trigger")
                self.position += self.u_i * dt



env = CustomMAEnvironment()

# 打印每个智能体的观测空间形状
for agent in env.agents:
    obs_space = env.observation_space(agent)
    print(f"Agent {agent}: Observation space shape = {obs_space.shape}")

print(env.agents)
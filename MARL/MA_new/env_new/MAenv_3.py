from sympy import AlgebraicField
from pettingzoo import ParallelEnv
import numpy as np
from gym import spaces
import numpy as np



class CustomMAEnvironment3(ParallelEnv):
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
    #     # 将所有智能体两两相连形成全连接邻居关系，方便一致性控制
    #     # 若不需要全连接，可根据实际需求修改
    #     for i in range(len(self.agent_objs)):
    #         for j in range(i+1, len(self.agent_objs)):
    #             self.agent_objs[i].add_neighbor(self.agent_objs[j])

    def init_neighbors(self):
        self.agent_objs[0].add_neighbor(self.agent_objs[1])
        self.agent_objs[0].add_neighbor(self.agent_objs[2])
        self.agent_objs[1].add_neighbor(self.agent_objs[2])
        self.agent_objs[2].add_neighbor(self.agent_objs[3])
        self.agent_objs[3].add_neighbor(self.agent_objs[4])

    # def init_neighbors(self):
    #     self.agent_objs[0].add_neighbor(self.agent_objs[1])
    #     self.agent_objs[0].add_neighbor(self.agent_objs[2])
    #     self.agent_objs[0].add_neighbor(self.agent_objs[3])
    #     self.agent_objs[0].add_neighbor(self.agent_objs[4])
    #     self.agent_objs[1].add_neighbor(self.agent_objs[2])
    #     self.agent_objs[1].add_neighbor(self.agent_objs[3])
    #     self.agent_objs[1].add_neighbor(self.agent_objs[4])
    #     self.agent_objs[2].add_neighbor(self.agent_objs[3])
    #     self.agent_objs[2].add_neighbor(self.agent_objs[4])
    #     self.agent_objs[3].add_neighbor(self.agent_objs[4])

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
        # 获取每个智能体的状态
        state = []
        for agent in self.agent_objs:
            state.append(agent.position)
            for neighbor in agent.neighbors:
                state.append(neighbor.position)
        return np.array(state)

    
    def get_observation(self, agent):
        """
        获取指定代理的动态观测，包括自身和邻居位置。
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
        triggers = np.array([actions[agent] for agent in self.agents])
        trigger_count = np.sum(triggers)
        self.total_trigger_count += trigger_count

        # 更新智能体位置
        for i, agent in enumerate(self.agent_objs):
            agent.update_position(self.current_iteration, self.dt, triggers[i])

        # 检查是否达到一致性（所有智能体间的差值都小于 epsilon）
        self.all_within_epsilon = all(all(abs(agent.position - neighbor.position) < self.epsilon for neighbor in agent.neighbors) for agent in self.agent_objs)

        if self.all_within_epsilon:
            if self.epsilon_violated:
                # 第一次达到一致性时刻
                self.time_to_reach_epsilon = self.current_iteration
                self.epsilon_violated = False
        else:
            self.epsilon_violated = True
            self.time_to_reach_epsilon = None

        self.current_iteration += 1
        done = self.current_iteration >= self.num_iterations

        rewards = {}
        
        if done:
            # 回合结束
            if self.time_to_reach_epsilon is not None:
                # 成功达成一致性
                # 全局奖励基于收敛时间和触发总数
                # 越早达成一致性、触发越少，奖励越高
                for agent in self.agents:
                    agent_index = self.agent_name_mapping[agent]
                    agent_obj = self.agent_objs[agent_index]
                    individual_trigger_count = len(agent_obj.trigger_points)
                    rewards[agent] = 300 - self.time_to_reach_epsilon - individual_trigger_count * 2
            else:
                # 未能达成一致性
                # 给一个大的负向奖励
                for agent in self.agents:
                    rewards[agent] = -600
        else:
            # 回合进行中
            for agent in self.agents:
                rewards[agent] = 0
        
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
                # 触发更新控制输入 u_i
                self.u_i = -sum((self.last_broadcast_position - neighbor.last_broadcast_position) for neighbor in self.neighbors if self.is_neighbor(neighbor))
                self.position += self.u_i * dt
                self.last_broadcast_position = self.position
                self.trigger_points.append((t, self.position))
            else:
                # 不触发则根据上一次的输入继续演化
                self.position += self.u_i * dt


# 以下为简单测试代码（不属于环境本身的一部分）

env = CustomMAEnvironment3()
for agent in env.agents:
    obs_space = env.observation_space(agent)
    print(f"Agent {agent}: Observation space shape = {obs_space.shape}")

print(env.agents)

from pettingzoo import ParallelEnv
import numpy as np
from gym import spaces


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
        
        self.initial_positions = [0.55, 0.4, -0.05, -0.1, -0.7]
        self.agent_objs = [self.Agent(pos, i) for i, pos in enumerate(self.initial_positions)]
        self.init_neighbors()
        
        self.epsilon = 0.005
        self.time_to_reach_epsilon = None
        self.epsilon_violated = True
        self.all_within_epsilon = False
        self.total_trigger_count = 0
        self.time_to_reach_epsilon_changes = 0
        
        # 计算最大邻居数量
        self.max_neighbors = max(len(agent.neighbors) for agent in self.agent_objs)
        # 观测维度：pos部分 + 掩码部分
        # pos部分 = 1(自己) + max_neighbors(邻居)；mask部分同样长度
        self.obs_dim = 2 * (1 + self.max_neighbors)
    
    # 根据需要修改邻居结构，如下只示范局部连接
    def init_neighbors(self):
        self.agent_objs[0].add_neighbor(self.agent_objs[1])
        self.agent_objs[0].add_neighbor(self.agent_objs[2])
        self.agent_objs[1].add_neighbor(self.agent_objs[2])
        self.agent_objs[2].add_neighbor(self.agent_objs[3])
        self.agent_objs[3].add_neighbor(self.agent_objs[4])

    def reset(self, seed=None, options=None):
        self.initial_positions = [0.55, 0.4, -0.05, -0.1, -0.7]
        # 如果想随机初始位置，可以使用：
        # self.initial_positions = np.round(np.random.uniform(-1, 1, size=5), 2)
        
        self.agent_objs = [self.Agent(pos, i) for i, pos in enumerate(self.initial_positions)]
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
        """
        生成“统一维度 + 掩码”的观测:
        - 前 half: 自己位置 + 邻居位置(若不足max_neighbors则补0)
        - 后 half: mask标识(1表示该位置存在邻居信息，0表示不存在)
        """
        agent_index = self.agent_name_mapping[agent]
        agent_obj = self.agent_objs[agent_index]

        # 1. 收集自身与邻居的位置
        positions = [agent_obj.position]
        for neighbor in agent_obj.neighbors:
            positions.append(neighbor.position)
        # 如果邻居数 < max_neighbors，则用 0 填充
        while len(positions) < 1 + self.max_neighbors:
            positions.append(0.0)

        # 2. 构建mask向量（与positions长度相同）
        #   自己这一个位置一定是1（可以表示为有效项），邻居有几个就在前几个位置写1，剩下写0
        mask = [1] * (1 + len(agent_obj.neighbors))
        while len(mask) < 1 + self.max_neighbors:
            mask.append(0)
        
        # 3. 整合 positions + mask
        obs_array = np.array(positions + mask, dtype=np.float32)
        return obs_array

    def step(self, actions):
        triggers = np.array([actions[agent] for agent in self.agents])
        trigger_count = np.sum(triggers)
        self.total_trigger_count += trigger_count

        # 更新每个智能体的位置
        for i, agent in enumerate(self.agent_objs):
            agent.update_position(self.current_iteration, self.dt, triggers[i])

        average_difference = self.compute_average_position_difference()

        # 判断收敛情况
        self.all_within_epsilon = all(all(abs(agent.position - neighbor.position) < self.epsilon 
                                          for neighbor in agent.neighbors) for agent in self.agent_objs)

        if self.all_within_epsilon:
            if self.epsilon_violated:
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
                for agent in self.agents:
                    agent_index = self.agent_name_mapping[agent]
                    agent_obj = self.agent_objs[agent_index]
                    individual_trigger_count = len(agent_obj.trigger_points)
                    rewards[agent] = 1000 - self.time_to_reach_epsilon - individual_trigger_count * 2
            else:
                # 未能达成一致性
                for agent in self.agents:
                    rewards[agent] = -800
        else:
            # 回合中
            threshold = 1.0 * self.epsilon
            time_penalty_factor = 0.01 * self.current_iteration
            for i, agent in enumerate(self.agents):
                if average_difference > threshold:
                    # 差距大，鼓励更多触发(此处给触发和不触发都为0作为示例)
                    rewards[agent] = 0
                else:
                    # 差距小，鼓励减少触发
                    if triggers[i] == 1:
                        rewards[agent] = -(1.0 + time_penalty_factor * 3)
                    else:
                        rewards[agent] = 1

        observations = {agent: self.get_observation(agent) for agent in self.agents}
        dones = {agent: done for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        return observations, rewards, dones, infos

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
    
    def render(self, mode='human'):
        positions = [agent.position for agent in self.agent_objs]
        print(f"Positions: {positions}")
    
    def observation_space(self, agent):
        """
        统一观测维度: 2 * (1 + max_neighbors)
        """
        return spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.obs_dim,), 
            dtype=np.float32
        )
    
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
                self.u_i = -sum((self.last_broadcast_position - neighbor.last_broadcast_position) 
                                for neighbor in self.neighbors if self.is_neighbor(neighbor))
                self.position += self.u_i * dt
                self.last_broadcast_position = self.position
                self.trigger_points.append((t, self.position))
            else:
                self.position += self.u_i * dt


# 简单测试
if __name__ == "__main__":
    env = CustomMAEnvironment()
    for agent in env.agents:
        obs_space = env.observation_space(agent)
        print(f"Agent {agent}: Observation space shape = {obs_space.shape}")
    
    # 检查一下reset返回的观测维度是否吻合
    obs = env.reset()
    for agent, ob in obs.items():
        print(agent, ob, ob.shape)

    # 简单走一步
    test_actions = {agent: 1 for agent in env.agents}
    obs, rew, dones, infos = env.step(test_actions)
    print("Step result:")
    for agent in env.agents:
        print(agent, "obs:", obs[agent], "reward:", rew[agent], "done:", dones[agent])

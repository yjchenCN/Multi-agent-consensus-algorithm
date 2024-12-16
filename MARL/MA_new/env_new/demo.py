import numpy as np
from pettingzoo import ParallelEnv
from gym import spaces


class SimpleMultiAgentEnv(ParallelEnv):
    metadata = {
        'render.modes': ['human'],
        'name': 'simple_multi_agent_env'
    }

    def __init__(self, num_agents=3, num_iterations=100, dt=0.1):
        self.agents = [f"agent_{i}" for i in range(num_agents)]
        self.num_iterations = num_iterations
        self.dt = dt
        self.current_iteration = 0

        # 初始化智能体位置
        self.agent_positions = {agent: np.random.uniform(-1.0, 1.0) for agent in self.agents}
        self.max_position = 1.0
        self.min_position = -1.0

        # 初始化状态维度
        self.state_dim = 1

        # 初始化奖励
        self.reward = 0
        self.done = False

        # 初始化邻居关系 (简单起见，所有智能体都是邻居)
        self.neighbors = {agent: [a for a in self.agents if a != agent] for agent in self.agents}

    def reset(self, seed=None, options=None):
        """ 重置环境，初始化状态 """
        self.current_iteration = 0
        self.agent_positions = {agent: np.random.uniform(-1.0, 1.0) for agent in self.agents}
        self.done = False
        observations = {agent: self.get_observation(agent) for agent in self.agents}
        return observations

    def get_observation(self, agent):
        """ 返回智能体的观察（当前位置和其邻居位置） """
        agent_pos = self.agent_positions[agent]
        neighbor_positions = [self.agent_positions[neighbor] for neighbor in self.neighbors[agent]]
        return np.array([agent_pos] + neighbor_positions, dtype=np.float32)

    def get_state(self):
        """ 获取所有智能体的状态 """
        # 返回所有智能体的位置作为状态
        state = np.array([self.agent_positions[agent] for agent in self.agents], dtype=np.float32)
        return state

    def step(self, actions):
        """ 执行每个智能体的动作并返回下一状态、奖励、是否结束等信息 """
        for agent in self.agents:
            # 根据动作更新智能体的位置（这里简单做了个线性更新）
            if actions[agent] == 1:  # 假设动作1表示移动，0表示不动
                self.agent_positions[agent] += np.random.uniform(-0.1, 0.1)

            # 确保智能体的位置不会超过边界
            self.agent_positions[agent] = np.clip(self.agent_positions[agent], self.min_position, self.max_position)

        # 计算奖励：每个智能体尽量接近所有其他智能体
        rewards = {}
        for agent in self.agents:
            avg_distance = np.mean([abs(self.agent_positions[agent] - self.agent_positions[neighbor]) for neighbor in self.neighbors[agent]])
            rewards[agent] = -avg_distance  # 惩罚越远，奖励越小

        # 检查是否结束
        self.current_iteration += 1
        if self.current_iteration >= self.num_iterations:
            self.done = True

        observations = {agent: self.get_observation(agent) for agent in self.agents}
        dones = {agent: self.done for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        return observations, rewards, dones, infos

    def render(self, mode='human'):
        """ 输出智能体的位置 """
        print(f"Iteration {self.current_iteration}:")
        positions = {agent: self.agent_positions[agent] for agent in self.agents}
        print(positions)

    def observation_space(self, agent):
        """ 返回每个智能体的观测空间大小 """
        # 每个智能体的状态由自己和它的所有邻居的位置信息组成
        return spaces.Box(low=-1.0, high=1.0, shape=(1 + len(self.neighbors[agent]),), dtype=np.float32)

    def action_space(self, agent):
        """ 返回每个智能体的动作空间大小（0：不动，1：移动） """
        return spaces.Discrete(2)


# 测试环境是否正确运行
if __name__ == "__main__":
    env = SimpleMultiAgentEnv(num_agents=3, num_iterations=10)

    # 打印每个智能体的观测空间形状
    for agent in env.agents:
        obs_space = env.observation_space(agent)
        print(f"Agent {agent}: Observation space shape = {obs_space.shape}")

    # 测试环境循环
    done = False
    obs = env.reset()
    while not done:
        actions = {agent: np.random.choice([0, 1]) for agent in env.agents}  # 随机选择动作
        obs, rewards, done, infos = env.step(actions)
        env.render()
        print(f"Rewards: {rewards}")

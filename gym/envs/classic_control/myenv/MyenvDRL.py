import numpy as np
import gym
from gym import spaces


class Consensus(gym.Env):
    """
    群体智能体环境，模拟智能体根据邻居位置的相对距离更新自己的位置
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, num_agents=4, num_iterations=150, dt=0.1):
        super(Consensus, self).__init__()

        self.num_agents = num_agents
        self.num_iterations = num_iterations
        self.dt = dt
        self.agents = [self.Agent(np.random.rand() * 10, i) for i in range(self.num_agents)]
        self.define_neighbors()

        # 定义观测空间和动作空间
        # 观测空间为所有智能体的位置
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_agents,), dtype=np.float32)
        # 动作空间为智能体的位置更新，这里简化为每个智能体的位移
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.num_agents,), dtype=np.float32)

    class Agent:
        def __init__(self, initial_position, index):
            self.position = initial_position
            self.index = index
            self.neighbors = []
            self.last_broadcast_position = self.position  # 存储了该智能体最近一次广播的位置
            self.sigma = np.random.uniform(0, 1)
            self.alpha = np.random.uniform(0, 1 / max(1, len(self.neighbors)))  # 另一个触发函数参数
            self.trigger_points = []
            self.trigger_points2 = []  # 用于记录事件触发时的位置
            self.delta_positions = []
            self.communication_times = []  # 记录每次通信的时间

    def define_neighbors(self):
        # 定义邻居关系
        self.agents[0].add_neighbor(self.agents[1])
        self.agents[0].add_neighbor(self.agents[2])
        self.agents[0].add_neighbor(self.agents[3])
        self.agents[2].add_neighbor(self.agents[3])
        self.agents[1].add_neighbor(self.agents[3])
        self.agents[2].add_neighbor(self.agents[1])

    def step(self, action):
        # 应用动作更新智能体位置
        for i, agent in enumerate(self.agents):
            agent.update_position_based_on_action(action[i], self.dt)

        # 计算新的观测和奖励
        new_obs = np.array([agent.position for agent in self.agents])
        reward = self.calculate_reward(new_obs)  # 定义奖励计算方法
        done = self.is_done()  # 定义终止条件

        return new_obs, reward, done, {}


    def reset(self):
        # 重置环境
        self.agents = [self.Agent(np.random.rand() * 10, i) for i in range(self.num_agents)]
        self.define_neighbors()
        return np.array([agent.position for agent in self.agents])

    def render(self, mode='human'):
        # 渲染环境
        pass



import numpy as np
import itertools
import random
import gym
from gym import spaces
from gym.utils import seeding

class MultiAgentEnv(gym.Env):
    metadata = {
        'render.modes': ['human'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        self.num_agents = 4
        self.N = self.num_agents
        self.M = 7
        self.forms = list(itertools.permutations(range(self.N, 0, -1)))
        self.formaction = np.array(self.forms).T
        self.numAct = self.formaction.shape[1]

        self.action_space = spaces.Tuple([spaces.Discrete(self.numAct) for _ in range(self.num_agents)])
        self.observation_space = spaces.Tuple([
            spaces.Box(low=np.array([0.0, 0.0, 0.0, 0.0, 0], dtype=np.float32),
                       high=np.array([1.0, 1.0, 1.0, 1.0, 7], dtype=np.float32), shape=(5,))
            for _ in range(self.num_agents)
        ])

        self.state = None
        self.SOC = np.zeros(self.num_agents)
        self.remRsq = None
        self.col = None
        self.Delta = np.array([[0.1302, 0.1334, 0.2522, 0.1868, 0.0787],
                               [0.1224, 0.1255, 0.2372, 0.1756, 0.0741],
                               [0.1170, 0.1199, 0.2266, 0.1678, 0.0708],
                               [0.1170, 0.1199, 0.2266, 0.1678, 0.0708]])
        self.seed()
        self.viewer = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.state = np.array([1, 1, 1, 1, 7])
        return tuple(self.state for _ in range(self.num_agents))

    def clcSM(self):
        SM = self.SOC.copy()
        SC = np.zeros(4)
        for indEV in range(self.N):
            X = SM[indEV]
            indices = np.where(self.form == indEV + 1)[0]
            if indices.size > 0 and 0 <= indices[0] < len(self.Delta) and 0 <= self.col - 1 < len(self.Delta[0]):
                Y = self.Delta[indices[0]][self.col - 1]
            else:
                Y = self.Delta[int(indices)][-1]  
            SC[indEV] = np.squeeze(np.sum(X) - np.sum(Y))
        return SC

    def calculate_efficiency(state, new_state):
        # 计算车辆之间的位置差
        position_diff = np.abs(new_state[:4] - state[:4])

        # 计算编队紧密度指标，利用位置差的均值
        efficiency = np.mean(position_diff)

        return efficiency


    def step(self, actions):
        '''for action in actions:
            assert self.action_space.contains(action), f"{action} is an invalid action"'''
        if not isinstance(actions, list):
            actions = [actions]

        # 处理每个智能体的动作
        for i, action in enumerate(actions):
            # 检查动作是否有效，如果无效，则随机选择一个有效动作
            if not self.action_space[i].contains(action):
                actions[i] = self.action_space[i].sample()

        
        self.remRsq = self.state[-1]
        self.col = self.M - self.remRsq + 1

        
        self.form = np.column_stack([self.formaction[:, action] for action in actions])
        self.SOC = self.clcSM()

        
        self.remRsq = max(self.remRsq - 1, 0)
        self.state = np.concatenate((self.SOC, [self.remRsq]))

        
        rewards = self.calculate_rewards()
        done = np.any(self.SOC <= 0)
        dones = [done for _ in range(self.num_agents)]
        infos = [{} for _ in range(self.num_agents)]

        observations = tuple(self.state for _ in range(self.num_agents))
        return observations, rewards, dones, infos

    def calculate_rewards(self):
        # 计算距离最后位置的距离
        distances = np.abs(self.SOC)
                
        # 计算惩罚，距离越远惩罚越大
        penalty = -np.sum(distances)

        position_changes = np.sum(np.abs(self.state[:4] - self.form))  # 使用self.form表示新状态
        position_change_penalty = -position_changes  # 负惩罚，越少位置变化越好
        
        fuel_efficiency_reward = self.calculate_efficiency(self.state, self.form)

        team_reward = np.std(self.SOC) + penalty + position_change_penalty + fuel_efficiency_reward

        
        individual_rewards = self.SOC

        
        total_rewards = [team_reward + individual_reward for individual_reward in individual_rewards]
        return total_rewards

    def render(self, mode='human'):
        
        pass

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


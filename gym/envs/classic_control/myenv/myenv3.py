from typing import Optional, Union
import numpy as np
import random
from gym.error import DependencyNotInstalled
import itertools
import math
import gym
from gym import spaces, logger
from gym.utils import seeding

class MyEnv3(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        self.N = 4
        self.M = 7  # 修改为7个地点
        self.forms = list(itertools.permutations(range(self.N, 0, -1)))  # 4辆车的全排列 所有可能
        self.formaction = np.array(self.forms).T  # 写成矩阵的形式
        self.numAct = self.formaction.shape[1]  # 转置矩阵的列数，即N的全排列的所有可能数
        self.state = None
        self.SOC = np.zeros(4)
        self.SC = np.zeros(4)
        self.remRsq = None  # 所剩余的可重排序地点数
        self.col = None  # col为当下处于第几个重排序地点
        self.Action = random.choice(range(self.formaction.shape[1]))  # 选择一个随机的列索引(随机选择一个动作)
        self.form = None
        self.fifth_order_vector = None
        self.Delta = np.array([[0.1302, 0.1334, 0.2522, 0.1868, 0.0787],
                               [0.1224, 0.1255, 0.2372, 0.1756, 0.0741],
                               [0.1170, 0.1199, 0.2266, 0.1678, 0.0708],
                               [0.1170, 0.1199, 0.2266, 0.1678, 0.0708]])
        # 定义观测空间和动作空间
        self.action_space = spaces.Discrete(self.numAct)
        self.observation_space = spaces.Box(low=np.array([0.0, 0.0, 0.0, 0.0, 0], dtype=np.float32),
                                            high=np.array([1.0, 1.0, 1.0, 1.0, 7], dtype=np.float32), shape=(5,))
        self.seed()
        self.viewer = None
        self.state = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    


    # 未到达终点前SOC的计算函数
    def clcSM(self):
        SM = self.SOC.copy()
        SC = np.zeros(4)
        
        for indEV in range(self.N):
            X = SM[indEV]
            indices = np.where(self.form == indEV + 1)[0]

            if indices and 0 <= int(indices) < len(self.Delta) and 0 <= int(self.col) - 1 < len(self.Delta[0]):
                Y = self.Delta[int(indices)][int(self.col) - 1]
            elif self.col <= 5:
                # 如果索引越界且在前五次位置排序，按照前一次的SOC情况排序
                order_indices = np.argsort(self.SOC) + 1
                Y = self.Delta[:, order_indices[-1] - 1]
            else:
                # 在第五次排序后，一直使用第五次排序的 Y 值
                if self.fifth_order_vector is None:
                    # 如果第五次排序的向量尚未记录，则记录
                    sorted_indices = np.argsort(self.SOC) + 1
                    self.fifth_order_vector = self.Delta[:, sorted_indices[-1] - 1]

                Y = self.fifth_order_vector

            SC[indEV % 4] = np.squeeze(np.sum(X) - np.sum(Y))
        
        return SC




    # 最后一个位置进行SOC排序
    def socOrderForm(self):
        sorted_indices = np.argsort(self.SOC) + 1
        sorted_SOC = self.SOC[sorted_indices - 1]
        return sorted_SOC
    

    def calculate_efficiency(state, new_state):
        # 计算车辆之间的位置差
        position_diff = np.abs(new_state[:4] - state[:4])

        # 计算编队紧密度指标，利用位置差的均值
        efficiency = np.mean(position_diff)

        return efficiency


    def step(self, action):
        # 检查动作的有效性
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg
        self.form = self.formaction[:, action]  # 提取当前动作索引所对应的向量
        self.SOC = self.state[:self.N]
        self.remRsq = self.state[self.N]
        self.col = self.M - self.remRsq + 1
        self.SOC = self.clcSM()  # 计算soc
        self.remRsq = self.remRsq - 1
        self.state = np.array([])
        self.state = np.concatenate((self.SOC, [self.remRsq]))

        # 计算位置变化次数的惩罚
        position_changes = np.sum(np.abs(self.state[:4] - self.form))  # 使用self.form表示新状态
        position_change_penalty = -position_changes  # 负惩罚，越少位置变化越好

        # 计算燃油节省效率奖励，例如，编队紧密度
        fuel_efficiency_reward = self.calculate_efficiency(self.state, self.form)

        done = bool(self.remRsq == 1 or any(self.SOC <= 0))  # 终止条件修改为SOC为零或任一SOC小于等于零
        if not done:
            reward = 0.0
        else:
            form_M = self.socOrderForm()
            SOC_M = self.clcSM()
            standard_deviation = np.std(SOC_M)
            if any(self.SOC <= 0):
                # 计算距离最后位置的距离
                distances = np.abs(self.SOC)
                
                # 计算惩罚，距离越远惩罚越大
                penalty = -np.sum(distances)
                
                # 设置最终奖励
                reward = standard_deviation + penalty + position_change_penalty + fuel_efficiency_reward
                
            else:
                # 如果所有车的SOC都大于零，奖励为SOC的标准差
                reward = standard_deviation
        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = np.array([1, 1, 1, 1, 7])  # 修改为初始SOC为1，可重排序地点为7
        return np.array(self.state)

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def render(self, mode='human'):
        pass
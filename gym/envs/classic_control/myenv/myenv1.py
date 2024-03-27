from typing import Optional, Union
import numpy as np
import random
#from gym.envs.classic_control import utils
from gym.error import DependencyNotInstalled
import itertools
import math
import gym
from gym import spaces, logger
from gym.utils import seeding

class MyEnv1(gym.Env):
    
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):

        self.N = 4
        self.M = 5
        self.forms = list(itertools.permutations(range(self.N, 0, -1)))  # 4辆车的全排列 所有可能
        self.formaction = np.array(self.forms).T  # 写成矩阵的形式
        self.numAct = self.formaction.shape[1]  # 转置矩阵的列数，即N的全排列的所有可能数
        self.state = None

        self.SOC = np.zeros(4)
        self.SC=np.zeros(4)
        self.remRsq = None # 所剩余的可重排序地点数
        self.col = None  # col为当下处于第几个重排序地点



        self.Action = random.choice(range(self.formaction.shape[1]))  # 选择一个随机的列索引(随机选择一个动作)
        self.form = None

        self.Delta = np.array([[0.1302, 0.1334, 0.2522, 0.1868, 0.0787],
                        [0.1224, 0.1255, 0.2372, 0.1756, 0.0741],
                        [0.1170, 0.1199, 0.2266, 0.1678, 0.0708],
                        [0.1170, 0.1199, 0.2266, 0.1678, 0.0708]])
        # 定义观测空间和动作空间
        self.action_space = spaces.Discrete(self.numAct)
        self.observation_space = spaces.Box(low=np.array([0.0, 0.0, 0.0, 0.0, 0], dtype=np.float32), high=np.array([1.0, 1.0, 1.0, 1.0, 10], dtype=np.float32), shape=(5,))


        self.seed()
        self.viewer = None
        self.state = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # 未到达终点前SOC的计算函数
    def clcSM(self):
        SM = self.SOC.copy()
        SC=np.zeros(4)
        for indRsq in range(1):
            for indEV in range(self.N):
                X = SM[indEV]
                indices = np.where(self.form == indEV + 1)[0]
                Y = self.Delta[int(indices) ][int(self.col)-1]
                SC[indEV%4]=np.squeeze(X-Y)
        #print("sm",SC)
        return SC

    # 最后一个位置进行SOC排序
    def socOrderForm(self):
        sorted_indices = np.argsort(self.SOC)+1
        sorted_SOC = self.SOC[sorted_indices-1]

        return sorted_SOC

    def step(self, action):
        #检查动作的有效性
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        self.form=self.formaction[:,action]  #提取当前动作索引所对应的向量
        #print(self.form)
        self.SOC=self.state[:self.N]
        self.remRsq=self.state[self.N]
        self.col=self.M-self.remRsq+1

        self.SOC=self.clcSM()  #计算soc
        #print("SOC",self.SOC)
        self.remRsq=self.remRsq-1

        self.state=np.array([])
        self.state=np.concatenate((self.SOC,[self.remRsq]))

        done=bool(self.remRsq==1)
        if not done:
            reward=0.0
        else:
            form_M=self.socOrderForm()
            SOC_M=self.clcSM()
            standard_deviation = np.std(SOC_M)
            #print("standard_deviation",standard_deviation)
            #print(SOC_M)
            reward=-1000*np.std(SOC_M,axis=0)

        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = np.array([1, 1, 1, 1, 5])
        return np.array(self.state)

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def render(self, mode='human'):
        pass
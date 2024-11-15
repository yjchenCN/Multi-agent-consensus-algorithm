import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random

class DistributedConsensusEnv(gym.Env):
    metadata = {
        'render_modes': ['human']
    }

    def __init__(self, num_iterations=20000, dt=0.001, max_neighbors=4):
        self.num_iterations = num_iterations  # 最大迭代次数
        self.dt = dt  # 时间步长
        self.current_iteration = 0  # 当前迭代步数
        self.max_neighbors = max_neighbors  # 每个智能体的最大邻居数量

        # 动作空间：0表示不触发更新，1表示触发更新
        self.action_space = spaces.Discrete(2)

        # 观测空间：当前智能体的位置和邻居的标准化相对位置差值
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(max_neighbors + 1,), dtype=np.float32)

        # 创建要训练的主智能体
        self.main_agent = self.Agent(0, initial_position=np.random.uniform(-1, 1))
        self.agents = [self.main_agent]  # 主智能体列表

        # 参数设置
        self.c_0 = 0.00068
        self.c_1 = 1.2
        self.alpha = 1.4

    def reset(self, seed=None, options=None):
        """重置环境并创建五个智能体，随机选择一个作为主智能体，确保没有孤立节点并生成稀疏图"""
        super().reset(seed=seed)
        self.current_iteration = 0
        self.total_trigger_count = 0

        # 创建五个智能体并确定初始位置
        self.agents = [self.Agent(i, initial_position=round(np.random.uniform(-1, 1), 2)) for i in range(5)]

        # 随机选择一个主智能体
        self.main_agent = random.choice(self.agents)

        # 确保没有孤立节点，构建稀疏图
        for agent in self.agents:
            potential_neighbors = [other_agent for other_agent in self.agents if other_agent != agent]

            # 确保当前节点至少有一个邻居
            if not agent.neighbors:
                chosen_neighbor = random.choice(potential_neighbors)
                agent.neighbors.append(chosen_neighbor)
                chosen_neighbor.neighbors.append(agent)

            # 随机分配额外的邻居，但限制总数以生成稀疏图
            num_neighbors = random.randint(0, min(self.max_neighbors // 2, len(potential_neighbors)))  # 控制最大邻居数量，生成稀疏图
            additional_neighbors = random.sample(potential_neighbors, num_neighbors)
            for neighbor in additional_neighbors:
                if neighbor not in agent.neighbors:
                    agent.neighbors.append(neighbor)
                    neighbor.neighbors.append(agent)

        return self.get_observation(0), {}

    def get_observation(self, agent_index=0):
        """生成主智能体的观测值，包括自身位置和邻居的标准化相对位置差值"""
        agent = self.main_agent
        obs = [agent.position]

        if agent.neighbors:
            neighbor_differences = [(neighbor.position - agent.position) for neighbor in agent.neighbors]
            neighbor_differences = [np.clip(diff, -1.0, 1.0) for diff in neighbor_differences]
            obs.extend(neighbor_differences)

        obs.extend([0.0] * (self.max_neighbors - len(agent.neighbors)))

        return np.array(obs, dtype=np.float32)
    
#邻居也会一起更新
    #def step(self, action):
        """根据动作更新环境，所有智能体进行更新"""
        main_agent = self.main_agent
        if self.current_iteration == 0:
            main_agent.update_position_1(self.dt, trigger=True)
            for agent in self.agents:
                if agent != main_agent:
                    agent.update_position_formula_with_hold_1(self.dt, self.c_0, self.c_1, self.alpha, self.current_iteration)
            self.total_trigger_count += 1
            reward = 0
        else:
            position_difference = abs(main_agent.position - main_agent.last_broadcast_position)
            time_scaling = np.exp(-self.alpha * self.current_iteration * self.dt)
            threshold = self.c_0 + self.c_1 * time_scaling

            if action == 1:
                self.total_trigger_count += 1
                main_agent.update_position(self.dt, trigger=True)
                reward = 2 if position_difference >= threshold else 0
            else:
                main_agent.update_position(self.dt, trigger=False)
                reward = 2 if position_difference < threshold else 0

            # 所有邻居根据公式或零阶保持器进行更新
            for agent in self.agents:
                if agent != main_agent:
                    agent.update_position_formula_with_hold(self.dt, self.c_0, self.c_1, self.alpha, self.current_iteration)
                   

        self.current_iteration += 1
        done = self.current_iteration >= self.num_iterations

        if done:
            self.t = self.total_trigger_count

        return self.get_observation(0), reward, done, False, {}


#随机邻居位置
    #def step(self, action):
        """根据动作更新环境，仅更新当前主智能体，其他智能体在模拟中被遍历决策"""
        main_agent = self.main_agent
        if self.current_iteration == 0:
            main_agent.update_position_1(self.dt, trigger=True)
            # for agent in self.agents:
            #     if agent != main_agent:
            #         agent.update_position_formula_with_hold_1(self.dt, self.c_0, self.c_1, self.alpha, self.current_iteration)
            self.total_trigger_count += 1
            reward = 0
        else:
            position_difference = abs(main_agent.position - main_agent.last_broadcast_position)
            time_scaling = np.exp(-self.alpha * self.current_iteration * self.dt)
            threshold = self.c_0 + self.c_1 * time_scaling

            if action == 1:
                self.total_trigger_count += 1
                main_agent.update_position(self.dt, trigger=True)
                reward = 2 if position_difference >= threshold else 0
            else:
                main_agent.update_position(self.dt, trigger=False)
                reward = 2 if position_difference < threshold else 0

            # 让非主智能体以逐渐降低的概率随机变换 u_i 并更新位置
            for agent in self.agents:
                if agent != main_agent:
                    # 随着时间步数增加，概率逐渐降低
                    #change_probability = max(0.1, 1.0 - 0.01 * self.current_iteration)  # 这里概率会随着时间步数逐渐减少，最低为 0.1
                    if random.random() < 0.01 * 0.0005 * (self.num_iterations - self.current_iteration + 1):
                        # 随机更新控制率 u_i
                        agent.u_i = np.random.uniform(-2.0, 2.0)
                    
                    # 通过控制率更新位置
                    agent.position += agent.u_i * self.dt

        self.current_iteration += 1
        done = self.current_iteration >= self.num_iterations

        if done:
            self.t = self.total_trigger_count

        return self.get_observation(0), reward, done, False, {}


#邻居不更新
    def step(self, action):
        """根据动作更新环境，仅更新当前主智能体，其他智能体在模拟中被遍历决策"""
        main_agent = self.main_agent
        if self.current_iteration == 0:
            main_agent.update_position_1(self.dt, trigger=True)
            # for agent in self.agents:
            #     if agent != main_agent:
            #         agent.update_position_formula_with_hold_1(self.dt, self.c_0, self.c_1, self.alpha, self.current_iteration)
            self.total_trigger_count += 1
            reward = 0
        else:
            position_difference = abs(main_agent.position - main_agent.last_broadcast_position)
            time_scaling = np.exp(-self.alpha * self.current_iteration * self.dt)
            threshold = self.c_0 + self.c_1 * time_scaling

            if action == 1:
                self.total_trigger_count += 1
                main_agent.update_position(self.dt, trigger=True)
                reward = 2 if position_difference >= threshold else 0
            else:
                main_agent.update_position(self.dt, trigger=False)
                reward = 2 if position_difference < threshold else 0

        self.current_iteration += 1
        done = self.current_iteration >= self.num_iterations

        if done:
            self.t = self.total_trigger_count

        return self.get_observation(0), reward, done, False, {}
    

    def render(self, action):
        """根据动作更新环境，仅更新当前主智能体，其他智能体在模拟中被遍历决策"""
        main_agent = self.main_agent
        if self.current_iteration == 0:
            main_agent.update_position_1(self.dt, trigger=True)
            # for agent in self.agents:
            #     if agent != main_agent:
            #         agent.update_position_formula_with_hold_1(self.dt, self.c_0, self.c_1, self.alpha, self.current_iteration)
            self.total_trigger_count += 1
            reward = 0
        else:
            position_difference = abs(main_agent.position - main_agent.last_broadcast_position)
            time_scaling = np.exp(-self.alpha * self.current_iteration * self.dt)
            threshold = self.c_0 + self.c_1 * time_scaling

            if action == 1:
                self.total_trigger_count += 1
                main_agent.update_position(self.dt, trigger=True)
                reward = 1 if position_difference >= threshold else -1
            else:
                main_agent.update_position(self.dt, trigger=False)
                reward = 1 if position_difference < threshold else -1

        self.current_iteration += 1
        done = self.current_iteration >= self.num_iterations

        if done:
            self.t = self.total_trigger_count

        return self.get_observation(0), reward, done, False, {}


    class Agent:
        def __init__(self, index, initial_position=0.0):
            self.position = initial_position
            self.index = index
            self.neighbors = []
            self.last_broadcast_position = self.position
            self.u_i = 0.0  # 上一次更新的控制率

        def add_neighbor(self, neighbor):
            if neighbor not in self.neighbors:
                self.neighbors.append(neighbor)

        def update_position(self, dt, trigger):
            if trigger:
                self.u_i = -sum((self.last_broadcast_position - neighbor.last_broadcast_position) for neighbor in self.neighbors)
                self.position += self.u_i * dt
                self.last_broadcast_position = self.position
            else:
                self.position += self.u_i * dt

        def update_position_1(self, dt, trigger):
            if trigger:
                self.u_i = -sum((self.last_broadcast_position - neighbor.last_broadcast_position) for neighbor in self.neighbors)
                self.position += self.u_i * dt
            else:
                self.position += self.u_i * dt

        def update_position_formula_with_hold(self, dt, c_0, c_1, alpha, current_iteration):
            """使用公式更新位置，未触发时使用零阶保持器更新"""
            e_i = self.position - self.last_broadcast_position  # 计算误差
            f_i = abs(e_i) - (c_0 + c_1 * np.exp(-alpha * current_iteration * dt))  # 计算 f_i

            #有0.1的概率进行随机更新控制率

            if f_i >= 0:  # 触发条件
                self.u_i = -sum((self.last_broadcast_position - neighbor.last_broadcast_position) for neighbor in self.neighbors)  # 更新控制率
                self.position += self.u_i * dt  # 更新位置
                self.last_broadcast_position = self.position
            else:
                self.position += self.u_i * dt  # 未触发时使用零阶保持器更新


        def update_position_formula_with_hold_1(self, dt, c_0, c_1, alpha, current_iteration):
            """使用公式更新位置，未触发时使用零阶保持器更新"""
            e_i = self.position - self.last_broadcast_position  # 计算误差
            f_i = abs(e_i) - (c_0 + c_1 * np.exp(-alpha * current_iteration * dt))  # 计算 f_i

            if f_i >= 0:  # 触发条件
                self.u_i = -sum((self.last_broadcast_position - neighbor.last_broadcast_position) for neighbor in self.neighbors)  # 更新控制率
                self.position += self.u_i * dt  # 更新位置
            else:
                self.position += self.u_i * dt  # 未触发时使用零阶保持器更新
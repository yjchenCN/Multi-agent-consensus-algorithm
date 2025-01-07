from pettingzoo import ParallelEnv
import numpy as np
from gym import spaces

class CustomMAEnvironmentSumDiff(ParallelEnv):
    """
    本环境与原先的 CustomMAEnvironment3 类似，只是在 get_observation() 中，
    返回一个 1 维观测：该智能体与邻居位置差值的总和。
    """

    metadata = {
        'render.modes': ['human'],
        'name': 'custom_environment_demo_sumdiff'
    }
    
    def __init__(self, num_agents=5, num_iterations=200, dt=0.1):
        # 与原先一样，初始化智能体
        self.agents = ["agent_" + str(i) for i in range(num_agents)]
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = dict(zip(self.agents, list(range(num_agents))))
        
        self.num_iterations = num_iterations
        self.dt = dt
        self.current_iteration = 0
        
        # 初始位置
        self.initial_positions = [0.55, 0.4, -0.05, -0.1, -0.7]
        self.agent_objs = [self.Agent(pos, i) for i, pos in enumerate(self.initial_positions)]
        
        # 初始化邻居关系（也可以改成全连接或者其他方式）
        self.init_neighbors()
        
        # 收敛判定用的 epsilon
        self.epsilon = 0.005
        # 记录第一次收敛的时间
        self.time_to_reach_epsilon = None
        # 当前是否违反 epsilon（尚未收敛）
        self.epsilon_violated = True
        # 是否所有agent均在 epsilon 范围内
        self.all_within_epsilon = False
        # 总触发次数（仅用于统计或奖励）
        self.total_trigger_count = 0
        # time_to_reach_epsilon 的变化次数
        self.time_to_reach_epsilon_changes = 0

    def init_neighbors(self):
        """
        在此只给出一个示例邻居结构。你可以自由修改或增添逻辑。
        例如，这里让0与1、2相连，1与2相连，2与3相连，3与4相连。
        """
        self.agent_objs[0].add_neighbor(self.agent_objs[1])
        self.agent_objs[0].add_neighbor(self.agent_objs[2])
        self.agent_objs[1].add_neighbor(self.agent_objs[2])
        self.agent_objs[2].add_neighbor(self.agent_objs[3])
        self.agent_objs[3].add_neighbor(self.agent_objs[4])

    def reset(self, seed=None, options=None):
        """
        回合开始时的重置，注意要重新设置智能体位置和相关标志。
        """
        # 这里你可以保持固定初始位置，也可以改成随机
        self.initial_positions = [0.55, 0.4, -0.05, -0.1, -0.7]
        # self.initial_positions = np.round(np.random.uniform(-1, 1, size=5), 2)

        self.agent_objs = [self.Agent(pos, i) for i, pos in enumerate(self.initial_positions)]
        self.init_neighbors()
        
        # 时间、收敛标志等
        self.current_iteration = 0
        self.epsilon_violated = True
        self.all_within_epsilon = False
        self.total_trigger_count = 0
        self.time_to_reach_epsilon_changes = 0
        self.time_to_reach_epsilon = None
        
        # 返回每个智能体的观测
        observations = {agent: self.get_observation(agent) for agent in self.agents}
        return observations
    
    def get_observation(self, agent):
        """
        关键改动：返回一个 1 维的值 = (该智能体位置 - 每个邻居位置) 的总和。
        这样每个智能体的观测空间就固定为 shape(1,)。
        """
        agent_index = self.agent_name_mapping[agent]
        agent_obj = self.agent_objs[agent_index]

        # 计算差值之和
        diff_sum = 0.0
        for neighbor in agent_obj.neighbors:
            diff_sum += (agent_obj.position - neighbor.position)

        # 作为 numpy array 返回
        obs = np.array([diff_sum], dtype=np.float32)
        return obs

    def step(self, actions):
        """
        环境核心循环，与原先一样：接收每个智能体的动作(是否触发)，更新位置、计算奖励、done等。
        """
        # 将 actions(字典) 转为 array 方便处理
        triggers = np.array([actions[agent] for agent in self.agents])
        trigger_count = np.sum(triggers)
        self.total_trigger_count += trigger_count

        # 先更新所有智能体的位置
        for i, agent in enumerate(self.agent_objs):
            agent.update_position(self.current_iteration, self.dt, triggers[i])

        # 计算平均差值，用于一些中间的惩罚或奖励逻辑
        average_difference = self.compute_average_position_difference()

        # 检查是否达到一致性（所有智能体间的差值都小于 epsilon）
        self.all_within_epsilon = all(
            all(abs(agent.position - neighbor.position) < self.epsilon 
                for neighbor in agent.neighbors) 
            for agent in self.agent_objs
        )

        # 如果当前时刻达成一致性，而之前还未达成，则记录 time_to_reach_epsilon
        if self.all_within_epsilon:
            if self.epsilon_violated:
                self.time_to_reach_epsilon = self.current_iteration
                self.epsilon_violated = False
        else:
            self.epsilon_violated = True
            self.time_to_reach_epsilon = None

        self.current_iteration += 1
        done = self.current_iteration >= self.num_iterations

        # 计算奖励
        rewards = {}
        
        if done:
            # 回合结束
            if self.time_to_reach_epsilon is not None:
                # 成功达成一致性，给一个基于收敛时间和触发次数的全局奖励
                for agent in self.agents:
                    agent_index = self.agent_name_mapping[agent]
                    agent_obj = self.agent_objs[agent_index]
                    individual_trigger_count = len(agent_obj.trigger_points)
                    rewards[agent] = 1000 - self.time_to_reach_epsilon - individual_trigger_count * 2
            else:
                # 未能收敛
                for agent in self.agents:
                    rewards[agent] = -800
        else:
            # 回合进行中
            threshold = 1.0 * self.epsilon
            time_penalty_factor = 0.01 * self.current_iteration

            for i, agent in enumerate(self.agents):
                if average_difference > threshold:
                    # 差值还较大，鼓励触发(帮助尽快收敛)
                    if triggers[i] == 1:
                        rewards[agent] = 0
                    else:
                        rewards[agent] = 0
                else:
                    # 进入细调阶段，差值小，鼓励少触发
                    if triggers[i] == 1:
                        rewards[agent] = -(1.0 + time_penalty_factor * 3)
                    else:
                        rewards[agent] = 1
        
        # 新的观察、done、info
        observations = {agent: self.get_observation(agent) for agent in self.agents}
        dones = {agent: done for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        
        return observations, rewards, dones, infos

    def compute_average_position_difference(self):
        """
        一个辅助函数，用于计算所有智能体之间的平均位置差值。
        仅在 step 中给奖励或惩罚参考，可改可不改。
        """
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
        关键改动：现在每个智能体的观测是 (1,) 大小，取值范围可根据需求调整。
        """
        return spaces.Box(low=-100, high=100, shape=(1,), dtype=np.float32)
    
    def action_space(self, agent):
        return spaces.Discrete(2)

    class Agent:
        """
        智能体类，与之前相同。
        """
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
            """
            如果触发(=1)或t=0(初始时刻)，则计算新的控制输入并更新位置；
            否则用上一次的控制输入持续演化。
            """
            if trigger == 1 or t == 0:
                # 触发更新 u_i
                self.u_i = -sum(
                    (self.last_broadcast_position - neighbor.last_broadcast_position) 
                    for neighbor in self.neighbors if self.is_neighbor(neighbor)
                )
                self.position += self.u_i * dt
                self.last_broadcast_position = self.position
                self.trigger_points.append((t, self.position))
            else:
                # 不触发则根据上一次的输入继续演化
                self.position += self.u_i * dt

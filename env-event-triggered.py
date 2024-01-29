import numpy as np
import matplotlib.pyplot as plt


# 定义触发函数
def triggered_communication(agent, threshold):
    for neighbor in agent.neighbors:
        if abs(agent.get_position() - neighbor.get_position()) > threshold:
            return True
    return False


# 更新Agent类，增加触发通信的功能
class Agent:
    def __init__(self, initial_position):
        self.position = initial_position
        self.velocity = 0
        self.neighbors = []
        self.triggered = False

    def get_position(self):
        return self.position

    def set_position(self, new_position):
        self.position = new_position

    def is_neighbor(self, agent):
        return agent in self.neighbors

    def add_neighbor(self, neighbor):
        if neighbor not in self.neighbors:
            self.neighbors.append(neighbor)
            neighbor.neighbors.append(self)

    def update_position(self, dt):
        if triggered_communication(self, threshold=0.1):
            self.triggered = True  # 触发通信
            position_update = sum(self.is_neighbor(neighbor) * (neighbor.get_position() - self.position) for neighbor in self.neighbors)
            self.velocity = position_update
        else:
            self.triggered = False  # 未触发通信，保持当前速度不变

        # 更新位置
        self.position += self.velocity * dt


# 初始化智能体和模拟参数
num_agents = 3  # 智能体的数量
num_iterations = 400  # 迭代次数
dt = 0.01  # 时间增量
alpha = 1  # alpha系数

# 创建智能体并随机初始化位置
np.random.seed(1)
agents = [Agent(np.random.rand() * 10) for _ in range(num_agents)]

# 定义邻居关系
agents[0].add_neighbor(agents[1])
agents[1].add_neighbor(agents[2])
agents[0].add_neighbor(agents[2])

# 运行模拟
positions = np.zeros((num_iterations, num_agents))

for t in range(num_iterations):
    for i, agent in enumerate(agents):
        agent.update_position(dt)
        positions[t, i] = agent.get_position()

    # 如果所有智能体的位置都很接近，则认为达成一致，结束模拟
    if np.std([agent.get_position() for agent in agents]) < 0.01:
        positions[t + 1:, :] = positions[t, :]
        break

# 绘制智能体位置随时间变化的图像
plt.figure(figsize=(10, 5))
for i, agent in enumerate(agents):
    plt.plot(positions[:, i], label=f'Agent {i + 1}')
plt.xlabel('Time step')
plt.ylabel('Position')
plt.title('Agent Positions Over Time')
plt.legend()
plt.grid(True)
plt.show()

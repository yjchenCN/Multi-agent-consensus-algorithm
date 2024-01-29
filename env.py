import numpy as np
import matplotlib.pyplot as plt
import matplotlib


class Agent:
    def __init__(self, initial_position):
        self.position = initial_position
        self.velocity = 0
        self.neighbors = []

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
        # 根据给定的速度更新方程来更新位置
        position_update = sum(self.is_neighbor(neighbor) * (neighbor.get_position() - self.position) for neighbor in self.neighbors)
        self.velocity = position_update  # 更新速度
        self.position += self.velocity * dt  # 更新位置


# 初始化参数
num_agents = 4  # 智能体的数量
num_iterations = 1000  # 迭代次数
dt = 0.01  # 时间增量
alpha = 1  # alpha系数，简化模型中为常数

# 创建智能体并随机初始化位置
np.random.seed(0)  # 为了可复现性设置随机种子
agents = [Agent(np.random.rand() * 10) for _ in range(num_agents)]

agents[0].add_neighbor(agents[1])
agents[1].add_neighbor(agents[2])
agents[0].add_neighbor(agents[2])
agents[0].add_neighbor(agents[3])


# 运行模拟
positions = np.zeros((num_iterations, num_agents))

for t in range(num_iterations):
    for i, agent in enumerate(agents):
        agent.update_position(dt)
        positions[t, i] = agent.get_position()

# 绘制结果图
plt.figure(figsize=(10, 5))
for i, agent in enumerate(agents):
    plt.plot(positions[:, i], label=f'智能体 {i+1}')
plt.xlabel('时间步')
plt.ylabel('位置')
plt.title('智能体随时间变化的位置')
plt.legend()
plt.grid(True)

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号 '-' 显示为方块的问题

plt.show()


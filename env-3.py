import numpy as np
import matplotlib.pyplot as plt


class Agent:
    def __init__(self, position, velocity):
        self.position = position
        self.velocity = velocity
        self.neighbors = []

    def get_position(self):
        return self.position

    def set_position(self, new_position):
        self.position = new_position

    def get_velocity(self):
        return self.velocity

    def set_velocity(self, new_velocity):
        self.velocity = new_velocity

    def add_neighbor(self, neighbor):
        self.neighbors.append(neighbor)

    def update_velocity(self, alpha):
        # Apply the consensus algorithm
        # \dot{x}_i(t) = -\sum_{j=1}^n \alpha_{ij}(t)[x_i(t) - x_j(t)]
        velocity_change = sum(alpha * (neighbor.get_velocity() - self.velocity) for neighbor in self.neighbors)
        self.velocity += velocity_change


# Simulation parameters
time_steps = 1000
alpha = 0.01  # Coefficient for the consensus algorithm
agents_count = 3  # Number of agents

# Initialize agents with random positions and velocities
agents = [Agent(np.random.rand(), np.random.rand()) for _ in range(agents_count)]

# Every agent is a neighbor of each other for this scenario
for agent in agents:
    agent.neighbors = [neighbor for neighbor in agents if neighbor is not agent]

# Record the velocity of each agent at each time step
velocity_history = np.zeros((time_steps, agents_count))

# Run the simulation
for t in range(time_steps):
    for i, agent in enumerate(agents):
        agent.update_velocity(alpha)
        velocity_history[t, i] = agent.get_velocity()

# Plotting the velocity of agents over time
plt.figure(figsize=(10, 5))
for i in range(agents_count):
    plt.plot(velocity_history[:, i], label=f'Agent {i + 1}')
plt.xlabel('Time step')
plt.ylabel('Velocity')
plt.title('Velocity of agents over time')
plt.legend()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from cv2 import threshold


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
time_steps = 400
alpha = 0.01  # Coefficient for the consensus algorithm
agents_count = 5  # Number of agents

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


class AgentETC(Agent):
    def __init__(self, position, velocity, threshold):
        super().__init__(position, velocity)
        self.threshold = threshold
        self.triggered_last_time = True  # To indicate if the event was triggered last time

    def update_velocity(self, alpha):
        # Calculate the velocity change based on the consensus algorithm
        velocity_change = sum(alpha * (neighbor.get_velocity() - self.velocity) for neighbor in self.neighbors)
        # Check if the update is necessary based on the event-triggered condition
        if self.triggered_last_time or np.abs(velocity_change) > self.threshold:
            self.velocity += velocity_change
            self.triggered_last_time = True
        else:
            self.triggered_last_time = False


# Reset the simulation with new agents
agents_etc = [AgentETC(np.random.rand(), np.random.rand(), threshold) for _ in range(agents_count)]

for agent in agents_etc:
    agent.neighbors = [neighbor for neighbor in agents_etc if neighbor is not agent]

# Record the velocity of each agent at each time step
velocity_history_etc = np.zeros((time_steps, agents_count))

# Run the simulation with event-triggered control
for t in range(time_steps):
    for i, agent in enumerate(agents_etc):
        agent.update_velocity(alpha)
        velocity_history_etc[t, i] = agent.get_velocity()

# Plotting the velocity of agents over time with ETC
plt.figure(figsize=(10, 5))
for i in range(agents_count):
    plt.plot(velocity_history_etc[:, i], label=f'Agent {i + 1} ETC')

plt.xlabel('Time step')
plt.ylabel('Velocity')
plt.title('Velocity of agents over time with Event-Triggered Control (Adjusted)')
plt.legend()
plt.show()
from pettingzoo import ParallelEnv
from gym import spaces
import numpy as np


class SimpleTagEnvironment(ParallelEnv):
    metadata = {
        'render.modes': ['human'],
        'name': 'simple_tag_env'
    }
    
    def __init__(self, num_runners=3, num_steps=100, dt=0.1):
        self.agents = ["chaser"] + ["runner_" + str(i) for i in range(num_runners)]
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = dict(zip(self.agents, list(range(len(self.agents)))))
        
        self.num_steps = num_steps
        self.dt = dt
        self.current_step = 0
        self.size = 10.0  # Size of the environment
        
        self.positions = {agent: np.random.uniform(-self.size/2, self.size/2, size=(2,)) for agent in self.agents}
        self.max_speed = 1.0
        
    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.positions = {agent: np.random.uniform(-self.size/2, self.size/2, size=(2,)) for agent in self.agents}
        observations = {agent: self.positions[agent] for agent in self.agents}
        return observations
    
    def step(self, actions):
        for agent in self.agents:
            move = np.array(actions[agent]) * self.dt * self.max_speed
            self.positions[agent] = np.clip(self.positions[agent] + move, -self.size/2, self.size/2)
        
        # Check if chaser catches any runner
        done = False
        rewards = {agent: -1 for agent in self.agents}
        for runner in self.agents[1:]:
            distance = np.linalg.norm(self.positions["chaser"] - self.positions[runner])
            if distance < 0.5:  # Catch radius
                rewards["chaser"] = 10
                rewards[runner] = -10
                done = True
                break
        
        self.current_step += 1
        if self.current_step >= self.num_steps:
            done = True
        
        observations = {agent: self.positions[agent] for agent in self.agents}
        dones = {agent: done for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        
        return observations, rewards, dones, infos
    
    def render(self, mode='human'):
        print("Positions:", self.positions)
    
    def observation_space(self, agent):
        return spaces.Box(low=-self.size/2, high=self.size/2, shape=(2,), dtype=np.float32)
    
    def action_space(self, agent):
        return spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

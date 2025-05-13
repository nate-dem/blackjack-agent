import gymnasium as gym
import numpy as np
from blackjack_env import MakeEnvironment
from collections import defaultdict

class TabularQLearning:
    def __init__(self, 
        env: MakeEnvironment,
        learning_rate: float,
        init_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount: float = 0.8,
    ):
        self.env = env
        self.learning_rate = learning_rate
        self.discount = discount
        self.init_epsilon = init_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.Q = defaultdict(lambda: np.zeros(env.action_space.n))

    def select_action(self, state):
        rand_choice = np.random.rand()
        if rand_choice < self.epsilon:
            return self.env.action_space.sample()
        return int(np.argmax(self.Q[state]))


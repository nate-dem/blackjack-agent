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



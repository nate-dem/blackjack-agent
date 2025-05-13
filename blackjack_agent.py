import gymnasium as gym
import numpy as np
from blackjack_env import MakeEnvironment
from collections import defaultdict

class TabularQLearning:
    def __init__(self, 
        env: MakeEnvironment,
        learningRate: float,
        initEpsilon: float,
        epsilonDecay: float,
        finalEpsilon: float,
        discount: float = 0.8,
    ):
        self.env = env
        self.learning_rate = learningRate
        self.discount = discount
        self.init_epsilon = initEpsilon
        self.epsilon_decay = epsilonDecay
        self.final_epsilon = finalEpsilon
        self.Q = defaultdict(lambda: np.zeros(env.action_space.n))



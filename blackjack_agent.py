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
        discount: float = 0.95,
    ):
        self.env = env
        self.learning_rate = learning_rate
        self.discount = discount
        self.init_epsilon = init_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.epsilon = init_epsilon
        self.Q = defaultdict(lambda: np.zeros(env.action_space.n))

    def select_action(self, state):
        rand_choice = np.random.rand()
        if rand_choice < self.epsilon:
            return self.env.action_space.sample()
        return int(np.argmax(self.Q[state]))

    def learn_policy(self, num_iterations: int):
        self.epsilon = self.init_epsilon
        
        for i in range(1, num_iterations + 1):
            state, _ = self.env.reset()
            done = False

            while not done:
                action = self.select_action(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                best_next = max(self.Q[next_state])
                cur_q = (((1 - self.learning_rate) * self.Q[state][action]) + (self.learning_rate * (reward + (self.discount * best_next))))
                self.Q[state][action] = cur_q

                state = next_state
            
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.final_epsilon)

    def evaluation(self, num_iterations: int):
        prev_epsilon = self.epsilon
        self.epsilon = 0.0

        total_score, wins, losses, pushes = 0, 0, 0, 0
        for _ in range(num_iterations):
            state, _ = self.env.reset()
            
            done = False
            cur_score = 0
            while not done:
                action = self.select_action(state)
                state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                cur_score += reward

            if cur_score > 0:
                wins += 1
            elif cur_score == 0:
                pushes += 1
            else:
                losses += 1

            total_score += cur_score

        avg_score = total_score / num_iterations
        win_percentage = wins / num_iterations
        loss_percentage = losses / num_iterations
        push_percentage = pushes / num_iterations

        self.epsilon = prev_epsilon

        results = {
            'avg_score': avg_score,
            'win_percentage': win_percentage,
            'loss_percentage': loss_percentage,
            'push_percentage': push_percentage
        }

        return results


if __name__ == "__main__":
    env = MakeEnvironment()
    agent = TabularQLearning(learning_rate=0.01, init_epsilon=1.0, epsilon_decay=0.99, final_epsilon=0.1, discount=discount)






import gymnasium as gym

class MakeEnvironment:
    def __init__(self, 
        env_name: str = 'Blackjack-v1',
        env_params: dict = None
    ):
        default_params = {'natural': True, 'sab': False, 'render_mode': None}
        params = {**default_params, **(env_params or {})}
        self.env = gym.make(env_name, **params)
        self.observation = None
        self.info = None

    def reset(self, seed: int = None):
        if seed is not None:
            self.observation, self.info = self.env.reset(seed=seed)
        else:
            self.observation, self.info = self.env.reset()
        return self.observation, self.info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.observation, self.info = obs, info
        return obs, reward, terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()

    @property
    def action_space(self):
        return self.env.action_space
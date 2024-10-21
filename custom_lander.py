import gymnasium as gym
import numpy as np
from gymnasium import spaces

class LunarLanderImageAndStateWrapper(gym.Wrapper):
    def __init__(self, env):
        super(LunarLanderImageAndStateWrapper, self).__init__(env)
        self.env = env
        # Assuming the image observation is a 64x64 RGB image
        image_shape = (400, 600, 3)
        # The original raw observation space
        
        # Combined observation space (image + raw observations)
        self.observation_space = spaces.Dict({
            'image': spaces.Box(low=0, high=255, shape=image_shape, dtype=np.uint8),
            'raw_observation': self.env.observation_space
        })
        self.action_space = self.env.action_space
        
        self.last_observation = None
        self.observation = None
        self.action = None
        self.reward = None

    def reset(self, seed = None):
        raw_obs, info = self.env.reset()
        image_obs = self.render()  # Get the image observation
        self.last_observation = {'image': image_obs, 'raw_observation': raw_obs}
        return self.last_observation, info

    def step(self, action):
        raw_obs, reward, done, trunc, info = self.env.step(action)
        image_obs = self.render()
        self.observation = self.last_observation
        self.action = action
        self.reward = reward
        self.last_observation = {'image': image_obs, 'raw_observation': raw_obs}
        return self.last_observation, reward, done, trunc, info

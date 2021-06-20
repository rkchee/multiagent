import gym
from gym.utils import seeding
import numpy as np
from gym.spaces import Discrete, Box

class Med_v0(gym.Env):
   def __init__(self):
        import tensorflow as tf
        low = np.full((128, 256, 10), -1000)
        high = np.full((128, 256, 10), 1000)
        # output_low = np.full((, 1), 0)
        # output_high = np.full((, 1), 10)

        # self.action_space = Box(low=0, high=10.0, shape=(1, ), dtype=np.float32)
        self.action_space = Discrete(10)
        # self.action_space = Box(low=output_low, high=output_high, shape=(, 1), dtype=np.int32)

        self.observation_space = Box(low=low, high=high, shape=(128, 256, 10), dtype=np.float32)
        # self.observation_space = Discrete(1)
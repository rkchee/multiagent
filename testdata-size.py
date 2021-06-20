from ray.rllib.models.tf.tf_modelv2 import TFModelV2
import tensorflow as tf
import ray 
from ray.tune.logger import pretty_print
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models import ModelCatalog
from ray.rllib.agents.dqn import DQNTrainer
from ray.rllib.agents.dqn import ApexTrainer
from ray.rllib.utils import merge_dicts
from ray.rllib.agents.trainer import with_common_config
import gym
from gym.spaces import Discrete, Box
import requests
import pandas as pd
import json
import numpy as np
import os 
import random
import pdb
import dask.array as da
import h5py

fdata = h5py.File('../../combinedxy.hdf5', 'r')
d = fdata['/x']
ys = fdata['/y']   

print(len(ys))
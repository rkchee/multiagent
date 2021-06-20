import tensorflow as tf
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
import ray 
from ray.tune.logger import pretty_print
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models import ModelCatalog
from ray.rllib.agents.dqn import DQNTrainer
from ray.rllib.agents.dqn import ApexTrainer
import gym
from gym.spaces import Discrete, Box
import requests
import pandas as pd
import json
import numpy as np
import os 
import random
import pdb


ray_head_ip =os.environ.get('RAY_HEAD_SERVICE_HOST')
ray_redis_port = os.environ.get('RAY_HEAD_SERVICE_PORT_REDIS_PRIMARY')
url = '35.247.84.216'



headers = {"content-type": "application/json"}
response = requests.get("http://" + url + ":8000/cube", headers=headers)
t = response.text 
t = json.loads(t)
ET = tf.data.Dataset.from_tensor_slices(t['Electra_Tensors'])
a = tf.data.Dataset.from_tensor_slices(t['answers'])
# # print(a)
pdb.set_trace()
# c = tf.data.Dataset.zip((ET, a))
# # c = c.range(1)
# c = c.as_numpy_iterator()
print(list(a)[0])

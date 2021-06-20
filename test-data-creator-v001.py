import ray 
from ray.tune.logger import pretty_print
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models import ModelCatalog
from ray.rllib.agents.dqn import DQNTrainer
from ray.rllib.agents.dqn import ApexTrainer
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
import gym
from gym.spaces import Discrete, Box
import requests
import tensorflow as tf
import tensorflow.keras.metrics
import tensorflow.keras.losses
import pandas as pd
import json
import numpy as np
import os 
import dask
import dask.array as da
import dask.dataframe as dd
from dask.delayed import delayed
import dask.bag as db
from distributed import Client
# client = Client()
import random

ray_head_ip =os.environ.get('RAY_HEAD_SERVICE_HOST')
ray_redis_port = os.environ.get('RAY_HEAD_SERVICE_PORT_REDIS_PRIMARY')
url = '34.83.251.186'


# # train test split
sheets_cards = pd.read_excel('training_medical.xlsx', sheet_name='Cardiology and Infectious Disea')
cardiology = sheets_cards.copy()
train_set = cardiology.sample(frac=0.65, random_state=0)
train_set.to_csv('train_set.csv', index=False)
test_set = cardiology.drop(train_set.index)
test_set.to_csv('test_set.csv', index=False)

data = pd.read_csv("train_set.csv")
print(len(data.index))
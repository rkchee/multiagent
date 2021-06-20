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
import cProfile

print(tf.__version__)


ray_head_ip =os.environ.get('RAY_HEAD_SERVICE_HOST')
ray_redis_port = os.environ.get('RAY_HEAD_SERVICE_PORT_REDIS_PRIMARY')
url = '34.83.105.129'


# # train test split
columns = ['Answer.answerA', 'Answer.answerB', 'Answer.answerC', 'Answer.answerD', 'Answer.answerE', 'Answer.answerF', 'Answer.image.label', 'Answer.question']
# sheets_cards = pd.read_excel('../training_medical.xlsx', usecols=columns, sheet_name='Cardiology and Infectious Disea')
# cardiology = sheets_cards.copy()

# tf.config.experimental_run_functions_eagerly(True)

# data = tf.data.experimental.make_csv_dataset('train_set.csv', batch_size=1, select_columns=columns, label_name='Answer.image.label', num_epochs=1, ignore_errors=True)
# it = iter(data)
# print(next(it).numpy())

# dataset = tf.data.Dataset.from_tensor_slices([8, 3, 0, 8, 2, 1])
# dataset

# for elem in dataset:
#   print(elem.numpy())

# train_set = cardiology.sample(frac=0.65, random_state=0)
# train_set.to_csv('train_set.csv', index=False)
# test_set = cardiology.drop(train_set.index)
# test_set.to_csv('test_set.csv', index=False)

# data = pd.read_csv("train_set.csv")
# print(len(data.index))
# print(sheets_cards.info(verbose=False, memory_usage='deep'))
# print(sheets_cards.columns)

tf.compat.v1.enable_eager_execution(
    config=None, device_policy=None, execution_mode=None
)

def sequence(seq):
    if not type(seq) == str:
        seq = str(seq)
    
    js = {
        "text":seq
    }

    dumps = json.dumps(js)
    payload = json.loads(dumps)
    headers = {"content-type": "application/json"}
    response = requests.post("http://" + url + ":8000/electratensors", json=payload, headers=headers)
    t = response.text 
    t = json.loads(t)
    t = t['Electra_Tensors']
    return np.asarray(t)

def form_matrix(sequences):
    matrix_max_size = 10
    matrix = []


    for i in sequences:
        answer = sequence(i)
        matrix.append(answer)
    
    matrix = np.asarray(matrix)
    matrix = np.squeeze(matrix)
    # print(np.shape(matrix))
    state_matrix = np.pad(matrix, [(0, matrix_max_size - matrix.shape[0]), (0,0), (0,0)])
    state_matrix = np.moveaxis(state_matrix, 0, 2)
    return state_matrix


batch_size = 5
data = tf.data.experimental.make_csv_dataset('train_set.csv', batch_size=batch_size, select_columns=columns, label_name='Answer.image.label', num_epochs=1, ignore_errors=True)


columns = ['Answer.answerA', 'Answer.answerB', 'Answer.answerC', 'Answer.answerD', 'Answer.answerE', 'Answer.answerF', 'Answer.image.label', 'Answer.question']


@tf.autograph.experimental.do_not_convert
def create_matrix(samples, targets):
    state_cube = ['Answer.question', 'Answer.answerA', 'Answer.answerB', 'Answer.answerC', 'Answer.answerD', 'Answer.answerE', 'Answer.answerF']
  # for i in targets:
  #   print(i)
  # batch = [[[],[]],[[],[]],[[],[]],[[],[]],[[],[]]]
  # batch = []
  # print(batch)
  # y = []
  # for ij in range(batch_size):
    matrix = []
    for k in state_cube:
        matrix.append(samples[k])
        # matrix = form_matrix(matrix)
        # batch.append(matrix)
        # y.append(targets[ij]) 
    return matrix, targets

  # return targets, targets
    
data = data.map(create_matrix)
print('data is', data)
# y = data.map(create_matrix)


x, y = next(iter(data.take(1)))
# y = next(iter(y.take(1)))


store_list = [[],[]]
for i in range(len(x[0])):
    seq = [x[0][i], x[1][i], x[2][i], x[3][i], x[4][i], x[5][i], x[6][i]]
    obs = form_matrix(seq)
    store_list[0].append(obs)

# print(y[0])


# y = next(iter(y))
# print(y)
# print()
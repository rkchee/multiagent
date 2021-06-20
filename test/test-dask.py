from ray.rllib.models.tf.tf_modelv2 import TFModelV2
import tensorflow as tf
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
import dask
import dask.array as da
import dask.dataframe as dd
from dask.delayed import delayed

ray_head_ip =os.environ.get('RAY_HEAD_SERVICE_HOST')
ray_redis_port = os.environ.get('RAY_HEAD_SERVICE_PORT_REDIS_PRIMARY')
url = '34.83.105.129'

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
    state_matrix = np.pad(matrix, [(0, matrix_max_size - matrix.shape[0]), (0,0), (0,0)])
    state_matrix = np.moveaxis(state_matrix, 0, 2)
    return state_matrix


class create_dask_func():
    def __init__(self):
        columns = ['Answer.answerA', 'Answer.answerB', 'Answer.answerC', 'Answer.answerD', 'Answer.answerE', 'Answer.answerF', 'Answer.image.label', 'Answer.question']
        df = pd.read_csv('train_set.csv', usecols=columns)
        # print(np.shape(df))
        parts = dask.delayed(pd.read_csv)('train_set.csv', usecols=columns)
        # data = dd.from_delayed(parts).sample(frac=0.04)
        data = dd.from_delayed(parts).sample(frac=0.04)

        batch_size=5
        store_list = []
        eps_id = random.randrange(0, batch_size - 1, 1)

        for i in range(batch_size):
            sq = [data['Answer.question'].loc[i], data['Answer.answerA'].loc[i], data['Answer.answerB'].loc[i], data['Answer.answerC'].loc[i], data['Answer.answerD'].loc[i], data['Answer.answerE'].loc[i], data['Answer.answerF'].loc[i]]
            obs = dask.delayed(form_matrix)(sq)
            store_list.append(obs)

        store_list = dask.compute(*store_list)
        print(store_list[1])


if __name__ == "__main__":
    create_dask_func()    

# seq_element1 = sequence(data['Answer.question'].loc[1])
# print(seq_element1)
# seq_element1 = dask.delayed(sequence)(data['Answer.question'].loc[1])
# print(seq_element1.compute())





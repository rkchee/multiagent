import numpy as np
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
# import modin.pandas as pd
import json
import numpy as np

import os 

url ='34.82.238.59'
# data = pd.read_csv('/home/rkchee/nxtopinion/rl/rheumatology_4199362_batch_results.csv')[0:5]

# train test split
# sheets_cards = pd.read_excel('../training dataset - medical.xlsx', sheet_name='Cardiology and Infectious Disea')
# cardiology = sheets_cards.copy()
# train_set = cardiology.sample(frac=0.001, random_state=0)
# train_set.to_csv('train_set.csv', index=False)
# test_set = cardiology.drop(train_set.index)
# test_set.to_csv('test_set.csv', index=False)

# set the training data 
# data = pd.read_csv('train_set.csv')
# data = pd.DataFrame(sheets_cards)[0:10]

import dask
import dask.dataframe as dd
from dask.delayed import delayed
import dask.array as da



# data = dd.read_csv('training dataset - medical.xlsx', sheet_name='Cardiology and Infectious Disea')
parts = dask.delayed(pd.read_excel)('../training_medical.xlsx', sheet_name='Cardiology and Infectious Disea')
data = dd.from_delayed(parts)


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

@dask.delayed
def form_matrix(qa_index, question):
    matrix_max_size = 10
    matrix = []

    mc_choices = ["A", "B", "C", "D", "E"]

    matrix = question
    for letter in mc_choices:
        answer = data['Answer.answer' +  letter].loc[qa_index]
        answer = sequence(answer)
        matrix = np.concatenate((matrix, answer), axis=0)
    
    matrix = np.squeeze(matrix)
    state_matrix = np.pad(matrix, [(0, matrix_max_size - matrix.shape[0]), (0,0), (0,0)])
    state_matrix = np.moveaxis(state_matrix, 0, 2)
    return state_matrix

num_questions = len(data.index)
print(num_questions)



# nrows, ncols = num_questions, 2
# f = np.memmap('memmapped.dat', dtype=object,
#               mode='w+', shape=(nrows, ncols))

answers = data['Answer.image.label']


mcq_number ={
    'A': 0,
    'a': 0,
    'B': 1,
    'b': 1,
    'C': 2,
    'c': 2,
    'D': 3,
    'd': 3,
    'E': 4,
    'e': 4,
    'F': 5,
    'f': 5,
    'G': 6,
    'g': 6,
    'H': 7,
    'h': 7,
    'I': 8,
    'i': 8,
    'J': 9,
    'j': 9, 
}        

# f = pd.DataFrame(columns=[0])
# # f[0]= f[0].astype(object) 


# dask_array = dask.delayed(np.ones)((128,256,10))
# dask_array = da.from_delayed(dask_array, (128, 256, 10), dtype=float)

f =[]


for i in data.index:
# def create_matrix()
    q = data['Answer.question'].loc[i]
    # obs = sequence(obs)
    # df.map_partitions(train)
    obs = dask.delayed(sequence)(q)
    # obs = q.map_partitions(sequence)
    state = dask.delayed(form_matrix)(i, obs)
    # state = obs.map_partitions(form_matrix, i)
    f.append(state)

import dask.bag as db


if __name__ == '__main__':

    f = dask.persist(f[1])
    f = dask.compute(f)
    # b = db.from_sequence(f)
    # df = b.to_dataframe()
    print(f)
    c = dask.persist(f)
    c = dask.compute(f)
    print(c)




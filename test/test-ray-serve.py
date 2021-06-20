import ray
from ray.tune.logger import pretty_print
from ray.rllib.models.tf.misc import normc_initializer as rllib_normc_initializer
from ray.rllib.models import ModelCatalog # possibly causes a conflict with electra - roger
from ray.rllib.agents.dqn import DQNTrainer
from ray.rllib.models.tf.tf_modelv2 import TFModelV2 as rllib_TFModelV2

import gym
from gym.spaces import Discrete, Box
import requests 
import tensorflow as tf
import tensorflow.keras.metrics
import tensorflow.keras.losses
import pandas as pd
import json
import numpy as np

url ="35.230.63.77"
# sheets_cards = pd.read_excel('training dataset - medical.xlsx', sheet_name='Cardiology and Infectious Disea')
data = pd.read_csv('/home/rkchee/nxtopinion/rl/rheumatology_4199362_batch_results.csv')[0:30]
# data = pd.read_csv('/workspace/nxtopinion/rl/rheumatology_4199362_batch_results.csv')

class electra_serve:
    def sequence(seq):
        if not type(seq) == str:
            seq = str(seq)            
            print(seq)

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


def form_matrix(qa_index, question):
    matrix_max_size = 10
    matrix = []

    mc_choices = ["A", "B", "C", "D", "E"]

    matrix = question
    for letter in mc_choices:
        answer = data['Answer.answer' +  letter].iloc[qa_index]
        answer = electra_serve.sequence(answer)
        matrix = np.concatenate((matrix, answer), axis=0)
    
    matrix = np.squeeze(matrix)
    state_matrix = np.pad(matrix, [(0, matrix_max_size - matrix.shape[0]), (0,0), (0,0)])
    state_matrix = np.moveaxis(state_matrix, 0, 2)
    return state_matrix

questions = data['Answer.question']
n = len(data.index)
counter=0
while True:
    question = questions.iloc[counter]
    question = electra_serve.sequence(question)
    t = form_matrix(counter, question)
    # t = electra_serve.sequence('having fun with math and ML')
    print(t)
    if counter == n-1:
        counter =0
    else: 
        counter += 1
    
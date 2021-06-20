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

tf.compat.v1.enable_eager_execution(
    config=None, device_policy=None, execution_mode=None
)

ray_head_ip =os.environ.get('RAY_HEAD_SERVICE_HOST')
ray_redis_port = os.environ.get('RAY_HEAD_SERVICE_PORT_REDIS_PRIMARY')
url = '34.83.105.129'

def create_matrix(samples, targets):
    # import tensorflow as tf
    state_cube = ['Answer.question', 'Answer.answerA', 'Answer.answerB', 'Answer.answerC', 'Answer.answerD', 'Answer.answerE', 'Answer.answerF']
    matrix = []
    for k in state_cube:
        matrix.append(samples[k])
    return matrix, targets

def tensor_string(tensor_input):
    string_out = tensor_input.numpy().decode('utf-8')
    return string_out 

batch_size = 1
columns = ['Answer.answerA', 'Answer.answerB', 'Answer.answerC', 'Answer.answerD', 'Answer.answerE', 'Answer.answerF', 'Answer.image.label', 'Answer.question']
data = tf.data.experimental.make_csv_dataset('train_set.csv', batch_size=batch_size, select_columns=columns, label_name='Answer.image.label', num_epochs=1, ignore_errors=True)
data = data.map(create_matrix)

@tf.function
def datagen(data):
    # import tensorflow as tf
    qnatext, answers = next(iter(data.take(1)))
    return qnatext, answers

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

if __name__ == "__main__":
    # ModelCatalog.register_custom_model("my_model", neuronetwork)
    # ray.init(address=ray_head_ip + ":" + ray_redis_port)
    # ray.init()
    qnatext, answers = datagen(data)
    store_list = []
    for i in range(batch_size):
        seq = [qnatext[0][i], qnatext[1][i], qnatext[2][i], qnatext[3][i], qnatext[4][i], qnatext[5][i], qnatext[6][i]]
        obs = form_matrix(seq)
        store_list.append(obs)


    # print(tensor_string(qnatext[0][0]))
    # random_test=random.randrange(0, 10, 1)
    # print(random_test)

    # byte_code = answers[0].numpy().decode('utf-8')
    # print(byte_code)
    # print(byte_code =='A')

    print(answers)
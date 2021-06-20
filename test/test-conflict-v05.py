from transformers import ElectraTokenizer, TFElectraModel # conflict
tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator', max_length=128, pad_to_max_length=True)
model = TFElectraModel.from_pretrained('google/electra-small-discriminator')

import ray 
from ray.tune.logger import pretty_print
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models import ModelCatalog
from ray.rllib.agents.dqn import DQNTrainer
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

url = '35.230.63.77'
# data = pd.read_csv('/home/rkchee/nxtopinion/rl/rheumatology_4199362_batch_results.csv')

sheets_cards = pd.read_excel('training dataset - medical.xlsx', sheet_name='Cardiology and Infectious Disea')
data = pd.DataFrame(sheets_cards)

def sequence(seq):
    if not type(seq) == str:
        seq = str(seq)            
        print(seq)

    input_ids = tf.constant(tokenizer.encode(seq, max_length=128, pad_to_max_length=128))[None, :]
    outputs = model(input_ids)
    return np.asarray(outputs)

def form_matrix(qa_index, question):
    matrix_max_size = 10
    matrix = []

    mc_choices = ["A", "B", "C", "D", "E"]

    matrix = question
    for letter in mc_choices:
        answer = data['Answer.answer' +  letter].iloc[qa_index]
        answer = sequence(answer)
        matrix = np.concatenate((matrix, answer), axis=0)
    
    matrix = np.squeeze(matrix)
    state_matrix = np.pad(matrix, [(0, matrix_max_size - matrix.shape[0]), (0,0), (0,0)])
    state_matrix = np.moveaxis(state_matrix, 0, 2)
    return state_matrix

class realworldEnv(gym.Env):
    def __init__(self, env_config):
        low = np.full((128, 256, 10), -1000)
        high = np.full((128, 256, 10), 1000)
        self.action_space = Discrete(10)
        self.observation_space = Box(low=low, high=high, shape=(128, 256, 10), dtype=np.int32)
        self.obs = np.zeros((128, 256, 10))
        self.rew = 0 
        self.info = {}

        self.num_questions = len(data.index)
        print(self.num_questions)
        self.done=False
        self.eps_id = 0
        self.questions = data['Answer.question']
        self.answers = data['Answer.image.label']
        self.mcq_number ={
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
        self.reset()
    
    def reset(self):
        self.question = self.questions.iloc[self.eps_id]
        self.answer = self.answers.iloc[self.eps_id]

        if self.answer not in self.mcq_number:
            print(self.answer)
            self.answer = False
        else:
            self.answer = self.mcq_number[self.answer]

        obs = sequence(self.question)        
        self.obs = form_matrix(self.eps_id, obs)
        return self.obs

    def step(self, action):
        if action == self.answer:
            self.rew = 1
            self.eps_id += 1
            print(self.eps_id)

        elif not self.answer:
            self.rew = 0
        
        else:
            self.rew = -1

        try:
            assert self.observation_space.contains(self.obs)
        except AssertionError:
            print("INVALID STATE", self.obs)
        self.done =True
        if self.eps_id > self.num_questions-1:
            self.eps_id = 0
        return [self.obs, self.rew, self.done, self.info]

class neuronetwork(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(neuronetwork, self).__init__(obs_space, action_space, num_outputs, model_config, name)

        self.inputs = tf.keras.layers.Input(shape=obs_space.shape, name='observations', dtype=np.int32)

        layer_1 = tf.keras.layers.Conv2D(
            2, 3,
            name="my_layer1",
            padding="same",
            activation=tf.nn.relu,
            kernel_initializer=normc_initializer(1.0)
        )(self.inputs)

        layer_2 = tf.keras.layers.Flatten()(layer_1)
        
        layer_3 = tf.keras.layers.Dense(num_outputs)(layer_2)

        value_out = tf.keras.layers.Dense(
            1,
            name="value_out",
            activation=None,
            kernel_initializer=normc_initializer(0.01)
        )(layer_2)

        self.base_model = tf.keras.Model(self.inputs, [layer_3, value_out])
        self.register_variables(self.base_model.variables)
    
    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self.base_model(input_dict["obs"])
        return model_out, state
    
    def value_function(self):
        return tf.reshape(self._value_out, [-1])

if __name__ == "__main__":
    ModelCatalog.register_custom_model("my_model", neuronetwork)
    ray.init()

    trainer = DQNTrainer(
        env=realworldEnv,
        config=dict(
            **{
                "framework": "tf",
                "num_workers": 1,
                "model": {
                    "custom_model": "my_model",
                    "custom_model_config": {},
                },
            }
        )
    )
    while True:
        result = trainer.train()
        print(pretty_print(result))

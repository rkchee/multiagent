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
import pdb

ray_head_ip =os.environ.get('RAY_HEAD_SERVICE_HOST')
ray_redis_port = os.environ.get('RAY_HEAD_SERVICE_PORT_REDIS_PRIMARY')
url = '35.185.230.195'
    
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


def pullElement():
    headers = {"content-type": "application/json"}
    response = requests.get("http://" + url + ":8000/cube", headers=headers)
    t = response.text 
    t = json.loads(t)
    ET = t['Electra_Tensors']
    a = t['answers']
    return np.asarray(ET), a


class realworldEnv(gym.Env):
    def __init__(self, env_config):
        low = np.full((128, 256, 10), -1000)
        high = np.full((128, 256, 10), 1000)
        self.action_space = Discrete(10)
        self.observation_space = Box(low=low, high=high, shape=(128, 256, 10))        
        self.rew = 0 
        self.done=False
        self.info = {}

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
        self.done=False
        self.counter = 0
        self.obs, self.a = pullElement()
        self.n_elem = len(self.a)
        # print(self.n_elem)
        local_n_ran = random.choice(range(self.n_elem))
        self.answer = self.a[local_n_ran]
        while self.answer not in self.mcq_number:
            print("this is not an answer in the actions of mcq. We will move to next question and anwer: ", self.answer)
            # self.obs, self.answer = pullElement()
            local_n_ran = random.choice(range(self.n_elem))
            self.answer = self.a[local_n_ran]

        self.answer = self.mcq_number[self.answer]
        # self.obs = np.zeros((128, 256, 10))
        self.obs_curr = self.obs[local_n_ran]
        return self.obs_curr
    
    def step(self, action):
        if action == self.answer:
            self.rew = 1
            print("the answer is", self.answer, "the predicted answer is: ", action, "the rew: ", self.rew)
            # self.obs, a = pullElement()
            local_n_ran = random.choice(range(self.n_elem))
            self.answer = self.a[local_n_ran]
            while self.answer not in self.mcq_number:
                print("this is not an answer in the actions of mcq. We will move to next question and anwer: ", self.answer)
                # self.obs, self.answer = pullElement()
                local_n_ran = random.choice(range(self.n_elem))
                self.answer = self.a[local_n_ran]

            self.answer = self.mcq_number[self.answer]
            self.obs_next = self.obs[local_n_ran]
        else:
            self.rew = -1
            print("the answer is", self.answer, "the predicted answer is: ", action, "the rew: ", self.rew)
            self.obs_next = self.obs_curr
        self.counter += 1
        if self.counter >= 20:
            self.done = True

        return [self.obs_next, self.rew, self.done, self.info]



class neuronetwork(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(neuronetwork, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        import tensorflow as tf
        import tensorflow.keras.metrics
        import tensorflow.keras.losses

        self.inputs = tf.keras.layers.Input(shape=obs_space.shape, name='observations')

        layer_1 = tf.keras.layers.Conv2D(
            10, 3,
            name="my_layer1",
            padding="valid",
            activation=tf.nn.relu,
            kernel_initializer=normc_initializer(1.0)
        )(self.inputs)

        batchnorm_1 = tf.keras.layers.Dropout(.2)(layer_1)

        layer_2 = tf.keras.layers.Conv2D(
            50, 2, 
            strides=(2,2),
            padding="valid",
            activation=tf.nn.relu,
            kernel_initializer=normc_initializer(0.01)
        )(batchnorm_1)

        layer_dropout_2 = tf.keras.layers.Dropout(.2)(layer_2)

        layer_3 = tf.keras.layers.Conv2D(
            50, 3, 
            strides=(2,2),
            padding="valid",
            activation=tf.nn.relu,
            kernel_initializer=normc_initializer(0.01)
        )(layer_dropout_2)

        layer_dropout_3 = tf.keras.layers.Dropout(.2)(layer_3)

        layer_4 = tf.keras.layers.Conv2D(
            50, 3, 
            strides=(2,2),
            padding="valid",
            activation=tf.nn.relu,
            kernel_initializer=normc_initializer(0.01)
        )(layer_dropout_3)

        layer_dropout_4 = tf.keras.layers.Dropout(.2)(layer_4)

        layer_5 = tf.keras.layers.Conv2D(
            50, 3, 
            strides=(4,4),
            padding="valid",
            activation=tf.nn.relu,
            kernel_initializer=normc_initializer(0.01)
        )(layer_dropout_4)

        layer_dropout_5 = tf.keras.layers.Dropout(.2)(layer_5)

        layer_6 = tf.keras.layers.Conv2D(
            50, 2, 
            strides=(2,2),
            padding="valid",
            activation=tf.nn.relu,
            kernel_initializer=normc_initializer(0.01)
        )(layer_dropout_5)

        layer_dropout_6 = tf.keras.layers.Dropout(.2)(layer_6)

        layer_7 = tf.keras.layers.Conv2D(
            50, 2, 
            strides=(2,2),
            padding="valid",
            activation=tf.nn.relu,
            kernel_initializer=normc_initializer(0.01)
        )(layer_dropout_6)
    
        layer_8 = tf.keras.layers.Flatten()(layer_7)
        
        layer_9 = tf.keras.layers.Dense(num_outputs, kernel_initializer=normc_initializer(0.01))(layer_8)

        value_out = tf.keras.layers.Dense(
            1,
            name="value_out",
            activation=None,
            kernel_initializer=normc_initializer(0.01)
        )(layer_8)

        self.base_model = tf.keras.Model(self.inputs, [layer_9, value_out])
        self.register_variables(self.base_model.variables)
    
    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self.base_model(input_dict["obs"])
        return model_out, state
    
    def value_function(self):
        return tf.reshape(self._value_out, [-1])


if __name__ == "__main__":
    ModelCatalog.register_custom_model("my_model", neuronetwork)
    ray.init()
    # create_dask_func()    

    trainer = ApexTrainer(
        env=realworldEnv,
        config=dict(
            **{
                "exploration_config": {
                    "type": "EpsilonGreedy",
                    "initial_epsilon": 1.0,
                    "final_epsilon": 0.01,
                    # "epsilon_timesteps": 1000,
                    # "fraction": .1,
                },
                # "input": "/tmp/demo-out",
                # "input_evaluation": [],
                # "learning_starts": 100,
                # "timesteps_per_iteration": 200,
                # "log_level": "INFO",
                # "train_batch_size": 32,
                "framework": "tf",
                "num_workers": 6,
                "num_cpus_per_worker": 1,
                "buffer_size": 1000,
                # "double_q": False,
                # "dueling": False,
                # "num_atoms": 1,
                "noisy": False,
                "n_step": 3,
                # "lr": .0001,
                # "adam_epsilon": .00015,
                # "prioritized_replay_alpha": 0.5,
                "observation_filter": "MeanStdFilter",              
                # "lr": 1e-4,
                # "num_gpus": 4,
                "num_envs_per_worker": 1,
                "model": {
                    "custom_model": "my_model",
                    # Extra kwargs to be passed to your model's c'tor.
                    "custom_model_config": {},
                },                
            })
    )    

    # seq_element1 = sequence(data['Answer.question'].loc[1])
    # print(seq_element1)
    # seq_element1 = dask.delayed(sequence)(data['Answer.question'].loc[1])
    # print(seq_element1.compute())
    i=0
    while True:
        result = trainer.train()
        print(pretty_print(result))
        if i % 100 == 0:
            checkpoint = trainer.save()
            print("checkpoint saved at", checkpoint)
        else:
            i += 1




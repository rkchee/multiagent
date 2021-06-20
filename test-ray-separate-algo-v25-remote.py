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
import pdb
import dask.array as da
import h5py

#wing
ray_head_ip =os.environ.get('RAY_HEAD_SERVICE_HOST')
ray_redis_port = os.environ.get('RAY_HEAD_SERVICE_PORT_REDIS_PRIMARY')
url = '35.230.70.82'
    
def test(d, ys):
    c = random.choice(range(500))
    data = d[c]
    label = ys[c]
    return data, label

class realworldEnv(gym.Env):
    def __init__(self, env_config):
        import tensorflow as tf
        low = np.full((128, 256, 10), -1000)
        high = np.full((128, 256, 10), 1000)
        self.action_space = Discrete(10)
        self.observation_space = Box(low=low, high=high, shape=(128, 256, 10))        
        self.rew = 0 
        self.done=False
        self.info = {}
        self.buff_n = 4
        self.fdata = h5py.File('/combinedxy.hdf5', 'r')
        self.d = self.fdata['/x']
        self.ys = self.fdata['/y']        
 
        self.reset()
        
    def reset(self):
        self.obs, self.answer = test(self.d, self.ys)
        self.done=False
        self.counter = 0
        # self.obs_curr = self.array
        return self.obs
    
    def step(self, action):
        if action == self.answer:
            self.rew = 1
            print("the answer is", self.answer, "the predicted answer is: ", action, "the rew: ", self.rew)
            # self.obs = self.buffer_x
            self.obs, self.answer = test(self.d, self.ys)
        else:
            self.rew = -1
            print("the answer is", self.answer, "the predicted answer is: ", action, "the rew: ", self.rew)
            self.obs, self.answer = test(self.d, self.ys)


        self.counter += 1
        if self.counter >= 20:
            self.done = 1

        # self.obs_next = np.zeros((128, 256, 10))
        # self.rew = 0
        # self.done = False
        return [self.obs, self.rew, self.done, self.info]



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
    ray.init(address=ray_head_ip + ":" + ray_redis_port)

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
                "num_workers": 280,
                "num_cpus_per_worker": 1,
                "buffer_size": 20000,
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
                "num_gpus": 4,
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




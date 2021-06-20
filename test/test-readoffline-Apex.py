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

ray_head_ip =os.environ.get('RAY_HEAD_SERVICE_HOST')
ray_redis_port = os.environ.get('RAY_HEAD_SERVICE_PORT_REDIS_PRIMARY')

class realworldEnv(gym.Env):
    def __init__(self, env_config):
        low = np.full((128, 256, 10), -1000)
        high = np.full((128, 256, 10), 1000)
        self.action_space = Discrete(10)
        self.observation_space = Box(low=low, high=high, shape=(128, 256, 10), dtype=np.float32)
        

class neuronetwork(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(neuronetwork, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        import tensorflow as tf
        import tensorflow.keras.metrics
        import tensorflow.keras.losses

        self.inputs = tf.keras.layers.Input(shape=obs_space.shape, name='observations')

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
    ray.init(address=ray_head_ip + ":" + ray_redis_port)

    # trainer = DQNTrainer(
    #     env=realworldEnv,
    #     config=dict(
    #         **{
    #             "input": "/tmp/demo-out",
    #             "input_evaluation": [],
    #             "explore": False,
    #             "framework": "tf",
    #             "num_workers": 56,
    #             "model": {
    #                 "custom_model": "my_model",
    #                 "custom_model_config": {},
    #             },
    #         }
    #     )
    # )

    trainer = ApexTrainer(
        env=realworldEnv,
        config=dict(
            **{
                # "exploration_config": {
                #     "type": "EpsilonGreedy",
                #     "initial_epsilon": 1.0,
                #     "final_epsilon": 0.02,
                #     "epsilon_timesteps": 1000,
                # },
                "input": "/tmp/",
                "input_evaluation": [],
                "explore": False,
                # "learning_starts": 100,
                # "timesteps_per_iteration": 200,
                # "log_level": "INFO",
                # "train_batch_size": 32,
                "framework": "tf",
                "num_workers": 310,
                "buffer_size": 10000,
                "num_gpus": 4,
                "num_envs_per_worker": 1,
                "model": {
                    "custom_model": "my_model",
                    # Extra kwargs to be passed to your model's c'tor.
                    "custom_model_config": {},
                },                
            })
    )    

    for i in range(1000000000):
        result = trainer.train()
        print(pretty_print(result))
        if i % 100 == 0:
            checkpoint = trainer.save()
            print("checkpoint saved at", checkpoint)


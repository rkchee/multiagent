#!/usr/bin/env python
"""Example of running a policy server. Copy this file for your use case.
To try this out, in two separate shells run:
    $ python cartpole_server.py
    $ python cartpole_client.py --inference-mode=local|remote
"""

import argparse
import os

import ray
from ray.rllib.agents.dqn import DQNTrainer
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.a3c import A3CTrainer

from ray.rllib.env.policy_server_input import PolicyServerInput
from ray.tune.logger import pretty_print

import gym
from gym.spaces import Discrete, Box
import numpy as np

import tensorflow as tf
import tensorflow.keras.metrics
import tensorflow.keras.losses

# from ray.rllib.utils.framework import try_import_tf
# tf = try_import_tf()

from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.misc import normc_initializer

from ray.rllib.models import ModelCatalog # possibly causes a conflict with electra - roger


SERVER_ADDRESS = "0.0.0.0"
# SERVER_ADDRESS = "ntoml.canadacentral.cloudapp.azure.com"
SERVER_PORT = 9900
CHECKPOINT_FILE = "last_checkpoint_{}.out"

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=str, default="DQN")
parser.add_argument(
    "--framework", type=str, choices=["tf", "torch"], default="tf")

class realworldEnv(gym.Env):
   def __init__(self, env_config):
        import tensorflow as tf
        low = np.full((128, 256, 10), -1000)
        high = np.full((128, 256, 10), 1000)
        # output_low = np.full((32768, ), 0)
        # output_high = np.full((32768, ), 1)
        output_low = np.full((1, ), 0)
        output_high = np.full((1, ), 10)

        # self.action_space = Box(low=-1.0, high=2.0, shape=(1, ), dtype=np.float32)
        # self.action_space = Box(low=output_low, high=output_high, shape=(1, ), dtype=np.float32)
        self.action_space = Discrete(10)
        self.observation_space = Box(low=low, high=high, shape=(128, 256, 10), dtype=np.float32)

class vanillaKerasModel(TFModelV2):
    
    """Custom model for policy gradient algorithms."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        super(vanillaKerasModel, self).__init__(obs_space, action_space,
                                           num_outputs, model_config, name)
        import tensorflow as tf
        import tensorflow.keras.metrics
        import tensorflow.keras.losses
        self.inputs = tf.keras.layers.Input(
            shape=obs_space.shape, name="observations")

        layer_1 = tf.keras.layers.Conv2D(
            2, 3,
            name="my_layer1",
            padding = 'same',
            activation=tf.nn.relu,
            kernel_initializer=normc_initializer(1.0))(self.inputs)
        layer_2 = tf.keras.layers.Flatten()(layer_1)
        layer_3 = tf.keras.layers.Dense(
            num_outputs,
            name="model_out",
            activation=None,
            kernel_initializer=normc_initializer(0.01))(layer_2)
        # layer_out = tf.reshape(layer_1, (1, -1))
        value_out = tf.keras.layers.Dense(
            1,
            name="value_out",
            activation=None,
            kernel_initializer=normc_initializer(0.01))(layer_2)

        self.base_model = tf.keras.Model(self.inputs, [layer_3, value_out])
        
    
        # self.base_model = tf.keras.Model(self.inputs, [layer_out])

        self.register_variables(self.base_model.variables)

    # @tf.function
    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self.base_model(input_dict["obs"])
        return model_out, state

    # @tf.function
    def value_function(self):
        import tensorflow as tf
        import tensorflow.keras.metrics
        import tensorflow.keras.losses
        return tf.reshape(self._value_out, [-1])

    # @tf.function
    def metrics(self):
        import tensorflow as tf
        import tensorflow.keras.metrics
        import tensorflow.keras.losses

        return {"foo": tf.constant(42.0)}


if __name__ == "__main__":
    # url = "34.94.143.197"
    # url = "ntoml.canadacentral.cloudapp.azure.com"
    # url = "52.228.66.9"
    # url = "34.94.83.0"
    # url = "35.233.170.130"
    # url = "34.82.117.252"
    url = "localhost"
    args = parser.parse_args()
    env = realworldEnv
    ModelCatalog.register_custom_model("my_model", vanillaKerasModel)
    ray.init(address= url + ":6000", redis_password='5241590000000000')
    # ray.init()

    # env = "CartPole-v0"
    connector_config = {
        # Use the connector server to generate experiences.
        "input": (
            lambda ioctx: PolicyServerInput(ioctx, SERVER_ADDRESS, SERVER_PORT)
        ),
        # Use a single worker process to run the server.
        "num_workers": 0,
        # Disable OPE, since the rollouts are coming from online clients.
        "input_evaluation": [],
    }

    if args.run == "DQN":
        # Example of using DQN (supports off-policy actions).
        trainer = DQNTrainer(
            env=env,
            config=dict(
                connector_config, **{
                    "exploration_config": {
                        "type": "EpsilonGreedy",
                        "initial_epsilon": 1.0,
                        "final_epsilon": 0.02,
                        "epsilon_timesteps": 1000,
                    },
                    "learning_starts": 100,
                    "timesteps_per_iteration": 200,
                    "log_level": "INFO",
                    "framework": args.framework,
                    "model": {
                        "custom_model": "my_model",
                        # Extra kwargs to be passed to your model's c'tor.
                        "custom_model_config": {},
                    }                    
                }))
    elif args.run == "PPO":
        # Example of using PPO (does NOT support off-policy actions).
        
        trainer = PPOTrainer(
            env=env,
            config=dict(
                connector_config, **{
                    "sample_batch_size": 100,
                    "train_batch_size": 400,
                    "framework": args.framework,
                    "model": {
                        "custom_model": "my_model",
                        # Extra kwargs to be passed to your model's c'tor.
                        "custom_model_config": {},
                    }
                }))
    else:
        raise ValueError("--run must be DQN or PPO")

    checkpoint_path = CHECKPOINT_FILE.format(args.run)

    # Attempt to restore from checkpoint if possible.
    # if os.path.exists(checkpoint_path):
    #     checkpoint_path = open(checkpoint_path).read()
    #     print("Restoring from checkpoint path", checkpoint_path)
    #     trainer.restore(checkpoint_path)

    # # Serving and training loop
    while True:
        print(pretty_print(trainer.train()))
    #     checkpoint = trainer.save()
    #     print("Last checkpoint", checkpoint)
    #     with open(checkpoint_path, "w") as f:
    #         f.write(checkpoint)

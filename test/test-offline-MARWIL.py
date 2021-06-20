import os
import gym
from gym.spaces import Discrete, Box
import numpy as np


from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.misc import normc_initializer

from ray.rllib.models import ModelCatalog # possibly causes a conflict with electra - roger

from ray.rllib.agents.dqn import DQNTrainer
from ray.rllib.agents.dqn import dqn
from ray.rllib.agents.a3c import A3CTrainer
from ray.rllib.agents.marwil import MARWILTrainer




import ray
import ray.rllib.agents.ppo as ppo
from ray.tune.logger import pretty_print

import tensorflow as tf
import tensorflow.keras.metrics
import tensorflow.keras.losses


class realworldEnv(gym.Env):
   def __init__(self, env_config):
        import tensorflow as tf
        low = np.full((128, 256, 10), -1000)
        high = np.full((128, 256, 10), 1000)
        # output_low = np.full((, 1), 0)
        # output_high = np.full((, 1), 10)

        # self.action_space = Box(low=0, high=10.0, shape=(1, ), dtype=np.float32)
        self.action_space = Discrete(10)
        # self.action_space = Box(low=output_low, high=output_high, shape=(, 1), dtype=np.int32)

        self.observation_space = Box(low=low, high=high, shape=(128, 256, 10), dtype=np.float32)
        # self.observation_space = Discrete(1)

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
        layer_3 = tf.keras.layers.Dense(num_outputs)(layer_2)
        # layer_4 = tf.keras.layers.Dense(256)(layer_3)
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
    env = realworldEnv
    # env = 'CartPole-v0'
    ModelCatalog.register_custom_model("my_model", vanillaKerasModel)
    ray.init()

    # config = dqn.DEFAULT_CONFIG.copy()
    # config["num_gpus"] = 0
    # config["num_workers"] = 1
    # config["eager"] = False

    # trainer = DQNTrainer(config=config, env="CartPole-v0")

    trainer = MARWILTrainer(
        env=env,
        config=dict(
            **{
                # "exploration_config": {
                #     "type": "EpsilonGreedy",
                #     "initial_epsilon": 1.0,
                #     "final_epsilon": 0.02,
                #     "epsilon_timesteps": 1000,
                # },
                "input": "/tmp/demo-out",
                # "input_evaluation": [],
                # "explore": False,
                # # "learning_starts": 100,
                # # "timesteps_per_iteration": 200,
                # "log_level": "INFO",
                "train_batch_size": 32,
                "framework": "tf",
                "num_workers": 6,
                "model": {
                    "custom_model": "my_model",
                    # Extra kwargs to be passed to your model's c'tor.
                    "custom_model_config": {},
                },                
            })
    )


    for i in range(1000):
        # Perform one iteration of training the policy with PPO
        result = trainer.train()
        print(pretty_print(result))

        if i % 100 == 0:
            checkpoint = trainer.save()
            print("checkpoint saved at", checkpoint)



    # trainer = DQNTrainer(...)
    # train policy offline

    # from ray.rllib.offline.json_reader import JsonReader
    # from ray.rllib.offline.wis_estimator import WeightedImportanceSamplingEstimator

    # estimator = WeightedImportanceSamplingEstimator(trainer.get_policy(), gamma=0.99)
    # reader = JsonReader("/path/to/data")
    # for _ in range(1000):
    #     batch = reader.next()
    #     for episode in batch.split_by_episode():
    #         print(estimator.estimate(episode))
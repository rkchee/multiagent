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
    url = '34.83.105.129'
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

# def form_matrix(sequences):
#     matrix_max_size = 10
#     matrix = []

#     for i in sequences:
#         answer = sequence(i)
#         matrix.append(answer)
    
#     matrix = np.asarray(matrix)
#     matrix = np.squeeze(matrix)
#     state_matrix = np.pad(matrix, [(0, matrix_max_size - matrix.shape[0]), (0,0), (0,0)])
#     state_matrix = np.moveaxis(state_matrix, 0, 2)
#     return state_matrix


class realworldEnv(gym.Env):
    def __init__(self, env_config):
        import tensorflow as tf
        self.batch_size = 10
        self.random_number_size = 25
        low = np.full((128, 256, 10), -1000)
        high = np.full((128, 256, 10), 1000)
        self.action_space = Discrete(10)
        self.observation_space = Box(low=low, high=high, shape=(128, 256, 10))
        # self.obs = np.zeros((128, 256, 10))
        self.rew = 0 
        self.info = {}
        self.done=False

        
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
    

    def sequence(seq):
        url = '34.83.105.129'
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
            answer = self.sequence(i)
            matrix.append(answer)
        
        matrix = np.asarray(matrix)
        matrix = np.squeeze(matrix)
        state_matrix = np.pad(matrix, [(0, matrix_max_size - matrix.shape[0]), (0,0), (0,0)])
        state_matrix = np.moveaxis(state_matrix, 0, 2)
        return state_matrix
    
    def reset(self):
        self.done=False
        self.columns = ['Answer.answerA', 'Answer.answerB', 'Answer.answerC', 'Answer.answerD', 'Answer.answerE', 'Answer.answerF', 'Answer.image.label', 'Answer.question']
        self.parts = dask.delayed(pd.read_csv)('train_set.csv', usecols=self.columns)

        self.data = dd.from_delayed(self.parts).sample(frac=0.04)
        self.eps_id = random.randrange(0, self.batch_size - 1, 1)
        # self.data['Answer.question'].loc[1]
        store = []
        
        eel=1
        sq = [self.data['Answer.question'].loc[eel], self.data['Answer.answerA'].loc[eel], self.data['Answer.answerB'].loc[eel], self.data['Answer.answerC'].loc[eel], self.data['Answer.answerD'].loc[eel], self.data['Answer.answerE'].loc[eel], self.data['Answer.answerF'].loc[eel]]
        obs = dask.delayed(self.form_matrix)(sq)
        store.append(obs)

        store = dask.compute(*store)


        # self.answers = self.data['Answer.image.label'].reset_index(drop=True)
        # self.answer = dask.compute(self.answers.loc[self.eps_id].values)[0][0]

        # while self.answer not in self.mcq_number:
        #     print("this is not an answer in the actions of mcq. We will move to next question and anwer: ", self.answer)
        #     self.eps_id = random.randrange(0, self.batch_size-1, 1)
        #     self.answer = dask.compute(self.answers.loc[self.eps_id].values)[0][0]

        # self.answer = self.mcq_number[self.answer]
        self.answer=1
        # self.obs = np.squeeze(dask.compute(*self.store_list)[self.eps_id])
        self.obs = np.zeros((128, 256, 10))

        self.counter = 0
        return self.obs
    
    def step(self, action):
        if action == self.answer:
            self.rew = 1
            # print("score")
            print("The question id is: ", self.eps_id, "the answer is", self.answer, "the predicted answer is: ", action, "the rew: ", self.rew)
            self.eps_id = random.choice(range(self.batch_size))
            # self.answer = dask.compute(self.answers.loc[self.eps_id].values)[0][0]


            # while self.answer not in self.mcq_number:
            #     print("this is not an answer in the actions of mcq. We will move to next question and anwer: ", self.answer)
            #     self.eps_id = random.choice(range(self.batch_size))                
            #     # print("the episode is ", self.eps_id, "the answers is shape", np.shape(self.answers))
            #     self.answer = dask.compute(self.answers.loc[self.eps_id].values)[0][0]
            #     # self.answer = [1]
            # # print(np.shape(self.answers))
            # self.answer = self.mcq_number[self.answer]
            # self.obs = np.squeeze(dask.compute(*self.store_list)[self.eps_id])
            self.answer = 1


        else:
            self.rew = -1
            print("The question id is: ", self.eps_id, "the answer is", self.answer, "the predicted answer is: ", action, "the rew: ", self.rew)
            # print(self.random_test)
        self.counter += 1
        if self.counter >= 20:
            self.done = True

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
    # ray.init(address=ray_head_ip + ":" + ray_redis_port)
    ray.init()
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
                "num_gpus": 4,
                "num_envs_per_worker": 1,
                "model": {
                    "custom_model": "my_model",
                    # Extra kwargs to be passed to your model's c'tor.
                    "custom_model_config": {},
                },                
            })
    )    

    i=0
    while True:
        result = trainer.train()
        print(pretty_print(result))
        if i % 100 == 0:
            checkpoint = trainer.save()
            print("checkpoint saved at", checkpoint)
        else:
            i += 1

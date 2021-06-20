import ray 
from ray.tune.logger import pretty_print
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models import ModelCatalog
from ray.rllib.agents.dqn import DQNTrainer
from ray.rllib.agents.dqn import ApexTrainer
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
import dask.array as da


import gym
from gym.spaces import Discrete, Box
import requests
import tensorflow as tf
import tensorflow.keras.metrics
import tensorflow.keras.losses
# import pandas as pd
import pandas as pd
import json
import numpy as np

import os 

ray_head_ip =os.environ.get('RAY_HEAD_SERVICE_HOST')
ray_redis_port = os.environ.get('RAY_HEAD_SERVICE_PORT_REDIS_PRIMARY')

# ray.init(ignore_reinit_error=True)
# ray.init()

url ='35.233.198.75'
# data = pd.read_csv('/home/rkchee/nxtopinion/rl/rheumatology_4199362_batch_results.csv')[0:5]

# train test split
# sheets_cards = pd.read_excel('training dataset - medical.xlsx', sheet_name='Cardiology and Infectious Disea')
# cardiology = sheets_cards.copy()
# train_set = cardiology.sample(frac=0.01, random_state=0)
# train_set.to_csv('train_set.csv', index=False)
# test_set = cardiology.drop(train_set.index)
# test_set.to_csv('test_set.csv', index=False)

# set the training data 
# data = pd.read_csv('train_set.csv')
# data = pd.DataFrame(sheets_cards)[0:10]
# data = da.random.random((128,256,10), chunks=(128,256, 5))

import dask
import dask.dataframe as dd
from dask.delayed import delayed

# data = dd.read_csv('training dataset - medical.xlsx', sheet_name='Cardiology and Infectious Disea')
parts = dask.delayed(pd.read_excel)('training dataset - medical.xlsx', sheet_name='Cardiology and Infectious Disea')
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
        self.observation_space = Box(low=low, high=high, shape=(128, 256, 10))
        self.obs = np.zeros((128, 256, 10))
        self.rew = 0 
        self.info = {}
        self.num_questions = len(data.index)
        print(self.num_questions)
        self.done=False
        self.eps_id = 0


        # nrows, ncols = self.num_questions, 1
        # self.store_list = np.memmap('memmapped.dat', dtype=object,
        #             mode='w+', shape=(nrows, ncols))

        self.store_list = []
        # self.store_list = []

        for i in data.index:
            q = data['Answer.question'].loc[i]
            obs = dask.delayed(sequence)(q)
            state = dask.delayed(form_matrix)(i, obs)
            self.store_list.append(obs)
        


        # self.store_list = np.asarray(self.store_list)

        # self.store_memmap = np.asarray(self.store_memmap)
        # self.questions = data['Answer.question']
        self.answers = np.asarray(data['Answer.image.label'])

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
        # self.question = self.store_memmap[self.eps_id]
        self.answer = self.answers[self.eps_id] #this line is likely not needed

        while self.answer not in self.mcq_number:
            print("this is not an answer in the actions of mcq. We will move to next question and anwer: ", self.answer)
            self.eps_id += 1
            # self.question = self.questions.iloc[self.eps_id]
            self.answer = self.answers[self.eps_id]

        self.answer = self.mcq_number[self.answer]
        # obs = sequence(self.question)        
        # self.obs = form_matrix(self.eps_id, obs)
        # self.obs = np.asarray(self.store_memmap[self.eps_id, 0], dtype=np.float32)
        self.obs = np.squeeze(dask.compute(self.store_list[self.eps_id]))
        self.obs = np.zeros((126, 256, 10))
        return [self.obs]
    
    def step(self, action):
        if action == self.answer:
            self.rew = 1
            print("The question id is: ", self.eps_id, "the answer is", self.answer, "the predicted answer is: ", action, "the rew: ", self.rew)
            self.eps_id += 1
            self.done =True

        else:
            self.rew = -10
            print("The question id is: ", self.eps_id, "the answer is", self.answer, "the predicted answer is: ", action, "the rew: ", self.rew)
        


        if self.eps_id > self.num_questions-1:
            self.eps_id = 0
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


        layer_2 = tf.keras.layers.Conv2D(
            50, 2, 
            strides=(2,2),
            padding="valid",
            activation=tf.nn.relu,
        )(layer_1)

        layer_3 = tf.keras.layers.Conv2D(
            50, 3, 
            strides=(2,2),
            padding="valid",
            activation=tf.nn.relu,
        )(layer_2)

        layer_4 = tf.keras.layers.Conv2D(
            50, 3, 
            strides=(2,2),
            padding="valid",
            activation=tf.nn.relu,
        )(layer_3)

        layer_5 = tf.keras.layers.Conv2D(
            50, 3, 
            strides=(4,4),
            padding="valid",
            activation=tf.nn.relu,
        )(layer_4)

        layer_6 = tf.keras.layers.Conv2D(
            50, 2, 
            strides=(2,2),
            padding="valid",
            activation=tf.nn.relu,
        )(layer_5)

        layer_7 = tf.keras.layers.Conv2D(
            50, 2, 
            strides=(2,2),
            padding="valid",
            activation=tf.nn.relu,
        )(layer_6)
    
        layer_8 = tf.keras.layers.Flatten()(layer_7)
        
        layer_9 = tf.keras.layers.Dense(num_outputs)(layer_8)

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
    trainer = DQNTrainer(
        env=realworldEnv,
        config=dict(
            **{
                "framework": "tf",
                "num_workers": 2,
                "buffer_size": 1000,
                "observation_filter": "MeanStdFilter",
                "lr": 1e-4,
                # "num_gpus": 4,
                "model": {
                    "custom_model": "my_model",
                    "custom_model_config": {},
                },
            }
        )
    )

    # trainer = ApexTrainer(
    #     env=realworldEnv,
    #     config=dict(
    #         **{
    #             "exploration_config": {
    #                 "type": "EpsilonGreedy",
    #                 "initial_epsilon": 1.0,
    #                 "final_epsilon": 0.01,
    #                 # "epsilon_timesteps": 1000,
    #                 # "fraction": .1,
    #             },
    #             # "input": "/tmp/demo-out",
    #             # "input_evaluation": [],
    #             # "learning_starts": 100,
    #             # "timesteps_per_iteration": 200,
    #             # "log_level": "INFO",
    #             # "train_batch_size": 32,
    #             "framework": "tf",
    #             "num_workers": 2,
    #             "buffer_size": 2000,
    #             "double_q": False,
    #             "dueling": False,
    #             "num_atoms": 1,
    #             "noisy": False,
    #             "n_step": 3,
    #             "lr": .0001,
    #             "adam_epsilon": .00015,
    #             "prioritized_replay_alpha": 0.5,
    #             "observation_filter": "MeanStdFilter",              
    #             # "lr": 1e-4,
    #             # "num_gpus": 4,
    #             "num_envs_per_worker": 1,
    #             "model": {
    #                 "custom_model": "my_model",
    #                 # Extra kwargs to be passed to your model's c'tor.
    #                 "custom_model_config": {},
    #             },                
    #         })
    # )    

    i=0
    while True:
        i +=1
        result = trainer.train()
        print(pretty_print(result))
        if i % 100 == 0:
            checkpoint = trainer.save()
            print("checkpoint saved at", checkpoint)


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
import dask
import dask.array as da
import dask.dataframe as dd
from dask.delayed import delayed
import dask.bag as db
from distributed import Client
# client = Client()
import random
# tf.compat.v1.enable_eager_execution(
#     config=None, device_policy=None, execution_mode=None
# )

ray_head_ip =os.environ.get('RAY_HEAD_SERVICE_HOST')
ray_redis_port = os.environ.get('RAY_HEAD_SERVICE_PORT_REDIS_PRIMARY')
url = '34.83.105.129'

@tf.autograph.experimental.do_not_convert
def create_matrix(samples, targets):
    state_cube = ['Answer.question', 'Answer.answerA', 'Answer.answerB', 'Answer.answerC', 'Answer.answerD', 'Answer.answerE', 'Answer.answerF']
    matrix = []
    for k in state_cube:
        matrix.append(samples[k])
    return matrix, targets


batch_size = 5
columns = ['Answer.answerA', 'Answer.answerB', 'Answer.answerC', 'Answer.answerD', 'Answer.answerE', 'Answer.answerF', 'Answer.image.label', 'Answer.question']
data = tf.data.experimental.make_csv_dataset('train_set.csv', batch_size=batch_size, select_columns=columns, label_name='Answer.image.label', num_epochs=1, ignore_errors=True)
data = data.map(create_matrix)
# state_cube = ['Answer.question', 'Answer.answerA', 'Answer.answerB', 'Answer.answerC', 'Answer.answerD', 'Answer.answerE', 'Answer.answerF']


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

    for i in sequences:
        answer = sequence(i)
        matrix.append(answer)
    
    matrix = np.asarray(matrix)
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
        # self.num_questions = len(data.index)
        # print(self.num_questions)
        self.done=False
        
        #must be a list to use the persist function of dask
        self.store_list = []
        
        # for i in data.index:
        #     q = data['Answer.question'].loc[i]
        #     obs = dask.delayed(sequence)(q)
        #     state = dask.delayed(form_matrix)(i, obs)
        #     self.store_list[0].append(state)
           
        #     ans = data['Answer.image.label'].loc[i].to_dask_array()
        #     self.store_list[1].append(ans)
        

        # self.store_list[0] = dask.persist(*self.store_list[0])
        # self.store_list[1] = dask.persist(*self.store_list[1])
        # print(np.shape(self.store_list[0]))




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
        # for i in range(len(self.qnatext[0])):
        #     seq = [self.qnatext[0][i], self.qnatext[1][i], self.qnatext[2][i], self.qnatext[3][i], self.qnatext[4][i], self.qnatext[5][i], self.qnatext[6][i]]
        #     obs = form_matrix(seq)
        #     self.store_list.append(obs)

        self.eps_id = random.randrange(0, batch_size-1, 1)
        # # self.answer = dask.compute(self.store_list[1][self.eps_id])
        # # print(self.answer[0])

        # self.answer = self.answers[self.eps_id]

        # while self.answer not in self.mcq_number:
        #     print("this is not an answer in the actions of mcq. We will move to next question and anwer: ", self.answer)
        #     self.eps_id = random.randrange(0, batch_size-1, 1)
        #     self.answer = self.answers[self.eps_id]
        #     # print(self.answer)

        # self.answer = self.mcq_number[self.answer]

        # self.obs = np.squeeze(self.store_list[self.eps_id])
        self.obs = np.zeros((128, 256, 10))
        self.answer = np.array([1])
        self.counter = 0
        return self.obs
    
    def step(self, action):
        if action == self.answer:
            self.rew = 1
            print("The question id is: ", self.eps_id, "the answer is", self.answer, "the predicted answer is: ", action, "the rew: ", self.rew)
            self.eps_id = random.choice(range(batch_size))
            # self.obs = self.store_list[eps_id]


        else:
            self.rew = -1
            print("The question id is: ", self.eps_id, "the answer is", self.answer, "the predicted answer is: ", action, "the rew: ", self.rew)

        self.counter += 1
        if self.counter >= 20:
            self.done = True

        # if self.eps_id > self.num_questions-1:
        #     self.eps_id = 0
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
    qnatext, answers = next(iter(data.take(1)))


    # trainer = DQNTrainer(
    #     env=realworldEnv,
    #     config=dict(
    #         **{
    #             "framework": "tf",
    #             "num_workers": 2,
    #             "buffer_size": 1000,
    #             "observation_filter": "MeanStdFilter",
    #             "lr": 1e-4,
    #             # "num_gpus": 4,
    #             "model": {
    #                 "custom_model": "my_model",
    #                 "custom_model_config": {},
    #             },
    #         }
    #     )
    # )

    # batch_size = 20
    # n = len(data.index)
    # total_batches = n/batch_size
    # batch_num = 1
    # data = dataset(batch_size, data)
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
        if i % 20 == 0:
            qnatext, answers = next(iter(data.take(1)))

        if i % 100 == 0:
            checkpoint = trainer.save()
            print("checkpoint saved at", checkpoint)
        else:
            i += 1

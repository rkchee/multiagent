import ray
from ray.tune.logger import pretty_print
from ray.rllib.models.tf.misc import normc_initializer as rllib_normc_initializer
from ray.rllib.models import ModelCatalog # possibly causes a conflict with electra - roger
from ray.rllib.agents.dqn import DQNTrainer
from ray.rllib.models.tf.tf_modelv2 import TFModelV2 as rllib_TFModelV2

import gym
from gym.spaces import Discrete, Box
import requests 
import tensorflow as tf
import tensorflow.keras.metrics
import tensorflow.keras.losses
import pandas as pd
import json
import numpy as np

url ="35.230.63.77"
# sheets_cards = pd.read_excel('training dataset - medical.xlsx', sheet_name='Cardiology and Infectious Disea')
data = pd.read_csv('/home/rkchee/nxtopinion/rl/rheumatology_4199362_batch_results.csv')[0:30]
# data = pd.read_csv('/workspace/nxtopinion/rl/rheumatology_4199362_batch_results.csv')


class electra_serve:
    def sequence(seq):
        if not type(seq) == str:
            seq = str(seq)            
            print(seq)

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

class state_obs:
    def form_matrix(qa_index, question):

        matrix_max_size = 10
        matrix = []


        answerA = data['Answer.answer' + 'A'].iloc[qa_index]
        answerA = electra_serve.sequence(answerA)
        matrix = np.concatenate((question, answerA), axis=0)

        answerB = data['Answer.answer' + 'B'].iloc[qa_index]
        answerB = electra_serve.sequence(answerB)
        matrix = np.concatenate((matrix, answerB), axis=0)

        answerC = data['Answer.answer' + 'C'].iloc[qa_index]
        answerC = electra_serve.sequence(answerC)
        matrix = np.concatenate((matrix, answerC), axis=0)

        answerD = data['Answer.answer' + 'D'].iloc[qa_index]
        answerD = electra_serve.sequence(answerD)
        matrix = np.concatenate((matrix, answerD), axis=0)

        answerE = data['Answer.answer' + 'E'].iloc[qa_index]
        answerE = electra_serve.sequence(answerE)
        matrix = np.concatenate((matrix, answerE), axis=0)

        matrix = np.squeeze(matrix)
        state_matrix = np.pad(matrix,[(0,matrix_max_size - matrix.shape[0]), (0,0), (0,0)])
        state_matrix = np.moveaxis(state_matrix, 0, 2)
        return state_matrix

class realworldEnv(gym.Env):
    def __init__(self, env_config):
        low = np.full((128, 256, 10), -1000)
        high = np.full((128, 256, 10), 1000)
        self.action_space = Discrete(10)
        self.observation_space = Box(low=low, high=high, shape=(128, 256, 10), dtype=np.float32)
        self.num_questions = len(data.index)
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

        # self.eps_id = 0
        # self.question = self.questions.iloc[self.eps_id]
        # # question = np.zeros((128, 256, 10))

        # self.answer = self.answers.iloc[self.eps_id]
        # if self.answer not in self.mcq_number:
        #     print(self.answer)
        #     self.answer = False
        # else:
        #     self.answer = self.mcq_number[self.answer]

        # obs = electra_serve.sequence(self.question)        
        # self.obs = state_obs.form_matrix(self.eps_id, obs)
        self.reset()

    
    def reset(self):
        self.done=False
        self.question = self.questions.iloc[self.eps_id]
        self.answer = self.answers.iloc[self.eps_id]

        if self.answer not in self.mcq_number:
            print(self.answer)
            self.answer = False
        else:
            self.answer = self.mcq_number[self.answer]

        obs = electra_serve.sequence(self.question)        
        self.obs = state_obs.form_matrix(self.eps_id, obs)
        self.obs = np.zeros((128, 256, 10))
        return np.zeros((128, 256, 10))
        # return self.obs

    def step(self, action):
        assert action in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], action

        if action == self.answer:
            rew = 1
            self.eps_id += 1
                # self.question = self.questions.iloc[self.eps_id]
                # # # question = np.zeros(128, 256, 10)
                # self.answer = self.mcq_number[self.answers.iloc[self.eps_id]]

            # qTokens = tf.constant(tokenizer.encode(str(question), max_length=128, pad_to_max_length=128))[None, :]
            # obs = model(qTokens)
            # self.obs = state_obs.form_matrix(self.eps_id, obs)

        elif not self.answer:
            rew = 0
        
        else:
            rew = -1

        
        self.done = True      
        # print(self.eps_id)      
        if self.eps_id > self.num_questions-1:
            self.eps_id = 0
        # self.reward = self.reward + rew
        # print(self.reward)

        return [self.obs, rew, self.done, {}]

class vanillaKerasModel(rllib_TFModelV2):
    
    """Custom model for policy gradient algorithms."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        super(vanillaKerasModel, self).__init__(obs_space, action_space,
                                           num_outputs, model_config, name)
        self.inputs = tf.keras.layers.Input(
            shape=obs_space.shape, name="observations")


        layer_1 = tf.keras.layers.Conv2D(
            2, 3,
            name="my_layer1",
            padding = 'same',
            activation=tf.nn.relu,
            kernel_initializer=rllib_normc_initializer(1.0)
            )(self.inputs)
        layer_2 = tf.keras.layers.Flatten()(layer_1)
        layer_3 = tf.keras.layers.Dense(num_outputs)(layer_2)
        # layer_4 = tf.keras.layers.Dense(256)(layer_3)
        value_out = tf.keras.layers.Dense(
            1,
            name="value_out",
            activation=None,
            kernel_initializer=rllib_normc_initializer(0.01)
            )(layer_2)

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
    ModelCatalog.register_custom_model("my_model", vanillaKerasModel)
    ray.init()
    trainer = DQNTrainer(
        env=env,
        config=dict(
            **{
                # "exploration_config": {
                #     "type": "EpsilonGreedy",
                #     "initial_epsilon": 1.0,
                #     "final_epsilon": 0.02,
                #     "epsilon_timesteps": 1000,
                # },
                # "input": "/tmp/demo-out",
                # "input_evaluation": [],
                # "explore": False,
                # "learning_starts": 100,
                # "timesteps_per_iteration": 200,
                # "log_level": "INFO",
                # "train_batch_size": 32,
                "framework": "tf",
                "num_workers": 6,
                "model": {
                    "custom_model": "my_model",
                    # Extra kwargs to be passed to your model's c'tor.
                    "custom_model_config": {},
                },                
            })
    )
    for i in range(10000):
        # Perform one iteration of training the policy with DQN
        result = trainer.train()
        print(pretty_print(result))

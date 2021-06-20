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

url = '35.199.191.27'

# restore the training data from the prior training examples

# use this to test of the train set. 
# data = pd.read_csv('~/cluster-apex-40cpu-testdata/train_set.csv')

#use this for testin on the test set. 
# data = pd.read_csv('/home/rkchee/cluster-apex310cpu-dataset/train_set.csv')
data = pd.read_csv('/home/rkchee/nxtopinion/rl/test_set.csv')

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
    ray.init()

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
                # "input": "/tmp/demo-out",
                # "input_evaluation": [],
                # "explore": False,
                # "learning_starts": 100,
                # "timesteps_per_iteration": 200,
                # "log_level": "INFO",
                # "train_batch_size": 32,
                "framework": "tf",
                "num_workers": 6,
                "buffer_size": 2000,
                # "num_gpus": 1,
                "num_envs_per_worker": 1,
                "model": {
                    "custom_model": "my_model",
                    # Extra kwargs to be passed to your model's c'tor.
                    "custom_model_config": {},
                },                
            })
    )    

    # trainer.restore('/home/rkchee/cluster-apex70cpu-check201/APEX_realworldEnv_2020-11-26_23-46-53xxniggl1/checkpoint_201/checkpoint-201')
    # trainer.restore('/home/rkchee/cluster-apex150cpu-r001/APEX_realworldEnv_2020-11-27_17-56-20mhb1r10s/checkpoint_1/checkpoint-1')
    # trainer.restore('/home/rkchee/cluster-apex310cpu-r001/APEX_realworldEnv_2020-11-27_20-17-34lb_lnvo8/checkpoint_1/checkpoint-1')
    # trainer.restore('/home/rkchee/cluster-apex310cpu-rewards-r005/APEX_realworldEnv_2020-11-29_08-35-02x0c914vb/checkpoint_1/checkpoint-1')
    trainer.restore('/home/rkchee/ray_results/DQN_realworldEnv_2020-11-30_06-09-31p8_ftq14/checkpoint_601/checkpoint-601')
    questions = data['Answer.question']
    answers = data['Answer.image.label']

    n = len(questions.index)
    # print(n)

    mcq_number ={
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

    t = 0 
    rewards = 0         
    index_answer = 0 
    
    for question in questions:
        obs = sequence(question)        
        obs = form_matrix(index_answer, obs)
        answer = answers.iloc[index_answer]
        if answer not in mcq_number:
            # print(answer)
            answer = False
        else:
            answer = mcq_number[answer]
        # print(answer)
        action = trainer.compute_action(obs)
        if action == answer:
            rew = 1
            
        elif not answer:
            rew = 0
        else:
            rew = 0
        
        rewards += rew
        index_answer += 1
        acc = rewards/index_answer*100
        print("question number", index_answer, "the answer is ", answer, "the action is ", action, "the acc is ", acc)

        # print("episode: " + str(index_answer) + "  score: " + str(rewards))
    print('completed set of questions')






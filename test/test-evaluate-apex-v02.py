"""Simple example of writing experiences to a file using JsonWriter."""

# __sphinx_doc_begin__
import gym
import numpy as np
import os
import pandas as pd


import tensorflow as tf
from transformers import ElectraTokenizer, TFElectraModel
tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator', max_length=128, pad_to_max_length=True)
model = TFElectraModel.from_pretrained('google/electra-small-discriminator')

import ray.utils

from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.evaluation.sample_batch_builder import SampleBatchBuilder
from ray.rllib.offline.json_writer import JsonWriter


import tensorflow.keras.metrics
import tensorflow.keras.losses



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
                "num_gpus": 1,
                "num_envs_per_worker": 1,
                "model": {
                    "custom_model": "my_model",
                    # Extra kwargs to be passed to your model's c'tor.
                    "custom_model_config": {},
                },                
            })
    )    

trainer.restore('/home/rkchee/cluster-apex-40cpu2/APEX_realworldEnv_2020-11-21_09-28-12637u8x8s/checkpoint_201/checkpoint-201')

if __name__ == "__main__":

    # restore the training data from the prior training examples
    data = pd.read_csv('~/cluster-apex-40cpu-testdata/train_set.csv')
    
    def form_matrix(qa_index, question):
        matrix_max_size = 10
        matrix = []

        mc_choices = ["A", "B", "C", "D", "E"]

        matrix = question
        for letter in mc_choices:
            answer = data['Answer.answer' +  letter].iloc[qa_index]
            ans_Tokens = tf.constant(tokenizer.encode(str(answer), max_length=128, pad_to_max_length=128))[None, :]
            answer = model(ans_Tokens)
            matrix = np.concatenate((matrix, answer), axis=0)
        
        matrix = np.squeeze(matrix)
        state_matrix = np.pad(matrix, [(0, matrix_max_size - matrix.shape[0]), (0,0), (0,0)])
        state_matrix = np.moveaxis(state_matrix, 0, 2)
        return state_matrix


    questions = data['Answer.question']
    answers = data['Answer.image.label']

    n = len(questions.index)
    print(n)

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

    eps_id = 0
    prev_reward = 0
    t = 0 
    rewards = 0         
    done = False
    index_answer = 0 
    
    for question in questions:
        qTokens = tf.constant(tokenizer.encode(str(question), max_length=128, pad_to_max_length=128))[None, :]
        obs = model(qTokens)
        obs = form_matrix(index_answer, obs)
        answer = answers.iloc[index_answer]
        if answer not in mcq_number:
            print(answer)
            answer = False
        else:
            answer = mcq_number[answer]

        prev_action = np.zeros_like(answer)
        # print("the answer is ", answer)


        while True:
            action = mcq_number[letter]
            if action == answer:
                rew = 1
                index_answer += 1
            elif not answer:
                rew = 0
            else:
                rew = -1

            
            # batch_builder.add_values(
            #     t=t,
            #     eps_id=eps_id,
            #     agent_index=0,
            #     # # obs=prep.transform(obs),
            #     obs = obs,
            #     actions=action,
            #     action_prob=1.0,  # put the true action probability here
            #     # action_logp=0.0,
            #     rewards=rew,
            #     prev_actions=prev_action,
            #     prev_rewards=prev_reward,
            #     dones=True,
            #     # infos=""
            #     new_obs=obs #need to think about this
            # )
            # prev_action = action
            # prev_reward = rew
            # writer.write(batch_builder.build_and_reset())

    print('completed set of questions')

   
# __sphinx_doc_end__
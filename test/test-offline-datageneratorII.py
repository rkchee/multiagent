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



if __name__ == "__main__":

    sheets_cards = pd.read_excel('../training dataset - medical.xlsx', sheet_name='Cardiology and Infectious Disea')
    cardiology = sheets_cards.copy()
    train_set = cardiology.sample(frac=0.75, random_state=0)
    train_set.to_csv('train_set.csv', index=False)
    test_set = cardiology.drop(train_set.index)
    test_set.to_csv('test_set.csv', index=False)

    # set the training data 
    data = pd.read_csv('train_set.csv')
    
    batch_builder = SampleBatchBuilder()  # or MultiAgentSampleBatchBuilder
    writer = JsonWriter(
        os.path.join(ray.utils.get_user_temp_dir(), "demo-out"))
    print(ray.utils.get_user_temp_dir())



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


    # You normally wouldn't want to manually create sample batches if a
    # simulator is available, but let's do it anyways for example purposes:
    # env = gym.make("CartPole-v0")

    # RLlib uses preprocessors to implement transforms such as one-hot encoding
    # and flattening of tuple and dict observations. For CartPole a no-op
    # preprocessor is used, but this may be relevant for more complex envs.
    # prep = get_preprocessor(env.observation_space)(env.observation_space)
    # print("The preprocessor is", prep)
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

    # while True:
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



        mc_choices = ["A", "B", "C", "D", "E"]
        for letter in mc_choices:
            action = mcq_number[letter]
            if action == answer:
                rew = 1
            elif not answer:
                rew = 0
            else:
                rew = -1

            
            batch_builder.add_values(
                t=t,
                eps_id=eps_id,
                agent_index=0,
                # # obs=prep.transform(obs),
                obs = obs,
                actions=action,
                action_prob=1.0,  # put the true action probability here
                # action_logp=0.0,
                rewards=rew,
                prev_actions=prev_action,
                prev_rewards=prev_reward,
                dones=True,
                # infos=""
                new_obs=obs #need to think about this
            )
            prev_action = action
            prev_reward = rew
            writer.write(batch_builder.build_and_reset())
        index_answer += 1
    print('completed set of questions')

   
# __sphinx_doc_end__
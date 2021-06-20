import gym
# from gym.spaces import Discrete, Box
import gym_example
import numpy as np

import os
import tensorflow as tf
from transformers import ElectraTokenizer, TFElectraModel
tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator', max_length=128, pad_to_max_length=True)
model = TFElectraModel.from_pretrained('google/electra-small-discriminator')


import ray.utils

from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.evaluation.sample_batch_builder import SampleBatchBuilder
from ray.rllib.offline.json_writer import JsonWriter
import pandas as pd

class state_obs:
    def form_matrix(qa_index, question):
        matrix_max_size = 10
        matrix = []


        answerA = data['Answer.answer' + 'A'][qa_index]
        aTokens = tf.constant(tokenizer.encode(str(answerA), max_length=128, pad_to_max_length=128))[None, :]
        answerA = model(aTokens)
        # print(answerA)
        matrix = np.concatenate((question, answerA), axis=0)

        answerB = data['Answer.answer' + 'B'][qa_index]
        # print(type(answerB))
        aTokens = tf.constant(tokenizer.encode(str(answerB), max_length=128, pad_to_max_length=128))[None, :]
        answerB = model(aTokens)
        matrix = np.concatenate((matrix, answerB), axis=0)

        answerC = data['Answer.answer' + 'C'][qa_index]
        aTokens = tf.constant(tokenizer.encode(str(answerC), max_length=128, pad_to_max_length=128))[None, :]
        answerC = model(aTokens)
        matrix = np.concatenate((matrix, answerC), axis=0)

        answerD = data['Answer.answer' + 'D'][qa_index]
        aTokens = tf.constant(tokenizer.encode(str(answerD), max_length=128, pad_to_max_length=128))[None, :]
        answerD = model(aTokens)            
        matrix = np.concatenate((matrix, answerD), axis=0)

        answerE = data['Answer.answer' + 'E'][qa_index]
        aTokens = tf.constant(tokenizer.encode(str(answerE), max_length=128, pad_to_max_length=128))[None, :]
        answerE = model(aTokens)
        matrix = np.concatenate((matrix, answerE), axis=0)

        matrix = np.squeeze(matrix)
        state_matrix = np.pad(matrix,[(0,matrix_max_size - matrix.shape[0]), (0,0), (0,0)])
        state_matrix = np.moveaxis(state_matrix, 0, 2)
        # print(state_matrix.dtype)
        return state_matrix


mcq_number ={
    'A': 0,
    'B': 1,
    'C': 2,
    'D': 3,
    'E': 4,
    'F': 5,
    'G': 6,
    'H': 7,
    'I': 8,
    'J': 9, 
}

index_answer = 0
rew=10
eps_id = 0


if __name__ == "__main__":
    env = gym.make("med-v0")
    num_questions = 30
    batch_builder = SampleBatchBuilder()  # or MultiAgentSampleBatchBuilder
    writer = JsonWriter(
        os.path.join(ray.utils.get_user_temp_dir(), "demo-out"))

    # data = pd.read_csv('/home/rkchee/nxtopinion/rl/rheumatology_4199362_batch_results.csv')
    data = pd.read_csv('/workspace/nxtopinion/rl/rheumatology_4199362_batch_results.csv')

    questions = data['Answer.question'][0:num_questions]
    answers = data['Answer.image.label']

    prep = get_preprocessor(env.observation_space)(env.observation_space)
    print("The preprocessor is", prep)
    # action = 3
    # action = np.squeeze(action)
    for question in questions:
        qTokens = tf.constant(tokenizer.encode(str(question), max_length=128, pad_to_max_length=128))[None, :]
        obs = model(qTokens)
        obs = state_obs.form_matrix(index_answer, obs)
        answer = mcq_number[answers[index_answer]]        
        prev_action=np.zeros_like(answer)
        # print(answer)


        prev_reward = 0
        # done = False
        t = 0
        # action = answer

        for i in range(10):
            if i == answer:
                print(i, "and ", answer)
                rew = 1
            else:
                rew = -1
                print(i, "and", answer, rew)
        
            batch_builder.add_values(
                t=t,
                eps_id=eps_id,
                agent_index=0,
                obs = obs,
                actions=i,
                action_prob=1.0,  # put the true action probability here
                # action_logp=0.0,
                rewards=rew,
                prev_actions=prev_action,
                prev_rewards=prev_reward,
                dones=True,
                # infos=""
                new_obs=obs)
            writer.write(batch_builder.build_and_reset())
#         obs = new_obs
        prev_action = prev_action
        prev_reward = rew
#         t += 1
        eps_id += 1
        index_answer += 1

        




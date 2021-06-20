import gym
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
from ray.tune.registry import register_env

class realworldEnv(gym.Env):
   def __init__(self):
        import tensorflow as tf
        low = np.full((128, 256, 10), -1000)
        high = np.full((128, 256, 10), 1000)
        # output_low = np.full((, 1), 0)
        # output_high = np.full((, 1), 10)

        # self.action_space = Box(low=0, high=10.0, shape=(1, ), dtype=np.float32)
        self.action_space = Discrete(2)
        # self.action_space = Box(low=output_low, high=output_high, shape=(, 1), dtype=np.int32)

        # self.observation_space = Box(low=0, high=1, shape=(1, ), dtype=np.float32)
        self.observation_space = Discrete(2)



# def env_creator(env_name):
#     if env_name == 'CustomEnv-v0':
#         from custom_gym.envs.custom_env import CustomEnv0 as env
#     elif env_name == 'CustomEnv-v1':
#         from custom_gym.envs.custom_env import CustomEnv1 as env
#     else:
#         raise NotImplementedError
#     return env



if __name__ == "__main__":
    # register_env("realworldEnv", realworldEnv)
    # env = realworldEnv
    # env = 'CartPole-v0'
    # env = gym.make(env)
    # env = gym.make("CartPole-v0")
    register_env("realworldEnv", lambda config: realworldEnv(config))
    # env = gym.make('realworldEnv')



    num_questions = 30
    batch_builder = SampleBatchBuilder()  # or MultiAgentSampleBatchBuilder
    writer = JsonWriter(
        os.path.join(ray.utils.get_user_temp_dir(), "demo-out"))

    data = pd.read_csv('/home/rkchee/nxtopinion/rl/rheumatology_4199362_batch_results.csv')
    # print(data['Answer.question'][0:num_questions])
    questions = data['Answer.question'][0:num_questions]
    # print(len(data))
    # print(list(data.columns))

    # i=0
    # for question in questions:
    #     print(i)
    #     i += 1
   
    # print(qTokens)
    answers = data['Answer.image.label']
    # qTokens = tf.constant(tokenizer.encode(str(data['Answer.question'][0:1]), max_length=128, pad_to_max_length=128))[None, :]
    # outputs = model(qTokens)
    # print(qTokens)
    # print(outputs)
    # print(answers[0:num_questions])

    # print(len(data))
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
            # print(state_matrix.shape)
            return state_matrix



    # print(list(data.columns))
    # print(answers)
    # print(mcq_number['A'])

    # RLlib uses preprocessors to implement transforms such as one-hot encoding
    # and flattening of tuple and dict observations. For CartPole a no-op
    # preprocessor is used, but this may be relevant for more complex envs.
    prep = get_preprocessor(env.observation_space)(env.observation_space)
    print("The preprocessor is", prep)

    index_answer = 0
    rew=10
    eps_id = 0
    for question in questions:
        qTokens = tf.constant(tokenizer.encode(str(question), max_length=128, pad_to_max_length=128))[None, :]
        # qTokens = np.expand_dims(qTokens, axis=0)
        obs = model(qTokens)
        # obs = np.array(obs).astype(np.float)
        obs = state_obs.form_matrix(index_answer, obs)
        # print(np.shape(obs))
        # print(question.shape)
    #     # obs = env.reset()
        # print[answers]
        # answer = data['Answer.answer' + answers[index_answer]][index_answer]
        answer = mcq_number[answers[index_answer]]
        # print(answer)
        # aTokens = tf.constant(tokenizer.encode(str(answer), max_length=128, pad_to_max_length=128))[None, :]
        # answer = model(aTokens)
        # answer = np.array(answer).astype(np.float)
        # print(mcq_number[answer])

        # print('This is the created answer:  ' +'Answer.answer' + answers[index_answer])
        # print(type(obs))
        index_answer += 1
        # prev_action = np.zeros_like(answer)
        prev_action = np.zeros_like([])

        prev_reward = 0
        done = False
        t = 0
        action = answer
    # #         new_obs, rew, done, info = env.step(action)
        batch_builder.add_values(
            t=t,
            eps_id=eps_id,
            agent_index=0,
            # obs=prep.transform(obs),
            obs = prep.transform(obs),
            actions=0,
            action_prob=1.0,  # put the true action probability here
            action_logp=0.0,
            rewards=rew,
            prev_actions=0,
            prev_rewards=prev_reward,
            dones=False,
            infos="")
            # new_obs=prep.transform(new_obs))
    #         obs = new_obs
            # prev_action = action
            # prev_reward = rew
    #         t += 1
        eps_id += 1
        writer.write(batch_builder.build_and_reset())



    # state_obs.form_matrix()

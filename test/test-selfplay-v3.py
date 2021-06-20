import os
import argparse
import gym
# from gym.spaces import Discrete, Box
import numpy as np

import tensorflow as tf
from transformers import ElectraTokenizer, TFElectraModel
tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator', max_length=128, pad_to_max_length=True)
model = TFElectraModel.from_pretrained('google/electra-small-discriminator')
import tensorflow.keras.metrics
import tensorflow.keras.losses


# # from ray.rllib.models.tf.tf_modelv2 import TFModelV2
# from ray.rllib.models.tf.misc import normc_initializer

# from ray.rllib.models import ModelCatalog # possibly causes a conflict with electra - roger

# import ray.utils

# from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.evaluation.sample_batch_builder import SampleBatchBuilder
from ray.rllib.offline.json_writer import JsonWriter
import pandas as pd

# from ray.rllib.agents.dqn import DQNTrainer
# from ray.rllib.agents.dqn import dqn
# from ray.rllib.agents.a3c import A3CTrainer


import ray
# import ray.rllib.agents.ppo as ppo
# from ray.tune.logger import pretty_print
from ray.rllib.env.policy_client import PolicyClient




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

# # class realworldEnv(gym.Env):
# #    def __init__(self, env_config):
# #         import tensorflow as tf
# #         low = np.full((128, 256, 10), -1000)
# #         high = np.full((128, 256, 10), 1000)
# #         # output_low = np.full((, 1), 0)
# #         # output_high = np.full((, 1), 10)

# #         # self.action_space = Box(low=0, high=10.0, shape=(1, ), dtype=np.float32)
# #         self.action_space = Discrete(10)
# #         # self.action_space = Box(low=output_low, high=output_high, shape=(, 1), dtype=np.int32)

# #         self.observation_space = Box(low=low, high=high, shape=(128, 256, 10), dtype=np.float32)
# #         # self.observation_space = Discrete(1)

# # class vanillaKerasModel(TFModelV2):
    
# #     """Custom model for policy gradient algorithms."""

# #     def __init__(self, obs_space, action_space, num_outputs, model_config,
# #                  name):
# #         super(vanillaKerasModel, self).__init__(obs_space, action_space,
# #                                            num_outputs, model_config, name)
# #         import tensorflow as tf
# #         import tensorflow.keras.metrics
# #         import tensorflow.keras.losses
# #         self.inputs = tf.keras.layers.Input(
# #             shape=obs_space.shape, name="observations")


# #         layer_1 = tf.keras.layers.Conv2D(
# #             2, 3,
# #             name="my_layer1",
# #             padding = 'same',
# #             activation=tf.nn.relu,
# #             kernel_initializer=normc_initializer(1.0))(self.inputs)
# #         layer_2 = tf.keras.layers.Flatten()(layer_1)
# #         layer_3 = tf.keras.layers.Dense(num_outputs)(layer_2)
# #         # layer_4 = tf.keras.layers.Dense(256)(layer_3)
# #         value_out = tf.keras.layers.Dense(
# #             1,
# #             name="value_out",
# #             activation=None,
# #             kernel_initializer=normc_initializer(0.01))(layer_2)

# #         self.base_model = tf.keras.Model(self.inputs, [layer_3, value_out])
        
    
# #         # self.base_model = tf.keras.Model(self.inputs, [layer_out])

# #         self.register_variables(self.base_model.variables)

# #     # @tf.function
# #     def forward(self, input_dict, state, seq_lens):
# #         model_out, self._value_out = self.base_model(input_dict["obs"])
# #         return model_out, state

# #     # @tf.function
# #     def value_function(self):
# #         import tensorflow as tf
# #         import tensorflow.keras.metrics
# #         import tensorflow.keras.losses
# #         return tf.reshape(self._value_out, [-1])

# #     # @tf.function
# #     def metrics(self):
# #         import tensorflow as tf
# #         import tensorflow.keras.metrics
# #         import tensorflow.keras.losses

# #         return {"foo": tf.constant(42.0)}

parser = argparse.ArgumentParser()
parser.add_argument(
    "--no-train", action="store_true", help="Whether to disable training.")

if __name__ == "__main__":
    args = parser.parse_args()
    # env = realworldEnv
    # url = "35.233.170.130"
    url = "34.82.117.252"

    client = PolicyClient("http://" + url + ":9900", inference_mode="remote")

    num_questions = 30  #total of 55 questions in rheumatology
    index_answer = 0
    rew=0
    # eps_id = 0 # considered the question ID

    batch_builder = SampleBatchBuilder()  # or MultiAgentSampleBatchBuilder
    writer = JsonWriter(
        os.path.join(ray.utils.get_user_temp_dir(), "demo-out"))

    data = pd.read_csv('/home/rkchee/nxtopinion/rl/rheumatology_4199362_batch_results.csv') #google vm
    # data = pd.read_csv('/workspace/nxtopinion/rl/rheumatology_4199362_batch_results.csv') #kubernettes

    questions = data['Answer.question'][0:num_questions]
    answers = data['Answer.image.label']

#     # prep = get_preprocessor(env.observation_space)(env.observation_space)
#     # print("The preprocessor is", prep)


#     # env = 'CartPole-v0'
#     # ModelCatalog.register_custom_model("my_model", vanillaKerasModel)
#     # ray.init()

#     # config = dqn.DEFAULT_CONFIG.copy()
#     # config["num_gpus"] = 0
#     # config["num_workers"] = 1
#     # config["eager"] = False

#     # trainer = DQNTrainer(config=config, env="CartPole-v0")
#     # trainer = DQNTrainer(
#     #     env=env,
#     #     config=dict(
#     #         **{
#     #             # "exploration_config": {
#     #             #     "type": "EpsilonGreedy",
#     #             #     "initial_epsilon": 1.0,
#     #             #     "final_epsilon": 0.02,
#     #             #     "epsilon_timesteps": 1000,
#     #             # },
#     #             "input": "/tmp/demo-out",
#     #             "input_evaluation": [],
#     #             "explore": False,
#     #             # "learning_starts": 100,
#     #             # "timesteps_per_iteration": 200,
#     #             "log_level": "INFO",
#     #             "train_batch_size": 32,
#     #             "framework": "tf",
#     #             "num_workers": 6,
#     #             "model": {
#     #                 "custom_model": "my_model",
#     #                 # Extra kwargs to be passed to your model's c'tor.
#     #                 "custom_model_config": {},
#     #             },                
#     #         })
#     # )


    while True:
        eps_id = client.start_episode(training_enabled=not args.no_train) #logic of the args are not intuitive
        prev_reward = 0
        t = 0 
        rewards = 0         
        done = False
        index_answer = 0 
        
        for question in questions:
            if t == (num_questions - 1):
                done = True
            qTokens = tf.constant(tokenizer.encode(str(question), max_length=128, pad_to_max_length=128))[None, :]
            obs = model(qTokens)
            obs = state_obs.form_matrix(index_answer, obs)
            answer = mcq_number[answers[index_answer]]
            prev_action = np.zeros_like(answer)
            # print("the answer is ", answer)
            question_answered = False
            while not question_answered:
                action = client.get_action(eps_id, obs)
                # action = np.argmax(action)
                # print("agent answer: ", action)
                # print("action is: ", action)
                if action == answer:
                    rew = 1
                    question_answered = True
                    index_answer += 1
                else: 
                    rew = -1
                    done = False
                    new_obs = obs ## Roger to create new_obs function that randomizes the order of the answers
        
                rewards += rew
                client.log_returns(eps_id, rew, info="")
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
                    dones=done,
                    # infos=""
                    new_obs=obs
                )
                # obs = new_obs #used when new_obs function is ready - Roger
                prev_action = action
                prev_reward = rew
                t += 1
        writer.write(batch_builder.build_and_reset())
        print('Rewards after 30 question: ', rewards)
        client.end_episode(eps_id, obs)



  



#     # for i in range(1000):
#     #     # Perform one iteration of training the policy with DQN
#     #     result = trainer.train()
#     #     print(pretty_print(result))

#     #     if i % 100 == 0:
#     #         checkpoint = trainer.save()
#     #         print("checkpoint saved at", checkpoint)



#     # trainer = DQNTrainer(...)
#     # train policy offline

#     # from ray.rllib.offline.json_reader import JsonReader
#     # from ray.rllib.offline.wis_estimator import WeightedImportanceSamplingEstimator

#     # estimator = WeightedImportanceSamplingEstimator(trainer.get_policy(), gamma=0.99)
#     # reader = JsonReader("/path/to/data")
#     # for _ in range(1000):
#     #     batch = reader.next()
#     #     for episode in batch.split_by_episode():
#     #         print(estimator.estimate(episode))
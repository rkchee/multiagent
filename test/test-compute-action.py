import ray
from ray import tune
import ray.rllib.agents.ppo as ppo
import gym
import numpy as np
import tensorflow as tf

# sess = tf.compat.v1.InteractiveSession()

config={
    "env": "CartPole-v0",
    "num_gpus": 0,
    "num_workers": 1,
    "lr": tune.grid_search([0.01, 0.001, 0.0001]),
    "eager": False,
}

ray.init()
analysis = tune.run(
    "PPO",
    stop={"episode_reward_mean": 20},
    config={
        "env": "CartPole-v0",
        "num_gpus": 0,
        "num_workers": 1,
        "lr": tune.grid_search([0.01, 0.001, 0.0001]),
        "eager": False,
    },
    local_dir="~/nxtopinion/rl",
    checkpoint_at_end=True,
)

# tune.run() allows setting a custom log directory (other than ``~/ray-results``)
# and automatically saving the trained agent
# analysis = ray.tune.run(
#     ppo.PPOTrainer,
#     config=config,
#     local_dir='/',
#     stop=stop_criteria,
#     checkpoint_at_end=True)

# list of lists: one list per checkpoint; each checkpoint list contains
# 1st the path, 2nd the metric value
checkpoints = analysis.get_trial_checkpoints_paths(
    trial=analysis.get_best_trial("episode_reward_mean"),
    metric="episode_reward_mean")

agent = ppo.PPOTrainer(env="CartPole-v0")
print(checkpoints[0])
agent.restore("/home/rkchee/nxtopinion/rl/PPO/PPO_CartPole-v0_0_lr=0.01_2020-08-18_17-05-32e1qdwrhl/checkpoint_1/checkpoint-1")

policy = agent.get_policy()




env = gym.make('CartPole-v0')

# run until episode ends
episode_reward = 0
done = False
obs = env.reset()
print(obs)

logits, _ = policy.model.from_batch({"obs": np.array([[0.1, 0.2, 0.3, 0.4]])})

dist = policy.dist_class(logits, policy.model)

print(dist.sample())
act = dist.sample()

# with tf.compat.v1.Session() as sess:  print(act.eval(session=sess)) 

# action = agent.compute_action(obs, full_fetch=True)
# # obs, reward, done, info = env.step(action)
# # episode_reward += reward
# print(action)

#define a variable to hold normal random values 
normal_rv = tf.Variable( act )

#initialize the variable
init_op = tf.compat.v1.initialize_all_variables()

#run the graph
with tf.compat.v1.Session() as sess:
    sess.run(init_op) #execute init_op
    #print the random values that we sample
    print (sess.run(normal_rv))

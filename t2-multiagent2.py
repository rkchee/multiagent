import os
import ray
from ray.tune.registry import register_env
from ray.rllib.env import MultiAgentEnv
import gym
from ray import tune
import numpy as np
import pdb

class IrrigationEnv(MultiAgentEnv):
    def __init__(self, return_agent_actions = False, part=False):
        self.num_agents = 400
        self.reward = 0
        self.observation_space = gym.spaces.Box(low=200, high=800, shape=(1,))
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(1,))
        self.water = np.random.uniform(200,800)

    def reset(self):
        obs = {}
        self.water = np.random.uniform(200,800)
        for i in range(self.num_agents):
            obs[i] = np.array([self.water])
        return obs
    
    def step(self, action_dict):
        obs, rew, done, info = {}, {}, {}, {}
        for i in range(self.num_agents):

            obs[i], rew[i], done[i], info[i] = np.array([1]), 0, True, {}
        done["__all__"] = True
        pdb.set_trace()
        return obs, rew, done, info

if __name__ == "__main__":
    def env_creator(_):
        return IrrigationEnv()
    single_env = IrrigationEnv()
    env_name = "IrrigationEnv"
    register_env(env_name, env_creator)
    obs_space = single_env.observation_space
    act_space = single_env.action_space
    num_agents = single_env.num_agents
    def gen_policy():
        return (None, obs_space, act_space, {})
    policy_graphs = {}
    for i in range(num_agents):
        policy_graphs['agent-' + str(i)] = gen_policy()
    def policy_mapping_fn(agent_id):
            return 'agent-' + str(agent_id)

    config={
        "log_level": "WARN",
        "num_workers": 3,
        "num_cpus_for_driver": 1,
        "num_cpus_per_worker": 1,
        "lr": 5e-3,
        "model":{"fcnet_hiddens": [8, 8]},
        "multiagent": {
            "policies": policy_graphs,
            "policy_mapping_fn": policy_mapping_fn,
        },
        "env": "IrrigationEnv"

    }

    exp_name = 'more_corns_yey'
    exp_dict = {
            'name': exp_name,
            'run_or_experiment': 'PG',
            "stop": {
                "training_iteration": 100
            },
            'checkpoint_freq': 20,
            "config": config,
    }

ray.init()
# pdb.set_trace()
tune.run(**exp_dict)
    




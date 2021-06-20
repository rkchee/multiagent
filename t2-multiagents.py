import numpy as np
import pandas as pd
import pdb
import ray 
import os 

# ray_head_ip =os.environ.get('RAY_HEAD_SERVICE_HOST')
# ray_redis_port = os.environ.get('RAY_HEAD_SERVICE_PORT_REDIS_PRIMARY')
# url = '34.127.61.217'
# if __name__ == "__main__":
#     ray.init()



# # m1 = pd.read_csv('/home/rkchee/nxtopinion/matrix-full.csv')
# # print(m1.head())


from ray import tune
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from pettingzoo.sisl import waterworld_v3

# Based on code from github.com/parametersharingmadrl/parametersharingmadrl

if __name__ == "__main__":
    # RDQN - Rainbow DQN
    # ADQN - Apex DQN
    def env_creator(args):
        return PettingZooEnv(waterworld_v3.env())

    env = env_creator({})
    register_env("waterworld", env_creator)
    ray.init()

    obs_space = env.observation_space
    act_spc = env.action_space

    policies = {agent: (None, obs_space, act_spc, {}) for agent in env.agents}

    tune.run(
        "APEX_DDPG",
        stop={"episodes_total": 60000},
        checkpoint_freq=10,
        config={
            # Enviroment specific
            "env": "waterworld",
            # General
            "framework": "tf",
            "num_gpus": 1,
            "num_workers": 2,
            "num_cpus_per_worker": 32,
            # Method specific
            "multiagent": {
                "policies": policies,
                "policy_mapping_fn": (lambda agent_id: agent_id),
            },
        },
    )
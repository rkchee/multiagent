# nxtopinion
Project 1.0 Next Opinion

This readme describes how to start the nitinol RL client and server. My python version is 3.7.1

1. install the dependencies.  I have used pipenv to package everything in the pipfiles.  Please run the following at the command prompt in the /rl directory. You will also need to have python 3.7.1

pipenv install

then enter the virtual environment:

pipenv shell

2. Need to change the url variable in thor9-client.py and nitinol5-server.py to either:

your external ip if you are using your own computer or vm. Or use the Azure VM for which the environment is already installed:

ntoml.canadacentral.cloudapp.azure.com 

- ray serve will use your external ip and serve both backends and endpoints with your external ip and port 8000.  

- ray.init will initiate a cluster specifically using your external ip and port 6000

- the policyserver is rllibs special endpoint for external environments that expose the rllib RL agent to the real world.  It will use your external ip and port 9900 as endpoint for the policyclient to access its api.  

3. run this line at the command prompt to initiate the ray cluster.  adding clusters to ray is possible by using ray start ray start --address='10.168.0.7:6000' --redis-password='5241590000000000' moving forward. Replace the ip with your own external ip

ray start --head --redis-port=6000

4. Initiate the rllib policy server by running this at the command line

python nitinol5-server.py --run=PPO

FYI: nitinol2-server.py has been added to be used in combination with thor7-client.py.  This version changes the observational space to a 3D matrix of 1, 128, 256 and will no longer use single answer choices as input for getting actions.  It will only take the state matrix to create actions.  


5. Open a new terminal while leaving the rllib policy server running (FYI: you can use screen here which will also work)

6. intiate the rllib policy client, electra backend and endpoint, and the endpoint that exposes the rllib policy client API. Run the line at the command prompt (be sure to run the pipenv shell on the second terminal in the rl directory to getinto the virtual environment)

python thor9-client.py --inference-mode=remote

7. Endpoints that can be used in postman

url:8000/serve

This is GET method that will take a {key:value} pair using "text" as the key and your string input as the value for the algo to ingest. This is will return a json that will provide the tensors that will need to be mapped to the answers. At the moment, the output is a json with a array of 2 tensors and a eid number that corresponds to the action generated.  Each tensor respresents an answer choice for the user to select.  I will be changing the action.space so that each tensor output will be an array with shape of (1, 512, 256).  THis is the shape of the output of the electra model.  Each answer choice will need to be passed through electra as well. A nearest neighbor search will be used to map each of the node outputs of the RL algo to the answer choices provided by the physicians. The reward can be provided once the user selects that answer choice.   The eid will be used to identify which episode the reward goes to.

url:8000/eid_create

This is a GET method to create an new episode number.  The episode number is required to log a reward and link it to the correct episode of a  experiences collected. It returns a json.

url:8000/reward

This is a POST method that will take a json containing the key:value pairs of "reward":value where the value is a reward number awarded to the algorithm. The reward can be a negative or positive integer. It will also take in a "eid":id where id is the unique episode id (eid) for the experiences collected.  


url:8000/nearest_neighbors

This is a POST method that will take in a json containing a "sequences" key and a "predicted_sequence" key. The "sequences" key contain a few arrays of sequences collected from the answer choices like this and the "predicted_sequence" key is predicted from the nitinol RL algorithm:

{
    "sequences": [[100, 10, 20], [20, 20, 10], [30, 40, 30]],
    "predicted_sequence": [[20, 20, 10]]
}

The arrays are then input into the sklearn nearest neighbors algo.  The output is a a json with indices and distances like this obtained from the example above. As you can see, the predicted sequence matches the 1st index of sequences.  Index starts at 0.  Since it is a perfect match the distance will be 0:

{
  "distances": [
    [
      0.0
    ]
  ],
  "indices": [
    [
      1
    ]
  ]
}


url:8000/electratensors

This is a POST method that takes in text and converts it to tensors by the Electra Model via json with a key:value pair with the key of "text"

{
  "text": "what is the cause of hashimoto thyroiditis"
}

Output with return json with 

{
  "Electra_Tensors": list of tensors
}

url:8000\action_eid

This is a POST method that takes a json of electra tensors and eid to generate an action.


url:8000/state_obs

This is a POST method that takes a json of question and answers. It will automatically convert the question string and the answer strings into arrays of tensors and stack each answer together into a cube configuration.  Currently the max number of answers of set to 10 but can be set high by increasing the self.matrix_max_size = 10 (default at 10).  

Example: 
input json:

{
  "question": "what is the meaning of HHT?",
  "answer1": "hereditary hemorrhagic telengectasia",
  "answer2": "it is condition characterized by having AV malformations all over the body including the brain and the lungs. Patients get recurrent bleeding and anemia with this disorder. "
}

The results is matrix with a shape [10, 128, 256] - I think the shape here is incorrect. THis will need to be checked. 

The shape of the matrix is more likely [128, 256, 10]

-------------------

run:
python gym_example/test-medEnv2.py

file: gym_example/test-medEnv2.py

scans the dataset in the form of a csv file. Currently intakes answers from A-E and concatenates the question and the answers.  It will automatically convert the question string and the answer strings into arrays of tensors and stack each answer together into a cube configuration.  Currently the max number of answers of set to 10 but can be set high by increasing the matrix_max_size = 10 (default at 10). 

execute program by changing the file.csv in the line read by pd.read_csv to the file that has your dataset.  The concentenation is hard coded in sequence taking in A to E.  It calls the answer column as the data set is looped.  

To run, simply execute at command prompt:

python test-medEnv2.py


For testing purposes, the script is set to only scan 30 and create a json of 30 experiences.  THis will need to be set to the number of examples in the dataset.  The script then creates the json file as an output in the /tmp/demo-out directory.  Thus far there is no official json created. So feel free to delete what is there. I will use a different fold for official outputs. 


You can now set the number of number of workers (parallel agents running their own datasets).  There is a known error that occurs. If you see the follow error:

...
2020-06-16 17:59:38,089	ERROR worker.py:1029 -- Possible unhandled error from worker: ray::RolloutWorker.set_weights() (pid=2487, ip=192.168.178.24)
  File "python/ray/_raylet.pyx", line 463, in ray._raylet.execute_task
  File "python/ray/_raylet.pyx", line 417, in ray._raylet.execute_task.function_executor
  File "/mnt/c/Users/Stefan/git-repos/work/deep-rl-mobility-management/venv-wsl/lib/python3.8/site-packages/ray/rllib/evaluation/rollout_worker.py", line 571, in set_weights
    self.policy_map[pid].set_weights(w)
  File "/mnt/c/Users/Stefan/git-repos/work/deep-rl-mobility-management/venv-wsl/lib/python3.8/site-packages/ray/rllib/policy/tf_policy.py", line 357, in set_weights
    return self._variables.set_weights(weights)
  File "/mnt/c/Users/Stefan/git-repos/work/deep-rl-mobility-management/venv-wsl/lib/python3.8/site-packages/ray/experimental/tf_utils.py", line 182, in set_weights
    self.assignment_nodes[name] for name in new_weights.keys()
AttributeError: 'list' object has no attribute 'keys'


The program doesn't crash though. It keeps on running after the error.
The error occurs shortly after calling trainer.train(). 

Please rearrage the tensorflow import to be after the ray import as explained here by someone else. 

I finally narrowed the error down and found what's causing it. It has nothing to do with my environment and it's very strange:
It seems to happen because I imported tensorflow before ray.

See a minimal example here:

import tensorflow
import ray
import ray.rllib.agents.ppo as ppo
from ray.tune.logger import pretty_print


ray.init()

# ray config
config = ppo.DEFAULT_CONFIG.copy()
config["num_workers"] = 1

# train on cartpole
trainer = ppo.PPOTrainer(config=config, env="CartPole-v0")
result = trainer.train()
pretty_print(result)
This gives the error I posted above. If you remove the (unused) tensorflow import OR import it last (after all ray imports), the error does not occur! Really strange.

For me, that answers my question. I'll just (re-)move the tensorflow import. I'll close this "question issue", but I'll open a "bug issue" since I believe this should not happen - or at least lead to a more understandable error. This was really hard to debug.

-----------------------------

run:
python test-offline-dummytrainer.py

file: test-offline-dummytrainer.py

This script takes the json file that was create by the test-medEnv2.py script and runs offline training using rllib.  RLlib is configured to use the json file by adding the line "input": "/tmp/demo-out" in the configuration.  RLlib will use our modified model and pass the matrix through it.  In the /tmp/demo-out directory, there may be some sample output json files. please feel free to delete them and create your own before commencing offline training.  The script will uses all the files in the folder.  In the future I will use a different folder for official dataset output json.

Update: Everything runs but will crash at a certain iteration because of out of memory issue.  This is a known problem and is solved by increasing the memory of the VM.  The OOM will resolve and will not get any larger once it reaches steady state.  


---------------------
run:
1. ray start --head --redis-port=6000
2. python nitinol5-server.py --run=PPO
# please configure your url to match the external url that you are using. Otherwise the test-selfplay-v2.py script will not be able to access the ML algos
# please configure the path to ~/nxtopinion/rl/rheumatology_4199362_batch_results.csv which is the toy dataset at this time.  

3. python test-selfplay-v2.py #please configure the url to match the external url that are using.  Otherwise the nitinol5-server.py cannot be accessed.  

files:
nitinol5-server.py
test-selfplay-v2.py


This requires turning on the ML server.  It uses the ML server to obtain answer choices.  This script (test-selfplay-v2.py) now runs the algorithm in PPO mode and runs it against the question.  It continuously tries to answer the question until it gets the right answer and then terminates.  It captures the sequence in an on-policy method and batches the experiences for online training.  Currently it is set to fun the /home/rkchee/nxtopinion/rl/rheumatology_4199362_batch_results.csv file.  

----------------------------

run:
1. ray start --head --redis-port=6000
2. python nitinol5-server.py --run=PPO
# please configure your url to match the external url that you are using. Otherwise the test-selfplay-v3.py script will not be able to access the ML algos
# please configure the path to ~/nxtopinion/rl/rheumatology_4199362_batch_results.csv which is the toy dataset at this time.  

3. python test-selfplay-v3.py #please configure the url to match the external url that are using.  Otherwise the nitinol5-server.py cannot be accessed.  

files:
nitinol5-server.py
test-selfplay-v3.py

Refines each episode as a predetermined number of questions (currently set at 30 questions).  An answer correct is 1 point and an answer that is incorrect is -1 point.  The max score for each episode is therefore 30 points.  This allows us to evaluate whether the agent is learning the questions.     

------------------------------------
run:
1. ray start --head --redis-port=6000
2. python test.py --run=PPO
# please configure your url to match the external url that you are using. Otherwise the test.py script will not be able to access the ML algos

3. python test-ray-separate-algo.py #please configure the url to match the external url that are using.  Otherwise the nitinol5-server.py cannot be accessed.  
# please configure the path to ~/nxtopinion/rl/rheumatology_4199362_batch_results.csv which is the toy dataset at this time.  
# please configure your url to match the external url that you are using. Otherwise the test.py script will not be able to access the ML algos


files:
test.py
test-ray-separate-algo.py

Refines each episode as a predetermined number of questions (currently set at 30 questions).  An answer correct is 1 point and an answer that is incorrect is -1 point.  The max score for each episode is therefore 30 points.  This allows us to evaluate whether the agent is learning the questions in a set of 30.     

---------------
pip install -U pydantic==1.6

------------------------

We have updated to Ray v1.0

------------------------

run:
1. ray start --head --redis-port=6000
2. python test.py 
# please configure your url to match the external url that you are using. Otherwise the test.py script will not be able to access the ML algos

3. python test-ray-separate-algo-v06.py #please configure the url to match the external url that are using.  Otherwise the nitinol5-server.py cannot be accessed.  
# please configure the path to ~/nxtopinion/rl/rheumatology_4199362_batch_results.csv which is the toy dataset at this time.  
# please configure your url to match the external url that you are using. Otherwise the test.py script will not be able to access the ML algos


files:
test.py
test-ray-separate-algo-v06.py

Refines each episode as a single choice.  An answer correct is 1 point and an answer that is incorrect is -1 point.  The max score for each episode is 1 point.    
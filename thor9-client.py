from ray import serve
import requests
import ray

import numpy as np

from ray.rllib.env.policy_client import PolicyClient
import argparse

import json
from json import dumps

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

parser = argparse.ArgumentParser()
parser.add_argument(
    "--no-train", action="store_true", help="Whether to disable training.")
parser.add_argument(
    "--inference-mode", type=str, required=True, choices=["local", "remote"])
parser.add_argument(
    "--off-policy",
    action="store_true",
    help="Whether to take random instead of on-policy actions.")
parser.add_argument(
    "--stop-reward",
    type=int,
    default=9999,
    help="Stop once the specified reward is reached.")

args = parser.parse_args()
# url = "34.94.143.197"
# url = "34.94.83.0"
url = "ntoml.canadacentral.cloudapp.azure.com"
# url="52.228.66.9"

class ElectraNLP:
    def __init__(self):
        from transformers import ElectraTokenizer, TFElectraModel
        # configuration = ElectraConfig()
        self.tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator', max_length=128, pad_to_max_length=True)
        self.model = TFElectraModel.from_pretrained('google/electra-small-discriminator')

    def __call__(self, flask_request):
        import tensorflow as tf
        import json
        self.answer = flask_request.args.get("text")
        self.eid = flask_request.args.get("eid")
        self.input_ids = tf.constant(self.tokenizer.encode(self.answer, max_length=128, pad_to_max_length=128))[None, :]  # Batch size 1
        outputs = self.model(self.input_ids)
        results = outputs[0].numpy().tolist()

        payload = results
        payload = json.dumps(payload)

        headers = {"content-type": "application/json"}
        self.response = requests.post("http://" + url + ":8000/eid", json=payload, headers=headers)
        
        return self.response.text

class ElectraTensors:
    def __init__(self):
        from transformers import ElectraTokenizer, TFElectraModel
        self.tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator', max_length=128, pad_to_max_length=True)
        self.model = TFElectraModel.from_pretrained('google/electra-small-discriminator')

    def __call__(self, flask_request):
        import tensorflow as tf
        import json
        self.answer = flask_request.json
        self.input_ids = tf.constant(self.tokenizer.encode(self.answer['text'], max_length=128, pad_to_max_length=128))[None, :]  # Batch size 1
        outputs = self.model(self.input_ids)
        results = outputs[0].numpy().tolist()
        return {
            "Electra_Tensors": results
        }
        # return "wonderful"



class rllib_client:
    def __init__(self, args):
        # self.args = args
        self.client = PolicyClient("http://" + url + ":9900", inference_mode=args.inference_mode)
        self.eid = self.client.start_episode(training_enabled=not args.no_train)

    def __call__(self, flask_request):
        import json
        self.phrase_tensors = flask_request.json

        self.phrase_tensors = np.array(json.loads(self.phrase_tensors))
        self.action = self.client.get_action(self.eid, self.phrase_tensors)
        return {
            "model actions": self.action,
            "eid": self.eid,
        }
        # return self.phrase_tensors

class action_eid:
    def __init__(self):
        self.client = PolicyClient("http://" + url + ":9900", inference_mode=args.inference_mode)
        # self.eid = self.client.start_episode(training_enabled=not args.no_train)

    def __call__(self, flask_request):
        r = flask_request.json
        self.phrase_tensors = np.array(r['tensors'])
        self.action = self.client.get_action(r['eid'], self.phrase_tensors)
        return {
            "action":self.action.tolist(),
            "eid": r['eid']
        }
        # return {
        #     "shape": self.phrase_tensors.shape
        # }

        



class eid:
    def __init__(self):
        self.client = PolicyClient("http://" + url + ":9900", inference_mode=args.inference_mode)
    def __call__(self, flask_request):
        self.eid = self.client.start_episode(training_enabled=not args.no_train)
        return {"eid": self.eid}
        

class reward:
    def __init__(self):
        self.reward=0
        self.client = PolicyClient("http://" + url + ":9900", inference_mode=args.inference_mode)
    
    def __call__(self, flask_request):
        self.reward = flask_request.json
        self.client.log_returns(self.reward['eid'], self.reward['reward'], info='none')
        return 'reward logged with current eid'

class nearestneighbor:
    def __init__(self):
        self.sequences = []
    
    def __call__(self, flask_request):
        self.neighbors = flask_request.json
        matrix = np.asarray(self.neighbors['sequences'])
        pred_seq = np.asarray(self.neighbors['predicted_sequence'])
        matrix = np.moveaxis(matrix, 2, 0)
        matrix = np.delete(matrix, 0, 0)
        N = matrix.shape[0]
        B = matrix.shape
        if len(matrix) > 1:
            flatten_seq = np.reshape(matrix, (N, -1))
            flatten_predseq = np.reshape(pred_seq, (1, -1))

        nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(flatten_seq)
        distances, indices = nbrs.kneighbors(flatten_predseq)

        return {
            "distances": distances.tolist(), 
            "indices": indices.tolist()
            # "flatten_se": flatten_seq.shape,
            # "flatten_predseq": flatten_predseq.shape,
            # "N": B
        }

class state_present:
    def __init__(self):
        self.matrix = []
        self.matrix_max_size = 10
    
    def __call__(self, flask_request):
        self.state = flask_request.json


        flag_initialstate = 1
        for i in self.state:
            # self.state['text'] = self.state.pop(i)
            js = {
                "text":self.state[i]
            }

            dumps = json.dumps(js)
            payload = json.loads(dumps)

            headers = {"content-type": "application/json"}
            self.response = requests.post("http://" + url + ":8000/electratensors", json=payload, headers=headers)
            t = self.response.text
            t = json.loads(t)
            t = t['Electra_Tensors']

            self.state[i] = np.asarray(t)
            if flag_initialstate:
                state_matrix = self.state[i]
                flag_initialstate=0
            else:
                state_matrix = np.concatenate((state_matrix, self.state[i]), axis=0)

        state_matrix = np.pad(state_matrix,[(0,self.matrix_max_size - state_matrix.shape[0]), (0,0), (0,0)])
        state_matrix = np.moveaxis(state_matrix, 0, 2)
        # flatten = np.reshape(state_matrix.view(), (len(state_matrix), -1)))
        return {
            "state_matrix": state_matrix,
        }

class action_state_obs_eid:
    def __init__(self):
        self.client = PolicyClient("http://" + url + ":9900", inference_mode=args.inference_mode)

    def __call__(self, flask_request):
        request = flask_request.json
        headers = {"content-type": "application/json"}

        eid, question, answers = request["eid"], request["question"], request["answer"]

        if len(answers) > 10:
            return {
                "error": "10 answers maximum"
            }
        
        requestBody = {"question": question["text"]}

        for i, answer in enumerate(answers):
            key = "answer" + str(i+1)
            requestBody[key] = answer["text"]
        
        response = requests.post("http://" + url + ":8000/state_obs", json=requestBody, headers=headers)
        r = response.json()
        self.phrase_tensors = np.array(r["state_matrix"])
        self.action = self.client.get_action(request['eid'], self.phrase_tensors)
        action = self.action.tolist()
        if action >= len(answers):
            answer_id = "null"
        else:
            answer_id = answers[action]["id"]
        return {
            "answer_id": answer_id,
            "eid": request['eid']
        }

class QuestionRecommendation:
    def __init__(self):
        from transformers import ElectraTokenizer, TFElectraForSequenceClassification
        self.tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator', max_length=128, pad_to_max_length=True)
        self.model = TFElectraForSequenceClassification.from_pretrained('google/electra-small-discriminator', num_labels=1)

    def __call__(self, flask_request):
        import tensorflow as tf
        import json
        request = flask_request.json
        questions = request["questions"]
        if not isinstance(questions, list):
            return "Questions should contain a list of question text."
        questions_tokenized = self.tokenizer(questions, max_length=128, padding='max_length', return_tensors="tf")
        questions_tensor = questions_tokenized["input_ids"]
        outputs = self.model(questions_tensor)
        results = outputs[0].numpy().tolist()
        results = [x[0] for x in results]

        orders = [x for x, _ in sorted(enumerate(results), key=lambda p: p[1])]
        
        return json.dumps(orders)


#connect to ray start --head
ray.init(address= url + ":6000", redis_password='5241590000000000')
serve.init(http_host="0.0.0.0")

#creation of backend and endpoint 0.0.0.0/serve to post the text to key text
serve.create_backend('rllib_client', rllib_client, args)
serve.create_endpoint('rllib_client_endpoint', backend='rllib_client', route='/eid', methods=['POST'])

serve.create_backend('Electra_Backend', ElectraNLP)
serve.create_endpoint('endpoint', backend='Electra_Backend', route="/serve", methods=['GET'])

serve.create_backend('reward_backend', reward)
serve.create_endpoint('reward', backend='reward_backend', route="/reward", methods=['POST'])

serve.create_backend('eid_create', eid)
serve.create_endpoint('eid', backend='eid_create', route='/eid_create', methods=['GET'])

serve.create_backend('nearest_neighbors', nearestneighbor)
serve.create_endpoint('nearestneighbors', backend='nearest_neighbors', route='/nearest_neighbors', methods=['POST'])

serve.create_backend('electra_tensors', ElectraTensors)
serve.create_endpoint('ElectraTensors', backend='electra_tensors', route='/electratensors', methods=['POST'])

serve.create_backend('action_eid', action_eid)
serve.create_endpoint('actioneid', backend='action_eid', route='/action_eid', methods=['POST'])

serve.create_backend('state_present', state_present)
serve.create_endpoint('statepresent', backend='state_present', route='/state_obs', methods=['POST'])


serve.create_backend('action_state_obs_eid', action_state_obs_eid)
serve.create_endpoint('actionstateobseid', backend='action_state_obs_eid', route='/action_state_obs_eid', methods=['POST'])

serve.create_backend('question_recommendation', QuestionRecommendation)
serve.create_endpoint('questionrecommendation', backend='question_recommendation', route='/question_recommendation', methods=['POST'])

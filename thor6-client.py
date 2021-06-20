from ray import serve
import requests
import ray

import numpy as np

from ray.rllib.env.policy_client import PolicyClient
import argparse

import json
from json import dumps

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
url = "34.94.83.0"

class ElectraNLP:
    def __init__(self):
        from transformers import ElectraTokenizer, TFElectraModel
        self.tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator', max_length=128, pad_to_max_length=True)
        self.model = TFElectraModel.from_pretrained('google/electra-small-discriminator')

    def __call__(self, flask_request):
        import tensorflow as tf
        import json
        self.answer = flask_request.args.get("text")
        self.input_ids = tf.constant(self.tokenizer.encode(self.answer, max_length=128, pad_to_max_length=128))[None, :]  # Batch size 1
        outputs = self.model(self.input_ids)
        results = outputs[0].numpy().tolist()

        # payload = {"model_tensors": results}
        payload = results
        payload = json.dumps(payload)
        # payload = json.loads(payload)
        # data
        headers = {"content-type": "application/json"}
        self.response = requests.post("http://" + url + ":8000/eid", json=payload, headers=headers)
        # self.display = json.loads(self.response.text)
        # output_tensors = json.loads(self.response.text)
        # hidden_layer = self.outputs[0]
        # return np.array(hidden_layer).shape
        # return output_tensors
        # return {"test the outputs": results}
        return self.response.text

class rllib_client:
    def __init__(self, args):
        # self.args = args
        self.client = PolicyClient("http://" + url + ":9900", inference_mode=args.inference_mode)
        self.eid = self.client.start_episode(training_enabled=not args.no_train)

    def __call__(self, flask_request):
        import json
        self.phrase_tensors = flask_request.json
        
        # self.phrase_tensors = {"model_tensors": self.phrase_tensors}
        
        # self.phrase_tensors = json.dumps(self.phrase_tensors)
        self.phrase_tensors = np.array(json.loads(self.phrase_tensors))
        self.action = self.client.get_action(self.eid, self.phrase_tensors)
        return {"model actions": self.action}
        # return self.phrase_tensors


#connect to ray start --head
ray.init(address= url + ":6000", redis_password='5241590000000000')
serve.init(http_host="0.0.0.0")

#creation of backend and endpoint 0.0.0.0/serve to post the text to key text
serve.create_backend('rllib_client', rllib_client, args)
serve.create_endpoint('rllib_client_endpoint', backend='rllib_client', route='/eid', methods=['POST'])

serve.create_backend('Electra_Backend', ElectraNLP)
serve.create_endpoint('endpoint', backend='Electra_Backend', route="/serve", methods=['GET'])

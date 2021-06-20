import tensorflow as tf
from transformers import ElectraTokenizer, TFElectraModel
from ray import serve
import requests
import ray
import numpy as np
from transformers import ElectraTokenizer, TFElectraModel
import pdb; 
import random
import pandas as pd

class TrainingElement:
    def __init__(self):
        self.testvar = 0
        from transformers import ElectraTokenizer, TFElectraModel
        self.tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator', max_length=128, pad_to_max_length=True)
        self.model = TFElectraModel.from_pretrained('google/electra-small-discriminator')
        self.state_cube = ['Answer.question', 'Answer.answerA', 'Answer.answerB', 'Answer.answerC', 'Answer.answerD', 'Answer.answerE', 'Answer.answerF']
        columns = ['Answer.answerA', 'Answer.answerB', 'Answer.answerC', 'Answer.answerD', 'Answer.answerE', 'Answer.answerF', 'Answer.image.label', 'Answer.question']
        self.batch_size=1
        self.data = pd.read_csv('train_set.csv', usecols=columns)
        self.num_elem = len(self.data.index)
        self.mass_matrix = []
        matrix_max_size = 10
        # pdb.set_trace()
        for i in self.data.index:
            matrix = []
            for k in self.state_cube:
                # pdb.set_trace()
                matrix.append(self.tok_conv(self.data.iloc[i][k]))
            matrix = tf.squeeze(matrix)
            # print(matrix)
            matrix = np.asarray(matrix)
            matrix = np.pad(matrix, [(0, matrix_max_size - matrix.shape[0]), (0,0), (0,0)])
            matrix = np.moveaxis(matrix, 0, 2) 
            self.mass_matrix.append(matrix)
        # self.mass_matrix = tf.squeeze(self.mass_matrix)

        self.ans_targets = self.data['Answer.image.label']
        print(np.shape(self.mass_matrix[9].numpy()))

    def tok_conv(self, t):
        input_ids = tf.constant(self.tokenizer.encode(str(t), max_length=128, pad_to_max_length=128))[None, :]  # Batch size 1
        outputs = self.model(input_ids)
        return outputs

    def random_element(self):
        c = random.choice(range(self.num_elem))
        x = self.mass_matrix[c]
        y = self.ans_targets.iloc[c]
        # print(c)
        return x, y
            
    def __call__(self, flask_request):

        obs, l = self.random_element()
        

        return {
            "Electra_Tensors": list(obs.numpy()),
            # "answers": self.answers.numpy()
            "answers": list(l)
        }        


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


#connect to ray start --head
# url="34.83.237.208"
# url="35.233.235.19"
url = "35.247.84.216"
ray.init(address= url + ":6000", _redis_password='5241590000000000')
client = serve.start(detached=True, http_host="0.0.0.0")


config_ET = {
    "num_replicas": 1
}

ray_actor_options_ET = {
    "num_cpus": 1
}

config = {
    "num_replicas": 1
}

ray_actor_options = {
    "num_cpus": 6
}

client.create_backend('electra_tensors', ElectraTensors, config=config_ET, ray_actor_options=ray_actor_options_ET)
client.create_endpoint('ElectraTensors', backend='electra_tensors', route='/electratensors', methods=['POST'])

client.create_backend('matrix_cube', TrainingElement, config=config, ray_actor_options=ray_actor_options)
client.create_endpoint('TrainingElement', backend='matrix_cube', route='/cube', methods=['GET'])

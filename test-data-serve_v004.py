import tensorflow as tf
from transformers import ElectraTokenizer, TFElectraModel
from ray import serve
import requests
import ray
import numpy as np
from transformers import ElectraTokenizer, TFElectraModel
import pdb; 
import random

class TrainingElement:
    def __init__(self):
        self.testvar = 0
        from transformers import ElectraTokenizer, TFElectraModel
        self.tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator', max_length=128, pad_to_max_length=True)
        self.model = TFElectraModel.from_pretrained('google/electra-small-discriminator')
        self.state_cube = ['Answer.question', 'Answer.answerA', 'Answer.answerB', 'Answer.answerC', 'Answer.answerD', 'Answer.answerE', 'Answer.answerF']
        columns = ['Answer.answerA', 'Answer.answerB', 'Answer.answerC', 'Answer.answerD', 'Answer.answerE', 'Answer.answerF', 'Answer.image.label', 'Answer.question']
        self.batch_size=25
        self.data = tf.data.experimental.make_csv_dataset('train_set.csv', batch_size=self.batch_size, select_columns=columns, label_name='Answer.image.label', num_epochs=1, prefetch_buffer_size=30, ignore_errors=True)
        self.data = self.data.map(self.create_matrix, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.batchElement()
        # self.qbatch, self.labbatch = self.datagen(self.data)

    def create_matrix(self, samples, targets):
        matrix1 = []
        for k in self.state_cube:
            matrix1.append(samples[k])
        return matrix1, targets
    

    @tf.function
    def datagen(self, data):
        # import tensorflow as tf
        qnatext, answers = next(iter(data.take(1)))
        return qnatext, answers
    
    def add(self):
        self.testvar += 1

    def batchElement(self):
        self.n_ran = random.choice(range(self.batch_size))
        # self.qbatch, self.labbatch = self.datagen(self.data)
        
        matrix_max_size = 10
        self.batched_cubes = []
        qnatext, self.answers = self.datagen(self.data)
        self.seq = [qnatext[0], qnatext[1], qnatext[2], qnatext[3], qnatext[4], qnatext[5], qnatext[6]]
        for i in range(self.batch_size):
            matrix2 = []
            for element_text in self.seq:
                input_ids = tf.constant(self.tokenizer.encode(element_text[i].numpy().decode('utf-8'), max_length=128, pad_to_max_length=128))[None, :]  # Batch size 1
                outputs = self.model(input_ids)
                outputs = np.squeeze(outputs)
                matrix2.append(outputs)
            matrix2 = np.asarray(matrix2)
            state_matrix = np.pad(matrix2, [(0, matrix_max_size - matrix2.shape[0]), (0,0), (0,0)])
            state_matrix = np.moveaxis(state_matrix, 0, 2) 
            self.batched_cubes.append(state_matrix)




    def __call__(self, flask_request):
        self.add()
        # local_n_ran = random.choice(range(self.batch_size))
        if self.testvar > self.batch_size*self.n_ran:
            self.testvar = 0
            self.batchElement()

        return {
            "Electra_Tensors": self.batched_cubes,
            "answers": self.answers.numpy()
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
url = "35.185.230.195"
ray.init(address= url + ":6000", _redis_password='5241590000000000')
client = serve.start(detached=True, http_host="0.0.0.0")


config_ET = {
    "num_replicas": 1
}

ray_actor_options_ET = {
    "num_cpus": 6
}

config = {
    "num_replicas": 1
}

ray_actor_options = {
    "num_cpus": 10
}

client.create_backend('electra_tensors', ElectraTensors, config=config_ET, ray_actor_options=ray_actor_options_ET)
client.create_endpoint('ElectraTensors', backend='electra_tensors', route='/electratensors', methods=['POST'])

client.create_backend('matrix_cube', TrainingElement, config=config, ray_actor_options=ray_actor_options)
client.create_endpoint('TrainingElement', backend='matrix_cube', route='/cube', methods=['GET'])

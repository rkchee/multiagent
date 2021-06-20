import tensorflow as tf
from transformers import ElectraTokenizer, TFElectraModel
from ray import serve
import requests
import ray
import numpy as np
from transformers import ElectraTokenizer, TFElectraModel
import pdb; 
import random

class BatchElements:
    def __init__(self):
        self.testvar = 0
        from transformers import ElectraTokenizer, TFElectraModel
        self.tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator', max_length=128, pad_to_max_length=True)
        self.model = TFElectraModel.from_pretrained('google/electra-small-discriminator')
        self.state_cube = ['Answer.question', 'Answer.answerA', 'Answer.answerB', 'Answer.answerC', 'Answer.answerD', 'Answer.answerE', 'Answer.answerF']
        columns = ['Answer.answerA', 'Answer.answerB', 'Answer.answerC', 'Answer.answerD', 'Answer.answerE', 'Answer.answerF', 'Answer.image.label', 'Answer.question']
        self.batch_size=5
        self.data = tf.data.experimental.make_csv_dataset('train_set.csv', batch_size=self.batch_size, select_columns=columns, label_name='Answer.image.label', num_epochs=1, prefetch_buffer_size=30, ignore_errors=True)
        self.data = self.data.map(self.create_matrix, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.batchElement()

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
        # pdb.set_trace()
        matrix_max_size = 10
        self.n_ran = random.choice(range(self.batch_size))
        self.batched_cubes = []
        qnatext, self.answers = self.datagen(self.data)
        self.seq = [qnatext[0], qnatext[1], qnatext[2], qnatext[3], qnatext[4], qnatext[5], qnatext[6]]
        for i in range(self.batch_size):
            matrix2 = []
            for element_text in self.seq:
                input_ids = tf.constant(self.tokenizer.encode(element_text[i].numpy().decode('utf-8'), max_length=128, pad_to_max_length=128))[None, :]  # Batch size 1
                # input_ids = graph_token_layz(element_text[0].numpy().decode('utf-8'))
                # pdb.set_trace()
                outputs = self.model(input_ids)
                outputs = np.squeeze(outputs)
                matrix2.append(outputs)
            matrix2 = np.asarray(matrix2)
            state_matrix = np.pad(matrix2, [(0, matrix_max_size - matrix2.shape[0]), (0,0), (0,0)])
            state_matrix = np.moveaxis(state_matrix, 0, 2)            
            self.batched_cubes.append(state_matrix)


        # for x, y in self.joined_dataset:
        #     self.answer1 = y
        #     self.quest1 = x
        # self.testvar += 1

    def __call__(self, flask_request):
        # print(np.shape(self.batched_cubes[self.n_ran]))
        self.add()
        local_n_ran = random.choice(range(self.batch_size))
        if self.testvar > self.batch_size*self.n_ran:
            self.testvar = 0
            self.batchElement()
        return {
            # "test": [self.quest1],
            "random": [self.n_ran],
            "counter": [self.testvar],
            'answer': [self.answers[local_n_ran].numpy()],
            "Electra_Tensors": [self.batched_cubes[local_n_ran]],
        }



url = "35.185.230.195"
ray.init(address= url + ":6000", _redis_password='5241590000000000')
client = serve.start(detached=True, http_host="0.0.0.0")


client.create_backend('test', BatchElements)
client.create_endpoint('endpoint', backend='test', route='/test', methods=['GET'])

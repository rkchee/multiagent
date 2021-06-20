import tensorflow as tf
from transformers import ElectraTokenizer, TFElectraModel
from ray import serve
import requests
import ray
import numpy as np



@tf.autograph.experimental.do_not_convert
def create_matrix(samples, targets):
    state_cube = ['Answer.question', 'Answer.answerA', 'Answer.answerB', 'Answer.answerC', 'Answer.answerD', 'Answer.answerE', 'Answer.answerF']
    matrix = []
    for k in state_cube:
        matrix.append(samples[k])
    return matrix, targets

@tf.function
def datagen(data):
    # import tensorflow as tf
    qnatext, answers = next(iter(data.take(1)))
    return qnatext, answers

from transformers import ElectraTokenizer, TFElectraModel
tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator', max_length=128, pad_to_max_length=True)
model = TFElectraModel.from_pretrained('google/electra-small-discriminator')

columns = ['Answer.answerA', 'Answer.answerB', 'Answer.answerC', 'Answer.answerD', 'Answer.answerE', 'Answer.answerF', 'Answer.image.label', 'Answer.question']
batch_size=1
data = tf.data.experimental.make_csv_dataset('train_set.csv', batch_size=batch_size, select_columns=columns, label_name='Answer.image.label', num_epochs=1, prefetch_buffer_size=10, ignore_errors=True)
data = data.map(create_matrix, num_parallel_calls=tf.data.experimental.AUTOTUNE)
qnatext, answers = datagen(data)


# store_list= []

seq = [qnatext[0], qnatext[1], qnatext[2], qnatext[3], qnatext[4], qnatext[5], qnatext[6]]
# for i in range(batch_size):
matrix = []
matrix_max_size = 10
for element_text in seq:
    input_ids = tf.constant(tokenizer.encode(element_text[0].numpy().decode('utf-8'), max_length=128, pad_to_max_length=128))[None, :]  # Batch size 1
    outputs = model(input_ids)
    outputs = np.squeeze(outputs)
    matrix.append(outputs)
matrix = np.asarray(matrix)
# matrix = np.squeeze(matrix) 
state_matrix = np.pad(matrix, [(0, matrix_max_size - matrix.shape[0]), (0,0), (0,0)])
state_matrix = np.moveaxis(state_matrix, 0, 2)
# print(np.shape(state_matrix))
# store_list.append(state_matrix)

print(np.shape(state_matrix))


# class ServeData:
#     def __init__(self):



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
url = "35.247.47.188"
ray.init(address= url + ":6000", _redis_password='5241590000000000')
client = serve.start(detached=True, http_host="0.0.0.0")


config = {
    "num_replicas": 6
}
client.create_backend('electra_tensors', ElectraTensors, config=config)
client.create_endpoint('ElectraTensors', backend='electra_tensors', route='/electratensors', methods=['POST'])

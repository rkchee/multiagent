import tensorflow as tf
from transformers import ElectraTokenizer, TFElectraModel
from ray import serve
import requests
import ray
import numpy as np
from transformers import ElectraTokenizer, TFElectraModel
import pdb; 
import random
import tensorflow as tf
import pandas as pd
import dask.array as da
import h5py




from transformers import ElectraTokenizer, TFElectraModel, ElectraTokenizerFast
tokenizer = ElectraTokenizerFast.from_pretrained('google/electra-small-discriminator', max_length=128, pad_to_max_length=True)
model = TFElectraModel.from_pretrained('google/electra-small-discriminator')

columns = ['Answer.answerA', 'Answer.answerB', 'Answer.answerC', 'Answer.answerD', 'Answer.answerE', 'Answer.answerF', 'Answer.image.label', 'Answer.question']
data = pd.read_csv('train_set.csv', usecols=columns)
# print(data.iloc[1])

state_cube = ['Answer.question', 'Answer.answerA', 'Answer.answerB', 'Answer.answerC', 'Answer.answerD', 'Answer.answerE', 'Answer.answerF']

def tok_conv(t):
    # print(type(str(t)))
    input_ids = tf.constant(tokenizer.encode(str(t), max_length=128, pad_to_max_length=128))[None, :]  # Batch size 1
    outputs = model(input_ids)
    # print(input_ids)
    return outputs

mass_matrix = []
for i in data.index:
    matrix = []
    for k in state_cube:
        # print(data.iloc[i][k])
        # pdb.set_trace()
        matrix.append(tok_conv(data.iloc[i][k]))
    matrix = tf.squeeze(matrix)
    print(np.shape(matrix[1]))
    mass_matrix.append(matrix)
mass_matrix = tf.squeeze(mass_matrix)


pdb.set_trace()

# while True:

# data = tf.data.experimental.make_csv_dataset('train_set.csv', batch_size=1, select_columns=columns, label_name='Answer.image.label', num_epochs=1, prefetch_buffer_size=1, ignore_errors=True)



# for k in state_cube:
#     print(data[k])

# data = data.apply(lambda x: tok_conv(x), axis=1)
# print(data)


# def mat_append(mat):
#     mat.append(mat)
#     return mat


# def tokenize_model(inp_ids):
#     # print(inp_ids.numpy()[0])
#     outputs = model(inp_ids.numpy()[0])
    
#     return outputs

# def tokenize_encode(text):
#     input_ids = tf.constant(tokenizer.encode(tf.compat.as_str(text.numpy()[0]), max_length=128, pad_to_max_length=128))[None, :]  # Batch size 1
#     # outputs = model(input_ids)
#     # print(outputs)
#     # outputs = np.squeeze(outputs)
#     return input_ids


# def tok_map(samples, targets):
#     state_cube = ['Answer.question', 'Answer.answerA', 'Answer.answerB', 'Answer.answerC', 'Answer.answerD', 'Answer.answerE', 'Answer.answerF']
#     matrix = []
#     matrix_max_size = 10
#     for k in state_cube:
#         encoded = tf.py_function(tokenize_encode, # tokenize_encode is a wrapper around the Huggingface tokenizer and encoder
#                                     inp=[samples[k]],
#                                     Tout=[tf.int32])   
#         outputs = tf.py_function(tokenize_model, inp=[encoded], Tout=[tf.float32])
#         matrix.append(outputs)
#     # pdb.set_trace()
#     matrix = tf.squeeze(matrix)
#     # matrix = np.asarray(matrix)
#     # state_matrix = np.pad(matrix, [(0, matrix_max_size - matrix.shape[0]), (0,0), (0,0)])
#     # state_matrix = np.moveaxis(state_matrix, 0, 2) 
#     return matrix, targets

# data = data.take(1).map(tok_map, num_parallel_calls=tf.data.experimental.AUTOTUNE).prefetch(2).cache().repeat(5)
# # tf.data.experimental.AUTOTUNE
# while True:
#     print(next(iter(data)))
# # for x, y in data:
# #     print(x)
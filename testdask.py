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
# from mpi4py import MPI
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


mcq_number ={
        'A': 0,
        'a': 0,
        'B': 1,
        'b': 1,
        'C': 2,
        'c': 2,
        'D': 3,
        'd': 3,
        'E': 4,
        'e': 4,
        'F': 5,
        'f': 5,
        'G': 6,
        'g': 6,
        'H': 7,
        'h': 7,
        'I': 8,
        'i': 8,
        'J': 9,
        'j': 9, 
}  

mass_matrix = [[],[]]
matrix_max_size = 10
for i in data.index:
    test_answer = data.iloc[i]['Answer.image.label']
    if test_answer not in mcq_number:
        i += 1
    else: 
        matrix = []
        for k in state_cube:
            # print(data.iloc[i][k])
            # pdb.set_trace()
            matrix.append(tok_conv(data.iloc[i][k]))
        matrix = tf.squeeze(matrix)
        matrix = np.pad(matrix, [(0, matrix_max_size - matrix.shape[0]), (0,0), (0,0)])
        matrix = np.moveaxis(matrix, 0, 2) 
        mass_matrix[0].append(matrix)
        print(np.shape(mass_matrix[0]))
        mass_matrix[1].append(mcq_number[test_answer])






# mass_matrix[0] = tf.squeeze(mass_matrix[0])
# mass_matrix[1] = tf.squeeze(mass_matrix[1])
# pdb.set_trace()
# mass_matrix = da.stack(macss_matrix, axis=0)
# print(np.shape(mass_matrix))

# rank = MPI.COMM_WORLD.rank




d = da.from_array(mass_matrix[0])
y = da.from_array(mass_matrix[1])
da.to_hdf5('combinedxy.hdf5', {'/x': d, 'y': y})
pdb.set_trace()

# d.to_hdf5('myfile_x.hdf5', '/x')
# y.to_hdf5('myfile_y.hdf5', '/y')


# f = h5py.File('myfile_x.hdf5', 'r')
# y = h5py.File('myfile_y.hdf5', 'r')
# ds = f['/x']
# ys = y['/y']

# xy = h5py.File('combinedxy.hdf5', 'r')
# xy_y = xy['/y']
# xy_x = xy['/x']




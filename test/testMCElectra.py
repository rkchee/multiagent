import tensorflow as tf
from ray import serve
import requests
import ray
from transformers import ElectraTokenizer, TFElectraForMultipleChoice
import numpy as np
import pdb; 
import random
import tensorflow as tf
import pandas as pd
import dask.array as da
import dask.dataframe as dd
import h5py
# from mpi4py import MPI
import h5py
# from transformers import AutoModelForQuestionAnswering, AutoModelForMultipleChoice

# GLUE_TASKS = ["cola", "mnli", "mnli-mm", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]

# task = "cola"
# model_checkpoint = "distilbert-base-uncased"
# batch_size = 16







from transformers import ElectraTokenizer, TFElectraForMultipleChoice
tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')
model = TFElectraForMultipleChoice.from_pretrained('google/electra-small-discriminator')

# prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
# choice0 = "It is eaten with a fork and a knife."
# choice1 = "It is eaten while held in the hand."
# choice2 = "but cheeze"
# labels = [0]

columns = ['Answer.answerA', 'Answer.answerB', 'Answer.answerC', 'Answer.answerD', 'Answer.answerE', 'Answer.answerF', 'Answer.image.label', 'Answer.question']
state_cube = ['Answer.question', 'Answer.answerA', 'Answer.answerB', 'Answer.answerC', 'Answer.answerD', 'Answer.answerE', 'Answer.answerF']
# data = (pd.read_csv)('train_set.csv', usecols=columns)
ddf = dd.read_csv('train_set.csv', usecols=columns)
ddf = ddf.compute()

# pdb.set_trace()
train_text = []
train_label = []
train_dataset = []
for i in ddf.index:
    train_label.append(ddf.iloc[i]['Answer.image.label'])
    for k in state_cube:
        train_text.append(str(ddf.iloc[i][k]))
    # pdb.set_trace()
    train_dataset.append(tokenizer([train_text[0], train_text[0], train_text[0], train_text[0], train_text[0], train_text[0]], [train_text[1], train_text[2], train_text[3], train_text[4], train_text[5], train_text[6]], truncation=True, padding='max_length'))


# train_text = 


pdb.set_trace()



# encoding = tokenizer([[prompt, prompt, prompt], [choice0, choice1, choice2]], return_tensors='tf', padding=True)
# inputs = {k: tf.expand_dims(v, 0) for k, v in encoding.items()}
# outputs = model(inputs)  # batch size is 1

# # the linear classifier still needs to be trained
# logits = outputs.logits
# print(logits)
# # pdb.set_trace()

# encoding = tokenizer([prompt, prompt, prompt], [choice0, choice1, choice2], return_tensors='tf', padding=True)
# inputs = {k: tf.expand_dims(v, 0) for k, v in encoding.items()}
# outputs = model(**inputs)

# # the linear classifier still needs to be trained
# logits = outputs.logits
# print(logits)
# # pdb.set_trace()
# pdb.set_trace()
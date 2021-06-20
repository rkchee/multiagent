import pandas as pd
import numpy as np
import json
import requests


url = '35.199.191.27'

# train test split
sheets_cards = pd.read_excel('../training dataset - medical.xlsx', sheet_name='Cardiology and Infectious Disea')
cardiology = sheets_cards.copy()
train_set = cardiology.sample(frac=0.1, random_state=0)
train_set.to_csv('train_set.csv', index=False)
test_set = cardiology.drop(train_set.index)
test_set.to_csv('test_set.csv', index=False)


# set the training data 
data = pd.read_csv('train_set.csv')

def sequence(seq):
    if not type(seq) == str:
        seq = str(seq)
    
    js = {
        "text":seq
    }

    dumps = json.dumps(js)
    payload = json.loads(dumps)
    headers = {"content-type": "application/json"}
    response = requests.post("http://" + url + ":8000/electratensors", json=payload, headers=headers)
    t = response.text 
    t = json.loads(t)
    t = t['Electra_Tensors']
    return np.asarray(t)


def form_matrix(qa_index, question):
    matrix_max_size = 10
    matrix = []

    mc_choices = ["A", "B", "C", "D", "E"]

    matrix = question
    for letter in mc_choices:
        answer = data['Answer.answer' +  letter].iloc[qa_index]
        answer = sequence(answer)
        matrix = np.concatenate((matrix, answer), axis=0)
    
    matrix = np.squeeze(matrix)
    state_matrix = np.pad(matrix, [(0, matrix_max_size - matrix.shape[0]), (0,0), (0,0)])
    state_matrix = np.moveaxis(state_matrix, 0, 2)
    return state_matrix

# questions = data['Answer.question']
# answers = data['Answer.image.label']

store_list = []
for i in data.index:
    obs = data['Answer.question'].iloc[i]
    obs = sequence(obs)
    obs = form_matrix(i, obs)
    store_list.append(obs)
    

print(obs[0])
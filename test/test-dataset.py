import pandas as pd
import json
import requests
import numpy as np

# train test split
sheets_cards = pd.read_excel('../training dataset - medical.xlsx', sheet_name='Cardiology and Infectious Disea')
# cardiology = sheets_cards.copy()
# train_set = cardiology.sample(frac=0.75, random_state=0)
# train_set.to_csv('train_set.csv', index=False)
# test_set = cardiology.drop(train_set.index)
# test_set.to_csv('test_set.csv', index=False)

data_ans = pd.DataFrame(sheets_cards)
n = len(data_ans.index)

# data_allans = [x for x in data_ans['Answer.image.label']]
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

url ="35.230.63.77"

class electra_serve:
    def sequence(seq):
        if not type(seq) == str:
            seq = str(seq)            
            print(seq)


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


        # try: 
        #     dumps = json.dumps(js)
        #     payload = json.loads(dumps)

        #     headers = {"content-type": "application/json"}
        #     response = requests.post("http://" + url + ":8000/electratensors", json=payload, headers=headers)
        #     t = response.text
        #     t = json.loads(t)
        #     t = t['Electra_Tensors']
        # except:
        #     # print(seq)
        #     # t = np.zeros((1, 128, 256))
        #     t = seq

        # response = requests.post("http://" + url + ":8000/electratensors", json=payload, headers=headers)
        # t = response.text
        # t = json.loads(t)
        # t = t['Electra_Tensors']




        # print(t)
        # if t == None or t == '':
        #     print("there was something wierd here ", t)
        #     t = np.zeros((1, 128, 256))
        # t = json.loads(t)
        # t = t['Electra_Tensors']
        # try: 
        #     t = json.loads(t)
        #     t = t['Electra_Tensors']
        # except:
        #     t = np.zeros((1, 128, 256))

        return np.asarray(t)
        # print(payload)
        # return payload


questions = data_ans['Answer.question']
# obs = electra_serve.sequence(questions.iloc[1])
# obs = electra_serve.sequence(None)
# obs = json.loads(None)
# seq = ""
# if not isinstance(seq, str):
#     print(seq)
#     seq =""
# js = {
#     "text": seq
# }

# dumps = json.dumps(js)
# payload = json.loads(dumps)
# headers = {"content-type": "application/json"}
# response = requests.post("http://" + url + ":8000/electratensors", json=payload, headers=headers)
# t = response.text
# t = json.loads(t)

answerD = data_ans['Answer.answer' + 'A']
labels_ans = data_ans['Answer.image.label']
# for x in answerD:
#     t= electra_serve.sequence(x)
#     print(t.shape)

mcq = ['A', 'B', 'C', 'D', 'E']

counter = 1
for c in mcq:
    for answer in data_ans['Answer.answer' + c]:
        t= electra_serve.sequence(answer)        
        # if not type(answer) == str:
        #     print(str(counter) + " answer is not a string, it is " + str(type(answer)) + " " + str(answer) )
        #     counter += 1


# print(np.shape(t['Electra_Tensors']))
# print(np.shape(np.zeros((1, 128, 256))))


# episode = 0
# for x in data_ans['Answer.image.label']:
#     # if x not in ["A", "B", "C", "D", "E", "a", "F"]:
#     x =False
#     if x not in mcq_number:
#         print("this value is outside of the contraints was", x)
#         # mcq_number[x]
#         x = False
#     print(x)
#     if not x:
#         print("x is converted to false")
#     if episode >= (n-1):
#         print("episodes: ", episode)
#         print("n=", n)
#         print("n-1", n-1)

#     episode += 1

    


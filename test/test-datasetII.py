import pandas as pd
import json
import requests
import numpy as np

sheets_cards = pd.read_excel('../training dataset - medical.xlsx', sheet_name='Cardiology and Infectious Disea')
data_ans = pd.DataFrame(sheets_cards)
n = len(data_ans.index)

mcq = ['A', 'B', 'C', 'D', 'E']




answers = data_ans['Answer.answer' + 'A']
questions = data_ans['Answer.question']

labels_ans = data_ans['Answer.image.label']

counter = 1
# for c in mcq:
#     for answer in data_ans['Answer.answer' + c]:
#         if not type(answer) == str:
#             print(str(counter) + " answer is not a string, it is " + str(type(answer)) + " " + str(answer) )
#             counter += 1

for question in questions:
    if not type(question) == str:
        print(str(counter) + " answer is not a string, it is " + str(type(question)) + " " + str(question) )
        counter += 1

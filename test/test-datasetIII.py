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

for i in range(n):
    print(answers[i])
    print(answers.iloc[i])

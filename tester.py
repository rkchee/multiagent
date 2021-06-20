import pandas as pd
import numpy as np
import random

data = pd.DataFrame([1,2,3,4,5])
data = data.sample(n=2)
# print(data)
# print("   -----   ")
# print(data.iloc[2])

# for i in data.index:
#     print(data)
#     print("  ______   ")
#     # print(data[0])


l= [],[]
answers = []
for i in range(10):
    l[0].append(data.sample(1))
    l[1].append([1])

# s = random.randrange(0, 19, 1)
s = [np.array(['A'])]
print(s[0][0])

mcq = {
    'A': 0
}

if s[0][0] not in mcq: 
    print('not the same')
else:
    print('match')
# print(l[0][np.squeeze(s.index.values)])
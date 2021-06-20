import pandas as pd

sheets_cards = pd.read_excel('training dataset - medical.xlsx', sheet_names='Cardiology and Infectious Disea')


cardiology = sheets_cards.copy()
train_set = cardiology.sample(frac=0.75, random_state=0)
test_set = cardiology.drop(train_set.index)
filter = test_set['Answer.image.label']
# print(filter.iloc[0])

mcq = {
    "A": 0,
    "a": 0
}
print(mcq['a'])

# print(len(test_set.index))
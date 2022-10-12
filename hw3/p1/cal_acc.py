import pandas as pd

df = pd.read_csv('pred.csv')
correct = 0

for i in range(1, len(df)):
    pred = int(df['filename'][i].split('_')[0])
    if pred == df['label'][i]:
        correct += 1

print("Acc:", correct/(len(df)-1))
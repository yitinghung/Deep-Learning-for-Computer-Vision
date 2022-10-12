import pandas as pd

file_pth = '~/Documents/DLCV/hw4/hw4_data/office/val.csv'
df = pd.read_csv(file_pth)
labels = df['label'].tolist()

label_dic = {}
i = 0
for label in labels:
    if label not in label_dic:
        label_dic[label] = i
        i += 1

#print(labels)
print(label_dic)
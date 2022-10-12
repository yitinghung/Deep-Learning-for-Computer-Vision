import sys, csv
import pandas as pd
import numpy as np
args = sys.argv

# read your prediction file
pred_df = pd.read_csv(args[1])
pred = pred_df['label'].tolist()

# read ground truth data
gt_df = pd.read_csv(args[2])
gt = gt_df['label'].tolist()

if len(pred) != len(gt):
    sys.exit("Test case length mismatch.")

acc = 0.0
for i in range(len(pred)):
    if pred[i] == gt[i]:
        acc += 1
acc = acc / len(pred)

print('Accuracy: {:.4f}'.format(acc))

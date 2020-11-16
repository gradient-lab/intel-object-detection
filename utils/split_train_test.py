import sys
import numpy as np
import pandas as pd

np.random.seed(1)
sys.path.append("..")

# Read Custom Dataset Labels
full_labels = pd.read_csv('./data/raccoon_labels.csv')

# Group by filename
grouped = full_labels.groupby('filename')

# Split each file into a group in a list
gb = full_labels.groupby('filename')
grouped_list = [gb.get_group(x) for x in gb.groups]
train_index = np.random.choice(len(grouped_list), size=160, replace=False)
test_index = np.setdiff1d(list(range(200)), train_index)

# Take first 200 files
train = pd.concat([grouped_list[i] for i in train_index])
test = pd.concat([grouped_list[i] for i in test_index])

# Save
train.to_csv('./data/train_labels.csv', index=None)
test.to_csv('./data/test_labels.csv', index=None)

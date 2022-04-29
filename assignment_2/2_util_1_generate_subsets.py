#! /usr/bin/env python3

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from utils import *

random_state = 42

reset_timer()

dataset_name = 'training_set_VU_DM'
in_path = (Path(__file__).parent / 'dataset' / f'{dataset_name}.csv').resolve()
out_base_path = (Path(__file__).parent / 'dataset').resolve()

out_base_path.mkdir(exist_ok=True)

tprint(f'Loading dataset from {in_path}...')
df = pd.read_csv(in_path)

tprint(f'Splitting dataset into train and test sets...')
train_set, test_set = train_test_split(df, shuffle=True, test_size=0.2, random_state=random_state)

tprint(f'Writing training set to {out_base_path}...')
train_set.to_csv(out_base_path / f'{dataset_name}-train.csv', index=False)
test_set.to_csv(out_base_path / f'{dataset_name}-test.csv', index=False)

tprint(f'Create smaller training set for faster iterating...')
train_set_small = train_set.sample(frac=0.01, random_state=random_state)
train_set_small.to_csv(out_base_path / f'{dataset_name}-train-small.csv', index=False)

tprint('Done')

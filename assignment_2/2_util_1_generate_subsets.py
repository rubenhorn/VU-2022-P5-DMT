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
train_set, test_set = train_test_split(
    df, shuffle=True, test_size=0.2, random_state=random_state)

tprint(f'Writing train set to {out_base_path}...')
train_set.to_csv(out_base_path / f'{dataset_name}-train.csv', index=False)
test_set.to_csv(out_base_path / f'{dataset_name}-test.csv', index=False)

tprint(f'Create smaller train set for faster iterating...')
test_set.sort_values(by='srch_id', inplace=True)
train_set_small = test_set.head(n=int(len(test_set) * 0.01))
train_set_small = train_set_small.sample(frac=1, random_state=random_state)
train_set_small.to_csv(
    out_base_path / f'{dataset_name}-train-small.csv', index=False)

tprint(f'Create smaller test set for faster iterating...')
test_set.sort_values(by='srch_id', inplace=True)
test_set_small = test_set.head(n=int(len(test_set) * 0.01))
test_set_small = test_set_small.sample(frac=1, random_state=random_state)
test_set_small.to_csv(
    out_base_path / f'{dataset_name}-test-small.csv', index=False)

tprint(f'Create historical data of property bookings')
freq_df = train_set.groupby(['prop_id']).agg(
    {'booking_bool': ['sum', 'count']})
freq_df.columns = freq_df.columns.droplevel()
freq_df['booking_freq'] = freq_df['sum'] / freq_df['count']
freq_df = freq_df.rename(columns={'sum': 'booking_sum'})
freq_df.drop(['count'], axis=1, inplace=True)
freq_df.to_csv(out_base_path / 'prop_booking_freq.csv')

methods = ["mean", "std"]

tprint(f'Create historical data of property position')
no_rand_position = train_set.loc[train_set['random_bool'] == 0]

position_df = train_set.groupby('prop_id').agg({'position': methods})
position_df.columns = position_df.columns.droplevel()
position_df = position_df.rename(columns={'mean': 'position_mean', 'std': 'position_std'})

no_rand_position = no_rand_position.groupby('prop_id').agg({'position': methods})
no_rand_position.columns = no_rand_position.columns.droplevel()
no_rand_position = no_rand_position.rename(columns={'mean': 'no_rand_position_mean', 'std': 'no_rand_position_std'})

position_df = pd.merge(position_df, no_rand_position, on='prop_id')
position_df.to_csv(out_base_path / 'prop_position.csv')

tprint('Done')

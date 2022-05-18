#! /usr/bin/env python3

import pandas as pd
from pathlib import Path
from utils import *

reset_timer()

dataset_name = 'training_set_VU_DM'
in_path = (Path(__file__).parent / 'dataset' / f'{dataset_name}.csv').resolve()
out_base_path = (Path(__file__).parent / 'output').resolve()

out_base_path.mkdir(exist_ok=True)

tprint(f'Loading dataset from {in_path}...')
df = pd.read_csv(in_path)

tprint('Counting number of properties per search...')
prop_counts = df.groupby('srch_id').size().reset_index(name='count')['count']
tprint(f'Properties per query: {prop_counts.min()}-{prop_counts.max()}')

tprint('Creating summary of dataset...')
summary = df.describe()
tprint(f'Writing summary to {out_base_path}...')
summary.to_csv(out_base_path / f'{dataset_name}-summary.csv', index=True)
tprint('Summary:')
print(summary)

tprint('Creating info of dataset...')
info = pd.DataFrame(list(zip(df.columns, df.dtypes, df.isnull().sum(), df.isna().sum())),
                    columns=['columns', 'type', 'null_count', 'na_count'])
tprint(f'Writing info to {out_base_path}...')
info.to_csv(out_base_path / f'{dataset_name}-info.csv', index=True)
tprint('Info:')
print(info)

tprint('Creating correlation of dataset...')
corr = df.corr()
tprint(f'Writing correlation to {out_base_path}...')
corr.to_csv(out_base_path / f'{dataset_name}-corr.csv', index=True)
tprint('Correlation:')
print(corr)

tprint('Creating correlation of dataset in relation to \'booking_bool\'...')
booking_corr = df.corrwith(df['booking_bool'])
tprint(f'Writing correlation to {out_base_path}...')
booking_corr.to_csv(out_base_path / f'{dataset_name}-booking-corr.csv', index=True)
tprint('Correlation:')
print(booking_corr)

tprint('Done')

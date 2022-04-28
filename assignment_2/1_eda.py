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

tprint('Creating summary of dataset...')
summary = df.describe()
tprint(f'Writing summary to {out_base_path}...')
summary.to_csv(out_base_path / f'{dataset_name}-summary.csv')

tprint('Summary:')
print(summary)

tprint('Columns:')
print(', '.join(df.columns))

print('Done')

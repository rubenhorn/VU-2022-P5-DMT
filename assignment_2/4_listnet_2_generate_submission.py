#! /usr/bin/env python3

import pandas as pd
from utils import *
from pathlib import Path
import tensorflow as tf

from preprocessing import Preprocessing

reset_timer()

dataset_name = 'test_set_VU_DM'
model_in_path = (Path(__file__).parent / 'models' /
                 'training_set_VU_DM-ListNet').resolve()
prediction_out_path = (Path(__file__).parent / 'output' /
                       f'{dataset_name}-prediction.csv').resolve()

test_set = load_dataset(dataset_name)
tprint(f'Loading model from {model_in_path}...')
model = tf.keras.models.load_model(model_in_path)
tprint('Generating prediction...')
columns = ['srch_id', 'prop_id', 'score']
df = pd.DataFrame(columns=columns)
start_time = time.time()
grouped_test_set = test_set.groupby('srch_id')
n_groups = len(grouped_test_set)
group_idx = 0
pp = Preprocessing()

X_batches = []
for _, group in grouped_test_set:
        assert len(group) <= DOCS_PER_QUERY
        group = group.sample(DOCS_PER_QUERY, replace=True) # Pad input
        assert len(group) == DOCS_PER_QUERY
        X = pp.transform(group)
        X_batches.append(X)
y_batches = model.predict(X_batches)

tprint('Sort properties by predicted score...')
for i in range(len(grouped_test_set)):
    group = grouped_test_set.get_group(i)
    scores = y_batches[i]
    remaining_time = (time.time() - start_time) * (n_groups - i) / (i + 1)
    tprint(f'Sorting group {group_idx}/{n_groups} (Remaining: { format_time(remaining_time) })...', end='\r')
    group['score'] = scores
    group.sort_values(by='score', ascending=False, inplace=True)
    group.drop_duplicates(subset=['prop_id'], inplace=True) # Remove padding
    df = df.append(group[columns])

tprint(f'Saving prediction to {prediction_out_path}...')
prediction_out_path.parent.mkdir(exist_ok=True)
assert len(df) == len(test_set) # Check if all rows are present
df.to_csv(prediction_out_path, index=False)

tprint('Done')

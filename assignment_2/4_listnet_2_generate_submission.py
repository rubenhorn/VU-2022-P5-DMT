#! /usr/bin/env python3

import sys
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
if not use_full_dataset():
    test_set = test_set.sort_values(by='srch_id')
    test_set = test_set.head(int(len(test_set) * 0.01))

tprint(f'Loading model from {model_in_path}...')
model = tf.keras.models.load_model(model_in_path)
start_time = time.time()
tprint('Creating batches by grouping by srch_id...')
# With padding


def pad_group(group): return pd.concat(
    [group, group.sample(DOCS_PER_QUERY - len(group), replace=True)])


grouped_test_set = [pad_group(group)
                    for _, group in test_set.groupby('srch_id')]
n_groups = len(grouped_test_set)
tprint('Transforming input...')
pp = Preprocessing()
X_batches = []
y_batches = []
start_time = time.time()
for i in range(len(grouped_test_set)):
    group = grouped_test_set[i]
    remaining_time = (time.time() - start_time) * (n_groups - i) / (i + 1)
    tprint(
        f'Group {i + 1}/{n_groups} (Remaining: { format_time(remaining_time) })...', end='\r')
    X = pp.transform(group)
    X_batches.append(X)
    # Alternative: Run inference on a per query basis (not recommended)
    # X_minibatch = tf.expand_dims(tf.convert_to_tensor(X), axis=0)
    # y_batch = model.predict(X_minibatch, batch_size=1)
    # y_batches.append(y_batch)
print()
tprint('Generating prediction...')
BATCH_SIZE = 32
start_time = time.time()
for i in range(0, len(X_batches), BATCH_SIZE):
    remaining_time = (time.time() - start_time) * \
        (len(X_batches) - i) / (i + 1)
    tprint(
        f'Batch {int(i / BATCH_SIZE) + 1}/{len(X_batches / BATCH_SIZE)} (Remaining: { format_time(remaining_time) })...', end='\r')
    X_batch = X_batches[i:i + BATCH_SIZE] if i + \
        BATCH_SIZE < len(X_batches) else X_batches[i:]
    X_batch_tensor = tf.convert_to_tensor(np.array(X_batch))
    y_batch = model.predict(
        X_batch_tensor, batch_size=BATCH_SIZE, use_multiprocessing=True)
    y_batches.extend(y_batch)
print()

tprint('Sort properties by predicted score...')
result = np.empty(shape=(2,), dtype=int)
columns = ['srch_id', 'prop_id']
start_time = time.time()
for i in range(len(grouped_test_set)):
    group = grouped_test_set[i]
    scores = y_batches[i]
    remaining_time = (time.time() - start_time) * (n_groups - i) / (i + 1)
    tprint(
        f'Sorting group {i + 1}/{n_groups} (Remaining: { format_time(remaining_time) })...', end='\r')
    group['score'] = scores
    group.sort_values(by='score', ascending=False, inplace=True)
    group.drop_duplicates(subset=['prop_id'], inplace=True)  # Remove padding
    result = np.vstack((result, group[columns].values))
print()
df = pd.DataFrame(result[1:], columns=columns)

tprint(f'Saving prediction to {prediction_out_path}...')
prediction_out_path.parent.mkdir(exist_ok=True)
assert len(df) == len(
    test_set), f'Prediction size: {len(df)} != test set size: {len(test_set)}'
df.to_csv(prediction_out_path, index=False)

tprint('Done')

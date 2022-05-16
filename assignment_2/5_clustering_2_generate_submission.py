#! /usr/bin/env python3

import sys
import accelerate

from joblib import load
import pandas as pd
from utils import *
from pathlib import Path

reset_timer()

dataset_name = 'test_set_VU_DM'
model_in_path = (Path(__file__).parent / 'models' /
                 'training_set_VU_DM-clustering.joblib').resolve()
prediction_out_path = (Path(__file__).parent / 'output' /
                       f'{dataset_name}-prediction.csv').resolve()


def compute_search_result_scores(search_results, model, batch_size=1000):
    score = []
    n_search_results = len(search_results)
    n_batches = int(np.ceil(n_search_results / batch_size))
    start_time = time.time()
    for batch_idx in range(n_batches):
        remaining_time = (time.time() - start_time) * \
            (n_batches - batch_idx) / (batch_idx + 1)
        tprint(
            f'Predicting batch {batch_idx + 1}/{n_batches} (Remaining: { format_time(remaining_time) })...', end='\r')
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, n_search_results)
        batch = search_results.iloc[start_idx:end_idx]
        score.extend(model.predict(batch))
    print()
    tprint('Generating dataframe...')
    return pd.DataFrame({
        'srch_id': search_results['srch_id'],
        'prop_id': search_results['prop_id'],
        'score': score
    })


test_set = load_dataset(dataset_name)
if not use_full_dataset():
    test_set = test_set.sort_values(by='srch_id')
    test_set = test_set.head(int(len(test_set) * 0.01))

tprint(f'Loading model from {model_in_path}...')
model = load(model_in_path)
tprint('Generating prediction...')
df = compute_search_result_scores(test_set, model)
tprint('Sorting prediction...')
df.sort_values(['srch_id', 'score'], ascending=[True, False], inplace=True)
df.drop(['score'], axis=1, inplace=True)
tprint(f'Saving prediction to {prediction_out_path}...')
prediction_out_path.parent.mkdir(exist_ok=True)
assert len(df) == len(test_set)  # Check if all rows are present
df.to_csv(prediction_out_path, index=False)

tprint('Done')

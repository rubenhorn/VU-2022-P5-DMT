#! /usr/bin/env python3

import accelerate

from joblib import load
import pandas as pd
from utils import *
from pathlib import Path

reset_timer()

dataset_name = 'test_set_VU_DM'
model_in_path = (Path(__file__).parent / 'models' /
                 'training_set_VU_DM-pipeline.joblib').resolve()
prediction_out_path = (Path(__file__).parent / 'output' /
                       f'{dataset_name}-prediction.csv').resolve()

def compute_search_result_scores(search_results, model, batch_size=1000):
    ps_b = []
    ps_c = []
    n_search_results = len(search_results)
    n_batches = int(np.ceil(n_search_results / batch_size))
    start_time = time.time()
    for batch_idx in range(n_batches):
        remaining_time = (time.time() - start_time) * (n_batches - batch_idx) / (batch_idx + 1)
        tprint(f'Predicting batch {batch_idx + 1}/{n_batches} (Remaining: { format_time(remaining_time) })...', end='\r')
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, n_search_results)
        batch = search_results.iloc[start_idx:end_idx]
        y_probas = model.predict_proba(batch)
        ps_b.append(y_probas[0])
        ps_c.append(y_probas[1])
    for i in range(len(ps_b)):
        p_b = ps_b[i][1]
        p_c = ps_c[i][1]
        score = combine_booking_click_value(p_b, p_c)
        yield (search_results.iloc[i]['srch_id'], search_results.iloc[i]['prop_id'], score)

test_set = load_dataset(dataset_name)
tprint(f'Loading model from {model_in_path}...')
model = load(model_in_path)
tprint('Generating prediction...')
scored_results = compute_search_result_scores(test_set, model)
df_scored_results = pd.DataFrame(scored_results, columns=['srch_id', 'prop_id', 'score'])
tprint(f'Group data by search...')
grouped_scored_results = df_scored_results.groupby('srch_id')
df = pd.DataFrame(columns=['srch_id', 'prop_id'])
for search_id, group in grouped_scored_results:
    group.sort_values('score', ascending=False, inplace=True)
    group.reset_index(drop=True, inplace=True)
    group.drop(['score'], axis=1, inplace=True)
    df = pd.concat([df, group], ignore_index=True, axis=0, sort=False)
print()

tprint(f'Saving prediction to {prediction_out_path}...')
df.to_csv(prediction_out_path, index=False)

tprint('Done')

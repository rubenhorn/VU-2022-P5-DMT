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

test_set = load_dataset(dataset_name)
tprint(f'Loading model from {model_in_path}...')
model = load(model_in_path)
tprint('Generating prediction...')
scored_results = compute_search_result_scores(test_set, model, n_jobs=-1)
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

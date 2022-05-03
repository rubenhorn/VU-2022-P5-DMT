#! /usr/bin/env python3

from joblib import load
import pandas as pd
from utils import *
from pathlib import Path

reset_timer()

dataset_name = 'test_set_VU_DM'
model_in_path = (Path(__file__).parent / 'models' / 'training_set_VU_DM-pipeline.joblib').resolve()
prediction_out_path = (Path(__file__).parent / 'output' / f'{dataset_name}-prediction.csv').resolve()

tprint(f'Loading model from {model_in_path}...')
model = load(model_in_path)
test_set = load_dataset(dataset_name)

tprint('Generating prediction...')
df = pd.DataFrame(columns=['srch_id', 'prop_id'])
count = len(test_set)
for search_id, group in test_set.groupby('srch_id'):
    print(f'(Remaining: {count}) ', end='\r')
    prop_ids_and_scores = compute_search_result_scores(group, model)
    sorted_prop_ids_and_scores = sorted(prop_ids_and_scores, key=lambda x: x[1], reverse=True)
    sorted_prop_ids = [prop_id for prop_id, _ in sorted_prop_ids_and_scores]
    search_ids = [search_id] * len(sorted_prop_ids)
    df2 = pd.DataFrame({'srch_id': search_ids, 'prop_id': sorted_prop_ids})
    df = pd.concat([df, df2], ignore_index = True, axis = 0)
    count -= len(group)

tprint(f'Saving prediction to {prediction_out_path}...')
df.to_csv(prediction_out_path, index=False)

tprint('Done')

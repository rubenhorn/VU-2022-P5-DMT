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

tprint(f'Loading model from {model_in_path}...')
with accelerate.on_gpu():
    model = load(model_in_path)
    test_set = load_dataset(dataset_name)
    tprint(f'Group data by search...')
    grouped_dataset = test_set.groupby('srch_id')
    tprint('Generating prediction...')
    df = pd.DataFrame(columns=['srch_id', 'prop_id'])
    remaining_count = len(test_set)
    group_number = 0
    start_time = time.time()
    for search_id, group in grouped_dataset:
        # TODO explore alternative of parallelizing outer loop
        prop_ids_and_scores = compute_search_result_scores(group, model, n_jobs=-1)
        sorted_prop_ids_and_scores = sorted(
            prop_ids_and_scores, key=lambda x: x[1], reverse=True)
        sorted_prop_ids = [prop_id for prop_id, _ in sorted_prop_ids_and_scores]
        search_ids = [search_id] * len(sorted_prop_ids)
        df2 = pd.DataFrame({'srch_id': search_ids, 'prop_id': sorted_prop_ids})
        df = pd.concat([df, df2], ignore_index=True, axis=0)
        remaining_count -= len(group)
        group_number += 1
        remaining_time = ((time.time() - start_time) * remaining_count) / (len(test_set) - remaining_count)
        print(f'\r(Group: {group_number}/{len(grouped_dataset)}', end='', flush=False)
        print(f', Instances left: {remaining_count}', end='', flush=False)
        print(f', Time remaining: { format_time(remaining_time) })', end='', flush=True)
print()

tprint(f'Saving prediction to {prediction_out_path}...')
df.to_csv(prediction_out_path, index=False)

tprint('Done')

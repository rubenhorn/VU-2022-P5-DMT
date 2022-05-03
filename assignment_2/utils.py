from pathlib import Path
import time
from joblib import parallel_backend
import numpy as np

import pandas as pd

np.random.seed(0)
_start_time = time.time()


def reset_timer():
    global _start_time
    _start_time = time.time()

def format_time(seconds):
    hours = int(seconds / 3600)
    minutes = int((seconds - hours * 3600) / 60)
    seconds = int(seconds - hours * 3600 - minutes * 60)
    return f'{hours:02}h {minutes:02}m {seconds:02}s'

def print_elapsed_time(prefix='', suffix=': '):
    global _start_time
    elapsed_time = time.time() - _start_time
    print(f'{prefix}{format_time(elapsed_time)}{suffix}', end='')


def tprint(s, end='\n', flush=False):
    print_elapsed_time()
    print(s, end=end, flush=flush)


def load_dataset(dataset_name):
    in_path = (Path(__file__).parent / 'dataset' /
               f'{dataset_name}.csv').resolve()
    tprint(f'Loading dataset from {in_path}...')
    df = pd.read_csv(in_path)
    return df


def combine_booking_click_value(booking_value, click_value):
    w_booked = 5
    w_clicked = 1
    w_combined = w_booked + w_clicked
    return (booking_value * w_booked + click_value * w_clicked) / w_combined


def compute_search_result_scores(search_results, model, n_jobs=1):
    with parallel_backend('loky', n_jobs=n_jobs):
        y_probas = model.predict_proba(search_results)
    for i in range(len(y_probas[0])):
        p_b = y_probas[0][i][1]
        p_c = y_probas[1][i][1]
        score = combine_booking_click_value(p_b, p_c)
        yield (search_results.iloc[i]['prop_id'], score)

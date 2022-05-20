import os
from pathlib import Path
import sys
import time
import numpy as np

import pandas as pd
from sklearn.metrics import make_scorer

import tensorflow as tf
import tensorflow_ranking as tfr

DOCS_PER_QUERY = 50

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
    df = reduce_df_size(df)
    return df

def reduce_df_size(df):
    df = df.copy()
    for col in df.columns:
        if str(df[col].dtype).startswith('int'):
            if col.endswith('_id'):
                df[col] = df[col].astype(np.int32)
            elif col == 'srch_booking_window':
                df[col] = df[col].astype(np.int16)
            else:
                df[col] = df[col].astype(np.int8)
        elif str(df[col].dtype).startswith('float'):
            df[col] = df[col].astype(np.float32)
    return df


def combine_booking_click_value(booking_value, click_value):
    w_booked = 5
    w_clicked = 1
    w_combined = w_booked + w_clicked
    return (booking_value * w_booked + click_value * w_clicked) / w_combined


def ndcg_score_multivalue_booking_click(y_true, y_pred):
    values_true = combine_booking_click_value(y_true[:, 0], y_true[:, 1])
    values_pred = combine_booking_click_value(y_pred[:, 0], y_pred[:, 1])
    return ndcg_score(values_true, values_pred)

def use_full_dataset(show_warning=True):
    env_varname = 'USE_FULL_DATASET'
    use_full_dataset = env_varname in os.environ and os.environ[env_varname] == '1'
    if show_warning and not use_full_dataset:
        print('-' * 80)
        print('WARNING: Using small dataset', file=sys.stderr)
        print('Do not use for submission!', file=sys.stderr)
        print(f'(Use environment variable {env_varname}=1 to use full dataset)', file=sys.stderr)
        print('-' * 80)
    return use_full_dataset

# custom scorer for sci-kit learn using NDCG
def ndcg_score(y_true, y_pred):
    y_true = tf.expand_dims(y_true, 1)
    y_pred = tf.expand_dims(y_pred, 1)
    ndcg = tfr.keras.metrics.NDCGMetric()
    return ndcg(y_true, y_pred).numpy()
    
ndcg_sorer = make_scorer(ndcg_score, greater_is_better=True)

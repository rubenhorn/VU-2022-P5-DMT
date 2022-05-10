from pathlib import Path
import time
import numpy as np

import pandas as pd

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
    return df


def combine_booking_click_value(booking_value, click_value):
    w_booked = 5
    w_clicked = 1
    w_combined = w_booked + w_clicked
    return (booking_value * w_booked + click_value * w_clicked) / w_combined


def prediction_cost(y_true, y_pred):
    values_true = combine_booking_click_value(y_true[:, 0], y_true[:, 1])
    values_pred = combine_booking_click_value(y_pred[:, 0], y_pred[:, 1])
    residuals = values_true - values_pred
    return np.mean(np.abs(residuals))  # MAE

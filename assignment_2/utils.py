from pathlib import Path
import time

import pandas as pd

_start_time = time.time()

def reset_timer():
    global _start_time
    _start_time = time.time()

def print_elapsed_time(prefix='', suffix=': '):
    global _start_time
    elapsed_time = time.time() - _start_time
    elapsed_hours = int(elapsed_time / 3600)
    elapsed_minutes = int((elapsed_time - elapsed_hours * 3600) / 60)
    elapsed_seconds = int(elapsed_time - elapsed_hours * 3600 - elapsed_minutes * 60)
    formatted_time = f'{elapsed_hours:02}h {elapsed_minutes:02}m {elapsed_seconds:02}s'
    print(f'{prefix}{formatted_time}{suffix}', end='')

def tprint(s, end='\n'):
    print_elapsed_time()
    print(s, end=end)

def load_dataset(dataset_name):
    in_path = (Path(__file__).parent / 'dataset' / f'{dataset_name}.csv').resolve()
    tprint(f'Loading dataset from {in_path}...')
    df = pd.read_csv(in_path)
    return df

def compute_search_result_scores(search_results, model):
    y_probas = model.predict_proba(search_results)
    w_booked = 5
    w_clicked = 1
    w_combined = w_booked + w_clicked
    for i in range(len(y_probas[0])):
        p_b = y_probas[0][i][1]
        p_c = y_probas[1][i][1]
        score = (p_b * w_booked + p_c * w_clicked) / w_combined
        yield (search_results.iloc[i]['prop_id'], score)

class Preprocessing:
    # TODO choice of columns only for testing
    X_attrs = ['site_id', 'visitor_location_country_id', 'visitor_hist_starrating', 'visitor_hist_adr_usd', 'prop_country_id']

    def fit(self, X, y=None):
        return self
    def transform(self, X):
        # TODO implement actual preprocessing
        X = X[self.X_attrs]
        X = X.fillna(0)
        return X

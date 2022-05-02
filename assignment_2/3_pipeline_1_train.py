#! /usr/bin/env python3

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score
from sklearn.multioutput import MultiOutputClassifier
from utils import *
from pathlib import Path
from sklearn.pipeline import Pipeline

use_small_dataset = True
n_estimators=100
random_state=42

reset_timer()

dataset_name = 'training_set_VU_DM'
train_set_name = dataset_name + '-train'
test_set_name = dataset_name + '-test'
if use_small_dataset:
    train_set_name += '-small'
    test_set_name += '-small'

# TODO choice of columns only for testing
X_attrs = ['site_id', 'visitor_location_country_id', 'visitor_hist_starrating', 'visitor_hist_adr_usd', 'prop_country_id']
y_attrs = ['booking_bool', 'click_bool']

def load_dataset(dataset_name):
    in_path = (Path(__file__).parent / 'dataset' / f'{dataset_name}.csv').resolve()
    tprint(f'Loading dataset from {in_path}...')
    df = pd.read_csv(in_path)
    return df

train_set = load_dataset(train_set_name)
X_train = train_set[X_attrs]
y_train = train_set[y_attrs]

class Preprocessing:
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        # TODO implement actual preprocessing
        X = X.fillna(0)
        return X

clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
pipeline = Pipeline([
    ('preprocessing', Preprocessing()),
    ('classifier', MultiOutputClassifier(clf))
])

tprint('Fitting pipeline...')
pipeline.fit(X_train, y_train)

tprint('Evaluating pipeline...')
test_set = load_dataset(test_set_name)
X_test = test_set[X_attrs]
y_test = test_set[y_attrs]
recall = recall_score(y_test, pipeline.predict(X_test), average=None)
tprint(f'Booking recall: {recall[0]}')
tprint(f'Click recall: {recall[1]}')

tprint('Freezing pipeline...')
# TODO

tprint('Done')

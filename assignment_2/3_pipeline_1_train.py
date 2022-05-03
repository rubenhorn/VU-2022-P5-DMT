#! /usr/bin/env python3

import json
import sys
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, recall_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.multioutput import MultiOutputClassifier
from utils import *
from pathlib import Path
from sklearn.pipeline import Pipeline
from preprocessing import Preprocessing
import hyperparameters as hp

reset_timer()

model_out_path = (Path(__file__).parent / 'models').resolve()
model_out_path.mkdir(exist_ok=True)

dataset_name = 'training_set_VU_DM'
train_set_name = dataset_name + '-train'
test_set_name = dataset_name + '-test'
if hp.use_small_dataset:
    train_set_name += '-small'
    test_set_name += '-small'
    print('-' * 80)
    print('WARNING: Using small dataset', file=sys.stderr)
    print('Do not use for submission!', file=sys.stderr)
    print('-' * 80)

y_attrs = ['booking_bool', 'click_bool']

train_set = load_dataset(train_set_name)
X_train = train_set
y_train = train_set[y_attrs]
test_set = load_dataset(test_set_name)
X_test = test_set
y_test = test_set[y_attrs]

tprint('Creating pipeline...')
pipeline = Pipeline([
    ('preprocessing', Preprocessing()),
    ('classifier', MultiOutputClassifier(RandomForestClassifier()))
])

tprint('Optimizing hyperparameters...')
def combined_recall_score(y_true, y_pred):
    [recall_booking, recall_click] = recall_score(y_true, y_pred, average=None)
    return combine_booking_click_value(recall_booking, recall_click)

random_search = RandomizedSearchCV(
    pipeline,
    hp.param_grid,
    cv=hp.cv,
    n_iter=hp.n_iter,
    n_jobs=-1,
    verbose=1,
    random_state=hp.random_state,
    scoring=make_scorer(combined_recall_score),
    refit=True,
)   
random_search.fit(X_train, y_train)
best_hyperparams = random_search.best_params_
print('Best parameters:', best_hyperparams)
pipeline = random_search.best_estimator_

tprint('Evaluating optimized pipeline...')
[recall_booking, recall_click] = recall_score(y_test, pipeline.predict(X_test), average=None)
tprint(f'Booking recall: {recall_booking}')
tprint(f'Click recall: {recall_click}')

tprint('Freezing pipeline...')
dump(pipeline, model_out_path / f'{dataset_name}-pipeline.joblib')
with open(model_out_path / f'{dataset_name}-hyperparams.json', 'w') as f:
    json.dump(best_hyperparams, f, indent=4)

tprint('Done')

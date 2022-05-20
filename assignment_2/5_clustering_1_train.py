#! /usr/bin/env python3


import accelerate

from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeRegressor

import json
from joblib import dump
from sklearn.metrics import make_scorer, recall_score
from sklearn.model_selection import RandomizedSearchCV
from utils import *
from pathlib import Path
from sklearn.pipeline import Pipeline
from preprocessing import Preprocessing
import hyperparameters as hp
from utils import *

reset_timer()

model_out_path = (Path(__file__).parent / 'models').resolve()

dataset_name = 'training_set_VU_DM'
train_set_name = dataset_name + '-train'
test_set_name = dataset_name + '-test'
if not use_full_dataset():
    train_set_name += '-small'
    test_set_name += '-small'

y_attrs = ['booking_bool', 'click_bool']

y_to_scalar = lambda y: combine_booking_click_value(y[:, 0], y[:, 1])

train_set = load_dataset(train_set_name)
X_train = train_set
y_train = y_to_scalar(train_set[y_attrs].values)
test_set = load_dataset(test_set_name)
X_test = test_set
y_test = y_to_scalar(test_set[y_attrs].values)

tprint('Creating pipeline...')
pipeline = Pipeline([
    ('preprocessing', Preprocessing()),
    ('clustering', KMeans()),
    ('regression', DecisionTreeRegressor())
])
tprint('Optimizing hyperparameters...')
random_search = RandomizedSearchCV(
    pipeline,
    hp.param_grid_clustering,
    cv=hp.cv,
    n_iter=hp.n_iter,
    n_jobs=1, # Do not parallelize to avoid out-of-memory errors
    verbose=1,
    random_state=hp.random_state,
    refit=True,
    scoring=ndcg_sorer
)
random_search.fit(X_train, y_train)

best_hyperparams = random_search.best_params_
print('Best parameters:', best_hyperparams)
print('Best NDCG score:', random_search.best_score_)
pipeline = random_search.best_estimator_
tprint('Evaluating optimized pipeline...')
tprint(f'NDCG Score: { ndcg_score(y_test, pipeline.predict(X_test)) }')

tprint('Freezing pipeline...')
model_out_path.mkdir(exist_ok=True)
dump(pipeline, model_out_path / f'{dataset_name}-clustering.joblib')
with open(model_out_path / f'{dataset_name}-clustering-hyperparams.json', 'w') as f:
    json.dump(best_hyperparams, f, indent=4)

tprint('Done')

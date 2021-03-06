#! /usr/bin/env python3

from sklearn.ensemble import RandomForestClassifier
import accelerate

from sklearn.linear_model import SGDClassifier
from sklearn.decomposition import PCA
import json
from joblib import dump
from sklearn.metrics import make_scorer, recall_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.multioutput import MultiOutputClassifier
from utils import *
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.kernel_approximation import RBFSampler
from preprocessing import Preprocessing
import hyperparameters as hp

reset_timer()

model_out_path = (Path(__file__).parent / 'models').resolve()

dataset_name = 'training_set_VU_DM'
train_set_name = dataset_name + '-train'
test_set_name = dataset_name + '-test'
if not use_full_dataset():
    train_set_name += '-small'
    test_set_name += '-small'

y_attrs = ['booking_bool', 'click_bool']

train_set = load_dataset(train_set_name)
X_train = train_set
y_train = train_set[y_attrs].values.astype(np.float32)

tprint('Creating pipeline...')
clf = SGDClassifier()
pipeline = Pipeline([
    ('preprocessing', Preprocessing()),
    # ('pca', PCA()),
    # ('rbf', RBFSampler()), # Kernel trick (For non-linearly separable data)
    # ('classifier', MultiOutputClassifier(clf))
    ('classifier', RandomForestClassifier())
])
tprint('Optimizing hyperparameters...')
random_search = RandomizedSearchCV(
    pipeline,
    hp.param_grid_pipeline,
    cv=hp.cv,
    n_iter=hp.n_iter,
    n_jobs=1, # Do not parallelize to avoid out-of-memory errors
    verbose=1,
    random_state=hp.random_state,
    scoring=make_scorer(ndcg_score_multivalue_booking_click, greater_is_better=True),
    refit=True,
)
random_search.fit(X_train, y_train)

tprint('Deleting training set...')
del X_train
del y_train
del train_set

best_hyperparams = random_search.best_params_
print('Best parameters:', best_hyperparams)
print('Best NDCG score:', random_search.best_score_)
pipeline = random_search.best_estimator_
tprint('Evaluating optimized pipeline...')


test_set = load_dataset(test_set_name)
X_test = test_set
y_test = test_set[y_attrs].values.astype(np.float32)

y_pred = pipeline.predict(X_test)
[recall_booking, recall_click] = recall_score(
    y_test, y_pred, average=None)
tprint(f'Booking recall: {recall_booking}')
tprint(f'Click recall: {recall_click}')
tprint(f'NDCG Score: { ndcg_score_multivalue_booking_click(y_test, y_pred) }')

tprint('Freezing pipeline...')
model_out_path.mkdir(exist_ok=True)
dump(pipeline, model_out_path / f'{dataset_name}-pipeline.joblib')
with open(model_out_path / f'{dataset_name}-pipeline-hyperparams.json', 'w') as f:
    json.dump(best_hyperparams, f, indent=4)

tprint('Done')

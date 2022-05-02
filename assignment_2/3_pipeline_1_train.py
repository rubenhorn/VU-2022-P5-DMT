#! /usr/bin/env python3

from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score
from sklearn.multioutput import MultiOutputClassifier
from utils import *
from pathlib import Path
from sklearn.pipeline import Pipeline

# TODO set to False for actual training
use_small_dataset = True
n_estimators=100
random_state=42

reset_timer()

model_out_path = (Path(__file__).parent / 'models').resolve()
model_out_path.mkdir(exist_ok=True)

dataset_name = 'training_set_VU_DM'
train_set_name = dataset_name + '-train'
test_set_name = dataset_name + '-test'
if use_small_dataset:
    train_set_name += '-small'
    test_set_name += '-small'

y_attrs = ['booking_bool', 'click_bool']

train_set = load_dataset(train_set_name)
X_train = train_set
y_train = train_set[y_attrs]

clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
pipeline = Pipeline([
    ('preprocessing', Preprocessing()),
    ('classifier', MultiOutputClassifier(clf))
])

tprint('Fitting pipeline...')
pipeline.fit(X_train, y_train)

tprint('Evaluating pipeline...')
test_set = load_dataset(test_set_name)
X_test = test_set
y_test = test_set[y_attrs]
recall = recall_score(y_test, pipeline.predict(X_test), average=None)
tprint(f'Booking recall: {recall[0]}')
tprint(f'Click recall: {recall[1]}')

tprint('Freezing pipeline...')
dump(pipeline, model_out_path / f'{dataset_name}-pipeline.joblib')

tprint('Done')

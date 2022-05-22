#! /usr/bin/env python3

from joblib import dump
import pyltr
from utils import *
from preprocessing import Preprocessing
import hyperparameters as hp

reset_timer()

metric = pyltr.metrics.NDCG(k=10)

dataset_name = 'training_set_VU_DM'
model_out_path = (Path(__file__).parent / 'models').resolve()

train_set_name = dataset_name + '-train'
test_set_name = dataset_name + '-test'
if not use_full_dataset():
    train_set_name += '-small'
    test_set_name += '-small'

y_attrs = ['booking_bool', 'click_bool']

y_to_scalar = lambda y: combine_booking_click_value(y[:, 0], y[:, 1])

pp = Preprocessing()

train_set = load_dataset(train_set_name)
# Samples must be grouped by qid.
train_set = train_set.sort_values(by=['srch_id'])
X_train = pp.transform(train_set)
y_train = y_to_scalar(train_set[y_attrs].values).astype(np.float32)
sids_train = train_set['srch_id'].values

train_set = load_dataset(train_set_name)
# Samples must be grouped by qid.
test_set = train_set.sort_values(by=['srch_id'])
X_test = pp.transform(test_set)
y_test = y_to_scalar(test_set[y_attrs].values).astype(np.float32)
sids_test = test_set['srch_id'].values

monitor = pyltr.models.monitors.ValidationMonitor(X_test, y_test, sids_test, metric=metric, stop_after=hp.lm_stop_after)

model = pyltr.models.LambdaMART(
    metric=metric,
    n_estimators=hp.lm_n_estimators,
    learning_rate=hp.lm_learning_rate,
    max_features=hp.lm_max_features,
    query_subsample=hp.lm_query_subsample,
    max_leaf_nodes=hp.lm_max_leaf_nodes,
    min_samples_leaf=hp.lm_min_samples_leaf,
    verbose=1,
)

tprint('Training LambdaMART model...')
model.fit(X_train, y_train, sids_train, monitor=monitor)

tprint('Freezing pipeline...')
model_out_path.mkdir(exist_ok=True)
dump(model, model_out_path / f'{dataset_name}-lambdamart.joblib')

tprint('Done')

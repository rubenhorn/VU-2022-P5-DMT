#! /usr/bin/env python3

import tensorflow as tf

import sys
from utils import *
from pathlib import Path
from preprocessing import Preprocessing
import hyperparameters as hp

reset_timer()

dataset_name = 'training_set_VU_DM'
model_out_path = (Path(__file__).parent / 'models' /
                  f'{dataset_name}-ListNet').resolve()

train_set_name = dataset_name + '-train'
test_set_name = dataset_name + '-test'
if not use_full_dataset():
    train_set_name += '-small'
    test_set_name += '-small'


def create_batches(data):
    X_batches = []
    y_batches = []
    for _, group in data.groupby('srch_id'):
        group = group.sample(DOCS_PER_QUERY, replace=True)  # Pad input
        assert len(group) == DOCS_PER_QUERY
        X = pp.transform(group)
        y = group.apply(
            lambda row: row['booking_bool'] * 5 + row['click_bool'] * 1, axis=1)
        X_batches.append(X)
        y_batches.append(y)
    return X_batches, y_batches


pp = Preprocessing()
train_set = load_dataset(train_set_name)
test_set = load_dataset(test_set_name)

tprint('Create batches by query...')
X_train_batches, y_train_batches = create_batches(train_set)
X_test_batches, y_test_batches = create_batches(test_set)


def create_model(docs_per_query, embedding_dims):
    # TODO refine network architecture
    docs_input = tf.keras.layers.Input(
        shape=(docs_per_query, embedding_dims, ), dtype=tf.float32, name='docs')
    dense_1 = tf.keras.layers.Dense(
        units=3, activation='linear', name='dense_1')
    dense_out = tf.keras.layers.Dense(
        units=1, activation='linear', name='scores')
    scores_prob_dist = tf.keras.layers.Dense(
        units=docs_per_query, activation='softmax', name='scores_prob_dist')
    dense_1_out = dense_1(docs_input)
    scores = tf.keras.layers.Flatten()(dense_out(dense_1_out))
    model_out = scores_prob_dist(scores)
    model = tf.keras.models.Model(inputs=[docs_input], outputs=[model_out])
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=hp.learning_rate, momentum=hp.momentum)
    loss = tf.keras.losses.KLDivergence()
    model.compile(optimizer=optimizer, loss=loss)
    return model


tprint('Creating ListNet...')
embedding_dims = X_train_batches[0].shape[1]
model = create_model(DOCS_PER_QUERY, embedding_dims)
model.summary()

tprint('Fitting model...')
X_train_batches_array = np.array(X_train_batches)
y_train_batches_tensor = tf.constant(y_train_batches, dtype=tf.float32)
relevance_grades_prob_dist = tf.nn.softmax(y_train_batches_tensor, axis=-1)
hist = model.fit(
    [X_train_batches_array],
    relevance_grades_prob_dist,
    epochs=hp.tf_epochs,
    verbose=True,
)

tprint('Evaluating model...')
X_test_batches_array = np.array(X_test_batches)
y_test_batches_tensor = tf.constant(y_test_batches, dtype=tf.float32)
relevance_grades_prob_dist = tf.nn.softmax(y_test_batches_tensor, axis=-1)
loss = model.evaluate(
    [X_test_batches_array],
    relevance_grades_prob_dist,
    verbose=False
)
print('Test loss:', loss)

tprint('Freezing trained model...')
model_out_path.mkdir(exist_ok=True, parents=True)
model.save(str(model_out_path.absolute()))

tprint('Done')

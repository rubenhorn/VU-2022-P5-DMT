#! /usr/bin/env python3

import tensorflow as tf
import tensorflow_ranking as tfr

import sys
from utils import *
from pathlib import Path
from preprocessing import Preprocessing
import hyperparameters as hp
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import *

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
        group = pad_group(group)  # Pad input
        assert len(group) == DOCS_PER_QUERY
        X = pp.transform(group)
        y = group.apply(
            lambda row: (row['booking_bool'] * 5 + row['click_bool']) * 1 / 6, axis=1)
        X_batches.append(X)
        y_batches.append(y)
    return X_batches, y_batches


pp = Preprocessing()
train_set = load_dataset(train_set_name)

tprint('Create batches by query...')
X_train_batches, y_train_batches = create_batches(train_set)


def create_model(docs_per_query, embedding_dims):
    # Input layer
    docs_input = Input(
        shape=(docs_per_query, embedding_dims, ), dtype=tf.float32, name='docs')
    # Compute relevance scores
    hidden_1_booked = Dense(units=24, name='hidden_1_booked', activation='leaky_relu')
    hidden_1_clicked = Dense(units=24, name='hidden_1_clicked', activation='leaky_relu')
    dense_out_booked = Dense(units=1, name='scores_booked', activation='softmax')(hidden_1_booked(docs_input))
    dense_out_clicked = Dense(units=1, name='scores_clicked', activation='softmax')(hidden_1_clicked(docs_input))
    # Output layer (weighted sum of hidden output layers)
    dense_out_booked_weighted = Lambda(lambda x: x * 5, name='scores_booked_weighted')(dense_out_booked)
    dense_out = Add(name='combine_booked_clicked')([dense_out_booked_weighted, dense_out_clicked])
    dense_out_normalized = Lambda(lambda x: x / 6, name='normalize_combined')(dense_out)
    # Output (probability of relevance in query)
    scores = Flatten()(dense_out_normalized)
    scores_prob_dist = Dense(
        units=docs_per_query, activation='softmax', name='scores_prob_dist')
    model_out = scores_prob_dist(scores)
    # Build model
    model = tf.keras.models.Model(inputs=[docs_input], outputs=[model_out])
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=hp.lr, momentum=hp.momentum)
    # loss = tf.keras.losses.KLDivergence() # Suggested by original article
    loss = tfr.keras.losses.ApproxNDCGLoss()
    model.compile(optimizer=optimizer, loss=loss, metrics=[tfr.keras.metrics.NDCGMetric()])
    return model


tprint('Creating ListNet...')
embedding_dims = X_train_batches[0].shape[1]
model = create_model(DOCS_PER_QUERY, embedding_dims)
model.summary()

tprint('Fitting model...')
X_train_batches_array = np.array(X_train_batches)
y_train_batches_tensor = tf.constant(y_train_batches, dtype=tf.float32)
relevance_grades_prob_dist = tf.nn.softmax(y_train_batches_tensor, axis=-1)
model.fit(
    [X_train_batches_array],
    relevance_grades_prob_dist,
    epochs=hp.tf_epochs,
    verbose=True,
    callbacks=[
        ReduceLROnPlateau(monitor='loss', factor=hp.reduce_lr_factor, patience=hp.reduce_lr_patience, min_lr=hp.min_lr),
        # EarlyStopping(monitor='loss', patience=hp.early_stopping_patience, min_delta=hp.early_stopping_min_delta)
    ]
)

tprint('Delete training data to save memory...')
del X_train_batches_array
del y_train_batches_tensor
del relevance_grades_prob_dist
del X_train_batches
del y_train_batches
del train_set

test_set = load_dataset(test_set_name)

tprint('Create batches by query...')
X_test_batches, y_test_batches = create_batches(test_set)

tprint('Evaluating model...')
X_test_batches_array = np.array(X_test_batches)
y_test_batches_tensor = tf.constant(y_test_batches, dtype=tf.float32)
relevance_grades_prob_dist = tf.nn.softmax(y_test_batches_tensor, axis=-1)
scores = model.evaluate(
    [X_test_batches_array],
    relevance_grades_prob_dist,
    verbose=False
)
print('Test NDCG:', scores[1])

tprint('Freezing trained model...')
model_out_path.mkdir(exist_ok=True, parents=True)
model.save(str(model_out_path.absolute()))

tprint('Done')

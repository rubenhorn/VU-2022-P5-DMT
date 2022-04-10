from hyperparameters import *
import pandas as pd
from pathlib import Path
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC
from sklearn.utils import shuffle
from time import time

# Load pre-processed dataset
input_dataset_filename = (Path(__file__).parent / 'SmsCollection-pre-processed.csv').resolve()
df = pd.read_csv(input_dataset_filename)
X, y = shuffle(df['message'], df['is_spam'], random_state=random_state)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create vocabulary from training set
vocabulary = set()
for document in X_train:
    vocabulary.update(document.split(' '))

# Create pipeline to extract features
pipeline = Pipeline([
    ('vectorizer', CountVectorizer(vocabulary=vocabulary)),
    ('tfidf', TfidfTransformer()),
    # Cannot use PCA on sparse matrix
    ('svd', TruncatedSVD(n_components=svd_n_components, n_iter=svd_n_iter, random_state=random_state)),
    # Negative values do not work with all classifiers
    ('normalizer', MinMaxScaler([0, 1])),
])
pipeline = pipeline.fit(X_train, y_train)
X_train_vec = pipeline.transform(X_train)
X_test_vec = pipeline.transform(X_test)

# Evaluate multiple classifiers
df = pd.DataFrame(columns=['model', 'accuracy train', 'accuracy test', 'time fit', 'time inf'])
clfs = [
    LinearSVC(random_state=random_state),
    MultinomialNB(alpha=alpha),
    KNeighborsClassifier(n_neighbors=n_neighbors),
]

for clf in clfs:
    t0 = time()
    clf.fit(pipeline.transform(X_train), y_train)
    t_train = time() - t0
    t0 = time()
    accuracy_train = clf.score(X_train_vec, y_train)
    accuracy_test = clf.score(X_test_vec, y_test)
    t_inf = time() - t0
    df.loc[len(df)] = [type(clf).__name__, accuracy_train, accuracy_test, t_train, t_inf]
df.sort_values(by='accuracy test', ascending=False, inplace=True)

print(df.head().to_string(index=False))

print('\nDone')
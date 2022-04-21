import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from time import time

input_train_dataset_filename = (Path(__file__).parent / 'train-pre-processed.csv').resolve()
input_test_dataset_filename = (Path(__file__).parent / 'test-pre-processed.csv').resolve()
output_dataset_filename = (Path(__file__).parent / 'submission.csv').resolve()

train_df = pd.read_csv(input_train_dataset_filename, delimiter=',')
test_df = pd.read_csv(input_test_dataset_filename, delimiter=',')

X_train, X_test, Y_train, Y_test = train_test_split(train_df.drop("Survived", axis=1), train_df["Survived"], test_size=0.2)

# Evaluate multiple classifiers
n_neighbors = 3
n_estimators = 100
random_state = 42

classifier_df = pd.DataFrame(columns=['model', 'accuracy train', 'accuracy test', 'time fit', 'time inf'])
clfs = [
    LogisticRegression(),
    SVC(),
    GaussianNB(),
    Perceptron(),
    LinearSVC(random_state=random_state),
    SGDClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(n_estimators=n_estimators),
    KNeighborsClassifier(n_neighbors=n_neighbors),
]

for clf in clfs:
    t0 = time()
    clf.fit(X_train, Y_train)
    t_train = time() - t0
    t0 = time()
    accuracy_train = clf.score(X_train, Y_train)
    accuracy_test = clf.score(X_test, Y_test)
    t_inf = time() - t0
    classifier_df.loc[len(classifier_df)] = [type(clf).__name__, accuracy_train, accuracy_test, t_train, t_inf]
classifier_df.sort_values(by='accuracy test', ascending=False, inplace=True)

print(classifier_df.to_string(index=False))

# Use the top model predict the test dataset
chosen_clf = eval(classifier_df.iloc[0]['model'] + '()')
chosen_clf.fit(X_train, Y_train)
Y_pred = chosen_clf.predict(test_df.drop("PassengerId", axis=1))

# Output the results
submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv(output_dataset_filename, index=False)

print('Done')
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.utils import shuffle

input_dataset_filename = (Path(__file__).parent / 'german-pre-processed.csv').resolve()

random_state = 42
kfold_splits = 10
n_neighbors = 5
n_trees = 10

model_builders = {
    'LinearSVC': lambda: LinearSVC(random_state=random_state, dual=False),
    'KNeighborsClassifier': lambda: KNeighborsClassifier(n_neighbors=n_neighbors),
    'RandomForestClassifier': lambda: RandomForestClassifier(random_state=random_state, n_estimators=n_trees    ),
}

df = pd.read_csv(input_dataset_filename)
X = df.drop('good_customer', axis=1)
# X = df[['checking_account_status', 'credit_history', 'credit_duration', 'savings_status', 'credit_amount']]
y = df['good_customer']
X, y  = shuffle(X, y, random_state=random_state)

def cost(y_true, y_pred):
    costs = [5 if actual == 1 else 1 for expected, actual in zip(y_true, y_pred) if expected != actual]
    return np.array(costs).sum() / len(y_true)
penalty_scorer = make_scorer(cost, greater_is_better=False)

def fit_and_score_model(model, train, test):
    X_train, X_test, y_train, y_test = X.iloc[train], X.iloc[test], y.iloc[train], y.iloc[test]
    model.fit(X_train, y_train)
    return [abs(penalty_scorer(model, X_test, y_test)), model.score(X_test, y_test)]

kf = KFold(n_splits=kfold_splits, random_state=random_state, shuffle=True)
print('Model & Cost mean & Cost std & Accuracy mean & Accuracy std \\\\')
for model_name, model_builder in model_builders.items():
    scores = np.array([fit_and_score_model(model_builder(), train, test) for train, test in kf.split(X, y)])
    # Custom cost function
    print(model_name, '&', scores.mean(axis=0)[0], '&', scores.std(axis=0)[0], end=' ')
    # Accuracy
    print('&', scores.mean(axis=0)[1], '&', scores.std(axis=0)[1], end=' ')
    print('\\\\')

print('Done')

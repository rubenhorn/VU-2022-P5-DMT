import math
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import seaborn as sns
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import shuffle

dataset_filename = (Path(__file__).parent / 'insurance.csv').resolve()

random_state = 42
sgd_max_iter = 1000
tree_max_depth = 5
n_neighbors = 5


def plot_data():
    smokers = df[df['smoker'] == 1]
    non_smokers = df[df['smoker'] == 0]
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.scatter(smokers['age'], smokers['charges'], label='Smokers', color='red')
    ax1.scatter(non_smokers['age'], non_smokers['charges'], label='Smokers', color='blue')
    ax1.set_xlabel('Age')
    ax2.scatter(smokers['bmi'], smokers['charges'], label='Smokers', color='red')
    ax2.scatter(non_smokers['bmi'], non_smokers['charges'], label='Smokers', color='blue')
    ax2.set_xlabel('BMI')
    fig.suptitle('Charges for smokers (red) and non-smokers (blue)')
    plt.show()

def to_one_hot(df, column):
    df = df.copy()
    classes = df[column].unique()
    for cls in classes:
        df[f'{column}_{cls}'] = (df[column] == cls).astype(int)
    df.drop(column, axis=1, inplace=True)
    return df

df = pd.read_csv(dataset_filename)
df['sex'] = df['sex'].apply(lambda x: 1 if x == 'female' else 0)
df['smoker'] = df['smoker'].apply(lambda x: 1 if x == 'yes' else 0)

# Visualize data
# sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
# plt.show()
# plot_data()

df = to_one_hot(df, 'region')
#df = df[df['smoker'] == 1]
#df = df.drop('smoker', axis=1)

df = shuffle(df, random_state=random_state)

X = df[df.columns[(df.columns != 'charges')]]
# X = df[['bmi', 'age']]
y = df['charges']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

def eval_regressor(model):
    pipeline = make_pipeline(StandardScaler(), model)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    return mae, mse, y_pred

models = [
    LinearRegression(),
    SGDRegressor(random_state=random_state, max_iter=sgd_max_iter),
    DecisionTreeRegressor(random_state=random_state, max_depth=tree_max_depth),
    KNeighborsRegressor(n_neighbors=n_neighbors),
]

fig, axs = plt.subplots(1, len(models), sharey=True, sharex=True)
print('Model', '&', '\gls{mae}', '&', '\gls{mae}\\textsuperscript{2}', '&', '\gls{mse}', '&', '\gls{rmse}', '\\\\')
for i, model in enumerate(models):
    mae, mse, y_pred = eval_regressor(model)
    print(type(model).__name__, '&', f'{mae:.3f}', '&', f'{(mae**2):.3f}', '&', f'{mse:.3f}', '&', f'{math.sqrt(mse):.3f}' '\\\\')
    axs[i].scatter(y_pred, y_test)
    axs[i].plot(y_test, y_test, 'r-')
    axs[i].set_xlabel('Predicted')
    axs[i].set_ylabel('Actual')
    axs[i].set_title(type(model).__name__)
plt.show()

print('\nDone')

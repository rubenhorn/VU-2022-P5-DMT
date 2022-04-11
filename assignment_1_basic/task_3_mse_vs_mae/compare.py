import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

dataset_filename = (Path(__file__).parent / 'yield_df.csv').resolve()

df = pd.read_csv(dataset_filename, index_col=0)
# Drop average rainfall per region
df.drop('average_rain_fall_mm_per_year', axis=1, inplace=True)

# Merge columns 'Area' and 'Item'
df = df.loc[(df['Area'] == 'Brazil') & (df['Item'] == 'Maize')]
df.drop(['Area', 'Item'], axis=1, inplace=True)
df.columns = ['year', 'yield', 'pesticides', 'rainfall']

# Visualize data
# sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
# plt.show()
# sns.pairplot(df)
# plt.show()

df = shuffle(df, random_state=42)

X = pd.DataFrame(df['pesticides'])
y = df['yield']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#model = make_pipeline(StandardScaler(), SGDRegressor(random_state=42))

class Model:
    def __init__(self, learning_rate=0.1, decay_rate=0.01, n_epochs=1000, batch_size=20, random_state=42):
        self.learning_rate_ = learning_rate
        self.decay_rate_ = decay_rate
        self.n_epochs_ = n_epochs
        self.batch_size_ = batch_size
        self.rng_ = np.random.RandomState(random_state)

    def fit(self, X, y):
        self.b1_ = self.rng_.rand(X.shape[1])
        self.b0_ = self.rng_.rand(1)
        for _ in range(self.n_epochs_):
            self.learning_rate_ *= (1 - self.decay_rate_)
            batch_idxs = self.rng_.randint(X.shape[0], size=self.batch_size_)
            X_batch = X[batch_idxs]
            y_batch = y.values[batch_idxs]
            for x_i, y_i in zip(X_batch, y_batch):
                # TODO use custom error function?
                self.b1_ += self.learning_rate_ * (y_i - x_i.dot(self.b1_) - self.b0_) * x_i
                self.b0_ += self.learning_rate_ * (y_i - x_i.dot(self.b1_) - self.b0_)

    def predict(self, X):
        return [x.dot(self.b1_) + self.b0_ for x in X]

model = make_pipeline(StandardScaler(), Model(random_state=42))

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
MAE = mean_absolute_error(y_test, y_pred)
MSE = mean_squared_error(y_test, y_pred)
print('MAE:', MAE)
print('MSE:', MSE)

plt.scatter(X_train, y_train, color='blue')
plt.plot(X_train.iloc[:, 0], model.predict(X_train), color='red')
#plt.scatter(X_test, y_test, color='blue')
#plt.plot(X_test.iloc[:, 0], y_pred, color='red')
plt.show()

print('Done')

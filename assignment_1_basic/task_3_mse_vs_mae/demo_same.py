from matplotlib import pyplot as plt
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

max_years_experience = 10
base_salary = 30
max_bonus = 20

random_state = 42
n_samples = int(10e3)

X, y = make_regression(n_samples=n_samples, n_features=1, noise=0)
X = X + abs(X.min())
X = X / X.max() * max_years_experience
y = y + abs(y.min())
y = y / y.max() * max_bonus
y += base_salary
# Add gender pay gap
y[int(n_samples / 2):] += 2
is_male = [i >= int(n_samples / 2) for i in range(n_samples)]
X = pd.DataFrame(list(zip(X, is_male)), columns=['experience', 'is_male'])
X, y = shuffle(X, y, random_state=random_state)

# Remove gender from dataset
X = X.drop('is_male', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

# Fit a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print('MAE:', mean_absolute_error(y_test, y_pred))
print('MSE:', mean_squared_error(y_test, y_pred))

plt.scatter(X_test['experience'][:100], y_test[:100])
plt.plot(X_test['experience'], y_pred, color='red')
plt.xlabel('Years of experience')
plt.xlim(0, max_years_experience)
plt.ylim(0, max_bonus + base_salary)
plt.ylabel('Salary')
plt.show()

print('Done')

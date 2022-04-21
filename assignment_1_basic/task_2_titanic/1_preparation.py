import pandas as pd
import scipy.stats as st
from pathlib import Path

input_train_dataset_filename = (Path(__file__).parent / 'train.csv').resolve()
output_train_dataset_filename = (Path(__file__).parent / 'train-pre-processed.csv').resolve()
input_test_dataset_filename = (Path(__file__).parent / 'test.csv').resolve()
output_test_dataset_filename = (Path(__file__).parent / 'test-pre-processed.csv').resolve()

train_df = pd.read_csv(input_train_dataset_filename, delimiter=',')
train_df = train_df.drop_duplicates()

test_df = pd.read_csv(input_test_dataset_filename, delimiter=',')
test_df = test_df.drop_duplicates()

# Source: https://stackoverflow.com/questions/37487830/how-to-find-probability-distribution-and-parameters-for-real-data-python-3
def get_best_distribution(data):
    dist_names = ["norm", "exponweib", "weibull_max", "weibull_min", "pareto", "genextreme"]
    dist_results = []
    params = {}
    for dist_name in dist_names:
        dist = getattr(st, dist_name)
        param = dist.fit(train_df[data])

        params[dist_name] = param
        # Applying the Kolmogorov-Smirnov test
        D, p = st.kstest(train_df[data], dist_name, args=param)
        # print("p value for "+dist_name+" = "+str(p))
        dist_results.append((dist_name, p))

    # select the best fitted distribution
    best_dist, best_p = (max(dist_results, key=lambda item: item[1]))
    # store the name of the best fit and its p value
    print(f"Best fitting distribution for col ({data}): {str(best_dist)}")
    # print("Best p value: "+ str(best_p))
    # print("Parameters for the best fit: "+ str(params[best_dist]))

    return best_dist, best_p, params[best_dist]

# Find empty values
empty_vals = train_df.isnull().sum()

# Replace empty values
train_df['Age'].fillna(train_df['Age'].mean(), inplace=True)
train_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace=True)
test_df['Age'].fillna(test_df['Age'].mean(), inplace=True)
test_df['Embarked'].fillna(test_df['Embarked'].mode()[0], inplace=True)

test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)

# Find correlation between columns
col_corr = train_df.corr()

# ===========================Distibution=============================   ==
# print(train_df[['Age','SibSp','Parch','Fare']].describe())
# print(train_df.describe(include=['O']))
# col_names = ['Survived', 'Pclass', 'Sex', 'Embarked']
# for col in col_names:
#     # print(train_df.groupby(col).count())
#     print(train_df.groupby(["Survived", col]).size())


# col_names = ['Age', 'Fare', 'Parch', 'SibSp'] # Add Age column
# for col in col_names:
#     get_best_distribution(col)

# Drop unecessary columns
train_df = train_df.drop(['PassengerId','Name','Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Name','Ticket', 'Cabin'], axis=1)

# Convert Sex column to binary
train_df['Sex'] = train_df['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
test_df['Sex'] = test_df['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

# Convert Embarked column to ordinal values
train_df['Embarked'] = train_df['Embarked'].map( {'C': 0, 'Q': 1, 'S': 2} ).astype(int)
test_df['Embarked'] = test_df['Embarked'].map( {'C': 0, 'Q': 1, 'S': 2} ).astype(int)


# Cut the age column into 5 categories
# train_df['Age'] = pd.cut(train_df['Age'], 5)
# grouped__age_df = train_df[['Age', 'Survived']].groupby(['Age'], as_index=False).mean().sort_values(by='Age', ascending=True)
# print(grouped__age_df)

# Convert Age column to oridnal values
train_df.loc[ train_df['Age'] < 17, 'Age'] = 0
train_df.loc[(train_df['Age'] >= 17) & (train_df['Age'] < 33), 'Age'] = 1
train_df.loc[(train_df['Age'] >= 33) & (train_df['Age'] < 49), 'Age'] = 2
train_df.loc[(train_df['Age'] >= 49) & (train_df['Age'] < 65), 'Age'] = 3
train_df.loc[ train_df['Age'] >= 65, 'Age'] = 4

test_df.loc[ test_df['Age'] < 17, 'Age'] = 0
test_df.loc[(test_df['Age'] >= 17) & (test_df['Age'] < 33), 'Age'] = 1
test_df.loc[(test_df['Age'] >= 33) & (test_df['Age'] < 49), 'Age'] = 2
test_df.loc[(test_df['Age'] >= 49) & (test_df['Age'] < 65), 'Age'] = 3
test_df.loc[ test_df['Age'] >= 65, 'Age'] = 4


# Cut the Fare column into 3 categories
# train_df['Fare'] = pd.cut(train_df['Fare'], 3)
# grouped_fare_df = train_df[['Fare', 'Survived']].groupby(['Fare'], as_index=False).mean().sort_values(by='Fare', ascending=True)
# print(grouped_fare_df)

# Convert Fare column to oridnal values
train_df.loc[ train_df['Fare'] < 171, 'Fare'] = 0
train_df.loc[(train_df['Fare'] >= 171) & (train_df['Fare'] < 342), 'Fare'] = 1
train_df.loc[ train_df['Fare'] >= 342, 'Fare'] = 2

test_df.loc[ test_df['Fare'] < 171, 'Fare'] = 0
test_df.loc[(test_df['Fare'] >= 171) & (test_df['Fare'] < 342), 'Fare'] = 1
test_df.loc[ test_df['Fare'] >= 342, 'Fare'] = 2


train_df.to_csv(output_train_dataset_filename, index=False)
test_df.to_csv(output_test_dataset_filename, index=False)

print('Done')

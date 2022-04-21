import pandas as pd
import scipy.stats as st
from pathlib import Path

input_dataset_filename = (Path(__file__).parent / 'train.csv').resolve()
output_dataset_filename = (Path(__file__).parent / 'train-pre-processed.csv').resolve()

df = pd.read_csv(input_dataset_filename, delimiter=',')
df = df.drop_duplicates()
# Source: https://stackoverflow.com/questions/37487830/how-to-find-probability-distribution-and-parameters-for-real-data-python-3
def get_best_distribution(data):
    dist_names = ["norm", "exponweib", "weibull_max", "weibull_min", "pareto", "genextreme"]
    dist_results = []
    params = {}
    for dist_name in dist_names:
        dist = getattr(st, dist_name)
        param = dist.fit(df[data])

        params[dist_name] = param
        # Applying the Kolmogorov-Smirnov test
        D, p = st.kstest(df[data], dist_name, args=param)
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
empty_vals = df.isnull().sum()

# Replace empty values
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Find correlation between columns
col_corr = df.corr()

# ===========================Distibution=============================   ==
# print(df[['Age','SibSp','Parch','Fare']].describe())
# print(df.describe(include=['O']))
# col_names = ['Survived', 'Pclass', 'Sex', 'Embarked']
# for col in col_names:
#     # print(df.groupby(col).count())
#     print(df.groupby(["Survived", col]).size())


# col_names = ['Age', 'Fare', 'Parch', 'SibSp'] # Add Age column
# for col in col_names:
#     get_best_distribution(col)

# Drop unecessary columns
df = df.drop(['PassengerId','Name','Ticket', 'Cabin'], axis=1)

# Convert Sex column to binary
df['Sex'] = df['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

# Convert Embarked column to ordinal values
df['Embarked'] = df['Embarked'].map( {'C': 0, 'Q': 1, 'S': 2} ).astype(int)


# Cut the age column into 5 categories
# df['Age'] = pd.cut(df['Age'], 5)
# grouped__age_df = df[['Age', 'Survived']].groupby(['Age'], as_index=False).mean().sort_values(by='Age', ascending=True)
# print(grouped__age_df)

# Convert Age column to oridnal values
df.loc[ df['Age'] < 17, 'Age'] = 0
df.loc[(df['Age'] >= 17) & (df['Age'] < 33), 'Age'] = 1
df.loc[(df['Age'] >= 33) & (df['Age'] < 49), 'Age'] = 2
df.loc[(df['Age'] >= 49) & (df['Age'] < 65), 'Age'] = 3
df.loc[ df['Age'] >= 65, 'Age'] = 4

# Cut the Fare column into 3 categories
# df['Fare'] = pd.cut(df['Fare'], 3)
# grouped_fare_df = df[['Fare', 'Survived']].groupby(['Fare'], as_index=False).mean().sort_values(by='Fare', ascending=True)
# print(grouped_fare_df)

# Convert Fare column to oridnal values
df.loc[ df['Fare'] < 171, 'Fare'] = 0
df.loc[(df['Fare'] >= 171) & (df['Fare'] < 342), 'Fare'] = 1
df.loc[ df['Fare'] >= 342, 'Fare'] = 2

df.to_csv(output_dataset_filename, index=False)

print('Done')

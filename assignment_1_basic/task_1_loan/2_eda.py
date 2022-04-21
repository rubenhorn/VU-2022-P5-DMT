import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import seaborn as sns

input_dataset_filename = (Path(__file__).parent / 'german-pre-processed.csv').resolve()

df = pd.read_csv(input_dataset_filename)

print('Number of samples:', len(df))
print('Number of features:', len(df.columns) - 1)
print('Highest targte correlation coefficients:')
Xy_correlation = df.corrwith(df['good_customer'])
Xy_correlation = Xy_correlation.sort_values(ascending=False, key=abs)
# Exclude auto-correlation (1.0)
Xy_correlation = Xy_correlation.drop(Xy_correlation.index[0])
print(Xy_correlation.head(n=5).to_string())

print('Done'); exit()

# Only look at a few columns at a time or the plot will be too big
columns = ['age', 'credits_count', 'installment_rate_relative_to_income', 'registered_telephone', 'good_customer']

fig, axes = plt.subplots(nrows=1, ncols=len(columns), figsize=(20, 5))
for i, column in enumerate(columns):
    sns.distplot(df[column], ax=axes[i])
    axes[i].set_title(column)
plt.tight_layout()
plt.show()


sns.heatmap(df[columns].corr(), annot=True, cmap='coolwarm')
plt.show()

print('Done')

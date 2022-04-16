import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import seaborn as sns

input_dataset_filename = (Path(__file__).parent / 'german-pre-processed.csv').resolve()

df = pd.read_csv(input_dataset_filename)

print('Number of samples:', len(df))

# Only look at a few columns at a time or the plot will be too big
columns = ['age', 'credits_count', 'installment_rate_relative_to_income', 'providing_for_count', 'registered_telephone', 'good_customer']

# plot the distribution for each column
for column in columns:
    plt.figure()
    sns.distplot(df[column])
    plt.title(f'{column}')
    plt.show()


sns.heatmap(df[columns].corr(), annot=True, cmap='coolwarm')
plt.show()

print('Done')

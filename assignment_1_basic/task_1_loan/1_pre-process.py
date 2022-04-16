import pandas as pd
from pathlib import Path

input_dataset_filename = (Path(__file__).parent / 'german.data').resolve()
output_dataset_filename = (Path(__file__).parent / 'german-pre-processed.csv').resolve()

df = pd.read_csv(input_dataset_filename, header=None, sep=' ')

# Map categorical values to numeric values
for column_index in range(len(df.columns)):
    if df.dtypes[column_index] == 'object':
        df[column_index] = df[column_index].apply(lambda x: int(x[(2 if (column_index + 1) < 10 else 3):]))

df.columns = [
    # A1 - A5
    'checking_account_status', 'credit_duration', 'credit_history', 'credit_purpose', 'credit_amount',
    # A6 - A10
    'savings_status', 'employment_time', 'installment_rate_relative_to_income', 'sex_and_personal_status', 'guarantors',
    # A11 - A15
    'last_moved', 'property_ownership', 'age', 'other_installment_plans', 'housing',
    # A16 - A20, target 
    'credits_count', 'employment_status', 'providing_for_count', 'registered_telephone', 'foreign_worker', 'good_customer']

# Normalize boolean categorical values to 0 and 1
df['registered_telephone'] -= 1
df['foreign_worker'] = (df['foreign_worker'] - 2) * -1
# Predict good_customer
df['good_customer'] = (df['good_customer'] - 2) * -1

df.to_csv(output_dataset_filename, index=False)

print('Done')

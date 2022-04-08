import dateparser
from fuzzywuzzy import fuzz
import pandas as pd
from pathlib import Path
import re

input_dataset_filename = (Path(__file__).parent / 'ODI-2022.csv').resolve()
output_dataset_filename = input_dataset_filename.parent / 'ODI-2022-pre-processed.csv'

def tryParseInt(x, default=None):
    try:
        return int(x)
    except ValueError:
        return default

def tryParseFloat(x, default=None):
    if type(x) == str:
        x = x.strip().replace(',', '.') # TODO this only works for 10,5 -> 10.5, but is wrong for 3,500.0
    try:
        return float(x) 
    except ValueError:
        return default

def tryParseDate(x, default=None):
    try:
        return dateparser.parse(x)
    except TypeError:
        return default

def tryParseTime(x, default=None):
    # TODO this may be a bit broken and not work correctly with 12h vs 24h time
    date = tryParseDate(x)
    return None if date is None else date.time().strftime('%H:%M')

def is_program_cs(value):
    if fuzz.ratio('computer science', value) > 80:
        return True
    if 'cs' in value.split(' '):
        return True
    return False

def is_program_ai(value):
    if 'artificial intelligence' in value:
        return True
    if 'ai' in value.split(' '):
        return True
    return False

def is_program_econ(value):
    if 'econ' in value:
        return True
    return False

def is_program_bio(value):
    if 'bio' in value:
        return True
    return False

def is_program_data(value):
    if 'data' in value:
        return True
    return False

def is_program_ba(value):
    if 'business' in value:
        return True
    if 'ba' == value:
        return True
    return False

# Computational Science
def is_program_compsci(value):
    if 'computat' in value:
        return True
    if 'cls' == value:
        return True
    return False

# Duisenberg Honours Program (includes qrm and fintech)
def is_program_dh(value):
    if 'quant' in value:
        return True
    if 'qrm' in value.split(' '):
        return True
    if 'f&t' == value:
        return True
    if 'fint' in value:
        return True
    if 'finance' in value:
        return True
    return False

def program_map_value(value):
    words_to_remove = [' in ', ' of ', '@', 'vu', 'uva', 'masters', 'master', 'msc' ]
    new_value = value
    for word in words_to_remove:
        new_value = new_value.replace(word, '')
    new_value = re.sub('\(.*\)', '', new_value)
    new_value = new_value.strip()
    if is_program_cs(new_value):
        return 'cs'
    elif is_program_ai(new_value):
        return 'ai'
    elif is_program_econ(new_value):
        return 'econ'
    elif is_program_bio(new_value):
        return 'bio'
    elif is_program_data(new_value):
        return 'data'
    elif is_program_ba(new_value):
        return 'ba'
    elif is_program_dh(new_value):
        return 'dh'
    elif is_program_compsci(new_value):
        return 'compsci'
    else:
        return 'other'

def combine_enumerations_into_list(list1, list2):
    enumeration_to_list = lambda x: [xx.strip() for xx in x.split(',')]
    return [enumeration_to_list(x) + enumeration_to_list(y) for x, y in zip(list1, list2)]

df = pd.read_csv(input_dataset_filename, delimiter=';')

df.columns = ['timestamp','program','did_ml','did_ir','did_stat','did_db','gender','chocolate_makes_you','birthday','num_neighbors','stood_up','stress_level','keep_of_100_euros','random_number','bedtime','good_day','good_day_2']

# Convert all values in df to lowercase
df = df.applymap(lambda x: x.lower().strip() if type(x) == str else x)

col_timestamp = df.columns[0]
df[col_timestamp] = df[col_timestamp].apply(lambda x: dateparser.parse(x))

col_program = df.columns[1]

# Normalize program column
df[col_program] = df[col_program].apply(program_map_value)

# Map yes/no/unknown questions to boolean
col_did_ml = df.columns[2]
df[col_did_ml] = df[col_did_ml].apply(lambda x: None if x == 'unknown' else (True if x == 'yes' else False))

col_did_ir = df.columns[3]
df[col_did_ir] = df[col_did_ir].apply(lambda x: None if x == 'unknown' else (True if x == '1' else False))

col_did_stat = df.columns[4]
df[col_did_stat] = df[col_did_stat].apply(lambda x: None if x == 'unknown' else (True if x == 'mu' else False))

col_did_db = df.columns[5]
df[col_did_db] = df[col_did_db].apply(lambda x: None if x == 'unknown' else (True if x == 'ja' else False))

col_birthday = df.columns[8]
df[col_birthday] = df[col_birthday].apply(lambda x: dateparser.parse(x))

col_num_neighbors = df.columns[9]
df[col_num_neighbors] = df[col_num_neighbors].apply(tryParseInt)

col_stood_up = df.columns[10]
df[col_stood_up] = df[col_stood_up].apply(lambda x: None if x == 'unknown' else (True if x == 'yes' else False))

col_stress_level = df.columns[11]
df[col_stress_level] = df[col_stress_level].apply(tryParseInt)

col_keep_of_100_euros = df.columns[12]
# TODO this removes all textual responses or formulas
df[col_keep_of_100_euros] = df[col_keep_of_100_euros].apply(lambda x: tryParseFloat(str(x)
    .replace('euros', '')
    .replace('euro', '')
    .replace('eur', '')
    .replace('€', '')))

# Parse random number as float (or None if not possible)
col_random_number = df.columns[13]
df[col_random_number] = df[col_random_number].apply(tryParseFloat)

col_bedtime = df.columns[14]
df[col_bedtime] = df[col_bedtime].apply(tryParseTime)

# Combine last two columns
col_good_day_1 = df.columns[15]
col_good_day_2 = df.columns[16]
# TODO could do some more cleaning up like removing punctuation, stop words and redundant adjectives
df[col_good_day_1] = combine_enumerations_into_list(df[col_good_day_1], df[col_good_day_2])
df.drop(col_good_day_2, axis=1, inplace=True)

df.to_csv(output_dataset_filename, index=False)

print('Done')

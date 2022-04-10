import nltk
import pandas as pd
from pathlib import Path

input_dataset_filename = (Path(__file__).parent / 'SmsCollection.csv').resolve()
output_dataset_filename = (Path(__file__).parent / 'SmsCollection-pre-processed.csv').resolve()

lemmatizer = nltk.stem.WordNetLemmatizer()
stopwords = nltk.corpus.stopwords.words('english')
stopword_lems = [lemmatizer.lemmatize(word) for word in stopwords]

def lemmatize(text):
    tokens = nltk.word_tokenize(text.lower())
    try:
        lemmatizer.lemmatize('cats')
    except LookupError:
        nltk.download('wordnet')
        nltk.download('omw-1.4')
    # Lemmatization
    lems = [lemmatizer.lemmatize(token) for token in tokens]
    # Stop word removal
    lems = [lem for lem in lems if lem not in stopword_lems]
    return ' '.join(lems)

df = pd.DataFrame(columns=['is_spam', 'message'])

with open(input_dataset_filename, 'r') as file:
    file.readline()
    line_no = 1
    for line in file:
        stripped_line = line.strip()
        if stripped_line.startswith('spam;'):
            df.loc[len(df)] = [True, stripped_line[5:]]
        elif stripped_line.startswith('ham;'):
            df.loc[len(df)] = [False, stripped_line[4:]]
        else:
            print(f'Error: line {line_no} is not in the correct format')
            exit(1)
        line_no += 1

df['message'] = df['message'].apply(lemmatize)

df.to_csv(output_dataset_filename, index=False)

print('Done')

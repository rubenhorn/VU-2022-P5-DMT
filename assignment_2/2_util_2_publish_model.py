#! /usr/bin/env python3

from pathlib import Path
import sys

from requests import request
import requests
from utils import *

reset_timer()

base_upload_url = 'https://transfer.sh/'
model_base_path = (Path(__file__).parent / 'models').resolve()

model_files = model_base_path.glob('*.joblib')

if len(model_files) == 0:
    print('No model files found in models folder', file=sys.stderr)

for model_file in model_files:
    model_file_name = model_file.name
    upload_url = f'{base_upload_url}{model_file_name}'
    tprint(f'Uploading {model_file_name}...', end='', flush=True)
    res = requests.request('PUT', upload_url, data=open(model_file, 'rb'))
    if res.status_code == 200:
        print(f' => {res.text}')
    else:
        print(f' FAILED with status {res.status_code}')

tprint('Done')

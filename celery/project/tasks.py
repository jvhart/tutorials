
from celery import Celery, shared_task

from django.utils import timezone

import numpy as np
import os

app = Celery()

@app.task
def add(x, y, **kwargs):
    print(kwargs)
    print(f'adding {x} and {y}')
    return x + y

if __name__ == '__main__':
    args = ['worker', '--loglevel=INFO']
    app.worker_main(argv=args)

@app.task 
def write_random_data_file():
    fl = '{}.txt'.format(timezone.now().strftime('%Y%m%d%H%M%S%f'))
    alphabet = list('abcdefghijklmnopqrstuvwxyz') + list('abcdefghijklmnopqrstuvwxyz'.upper()) + list('0123456789.,!@#$%^&*()?<> ') 
    with open(os.path.join('data', 'random_files', fl), 'w') as f:
        f.write(''.join(np.random.choice(a=alphabet, replace=True, size=500)))
    return os.path.join('data', 'random_files', fl)

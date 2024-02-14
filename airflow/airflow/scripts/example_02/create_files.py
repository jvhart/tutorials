

import os
import numpy as np

if not os.path.exists('data'):
    os.mkdir('data')

for n in range(10):
    fl = os.path.join('data', f'file_{str(n).zfill(2)}.txt')
    with open(fl, 'w') as f:
        _ = f.write(''.join(np.random.choice(a=list('abcdefghijklmnopqrstuvwxyz'), size=100, replace=True)))
    print(f'Created file {fl}')

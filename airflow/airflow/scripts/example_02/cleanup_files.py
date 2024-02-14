
import os
from glob import glob

file_ls = glob(os.path.join('data', '*.txt'))

for fl in file_ls:
    os.remove(fl)
    print(f'Deleted file: {fl}')

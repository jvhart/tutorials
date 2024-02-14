
import os
from glob import glob
from cryptography.fernet import Fernet



with open(os.path.join('data', 'frenet_secret.txt'), 'rb') as f:
    key = f.read()

fernet = Fernet(key)
file_ls = glob(os.path.join('data', 'file_*.txt'))
for fl in file_ls:
    encrypted_fl = fl.replace('file_', 'encrypted_file_')
    with open(fl, 'r') as rf:
        message = rf.read()
        
    with open(encrypted_fl, 'wb') as wf:
        _  = wf.write(fernet.encrypt(bytes(message, 'utf-8')))
    
    print(f'Encoded message "{message}" in file {encrypted_fl}')
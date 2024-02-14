
import os
from glob import glob
from cryptography.fernet import Fernet



with open(os.path.join('data', 'frenet_secret.txt'), 'rb') as f:
    key = f.read()

fernet = Fernet(key)

file_ls = glob(os.path.join('data', 'file_*.txt'))
for fl in file_ls:
    encrypted_fl = fl.replace('file_', 'encrypted_file_')
    with open(fl, 'r') as f:
        original_message = f.read()
    
    with open(encrypted_fl, 'rb') as f:
        encrypted_message = f.read()
    decrypted_message = fernet.decrypt(encrypted_message).decode('utf-8')
    
    assert original_message == decrypted_message, 'The messages do not match!!!!'
    
    print(f'File {fl}: Message validated, "{decrypted_message}"')

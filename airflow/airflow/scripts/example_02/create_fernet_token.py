
import os
from cryptography.fernet import Fernet


if not os.path.exists('data'):
    os.mkdir('data')

key = Fernet.generate_key()
with open(os.path.join('data', 'frenet_secret.txt'), 'wb') as f:
    f.write(key)

print('Created secret Fernet key')

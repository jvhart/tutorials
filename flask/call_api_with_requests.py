

from requests import put, get


res = put('http://localhost:5000/todo1', data={'data': 'Remember the milk'})
res.json()


res = get('http://localhost:5000/todo1', data={'data': 'Remember the milk'})
res.json()

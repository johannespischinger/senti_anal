import time
import requests

url = 'https://us-central1-sensi-anal.cloudfunctions.net/senti_anal_2'
payload = {'message': 'this is bad'}

for _ in range(1000):
    r = requests.get(url, params=payload)
    print(r)

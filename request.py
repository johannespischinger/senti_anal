import time
import requests
import json

url = 'https://us-central1-sensi-anal.cloudfunctions.net/senti_anal_2'
payload = {'message': 'this is bad'}

newHeaders = {'Content-type': 'application/json', 'Accept': 'text/plain'}
headers = newHeaders
for _ in range(1000):
    #r = requests.post(url, data=payload, headers=newHeaders)
    r = requests.post(url, json=payload, headers=newHeaders)
    print(r)

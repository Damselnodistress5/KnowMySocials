import requests
import time

url = 'http://127.0.0.1:8000'
reviews = [
    {
        'review_id': '1',
        'text': 'The bike feels excellent and smooth.',
        'region': 'North',
        'model': 'Activa',
        'feature': 'Comfort',
        'campaign': 'Summer Launch',
        'price_tier': 'Midrange',
    },
    {
        'review_id': '2',
        'text': 'Service was slow and not helpful.',
        'region': 'East',
        'model': 'CB300R',
        'feature': 'Service',
        'campaign': 'Support Drive',
        'price_tier': 'Premium',
    },
]

resp = requests.post(url + '/analyze', json={'reviews': reviews}, timeout=20)
print('analyze status', resp.status_code)
print(resp.text)
if resp.status_code == 200:
    task_id = resp.json()['task_id']
    print('task_id', task_id)
    for _ in range(10):
        time.sleep(1)
        status = requests.get(url + f'/status/{task_id}', timeout=10).json()
        print('status', status)
        if status['status'] not in ('running', 'queued'):
            break
    results = requests.get(url + f'/results/{task_id}', timeout=15).json()
    print('results', results)

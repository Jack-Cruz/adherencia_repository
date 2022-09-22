import json
import requests

print('Making single prediction')
host = 'localhost'
port = '8501'
data = {'age': [72],
    'sex': ['female'],
    'marital_status': ['married'],
    'Education': [' General Certificate of Secondary Education (German Realschule)'],
    'Occupation': ['pensioned'],
    'Medication_preparation_by': ['Connected'],
    'medication': [1],
    'SAMS_item1': [0],
    'SAMS_item2': [4],
    'SAMS_item3': [1],
    'SAMS_item4': [0],
    'SAMS_item5': [3],
    'SAMS_item6': [1],
    'SAMS_item7': [2],
    'SAMS_item8': [0],
    'SAMS_item9': [0],
    'SAMS_item10': [0],
    'SAMS_item11': [0],
    'SAMS_item12': [0],
    'SAMS_item13': [0],
    'SAMS_item14': [0],
    'SAMS_item15': [4],
    'SAMS_item16': [0],
    'SAMS_item17': [0],
    'SAMS_item18': [0],
    'SAMS_item19': [0]
}

json = {"inputs": data}
if __name__ == '__main__':
    server_url = 'http://' + host + ':' + port + '/v1/models/adherencia:predict'
    response = requests.post(server_url, json=json)
    data = response.json()
    value = data['outputs'][0][0]
    print('Adherencia' if value > 0.80 else 'No adherencia')
import json

with open('data/vizwiz/dataset_vizwiz.json', 'r') as f:
    data = json.load(f)
    print(data.keys())
    if 'images' in data:
        print(data['images'][0])

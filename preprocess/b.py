import json

with open('accountCategory.json','r') as f:
    content = json.load(f)

for k,v in content.items():
    if k == 'VGJOPIIBM':
        print(len(content[k]))
        print(content[k])

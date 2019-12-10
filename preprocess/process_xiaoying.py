import numpy as np

with open('../data_hxy/DECODE_MAP_FILE','r') as f:
    decode = f.readlines()

decode_map = {}

for line in decode:
    line = line.strip()
    line = line.split(',')
    decode_map[int(line[0])] = line[1]

with open('../data_hxy/deep_walk_hour_emb','r') as ff:
    content = ff.readlines()

content = content[1:]
result = open('../data_hxy/result-vid-hxy.txt','w')
b = np.zeros(shape=(len(content),1,128), dtype=np.float32)
j= 0

for line in content:
    line = line.strip()
    line = line.split(' ')
    result.write(decode_map[int(line[0])]+'\n')
    a = np.zeros(shape=(1,128),dtype=np.float32)
    for i in range(128):
        a[0,i] = float(line[i+1])
    print(a)
    b[j] = a
    j = j + 1

np.save('../data_hxy/result-vid-hxy.npy', b)

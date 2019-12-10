with open('/data/2/chenyao/word2Vec/data_hxy/DECODE_MAP_FILE','r') as f:
    content = f.readlines()

result = {}
for line in content:
    line = line.strip()
    key = line.split(',')[0]
    value = line.split(',')[1]
    result[int(key)] = value

with open('/data/2/chenyao/word2Vec/deepwalk/deepwalk/walks','r') as ff:
    walks = ff.readlines()

with open('../train_data/sentences.txt','w') as fff:
    for line in walks:
        walk = line.strip()
        videos = walk.split(' ')
        if len(videos) < 3: continue
        for video in videos:
            fff.write(result[int(video)] + ' ')
        fff.write('\n')    



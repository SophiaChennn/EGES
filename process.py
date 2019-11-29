with open('video_dict.txt','r') as f:
    content = f.readlines()

result = {}
for line in content:
    line = line.strip()
    key = line.split(':')[1]
    value = line.split(':')[0]
    result[int(key)] = value

with open('hour_data.walks.0','r') as ff:
    walks = ff.readlines()

with open('sentences.txt','w') as fff:
    for line in walks:
        walk = line.strip()
        videos = walk.split(' ')
        if len(videos) < 3: continue
        for video in videos:
            fff.write(result[int(video)] + ' ')
        fff.write('\n')    



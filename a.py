import json

with open('video_category.json','r') as f:
    content = f.readlines()

category_dict = {}

for one in content:
    one = one.strip()
    video = one.split(',')[0].split('[')[1]
    video = video[1:-1]
    category = one.split(',')[1].split(']')[0]
    category = category[2:-1]
    if '/' in category:
        category_dict[video] = category.split('/')[-1]
    else:
        category_dict[video] = category
    print(video)

with open('re_video_category.json','w') as ff:
        ff.write(json.dumps(category_dict,ensure_ascii=False))


with open('video_interests.json','r') as f:
    content = f.readlines()

interests_dict = {}

for one in content:
    one = one.strip()
    video = one.split('\"')
    vid = video[1]
    interest = video[3]
    more_interests = interest.split(',')
    interests_dict[vid] = more_interests

with open('re_video_interests.json','w') as ff:
    ff.write(json.dumps(interests_dict,ensure_ascii=False))

    

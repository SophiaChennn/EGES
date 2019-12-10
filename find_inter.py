import numpy as np
cy = open('result-vid-cy.txt','r')
hxy = open('/data/2/chenyao/word2Vec/data_hxy/result-vid-hxy.txt','r')
cy_np = np.load('result-embedding-cy.npy',allow_pickle=True)
hxy_np = np.load('/data/2/chenyao/word2Vec/data_hxy/result-embedding-hxy.npy',allow_pickle=True)


i = 0
vid_dic = {}
for h in hxy.readlines():
    h = h.strip()
    vid_dic[h] = i
    i = i+1

j = 0
array_list_cy = []
array_list_hxy = []
result_vid = []

for c in cy.readlines():
    c = c.strip()
    if c in vid_dic:
        print(c)
        result_vid.append(c)
        array_list_cy.append(cy_np[j])
        array_list_hxy.append(hxy_np[vid_dic[c]])
    j = j+1

array_cy = np.zeros(shape=(len(array_list_cy),1,128),dtype=np.float32)
array_hxy = np.zeros(shape=(len(array_list_hxy),1,128),dtype=np.float32)

m = 0
for array1 in array_list_cy:
    array_cy[m] = array1
    m = m+1

n = 0
for array2 in array_list_hxy:
    array_hxy[n] = array2
    n = n+1

np.save('result-embedding-cy-inter.npy',array_cy)
np.save('result-embedding-hxy-inter.npy',array_hxy)

fff = open('result-vid-inter.txt','w')
for vid in result_vid:
    fff.write(vid + '\n')

print(hxy_np[0] == hxy_np[1])

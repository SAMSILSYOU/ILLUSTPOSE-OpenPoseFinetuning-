import  json
import math
import numpy as np

d = np.zeros((300, 17))

json_open = open('result_Illust100.json', 'r') #精度結果用のJsonfile
json_load = json.load(json_open)
json_open2 = open('illust_test.json', 'r')
json_load2 = json.load(json_open2)

print('json_dict:{}'.format(type(json_load2)))

for z in range(300): #半径10pxでキーポイントが重なっていれば可とする。

    for w in range(17):
        if json_load[z-1]['keypoints'][3 * w -1] == json_load2["annotations"][z-1]['keypoints'][3 * w -1] :

            a = json_load[z-1]['keypoints'][3 * w -3] - json_load2["annotations"][z-1]['keypoints'][3 * w -3]
            b = json_load[z-1]['keypoints'][3 * w -2] - json_load2["annotations"][z-1]['keypoints'][3 * w -2]
            c = math.sqrt(a * a + b * b)
            if c < 10:
                d[z-1][w-1] = 1
            else:
                d[z-1][w-1] = 0
        
        else:
            d[z-1][w-1] = 0

s = 0 
u = 0

for t in range(17):
    for k in range(300):
        s = s + d[k-1][t-1]
    
    u = u + s
    print(s/300) 
    s = 0

print('精度:')
print (u/5100)
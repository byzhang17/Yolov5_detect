import detect
import sys
import cv2
import math
import numpy as np
from PIL import Image
import os

def get_cobb(img):
    W = img.shape[1]
    H = img.shape[0]
    kernel = np.ones((6, 6), np.float32) / 25
    gray_cut_filter2D = cv2.filter2D(img, -1, kernel)
    gray_img = cv2.cvtColor(gray_cut_filter2D, cv2.COLOR_BGR2GRAY)
    ret, thresh1 = cv2.threshold(gray_img, 180, 255, cv2.THRESH_BINARY)
    tm = thresh1.copy()
    #test_main = tm[int(H / 5):int(H * 4 / 5), int(W / 5):int(W * 4 / 5)]
    #image = test_main.shape
    #w = image[1]
    #h = image[0]
    edges = cv2.Canny(tm, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 45, minLineLength=100, maxLineGap=10)
    cobb = 0
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if y2 >= H*2 / 5 and y2 <= H * 5 / 6 and x2 >= W *2/ 5 and x2 <= W * 5 / 6 and (x1 < W / 2 or y1 < H / 2):
                if x1 == x2:
                    cobb = 90
                else:
                    tan = (y2 - y1) / (x2 - x1)
                    cobb = math.atan(tan) / np.pi * 180
    return cobb

#AB,BC,AC相电压表
def read1(cobb):
    result = cobb * 12 / 90
    return result

#控母电流
def read2(cobb):
    result = cobb * 30 / 90
    return result

#控母电压，电池组电压
def read3(cobb):
    result = cobb * 300 / 90
    return result

#充放电电流
def read4(cobb):
    if cobb < 45:
        result = (45 - cobb) * 30 / 45
    else:
        result = (cobb - 45) * 30 / 45
    return result


#path = sys.argv[1]
path = sys.argv[1]
path2 = sys.argv[2]
path3 = sys.argv[3]
detect_api = detect.DetectAPI(weights=path3, project = path2, save_crop=True)
txt, crops = detect_api.run(source=path)
path1 = str(detect_api.save_dir) + '/labels/'
os.mkdir(path1)
path2 = path1 + str(os.path.basename(path)).split('.')[0] + '.txt'

dict = {}
for i in range(0, len(txt)):
    dict[txt[i]] = crops[i]
new_txt = []
new_txt = sorted(txt, key=lambda d: d[2])
with open(path2, "a") as f:
    f.truncate(0)
while new_txt:
    txt1 = []
    for i in range(0, len(new_txt)):
        if i == len(new_txt) - 1:
            txt1 = new_txt
            new_txt = []
            break
        else:
            if new_txt[i + 1][2] - new_txt[i][2] > 0.1:
                txt1 = new_txt[0:i + 1]
                new_txt = new_txt[i + 1:]
                break
    txt1 = sorted(txt1, key=lambda d: d[1])
    with open(path2, "a") as f:
        for text in txt1:
            cobb = get_cobb(dict[text][0])
            f.write(str(cobb)+' '+('%g ' * len(text)).rstrip() % text + '\n')

#result_path = str(detect_api.save_dir) + '/crops/result.txt'


#if flag == 1:
#    if crops is not None:
#        for crop in crops:
#            cobb = get_cobb(crop)
#            result = read1(cobb)
#            with open(result_path, 'a') as f:
#                f.write(str(result)+' ')

#else:
#    dict = {}
#    for i in range(0,len(txt)):
#        dict[txt[i]] = crops[i]
#    new_dict = sorted(dict.items(), key=lambda d: d[0])
#    crop1 = new_dict[0][1]
#    crop2 = new_dict[1][1]
#    crop3 = new_dict[2][1]
#    crop4 = new_dict[3][1]
#    cobb1 = get_cobb(crop1)
#    cobb2 = get_cobb(crop2)
#    cobb3 = get_cobb(crop3)
#    cobb4 = get_cobb(crop4)
#    result1 = read4(cobb1)
#    result2 = read2(cobb2)
#    result3 = read3(cobb3)
#    result4 = read3(cobb4)
#    with open(result_path, 'a') as f:
#        f.write(str(result1)+' '+str(result2)+' '+str(result3)+' '+str(result4))
#    print(result1)
#    print(result2)
#    print(result3)
#    print(result4)


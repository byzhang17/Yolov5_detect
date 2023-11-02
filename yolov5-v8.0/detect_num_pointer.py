import detect
import sys
import os
import cv2
import math
import numpy as np
from PIL import Image, ImageDraw

def xywh2xyxy(x):
    y = x.copy()
    y[0] = x[0] - x[2]/2
    y[1] = x[1] - x[3]/2
    y[2] = x[0] + x[2]/2
    y[3] = x[1] + x[3]/2
    return y

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


#path = sys.argv[1]
path = sys.argv[1]
path2 = sys.argv[2]
path3 = sys.argv[3]
path4 = sys.argv[4]
#path = 'D:/yolov5-v7.0/datasets/pointer/images/val/IMG20230905145008_BURST002.jpg'
detect_api_1 = detect.DetectAPI(weights=path3, project = path2, save_crop=True)
detect_api_2 = detect.DetectAPI(weights=path4, project = path2)
txts, crops = detect_api_1.run(source=path)
dict = {}
for i in range(0, len(txts)):
    dict[txts[i]] = crops[i]
path1 = str(detect_api_1.save_dir) + '/num_labels/'
path2 = str(detect_api_1.save_dir) + '/pointer_labels/'
os.mkdir(path1)
os.mkdir(path2)
path3 = path1 + str(os.path.basename(path)).split('.')[0] + '.txt'
path4 = path2 + str(os.path.basename(path)).split('.')[0] + '.txt'
open(path3, 'a')
open(path4, 'a')
txt1 = []
txt2 = []
for txt in txts:
    if txt[0] == 0:
        txt1.append(txt)
    else:
        txt2.append(txt)
new_txt1 = []
new_txt2 = []
new_txt1 = sorted(txt1, key=lambda d: d[2])
new_txt2 = sorted(txt2, key=lambda d: d[2])

if len(new_txt1):
    while new_txt1:
        texts1 = []
        for i in range(0, len(new_txt1)):
            if i == len(new_txt1) - 1:
                texts1 = new_txt1
                new_txt1 = []
                break
            else:
                if new_txt1[i + 1][2] - new_txt1[i][2] > 0.05:
                    texts1 = new_txt1[0:i + 1]
                    new_txt1 = new_txt1[i + 1:]
                    break
        texts1 = sorted(texts1, key=lambda d: d[1])
        for text1 in texts1:
            txt3, crop3 = detect_api_2.run(source=dict[text1][1])
            new_txt3 = []
            new_txt3 = sorted(txt3, key=lambda d: d[2])
            while new_txt3:
                texts2 = []
                for i in range(0, len(new_txt3)):
                    if i == len(new_txt3) - 1:
                        texts2 = new_txt3
                        new_txt3 = []
                        break
                    else:
                        if new_txt3[i + 1][2] - new_txt3[i][2] > 0.05:
                            texts2 = new_txt3[0:i + 1]
                            new_txt3 = new_txt3[i + 1:]
                            break
                texts2 = sorted(texts2, key=lambda d: d[1])
                with open(path3, "a") as f:
                    for text2 in texts2:
                        f.write(str(text2[0]))
                    f.write(' ' + ('%g ' * len(text1)).rstrip() % text1 + '\n')

if len(new_txt2):
    while new_txt2:
        texts3 = []
        for i in range(0, len(new_txt2)):
            if i == len(new_txt2) - 1:
                texts3 = new_txt2
                new_txt2 = []
                break
            else:
                if new_txt2[i + 1][2] - new_txt2[i][2] > 0.1:
                    texts3 = new_txt2[0:i + 1]
                    new_txt2 = new_txt2[i + 1:]
                    break
        texts3 = sorted(texts3, key=lambda d: d[1])
        with open(path4, "a") as f:
            for text in texts3:
                cobb = get_cobb(dict[text][0])
                f.write(str(cobb) + ' ' + ('%g ' * len(text)).rstrip() % text + '\n')

f1 = open(path3, 'r')
f2 = open(path4, 'r')

datas1 = f1.readlines()
l1 = []
flag = [0, 0, 0, 0]
string = ''
for data in datas1:
    a = data.strip('\n').split(' ')
    xywh = [float(a[2]), float(a[3]), float(a[4]), float(a[5])]
    if xywh == flag:
        string = string+a[0]+' '
    else:
        l1.append((string, flag))
        flag = xywh
        string = a[0]+' '
l1.append((string, flag))

datas2 = f2.readlines()
l2 = []
flag = [0, 0, 0, 0]
string = ''
for data in datas2:
    a = data.strip('\n').split(' ')
    xywh = [float(a[2]), float(a[3]), float(a[4]), float(a[5])]
    if xywh == flag:
        string = string+a[0]+' '
    else:
        l2.append((string, flag))
        flag = xywh
        string = a[0]+' '
l2.append((string, flag))

img = cv2.imread(path)
size = img.shape
w = size[1]
h = size[0]
if len(l1)>1:
    for i in range(1, len(l1)):
        string = l1[i][0]
        xywh = l1[i][1]
        xyxy = xywh2xyxy(xywh)
        x1 = int(xyxy[0] * w)
        y1 = int(xyxy[1] * h)
        x2 = int(xyxy[2] * w)
        y2 = int(xyxy[3] * h)
        p1 = (x1, y1)
        p2 = (x2, y2)
        cv2.rectangle(img, p1, p2, (0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
        tf = max(2 - 1, 1)  # font thickness
        w1, h1 = cv2.getTextSize(string, 0, fontScale=2 / 3, thickness=tf)[0]  # text width, height
        outside = p1[1] - h1 >= 3
        p2 = p1[0] + w1, p1[1] - h1 - 3 if outside else p1[1] + h1 + 3
        cv2.putText(img,
                    string, (p1[0], p1[1] - 2 if outside else p1[1] + h1 + 2),
                    0,
                    2 / 3,
                    (0, 0, 255),
                    thickness=tf,
                    lineType=cv2.LINE_AA)

if len(l2)>1:
    for i in range(1, len(l2)):
        string = l2[i][0]
        xywh = l2[i][1]
        xyxy = xywh2xyxy(xywh)
        x1 = int(xyxy[0] * w)
        y1 = int(xyxy[1] * h)
        x2 = int(xyxy[2] * w)
        y2 = int(xyxy[3] * h)
        p1 = (x1, y1)
        p2 = (x2, y2)
        cv2.rectangle(img, p1, p2, (0, 255, 0), thickness=3, lineType=cv2.LINE_AA)
        tf = max(3 - 1, 1)  # font thickness
        w1, h1 = cv2.getTextSize(string, 0, fontScale=3 / 3, thickness=tf)[0]  # text width, height
        outside = p1[1] - h1 >= 3
        p2 = p1[0] + w1, p1[1] - h1 - 3 if outside else p1[1] + h1 + 3
        cv2.putText(img,
                    string, (p1[0], p1[1] - 2 if outside else p1[1] + h1 + 2),
                    0,
                    3 / 3,
                    (0, 255, 0),
                    thickness=tf,
                    lineType=cv2.LINE_AA)

cv2.imwrite(str(detect_api_1.save_dir)+'/'+str(os.path.basename(path)), img)



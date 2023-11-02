import os
import numpy as np


def center_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    #计算每个框的上下左右边线的坐标
    y1_max = y1 + h1/2
    x1_max = x1 + w1/2
    y1_min = y1 - h1/2
    x1_min = x1 - w1/2

    y2_max = y2 + h2/2
    x2_max = x2 + w2/2
    y2_min = y2 - h2/2
    x2_min = x2 - w2/2

    #上取小下取大，右取小左取大
    xx1 = np.max([x1_min, x2_min])
    yy1 = np.max([y1_min, y2_min])
    xx2 = np.min([x1_max, x2_max])
    yy2 = np.min([y1_max, y2_max])

    #计算各个框的面积
    area1 = (x1_max-x1_min) * (y1_max-y1_min) 
    area2 = (x2_max-x2_min) * (y2_max-y2_min)

    #计算相交的面积
    inter_area = (np.max([0, xx2-xx1])) * (np.max([0, yy2-yy1]))
    #计算IoU
    iou = inter_area / (area1+area2-inter_area)
    return iou


def lightjudge_IOU(fread, light_label, light_box, iou_thres):
    fread.seek(0)
    while True: #读取文件内容
        line = fread.readline() #按行读取内容
        if len(line) > 0: #当该行为空，表明已经读取到文件末尾，退出循环
            detect_content = line.split()#将它们分开
            detect_label = int(detect_content[0])
            detect_box = list(map(float, detect_content[1:]))
            if center_iou(light_box, detect_box) > iou_thres:
                if light_label == detect_label:
                    return True
        else:
            break
    return False


def judge_IOU(pic_info, path):
    fread = open(path, 'r')
    for light_info in pic_info:
        if lightjudge_IOU(fread, light_info[0], light_info[1:], 0.5) == 0:
            fread.close()
            return False
    fread.close()
    return True

def judge_seq(pic_info, path):
    detect = []
    fread = open(path, 'r')
    while True:
        line = fread.readline()
        if len(line) > 0:
            detect.append([float(i) for i in line.split()])
        else:
            break
    
    detect.sort(key=lambda ele: ele[1])
    pic_info.sort(key=lambda ele: ele[1])

    detect_label = ''
    pic_label = ''
    for label in detect:
        detect_label = detect_label + str(int(label[0]))
    for label in pic_info:
        pic_label = pic_label + str(int(label[0]))
    
    if detect_label.find(pic_label) == -1:
        return False
    else:
        return True


fpath = r"E:\code\Server_Room_Judge\2023_02_17\labels\2023_02_17_18_18_07.txt"  #检测结果文件的路径
pic_info = [[0, 0.51, 0.52, 0.03, 0.04],
            [1, 0.45, 0.51, 0.03, 0.04],
            [3, 0.39, 0.51, 0.03, 0.04]]
if judge_seq(pic_info, fpath):
    print('True')
else:
    print('False')

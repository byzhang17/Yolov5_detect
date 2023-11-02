import detect
import sys
import os
import time
import cv2

time_start = time.time()
#path = 'D:/yolov5-v7.0/datasets/number1/images/val/IMG_20230829_151429.jpg'
path = sys.argv[1]
detect_api = detect.DetectAPI(weights='D:/yolov5-v7.0/runs/train/exp17/weights/best.pt', save_crop=True)
#rpath = 'D:/yolov5-v7.0/datasets/number1/images/val'
#files = os.listdir(rpath)
#i = 0
#for file in files:
#    detect_api.run(source=rpath + '/' + file)
#    print(i)
#    i = i+1
txt, crops = detect_api.run(source=path)
time_end = time.time()
time_sum = time_end - time_start
print(txt)
print(time_sum)
i = 1
for crop in crops:
    cv2.imshow("image"+str(i), crop)
    i = i+1
cv2.waitKey(0)
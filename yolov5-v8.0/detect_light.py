import detect
import sys
import os

path = sys.argv[1]
path2 = sys.argv[2]
path3 = sys.argv[3] #'/runs/train/exp13/weights/best.pt'
detect_api = detect.DetectAPI(weights=path3, project = path2, save_txt=True)
txt, crops = detect_api.run(source=path)
print(txt)


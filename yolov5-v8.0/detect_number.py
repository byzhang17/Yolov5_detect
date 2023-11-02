import detect
import sys
import os

#path = sys.argv[1]
path = sys.argv[1]
path2 = sys.argv[2]
path3 = sys.argv[3]
path4 = sys.argv[4]
#path = 'D:/yolov5-v7.0/datasets/number1/images/val/1691078553061.jpg'
detect_api_1 = detect.DetectAPI(weights=path3, project = path2, save_crop=True)
detect_api_2 = detect.DetectAPI(weights=path4, project = path2)
txt1, crop1 = detect_api_1.run(source=path)
dict = {}
for i in range(0, len(txt1)):
    dict[txt1[i]] = crop1[i]
path1 = str(detect_api_1.save_dir) + '/labels/'
os.mkdir(path1)
path2 = path1 + str(os.path.basename(path)).split('.')[0] + '.txt'
new_txt1 = []
new_txt1 = sorted(txt1, key=lambda d: d[2])
with open(path2, "a") as f:
    f.truncate(0)
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
        print(text1, dict[text1][1])
        txt2, crop2 = detect_api_2.run(source=dict[text1][1])
        new_txt2 = []
        new_txt2 = sorted(txt2, key=lambda d: d[2])
        while new_txt2:
            texts2 = []
            for i in range(0, len(new_txt2)):
                if i == len(new_txt2) - 1:
                    texts2 = new_txt2
                    new_txt2 = []
                    break
                else:
                    if new_txt2[i + 1][2] - new_txt2[i][2] > 0.05:
                        texts2 = new_txt2[0:i + 1]
                        new_txt2 = new_txt2[i + 1:]
                        break
            texts2 = sorted(texts2, key=lambda d: d[1])
            with open(path2, "a") as f:
                for text2 in texts2:
                    f.write(str(text2[0]))
                f.write(' ' + ('%g ' * len(text1)).rstrip() % text1 + '\n')


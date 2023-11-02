import detect
import sys
import os

#path = sys.argv[1]
path = sys.argv[1]
path2 = sys.argv[2]
path3 = sys.argv[3]
detect_api = detect.DetectAPI(weights=path3, project = path2, save_txt=True)
txt, crops = detect_api.run(source=path)
path1 = str(detect_api.save_dir) + '/labels/'

path2 = path1 + str(os.path.basename(path)).split('.')[0] + '.txt'

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
            if new_txt[i + 1][2] - new_txt[i][2] > 0.05:
                txt1 = new_txt[0:i + 1]
                new_txt = new_txt[i + 1:]
                break
    txt1 = sorted(txt1, key=lambda d: d[1])
    with open(path2, "a") as f:
        for text in txt1:
            f.write(('%g ' * len(text)).rstrip() % text + '\n')

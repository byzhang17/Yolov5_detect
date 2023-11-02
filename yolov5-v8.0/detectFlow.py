import os
from datetime import datetime
import detect
import robotAPI
import threading
import numpy as np
import parameter
import Route
import report
import math

# route1 = [2, 201, 203, 206, 210, 213, 215, 10, 1001, 1003, 1006, 1010, 1013, 1015, 2]
# route2 = [4, 401, 403, 406, 410, 413, 415, 12, 1201, 1203, 1206, 1210, 1213, 1215, 4]
# route3 = [7, 701, 703, 706, 710, 713, 715, 15, 1501, 1503, 1506, 1510, 1513, 1515, 7]

dev = 'cpu'
all_open_doors = [200, 400, 700, 216, 416, 716, 1000, 1016, 1200, 1216, 1500, 1516]
stay_time = 8  # 停留时间


def center_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # 计算每个框的上下左右边线的坐标
    y1_max = y1 + h1 / 2
    x1_max = x1 + w1 / 2
    y1_min = y1 - h1 / 2
    x1_min = x1 - w1 / 2

    y2_max = y2 + h2 / 2
    x2_max = x2 + w2 / 2
    y2_min = y2 - h2 / 2
    x2_min = x2 - w2 / 2

    # 上取小下取大，右取小左取大
    xx1 = np.max([x1_min, x2_min])
    yy1 = np.max([y1_min, y2_min])
    xx2 = np.min([x1_max, x2_max])
    yy2 = np.min([y1_max, y2_max])

    # 计算各个框的面积
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)

    # 计算相交的面积
    inter_area = (np.max([0, xx2 - xx1])) * (np.max([0, yy2 - yy1]))
    # 计算IoU
    iou = inter_area / (area1 + area2 - inter_area)
    return iou


def lightjudge(fread,  judgelist):
    fread.seek(0)
    while True:  # 读取文件内容
        line = fread.readline()  # 按行读取内容
        if len(line) > 0:  # 当改行为空，表明已经读取到文件末尾，退出循环
            detect_content = line.split()  # 将它们分开
            detect_label = int(detect_content[0])
            print("detect_label:({}), type:({})".format(detect_label, type(detect_label)))
            detect_box = list(map(float, detect_content[1:]))
            if detect_label == 0 or detect_label == 1:
                judgelist[detect_label] = 'T'
            elif detect_label == 3:
                judgelist[2] = 'T'

        else:
            break
    return True


def judge(pic_info, path):
    if not os.path.exists(path):
        print("错误：不存在对应的标签文件！")
        return False
    fread = open(path, 'r')
    judgeList = ['F', 'F', 'F']
    res = lightjudge(fread, judgeList)
    print(judgeList)

    if 'F' in judgeList:
        return False

    fread.close()
    return True


def make_today_dir():
    now = datetime.now()
    timeStr = now.strftime("%Y_%m_%d")
    dir = os.getcwd() + '/dataPool/' + timeStr
    if os.path.exists(dir):
        print('文件夹已存在')
    else:
        os.makedirs(dir)
    return timeStr


def inspectFlow(point, open_doors):
    if not parameter.point_pare.get(point):
        print("无效目标点: {}".format(point))
        return
    robotAPI.moveToPoint([point], open_doors)  # 移动到巡检点
    event = threading.Event()
    while robotAPI.getState() != "StandingBy" and robotAPI.getState() != "Charging":
        event.wait(5)
    robotAPI.postCtrlPanTilt(parameter.point_pare.get(point)[0], parameter.point_pare.get(point)[1])  # 相机高度，角度
    robotAPI.postCtrlLister(parameter.point_pare.get(point)[2])
    event.wait(stay_time)
    timeNow = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    # imageName = "data/images/test.jpg"
    today_dir = make_today_dir()
    imageName = "dataPool/{}/{}.jpg".format(today_dir, timeNow)
    robotAPI.getRGB(imageName)  # 获取巡检照片
    #print("准备调用yolov5 detect")
    detect.run(source=imageName,
               weights="models/trained.pt",
               view_img=False,
               device=dev,
               save_txt=True,
               project='dataPool/',
               name=today_dir,
               exist_ok=True)  # 送入yolov5检测
    # 根据检测结果推断异常
    #print("调用结束")
    return


def inspectLightFlow(point, open_doors, passage):
    if not parameter.point_pare.get(point):
        print("无效目标点: {}".format(point))
        return
    robotAPI.moveToPoint([point], open_doors)  # 移动到巡检点
    event = threading.Event()
    while robotAPI.getState() != "StandingBy" and robotAPI.getState() != "Charging":
        event.wait(5)
    robotAPI.postCtrlPanTilt(parameter.point_pare.get(point)[0], parameter.point_pare.get(point)[1])  # 相机高度，角度
    robotAPI.postCtrlLister(parameter.point_pare.get(point)[2])
    event.wait(stay_time)
    timeNow = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    # imageName = "data/images/test.jpg"
    today_dir = make_today_dir()
    imageName = "dataPool/{}/{}.jpg".format(today_dir, timeNow)
    robotAPI.getRGB(imageName)  # 获取巡检照片
    detect.run(source=imageName,
               weights="models/trained.pt",
               view_img=False,
               device=dev,
               save_txt=True,
               project='dataPool/',
               name=today_dir,
               exist_ok=True)  # 送入yolov5检测
    label_path = "dataPool/{}/labels/{}.txt".format(today_dir, timeNow)
    lightCorrect = judge(parameter.pic_info, label_path)
    path = "dataPool/{}".format(today_dir)
    report.CreateReport(robotAPI.getAllStatus(), passage, lightCorrect, path)
    # 根据检测结果推断异常
    return


def openDoorThread(event):
    while 1:
        if event.is_set():
            break
        loc = queryRobotLocation()
        if loc in [214, 215, 1000, 1001]:
            robotAPI.send_open_door(parameter.door_point_to_id_dict.get(215)[0],
                                    parameter.door_point_to_id_dict.get(215)[1])
            robotAPI.send_open_door(parameter.door_point_to_id_dict.get(1000)[0],
                                    parameter.door_point_to_id_dict.get(1000)[1])
        elif loc in [414, 415, 1200, 1201]:
            robotAPI.send_open_door(parameter.door_point_to_id_dict.get(415)[0],
                                    parameter.door_point_to_id_dict.get(415)[1])
            robotAPI.send_open_door(parameter.door_point_to_id_dict.get(1200)[0],
                                    parameter.door_point_to_id_dict.get(1200)[1])
        elif loc in [714, 715, 1500, 1501]:
            robotAPI.send_open_door(parameter.door_point_to_id_dict.get(715)[0],
                                    parameter.door_point_to_id_dict.get(715)[1])
            robotAPI.send_open_door(parameter.door_point_to_id_dict.get(1500)[0],
                                    parameter.door_point_to_id_dict.get(1500)[1])
        elif parameter.door_point_to_id_dict.get(loc):
            print("send open door {} order...".format(parameter.door_point_to_id_dict.get(loc)[1]))
            robotAPI.send_open_door(parameter.door_point_to_id_dict.get(loc)[0],
                                    parameter.door_point_to_id_dict.get(loc)[1])
        else:
            print("point :{} not found".format(loc))
            event.wait(3)


def Find_Recent(x, y, x_, y_):
    max_dis = 9999
    index = 0
    for i in range(len(x)):
        length = math.sqrt(pow(x[i] - x_, 2) + pow(y[i] - y_, 2))
        if length <= max_dis:
            max_dis = length
            index = i
        else:
            continue
    return index


def loadCoordinate():
    xlist = []
    ylist = []
    plist = []
    for key, value in parameter.point_to_xy.items():
        plist.append(key)
        xlist.append(value[0])
        ylist.append(value[1])
    return plist, xlist, ylist


def queryRobotLocation():
    allStatus = robotAPI.getAllStatus()
    if allStatus is None:
        return 0
    # moving_goal = allStatus["data"]["moving_goal"]
    # staying_loc = allStatus["data"]["staying_loc"]

    # state = allStatus["data"]["state"]
    '''
    air_smoke = allStatus["data"]["air"]["smoke"]
    air_pm1 = allStatus["data"]["air"]["pm1"]
    air_temperature = allStatus["data"]["air"]["temperature"]
    air_pm10 = allStatus["data"]["air"]["pm10"]
    air_pm25 = allStatus["data"]["air"]["pm25"]
    air_humidity = allStatus["data"]["air"]["humidity"]
    '''
    loc_y = allStatus["data"]["location"]["y"]
    loc_x = allStatus["data"]["location"]["x"]
    # loc_d = allStatus["data"]["location"]["d"]
    # loc_map = allStatus["data"]["location"]["map"]

    loc = 0
    plist, xlist, ylist = loadCoordinate()
    x = loc_x * 0.03 - 22.4893
    y = loc_y * 0.03 - 7.08015
    idx = Find_Recent(xlist, ylist, x, y)
    loc = plist[idx]
    # print("index:{}, point:{}, 当前x:{}, 当前y:{}, 最近点x:{}, 最近点y:{}".format(idx, loc, x, y, xlist[idx], ylist[idx]))
    '''
    print("移动目标点：{}, 类型: {}, 当前停留点：{}".format(moving_goal, staying_loc))
    
    if staying_loc != 0:
        loc = staying_loc
    if moving_goal != 0:
        loc = moving_goal
    print("当前起始点：{}".format(loc))
    '''
    return loc


def robotLocInMap():
    loc = queryRobotLocation()


def getDetectResult(imageName, today_dir):
    res = []
    f = open("dataPool/{}/{}.txt".format(today_dir, imageName))
    return res


def checkLight(res, standard):
    if res == standard:
        return True
    return False


def backToCharge():
    if robotAPI.getState() == "Error":
        print("机器人状态错误！请检查错误！")
        return
    elif robotAPI.getState() == "TakenOver":
        print("机器人处于人工接管状态！")
        return
    loc = queryRobotLocation()  #
    target = "-1"
    route_list = Route.queryBackRoute(loc)
    try:
        robotAPI.initState(0.1)
        robotAPI.moveToPoint(route_list, all_open_doors)
    except:
        return
    return


def stopMove():
    try:
        robotAPI.postStopMove()
    except:
        return


def exampleModule():
    robotAPI.moveToPoint([201], all_open_doors)
# point = 413
# print(point_pare.get(point)[0],point_pare.get(point)[1],point_pare.get(point)[2])
# robotAPI.postCtrlLister(1.0)
# robotAPI.postCtrlPanTilt(point_pare.get(point)[0], point_pare.get(point)[1])
# robotAPI.initState()
# robotAPI.moveToPoint([4, 2, -1], all_open_doors)
# robotAPI.moveToPoint([2, 4], all_open_doors)
def inspectRoute(route):
    if route not in parameter.route_str:
        print("Wrong route!")
        return
    if robotAPI.getState() == "Error":
        print("机器人状态错误！请检查错误！")
        return
    elif robotAPI.getState() == "TakenOver":
        print("机器人处于人工接管状态！")
        return
    robotAPI.initState(parameter.mianbanHeight)
    # e = threading.Event()
    # t1 = threading.Thread(target=openDoorThread, args=(e,))
    # t1.start()
    try:
        if route == parameter.route_str[0]:
            robotAPI.moveToPoint([2], all_open_doors)
            inspectFlow(203, all_open_doors)
            # print("执行到203点")
            inspectFlow(206, all_open_doors)
            # print("执行到206点")
            inspectFlow(210, all_open_doors)
            inspectFlow(213, all_open_doors)
            inspectLightFlow(215, all_open_doors, passage=2)

            robotAPI.moveToPoint([1000], all_open_doors)
            inspectLightFlow(1001, all_open_doors, passage=1)
            inspectFlow(1003, all_open_doors)
            inspectFlow(1006, all_open_doors)
            inspectFlow(1010, all_open_doors)
            inspectFlow(1013, all_open_doors)

            robotAPI.initState(parameter.defaultHeight)
            robotAPI.moveToPoint([1000, 215], all_open_doors)
            robotAPI.moveToPoint([201, 200], all_open_doors)
        elif route == parameter.route_str[1]:
            robotAPI.moveToPoint([4], all_open_doors)
            inspectFlow(403, all_open_doors)
            inspectFlow(406, all_open_doors)
            inspectFlow(410, all_open_doors)
            inspectFlow(413, all_open_doors)
            inspectLightFlow(415, all_open_doors, passage=4)
            robotAPI.moveToPoint([12], all_open_doors)
            inspectLightFlow(1201, all_open_doors, passage=3)
            inspectFlow(1203, all_open_doors)
            inspectFlow(1206, all_open_doors)
            inspectFlow(1210, all_open_doors)
            inspectFlow(1213, all_open_doors)
            robotAPI.initState(parameter.defaultHeight)
            robotAPI.moveToPoint([1213, 1200, 415, 401, 4, 200], all_open_doors)
        elif route == parameter.route_str[2]:
            robotAPI.moveToPoint([7], all_open_doors)
            inspectFlow(703, all_open_doors)
            inspectFlow(706, all_open_doors)
            inspectFlow(710, all_open_doors)
            inspectFlow(713, all_open_doors)
            inspectLightFlow(715, all_open_doors, passage=6)
            robotAPI.moveToPoint([1500], all_open_doors)
            inspectLightFlow(1501, all_open_doors, passage=5)
            inspectFlow(1503, all_open_doors)
            inspectFlow(1506, all_open_doors)
            inspectFlow(1510, all_open_doors)
            inspectFlow(1513, all_open_doors)
            robotAPI.initState(parameter.defaultHeight)
            robotAPI.moveToPoint([1500, 715, 701, 7, 200], all_open_doors)
        else:
            print("Wrong route!")
    finally:
        print("巡检异常！")
    # e.set()
    return

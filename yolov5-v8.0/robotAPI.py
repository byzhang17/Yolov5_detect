import requests
import threading
import socket
import json

ip1 = 'http://192.168.1.101:35281'
ip2 = 'http://192.168.1.102:35181'


def expo(rate):
    send_data = 'expo {}'.format(rate)
    ip = '192.168.1.101'  # change this value to the actual IP of pan-tilt camera
    port = 3579
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.sendto(send_data.encode('utf-8'), (ip, port))


# state string index: 43
# 获取机器人当前状态
def getState():
    status = '/beepatrol/status'
    headers = {
        'content-length': '0',
        'content-type': 'application/json',
    }
    try:
        ret = requests.get(ip2 + status, timeout=5)
    except requests.exceptions.RequestException as e:
        print(e)
    res = ret.text
    res_dict = json.loads(res)
    # index = res.find("state")
    # splited = res.split('"')
    # sta = res_dict[43]
    sta = res_dict["data"]["state"]
    # print("query status : ", sta)
    return sta


def getAllStatus():
    status = '/beepatrol/status'
    ret = None
    try:
        ret = requests.get(ip2 + status, timeout=5)
    except requests.exceptions.RequestException as e:
        print(e)
    if ret is None:
        return None
    res = ret.text
    res_dict = json.loads(res)
    return res_dict


# 获取RGB图像
def getRGB(output):
    rgb_shot = '/beepatrol/rgb_shot'
    ret = ""
    try:
        ret = requests.get(ip1 + rgb_shot, params=None, headers={'content-length': '0', }, stream=True, timeout=5)
    except requests.exceptions.RequestException as e:
        print(e)
    with open(output, 'wb') as ofile:
        for chuck in ret.iter_content(chunk_size=None):
            ofile.write(chuck)


# 控制补光灯
def postFillLight(lightOn=False):
    light = '/beepatrol/switch_fill_light'
    jdata = dict(turn_on=lightOn)
    try:
        ret = requests.post(ip1 + light, json=jdata, timeout=5)
    except requests.exceptions.RequestException as e:
        print(e)
    # print("Fill Light: ", ret.text)


def getPanTilt():
    panTilt = '/beepatrol/pan_tilt'
    try:
        ret = requests.get(ip1 + panTilt, timeout=5)
    except requests.exceptions.RequestException as e:
        print(e)
    # print("pan tilt: ", ret.text)


# 获取热度图
def getHeatMap(output):
    heatMap = '/beepatrol/heatmap_shot'
    try:
        ret = requests.get(ip1 + heatMap, timeout=5)
    except requests.exceptions.RequestException as e:
        print(e)
    with open(output, 'wb') as ofile:
        for chuck in ret.iter_content(chunk_size=None):
            ofile.write(chuck)
    # print("heat map: ", ret.text)


# 控制升降台高度
def postCtrlLister(height):
    lister = '/beepatrol/ctrl_lifter'
    try:
        ret = requests.post(ip2 + lister, json=dict(height=height), timeout=5)
    except requests.exceptions.RequestException as e:
        print(e)
    # print("check lister: ", ret.text)


# 调整云台旋转角度
def postCtrlPanTilt(panPara, tiltPara):
    panTilt = '/beepatrol/ctrl_pan_tilt'
    if panPara < -105 or panPara > 105 or tiltPara < -30 or tiltPara > 105:
        print("wrong parameter!")
        return
    try:
        ret = requests.post(ip1 + panTilt, json=dict(pan=float(panPara), tilt=float(tiltPara)), timeout=5)
    except requests.exceptions.RequestException as e:
        print(e)
    # print("pan tilt: ", ret.reason, ret.status_code)


# 调整云台方向
def postSetPanTiltPosture():
    posture = '/beepatrol/set_pan_tilt_posture'
    item = ['init', 'left', 'right']
    try:
        ret = requests.post(ip1 + posture, json=dict(preset=item[2]), timeout=5)  # left or right or init
    except requests.exceptions.RequestException as e:
        print(e)
    # print("set posture: ", ret.text)


# 发送移动指令
# goal代表机器人停点，充电桩为-1。open_doors代表已打开的门。


def postMove(point, opendoors):
    move = '/beepatrol/move'
    target = {
        'back': -1,
        'target1': 1
    }
    data = dict(goal=point, open_doors=opendoors)
    try:
        ret = requests.post(ip2 + move, json=data, timeout=5)
    except requests.exceptions.RequestException as e:
        print(e)
    # print("move: ", ret.status_code, ret.reason, ret.text)


# 发送停止指令
def postStopMove():
    stop = '/beepatrol/stop'
    try:
        ret = requests.post(ip2 + stop, timeout=5)
    except requests.exceptions.RequestException as e:
        print(e)
    # print("stop move")


# 人工接管
def postTakeOver():
    order = '/beepatrol/start_taking_over'
    try:
        ret = requests.post(ip2 + order, timeout=5)
    except requests.exceptions.RequestException as e:
        print(e)
    # print("robot take over, message: ", ret.status_code, ret.reason, ret.text)


# 停止人工接管
def postStopTakeOver():
    order = '/beepatrol/stop_taking_over'
    try:
        ret = requests.post(ip2 + order, timeout=5)
    except requests.exceptions.RequestException as e:
        print(e)
    # print("robot stop take over, message: ", ret.text)


# 清除错误
def clearError():
    order = '/beepatrol/clear_error'
    try:
        ret = requests.post(ip2 + order, timeout=5)
    except requests.exceptions.RequestException as e:
        print(e)
    # print("clean error, message: ", ret.text)


# 打印状态
def printState():
    state = getState()
    if state == "StandingBy":
        print("robot StandingBy")
    elif state == "Charging":
        print("robot Charging")
    elif state == "Moving":
        print("robot Moving")
    else:
        print("Unknown state")


def moveToPoint(route, open_doors):
    point_list = route
    # open_doors = [200, 400, 700, 216, 416, 716, 1000, 1016, 1200, 1216, 1500, 1516]
    # sem = threading.Semaphore(1)
    event = threading.Event()
    for point in point_list:
        while getState() != "StandingBy" and getState() != "Charging":
            event.wait(5)
        try:
            postMove(point, open_doors)
        except requests.exceptions.RequestException as e:
            print(e)
        event.wait(2)
        print("point : {} ".format(point))


def initState(height=0.1):
    try:
        postCtrlPanTilt(-90, 0)
        postCtrlLister(height)
        postFillLight(lightOn=False)
    except requests.exceptions.RequestException as e:
        print(e)


api_open_doors = [200, 400, 700, 216, 416, 716, 1000, 1016, 1200, 1216, 1500, 1516]


def checkRoute(route):
    try:
        postCtrlLister(1.3)
        postCtrlPanTilt(-90, 0)
        moveToPoint(route, api_open_doors)
    except requests.exceptions.RequestException as e:
        print(e)


def send_open_door(door_id, door_num):
    requests.post('http://10.255.254.1:18081/v1/ac/door', json=dict(
        method='open',
        param=dict(
            moid=4842000000005000 + door_id,
            name='C2-4F-冷通道0' + door_num + '-门禁'
        )
    ))


# print(getAllStatus())
# getState()
# getHeatMap(output='heatmap.png')
# getRGB(output='rgb.png')
# postFillLight(lightOn=True)
# postFillLight(lightOn=False)
# getPanTilt()
# postMove(2,[])
# postTakeOver()
# clearError()
# postCtrlLister(0.3)
# postSetPanTiltPosture()
# postCtrlPanTilt(-90, 0)
# initState()
# route1 = [2, 201, 203, 206, 210, 213, 215, 10, 1001, 1003, 1006, 1010, 1013, 1015, 2]
# route2 = [4, 401, 403, 406, 410, 413, 415, 12, 1201, 1203, 1206, 1210, 1213, 1215, 4]
# route3 = [7, 701, 703, 706, 710, 713, 715, 15, 1501, 1503, 1506, 1510, 1513, 1515, 7]
# checkRoute(route1)
# moveToPoint([7, 701, 714], api_open_doors)
# moveToPoint(route2, api_open_doors)
# moveToPoint(route3)
# moveToPoint([2], api_open_doors)
# initState()
# printState()
# post(798, '4-东')

'''
{
    "msg": "",
    "code": 0,
    "data": {"uptime": "230323170044",
             "battery": {"current": -2.440000057220459,
                         "state": "Discharging",
                         "temperature": 0.0,
                         "power": 62,
                         "voltage": 27.170000076293945},
             "air": {"smoke": 12.0,
                     "pm1": 0.0,
                     "temperature": 26.059999465942383,
                     "pm10": 0.0,
                     "pm25": 0.0,
                     "humidity": 49.93000030517578},
             "state": "StandingBy",
             "lifter": 0.09999491369464597,
             "location": {"y": 317.93473503584073,
                          "x": 727.2728150283964,
                          "d": 94.06631342202337,
                          "map": 0},
             "moving_goal": 0,
             "estop": false,
             "staying_loc": 2}
}
'''

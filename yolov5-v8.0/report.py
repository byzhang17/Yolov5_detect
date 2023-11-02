#-*- coding: utf-8 -*-
import json
import matplotlib.pyplot as plt
from datetime import datetime
plt.rcParams['font.sans-serif'] = ['SimHei']


def CreateReport(report, passage, light, path):
    data = report['data']
    moving_goal = data['moving_goal']
    staying_loc = data['staying_loc']
    battery = data['battery']
    location = data['location']
    lifter = data['lifter']
    air = data['air']
    estop = data['estop']
    uptime = data['uptime']

    plt.subplot(4, 1, 1)
    air_row = ["温度", "湿度", "烟雾浓度"]
    air_col = ["数据"]
    air_data = [[air['temperature']], [air['humidity']], [air['smoke']]]
    plt.axis('off')
    plt.table(cellText=air_data,
         colLabels=air_col,
         rowLabels=air_row,
         cellLoc='center',
         rowLoc='center',
         loc="center")
    plt.title("空气质量")

    plt.subplot(4, 1, 2)
    battery_row = ["电量", "温度"]
    battery_col = ["数据"]
    battery_data = [[battery['power']], [battery['temperature']]]
    plt.axis('off')
    plt.table(cellText=battery_data,
              colLabels=battery_col,
              rowLabels=battery_row,
              cellLoc='center',
              rowLoc='center',
              loc="center")
    plt.title("电池状态")

    plt.subplot(4, 1, 3)
    robot_row = ["巡检开始时间", "机器人位置", "升降机构高度"]
    robot_col = ["数据"]
    robot_data = [[uptime], [(location['x'], location['y'], location['d'])], [lifter]]
    plt.axis('off')
    plt.table(cellText=robot_data,
              colLabels=robot_col,
              rowLabels=robot_row,
              cellLoc='center',
              rowLoc='center',
              loc="center")
    plt.title("机器人状态")

    plt.subplot(4, 1, 4)
    light_row = ["指示灯是否正常"]
    light_col = ["结果"]
    light_data = [[light]]
    plt.axis('off')
    plt.table(cellText=light_data,
              colLabels=light_col,
              rowLabels=light_row,
              cellLoc='center',
              rowLoc='center',
              loc="center")
    plt.title("机柜巡检结果", pad=1)
    timeNow = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    plt.suptitle("机房巡检报告({}号通道)".format(passage))
    plt.savefig('./{}/机房巡检报告({}号通道)_{}.jpg'.format(path, passage, timeNow), dpi=300, bbox_inches = 'tight')


'''
if __name__=="__main__":
    status = 
    report = json.loads(status, strict=False)
    CreateReport(report, passage=2, light=True)
    
    
    
    


if __name__=="__main__":
    status = {"msg": "", "code": 0, "data": {"uptime": "23/03/23/17/00/44", "battery": {"current": -2, "state": "Discharging", "temperature": 20, "power": 62, "voltage": 27}, "air": {"smoke": 12.0, "pm1": 0.0, "temperature": 26, "pm10": 0.0, "pm25":
            0.0, "humidity": 49.9}, "state": "StandingBy", "lifter": 0.1, "location": {"y": 317, "x": 727,
            "d": 94, "map": 0}, "moving_goal": 0, "estop": false, "staying_loc": 2}}
    report = json.loads(status, strict=False)
    CreateReport(report, passage=2, light=True)
'''
import threading
import time
import robotAPI

point_to_id = [[133, '1-西'], [134, '1-东'],
               [212, '2-西'], [171, '2-东'],
               [143, '3-西'], [136, '3-东'],
               [797, '4-西'], [798, '4-东'],
               [161, '5-西'], [160, '5-东'],
               [139, '6-西'], [138, '6-东']]

near_door_point_list = [2, 200, 201, 214, 215, 10, 1000, 1001, 1014, 1015,
                        4, 400, 401, 414, 415, 12, 1200, 1201, 1214, 1215,
                        7, 700, 701, 714, 715, 15, 1500, 1501, 1514, 1515]

door_point_to_id_dict = {2: [133, '2-西'], 200: [133, '2-西'], 201: [133, '2-西'],
                         214: [134, '2-东'], 215: [134, '2-东'],

                         10: [212, '1-西'], 1000: [212, '1-西'], 1001: [212, '1-西'],
                         1014: [171, '1-东'], 1015: [171, '1-东'],

                         4: [797, '4-西'], 400: [797, '4-西'], 401: [797, '4-西'],
                         414: [798, '4-东'], 415: [798, '4-东'],

                         12: [143, '3-西'], 1200: [143, '3-西'], 1201: [143, '3-西'],
                         1214: [136, '3-东'], 1215: [136, '3-东'],

                         7: [139, '6-西'], 700: [139, '6-西'], 701: [139, '6-西'],
                         714: [138, '6-东'], 715: [138, '6-东'],

                         15: [161, '5-西'], 1500: [161, '5-西'], 1501: [161, '5-西'],
                         1514: [160, '5-东'], 1515: [160, '5-东']
                         }



import detectFlow
import parameter

# robotAPI.moveToPoint([2], robotAPI.api_open_doors)
robotAPI.send_open_door(parameter.door_point_to_id_dict.get(400)[0],
                                    parameter.door_point_to_id_dict.get(400)[1])



'''
event = threading.Event()
openlist = [200, 201, 214, 215, 213, 1000, 1015, 400, 1200]
for point in openlist:
    if door_point_to_id_dict.get(point):
        print("send open door order...")
        robotAPI.send_open_door(door_point_to_id_dict.get(point)[0],
                                door_point_to_id_dict.get(point)[1])
    else:
        print("point :{} not found".format(point))
        event.wait(1)


if __name__ == '__main__':
    e = threading.Event()
    t1 = threading.Thread(target=openDoorThread, args=(e,))
    t1.start()
    time.sleep(10)
    e.set()
    print("close thread")
'''


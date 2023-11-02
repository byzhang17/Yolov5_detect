from queue import PriorityQueue as PQ
import math

class Anode:  # 边表节点类
    def __init__(self, adjvex, weight=0):
        self.Adjvex = adjvex  # 邻接点在顶点列表中的下标
        self.Next = None
        self.Weight = weight


class Vnode:  # 顶点表节点类
    def __init__(self, data):
        self.Data = data  # 顶点的值
        self.Firstarc = None  # 指向边表（单链表）的表头节点


class Graph:
    def __init__(self):
        self.vertList = []  # 表头列表
        self.numVertics = 0  # 实际顶点数

    def add_vertex(self, key):
        vertex = Vnode(key)
        self.vertList.append(vertex)
        self.numVertics = self.numVertics + 1
        return vertex

    def add_edge(self, val1, val2, weight=0):  # 在val1 顶点和val2节点之间添加一个权值为weight的边
        i = 0
        while i < len(self.vertList):  # 判断val1是否存在于顶点表中
            if val1 == self.vertList[i].Data:
                vnode1 = self.vertList[i]
                break
            i = i + 1
        if i == len(self.vertList):  # if不在，生成val1节点加入到顶点表中
            vnode1 = self.add_vertex(val1)

        i = 0
        while i < len(self.vertList):  # 判断val2是否存在于顶点表中
            if val2 == self.vertList[i].Data:
                vnode2 = self.vertList[i]
                break
            i = i + 1
        if i == len(self.vertList):  # if不在，生成val2节点加入到顶点表中
            vnode2 = self.add_vertex(val2)

        v2id = self.vertList.index(vnode2)
        p = Anode(v2id, weight)
        p.Next = vnode1.Firstarc  # 头插法
        vnode1.Firstarc = p
        # 将val2 加入到val1的边表中,采用头插法

    def route(self, val1, val2):
        i = 0
        while i < len(self.vertList):  # 判断val1是否存在于顶点表中
            if val1 == self.vertList[i].Data:
                vnode1 = self.vertList[i]
                break
            i = i + 1
        if i == len(self.vertList):  # if不在，输出None
            return None

        i = 0
        while i < len(self.vertList):  # 判断val2是否存在于顶点表中
            if val2 == self.vertList[i].Data:
                vnode2 = self.vertList[i]
                break
            i = i + 1
        if i == len(self.vertList):  # if不在，输出None
            return None

        v1id = self.vertList.index(vnode1)
        v2id = self.vertList.index(vnode2)
        route = dijksrta(self, v1id, v2id)
        if route is None:
            return False

        for i in range(len(route)):
            route[i] = self.vertList[route[i]].Data

        return route


def dijksrta(graph, val1, val2):
    pq = PQ()
    distance = [float('inf')] * graph.numVertics
    visited = [0] * graph.numVertics
    parent = [-1] * graph.numVertics

    distance[val1] = 0
    parent[val1] = val1
    pq.put((0, val1))

    while not pq.empty():
        node = pq.get()[1]
        if node == val2:
            break
        if visited[node]:
            continue
        else:
            visited[node] = 1
        neighbor = graph.vertList[node].Firstarc
        while neighbor is not None:
            if distance[node] + neighbor.Weight < distance[neighbor.Adjvex]:
                distance[neighbor.Adjvex] = distance[node] + neighbor.Weight
                parent[neighbor.Adjvex] = node
                pq.put((distance[neighbor.Adjvex], neighbor.Adjvex))
            neighbor = neighbor.Next

    if parent[val2] == -1:
        return None
    route = []
    route_node = val2
    while route_node != parent[route_node]:
        route.insert(0, route_node)
        route_node = parent[route_node]
    route.insert(0, val1)

    return route


def findBestRoute(source, target):
    graph = Graph()
    graph.add_vertex('-1')
    for i in range(1, 17):
        for j in range(16):
            if j < 10:
                graph.add_vertex(str(i)+'0'+str(j))
            else:
                graph.add_vertex(str(i)+str(j))

    graph.add_edge('-1', '100', 1)
    graph.add_edge('100', '-1', 1)
    graph.add_edge('-1', '200', 1)
    graph.add_edge('200', '-1', 1)
    for i in range(1, 17):
        for j in range(15):
            if j < 9:
                graph.add_edge(str(i)+'0'+str(j), str(i)+'0'+str(j+1), 1)
                graph.add_edge(str(i)+'0'+str(j+1), str(i)+'0'+str(j), 1)
            elif j == 9:
                graph.add_edge(str(i)+'0'+str(j), str(i)+str(j+1), 1)
                graph.add_edge(str(i)+str(j+1), str(i)+'0'+str(j), 1)
            else:
                graph.add_edge(str(i)+str(j), str(i)+str(j+1), 1)
                graph.add_edge(str(i)+str(j+1), str(i)+str(j), 1)

    for i in range(1, 8):
        graph.add_edge(str(i)+'00', str(i+1)+'00', 1)
        graph.add_edge(str(i+1)+'00', str(i)+'00', 1)

    for i in range(1, 9):
        graph.add_edge(str(i)+'15', str(i+8)+'00', 1)
        graph.add_edge(str(i+8)+'00', str(i)+'15', 1)

    res = graph.route(str(source), str(target))
    print("起始点：{}, 目标点：{}, 路线： {}".format(source, target, res))
    if res is None:
        return [2, -1]
    route_list_int = list(map(int, res))
    return route_list_int


def queryBackRoute(p):
    if 1015 >= p >= 1000:
        return [1000, 215, 200, -1]
    elif 215 >= p >= 200:
        return [200, -1]
    elif 315 >= p >= 400:
        return [400, -1]
    elif 1215 >= p >= 1200:
        return [1200, 415, 400, -1]
    elif 715 >= p >= 700:
        return [700, 200, -1]
    elif 1515 >= p >= 1500:
        return [1500, 715, 700, 200, -1]
    else:
        return [2, -1]







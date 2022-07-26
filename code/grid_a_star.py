from sklearn.neighbors import KDTree
import numpy as np
import matplotlib.pyplot as plt
import heapq

class Node:     #节点类
    def __init__(self, x=0, y=0, cost=0.0, pind=-1):
        self.x = x
        self.y = y
        self.cost = cost
        self.pind = pind

def calc_obstacle_map(ox=np.array(()), oy=np.array(()), reso=0.0, vr=0.0):      #构建障碍物地图
    minx = int(round(min(ox)))      #最小x坐标
    miny = int(round(min(oy)))      #最小y坐标
    maxx = int(round(max(ox)))      #最大x坐标
    maxy = int(round(max(oy)))      #最大y坐标

    xwidth = int(round(maxx - minx))    #地图x宽度
    ywidth = int(round(maxy - miny))    #地图y宽度

    obmap = np.zeros((xwidth, ywidth))

    kdtree = KDTree(np.hstack((ox.reshape(-1, 1), oy.reshape(-1, 1))), leaf_size=10)
    for ix in range(xwidth):
        x = ix + minx
        for iy in range(ywidth):
            y = iy + miny
            onedist, idxs = kdtree.query(np.array([x, y]).reshape(1, -1), 1)
            if onedist[0] <= vr / reso:
                #obmap[ix-1, iy-1] = 1
                obmap[ix, iy] = 1       #有障碍物

    return obmap, minx, miny, maxx, maxy, xwidth, ywidth

def calc_index(node=Node(), xwidth=0, xmin=0, ymin=0):      #计算节点序号
    return (node.y - ymin)*xwidth + (node.x - xmin)

def get_motion_model():         #构建8种运动方式
    motion = np.array([[1, 0, 1],
                       [0, 1, 1],
                       [-1, 0, 1],
                       [0, -1, 1],
                       [-1, -1, np.sqrt(2)],
                       [-1, 1, np.sqrt(2)],
                       [1, -1, np.sqrt(2)],
                       [1, 1, np.sqrt(2)]])
    return motion

def verify_node(node=Node(), minx=0, miny=0, xw=0, yw=0, obmap=np.array(())):       #检测节点是否合理
    if (node.x - minx) >= xw:       #检测不超过地图大小
        return False
    elif (node.x - minx) <= 0:
        return False

    if (node.y - miny) >= yw:
        return False
    elif (node.y - miny) <= 0:
        return False

    if obmap[int(node.x - minx), int(node.y - miny)]:       #检测非障碍物节点
    #if obmap[int(node.x-minx)-1, int(node.y-miny)-1]:
        return False

    return True

def calc_policy_map(closed={}, xw=0, yw=0, minx=0, miny=0):     #返回所有节点到目标节点的A*距离
    pmap = np.inf * np.ones((xw, yw))

    for n in closed.values():
        #pmap[int(n.x-minx)-1, int(n.y-miny)-1] = n.cost
        pmap[int(n.x - minx), int(n.y - miny)] = n.cost

    return pmap

def calc_dist_policy(gx=0.0, gy=0.0, ox=[], oy=[], reso=0.0, vr=0.0):       #离线计算A*
    ngoal = Node(int(round(gx / reso)), int(round(gy / reso)), 0.0, -1)

    ox = [iox / reso for iox in ox]
    oy = [ioy / reso for ioy in oy]

    ox = np.array(ox)
    oy = np.array(oy)

    obmap, minx, miny, maxx, maxy, xw, yw = calc_obstacle_map(ox, oy, reso, vr)

    open, closed = {}, {}
    open[calc_index(ngoal, xw, minx, miny)] = ngoal

    motion = get_motion_model()
    nmotion = motion.shape[0]
    pq = []
    heapq.heappush(pq, (ngoal.cost, calc_index(ngoal, xw, minx, miny)))

    while(1):
        if len(open) == 0:
            print("Finish Search")
            break

        _, c_id = heapq.heappop(pq)
        current = open[c_id]

        del open[c_id]
        closed[c_id] = current

        for i in range(nmotion):        #扩展节点 遍历所有运动方式
            node = Node(current.x+motion[i, 0], current.y+motion[i, 1], current.cost+motion[i, 2], c_id)

            if not verify_node(node, minx, miny, xw, yw, obmap):
                continue

            node_ind = calc_index(node, xw, minx, miny)

            if node_ind in closed:
                continue
            if node_ind in open:
                if open[node_ind].cost > node.cost:
                    open[node_ind].cost = node.cost
                    open[node_ind].pind = c_id
            else:
                open[node_ind] = node
                heapq.heappush(pq, (node.cost, calc_index(node, xw, minx, miny)))

    pmap = calc_policy_map(closed, xw, yw, minx, miny)

    return pmap

def calc_cost(n=Node(), ngoal=Node()):      #路径代价
    return (n.cost + h(n.x - ngoal.x, n.y - ngoal.y))

def h(x=0, y=0):        #估计h函数
    return np.sqrt(x**2 + y**2)

def calc_astar_path(sx=0.0, sy=0.0, gx=0.0, gy=0.0, ox=[], oy=[], reso=0.0, vr=0.0):    #计算A*
    nstart = Node(int(round(sx / reso)), int(round(sy / reso)), 0.0, -1)
    ngoal = Node(int(round(gx / reso)), int(round(gy / reso)), 0.0, -1)

    ox = [iox / reso for iox in ox]
    oy = [ioy / reso for ioy in oy]

    ox = np.array(ox)
    oy = np.array(oy)

    obmap, minx, miny, maxx, maxy, xw, yw = calc_obstacle_map(ox, oy, reso, vr)

    open, closed = {}, {}
    open[calc_index(nstart, xw, minx, miny)] = nstart

    motion = get_motion_model()
    nmotion = motion.shape[0]
    pq = []
    heapq.heappush(pq, (calc_cost(nstart, ngoal), calc_index(nstart, xw, minx, miny)))

    while(1):
        if len(open) == 0:
            print("Error: No open set")
            break

        _, c_id = heapq.heappop(pq)
        current = open[c_id]

        if current.x == ngoal.x and current.y == ngoal.y:
            print("Goal")
            closed[c_id] = current
            break

        del open[c_id]
        closed[c_id] = current

        for i in range(nmotion):        #扩展节点 遍历所有运动方式
            node = Node(current.x+motion[i, 0], current.y+motion[i, 1], current.cost+motion[i, 2], c_id)

            if not verify_node(node, minx, miny, xw, yw, obmap):
                continue

            node_ind = calc_index(node, xw, minx, miny)

            if node_ind in closed:
                continue
            if node_ind in open:
                if open[node_ind].cost > node.cost:
                    open[node_ind].cost = node.cost
                    open[node_ind].pind = c_id
            else:
                open[node_ind] = node
                heapq.heappush(pq, (calc_cost(node, ngoal), calc_index(node, xw, minx, miny)))

    rx, ry = get_final_path(closed, ngoal, nstart, xw, minx, miny, reso)

    return rx, ry

def get_final_path(closed={}, ngoal=Node(), nstart=Node(), xw=0, minx=0, miny=0, reso=0.0):     #返回路径
    rx, ry = [ngoal.x], [ngoal.y]
    nid = calc_index(ngoal, xw, minx, miny)
    while(1):
        n = closed[nid]
        rx.append(n.x)
        ry.append(n.y)
        nid = n.pind

        if rx[-1] == nstart.x and ry[-1] == nstart.y:
            print("done")
            break

    rx.reverse()
    ry.reverse()

    rx = np.array(rx) * reso
    ry = np.array(ry) * reso

    return rx.tolist(), ry.tolist()

def search_min_cost_node(open={}, ngoal=Node()):        #寻找最小代价节点
    mnode = None
    mcost = np.inf
    for n in open.values():
        cost = n.cost + h(n.x - ngoal.x, n.y - ngoal.y)
        if mcost > cost:
            mnode = n
            mcost = cost

    return mnode

if __name__ == "__main__":      #测试主函数
    print("start")

    sx, sy = 10.0, 10.0
    gx, gy = 50.0, 50.0

    ox = []
    oy = []

    for i in range(61):
        ox.append(float(i))
        oy.append(0.0)
    for i in range(61):
        ox.append(60.0)
        oy.append(float(i))
    for i in range(61):
        ox.append(float(i))
        oy.append(60.0)
    for i in range(61):
        ox.append(0.0)
        oy.append(float(i))
    for i in range(41):
        ox.append(20.0)
        oy.append(float(i))
    for i in range(41):
        ox.append(40.0)
        oy.append(60.0-float(i))

    VEHICLE_RADIUS = 5.0
    GRID_RESOLUTION = 1.0

    rx, ry = calc_astar_path(sx, sy, gx, gy, ox, oy, GRID_RESOLUTION, VEHICLE_RADIUS)

    plt.plot(ox, oy, '.k')
    plt.plot(sx, sy, 'ro')
    plt.plot(gx, gy, 'rx')
    plt.plot(rx, ry, '-b')
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.show()
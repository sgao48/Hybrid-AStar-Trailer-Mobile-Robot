import matplotlib.pyplot as plt
import heapq
from sklearn.neighbors import KDTree
import rs_path
import grid_a_star
import trailerlib
import numpy as np

XY_GRID_RESOLUTION = 2.0    #栅格分辨率
YAW_GRID_RESOLUTION = np.radians(15.0)      #前车角度分辨率
GOAL_TYAW_TH = np.radians(5.0)      #目标误差容限
MOTION_RESOLUTION = 0.1     #扩展节点分辨率
N_STEER = 20.0      #转向角数量
EXTEND_AREA = 5.0   #地图扩展
SKIP_COLLISION_CHECK = 4    #碰撞检测分辨率

SB_COST = 100.0     #反转代价
BACK_COST = 5.0     #后退代价
STEER_CHANGE_COST = 5.0     #转向角改变代价
STEER_COST = 1.0    #转向角代价
JACKKNIF_COST = 200.0   #前后车转角差代价
H_COST = 5.0        #A*代价系数

WB = trailerlib.WB
LT = trailerlib.LT
MAX_STEER = trailerlib.MAX_STEER

class Node:     #节点类
    def __init__(self, xind=0, yind=0, yawind=0, direction=True, x=[], y=[],
                 yaw=[], yaw1=[], directions=[], steer=0.0, cost=0.0, pind=-1):
        self.xind = xind
        self.yind = yind
        self.yawind = yawind
        self.direction = direction
        self.x = x
        self.y = y
        self.yaw = yaw
        self.yaw1 = yaw1
        self.directions = directions
        self.steer = steer
        self.cost = cost
        self.pind = pind

class Config:       #地图配置
    def __init__(self, minx=0, miny=0, minyaw=0, minyawt=0, maxx=0, maxy=0,
                 maxyaw=0, maxyawt=0, xw=0, yw=0, yaww=0, yawtw=0, xyreso=0.0, yawreso=0.0):
        self.minx = minx
        self.miny = miny
        self.minyaw = minyaw
        self.minyawt = minyawt
        self.maxx = maxx
        self.maxy = maxy
        self.maxyaw = maxyaw
        self.maxyawt = maxyawt
        self.xw = xw
        self.yw = yw
        self.yaww = yaww
        self.yawtw = yawtw
        self.xyreso = xyreso
        self.yawreso = yawreso

class Path:     #路径类
    def __init__(self, x=[], y=[], yaw=[], yaw1=[], direction=[], cost=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.yaw1 = yaw1
        self.direction = direction
        self.cost = cost

def calc_config(ox=[], oy=[], xyreso=0.0, yawreso=0.0):     #计算地图属性
    min_x_m = min(ox) - EXTEND_AREA
    min_y_m = min(oy) - EXTEND_AREA
    max_x_m = max(ox) + EXTEND_AREA
    max_y_m = max(oy) + EXTEND_AREA

    ox.append(min_x_m)
    oy.append(min_y_m)
    ox.append(max_x_m)
    oy.append(max_y_m)

    minx = int(round(min_x_m / xyreso))
    miny = int(round(min_y_m / xyreso))
    maxx = int(round(max_x_m / xyreso))
    maxy = int(round(max_y_m / xyreso))

    xw = int(round(maxx - minx))
    yw = int(round(maxy - miny))

    minyaw = int(round(-np.pi / yawreso)) - 1
    maxyaw = int(round(np.pi / yawreso))
    yaww = int(round(maxyaw - minyaw))

    minyawt = minyaw
    maxyawt = maxyaw
    yawtw = yaww

    config = Config(minx, miny, minyaw, minyawt, maxx, maxy, maxyaw, maxyawt,
                    xw, yw, yaww, yawtw, xyreso, yawreso)

    return config

def pi_2_pi(angle=0.0):     #转换角至[-pi, pi]
    while angle > np.pi:
        angle -= 2.0 * np.pi
    while angle < -np.pi:
        angle += 2.0 * np.pi
    return angle

def calc_holonomic_with_obstacle_heuristic(gnode=None, ox=[], oy=[], xyreso=0.0):       #计算有障碍完整性约束代价
    h_dp = grid_a_star.calc_dist_policy(gnode.x[-1], gnode.y[-1], ox, oy, xyreso, 1.0)
    return h_dp

def calc_index(node=None, c=None):      #计算节点序号
    ind = (node.yawind - c.minyaw)*c.xw*c.yw + (node.yind - c.miny)*c.xw + (node.xind - c.minx)

    yaw1ind = int(round(node.yaw1[-1] / c.yawreso))
    ind += (yaw1ind - c.minyawt)*c.xw*c.yw*c.yaww

    if ind <= 0:
        print("Error(calc_index):", ind)

    return ind

def calc_cost(n=None, h_dp=np.array(()), ngoal=None, c=None):       #计算估计函数代价
    #return n.cost + H_COST*h_dp[n.xind - c.minx - 1, n.yind - c.miny - 1]
    return n.cost + H_COST * h_dp[n.xind - c.minx, n.yind - c.miny]

def calc_motion_inputs():       #扩展节点
    up = [i for i in np.arange(MAX_STEER/N_STEER, MAX_STEER+MAX_STEER/N_STEER, MAX_STEER/N_STEER)]      #转向

    u = np.hstack((np.zeros((1, 1)), np.array([i for i in up]).reshape(1, -1), np.array([-i for i in up]).reshape(1, -1)))      #左右转向
    d = np.hstack((np.array([1.0 for i in range(u.shape[1])]).reshape(1, -1), np.array([-1.0 for i in range(u.shape[1])]).reshape(1, -1)))      #前进或后退
    u = np.hstack((u, u))

    return u, d

def update_node_with_analystic_expantion(current=None, ngoal=None, c=None, ox=[], oy=[], kdtree=None, gyaw1=0.0):       #Reeds-Shepp曲线扩展
    apath = analystic_expantion(current, ngoal, c, ox, oy, kdtree)

    if apath != None:       #测试节点是否满足要求
        fx = apath.x[1:]
        fy = apath.y[1:]
        fyaw = apath.yaw[1:]
        steps = (MOTION_RESOLUTION * np.array(apath.directions)).tolist()
        yaw1 = trailerlib.calc_trailer_yaw_from_xyyaw(apath.x, apath.y, apath.yaw, current.yaw1[-1], steps)
        if abs(pi_2_pi(yaw1[-1] - gyaw1)) >= GOAL_TYAW_TH:      #满足目标姿态要求
            return False, None
        fcost = current.cost + calc_rs_path_cost(apath, yaw1)   #计算节点属性
        fyaw1 = yaw1[1:]
        fpind = calc_index(current, c)

        fd = []
        for d in apath.directions[1:]:
            if d >= 0:
                fd.append(True)
            else:
                fd.append(False)

        fsteer = 0.0

        fpath = Node(current.xind, current.yind, current.yawind, current.direction, fx, fy,
                     fyaw, fyaw1, fd, fsteer, fcost, fpind)

        return True, fpath

    return False, None

def analystic_expantion(n=None, ngoal=None, c=None, ox=[], oy=[], kdtree=None):     #Reeds-Shepp扩展
    sx = n.x[-1]
    sy = n.y[-1]
    syaw = n.yaw[-1]

    max_curvature = np.tan(MAX_STEER) / WB
    paths = rs_path.calc_paths(sx, sy, syaw, ngoal.x[-1], ngoal.y[-1], ngoal.yaw[-1],       #计算所有Reeds-Shepp曲线
                               max_curvature, step_size=MOTION_RESOLUTION)
    '''
    plt.plot(ox, oy, ".k")
    trailerlib.plot_trailer(sx, sy, syaw0, syaw1, 0.0)
    trailerlib.plot_trailer(gx, gy, gyaw0, gyaw1, 0.0)
    '''
    if len(paths) == 0:
        return None

    pathqueue = []
    for path in paths:      #计算路径代价
        steps = (MOTION_RESOLUTION * np.array(path.directions)).tolist()
        yaw1 = trailerlib.calc_trailer_yaw_from_xyyaw(path.x, path.y, path.yaw, n.yaw1[-1], steps)
        heapq.heappush(pathqueue, (calc_rs_path_cost(path, yaw1), path))

    for i in range(len(pathqueue)):
        _, path = heapq.heappop(pathqueue)

        steps = (MOTION_RESOLUTION * np.array(path.directions)).tolist()
        yaw1 = trailerlib.calc_trailer_yaw_from_xyyaw(path.x, path.y, path.yaw, n.yaw1[-1], steps)
        ind = [i for i in range(0, len(path.x), SKIP_COLLISION_CHECK)]
        if trailerlib.check_trailer_collision(ox, oy, (np.array(path.x)[ind]).tolist(), (np.array(path.y)[ind]).tolist(),       #碰撞检测
                                              (np.array(path.yaw)[ind]).tolist(), (np.array(yaw1)[ind]).tolist(), kdtree=kdtree):
            #plt.plot(path.x, path.y, "-^b")
            #plt.show()
            return path

    return None

def calc_rs_path_cost(rspath=rs_path.Path(), yaw1=0.0):     #计算Reeds-Shepp曲线路径代价
    cost = 0.0
    for l in rspath.lengths:        #倒车代价
        if l >= 0:
            cost += l
        else:
            cost += abs(l) * BACK_COST

    for i in range(len(rspath.lengths)-1):      #反转行车方向代价
        if rspath.lengths[i] * rspath.lengths[i+1] < 0.0:
            cost += SB_COST

    for ctype in rspath.ctypes:     #转向角代价
        if ctype != "S":
            cost += STEER_COST * abs(MAX_STEER)

    nctypes = len(rspath.ctypes)
    ulist = [0.0] * nctypes
    for i in range(nctypes):        #转向角改变定义
        if rspath.ctypes[i] == "R":
            ulist[i] = -MAX_STEER
        elif rspath.ctypes[i] == "L":
            ulist[i] = MAX_STEER

    for i in range(len(rspath.ctypes)-1):       #转向角改变代价
        cost += STEER_CHANGE_COST * abs(ulist[i+1] - ulist[i])

    cost += JACKKNIF_COST * np.sum(np.abs(np.array(rs_path.pi_2_pi((np.array(rspath.yaw) - yaw1).tolist()))))   #前后车姿态差别代价

    return cost

def calc_next_node(current=None, c_id=0, u=0.0, d=0.0, c=None):     #扩展下一节点
    arc_l = XY_GRID_RESOLUTION * 1.5

    nlist = int(np.floor(arc_l/MOTION_RESOLUTION)) + 1
    xlist = [0.0] * nlist
    ylist = [0.0] * nlist
    yawlist = [0.0] * nlist
    yaw1list = [0.0] * nlist

    xlist[0] = current.x[-1] + d * MOTION_RESOLUTION * np.cos(current.yaw[-1])      #计算初始小车四维位姿属性
    ylist[0] = current.y[-1] + d * MOTION_RESOLUTION * np.sin(current.yaw[-1])
    yawlist[0] = pi_2_pi(current.yaw[-1] + d * MOTION_RESOLUTION/WB * np.tan(u))
    yaw1list[0] = pi_2_pi(current.yaw1[-1] + d * MOTION_RESOLUTION/LT * np.sin(current.yaw[-1]-current.yaw1[-1]))

    for i in range(nlist-1):        #计算路径上所有位姿
        xlist[i+1] = xlist[i] + d * MOTION_RESOLUTION * np.cos(yawlist[i])
        ylist[i+1] = ylist[i] + d * MOTION_RESOLUTION * np.sin(yawlist[i])
        yawlist[i+1] = pi_2_pi(yawlist[i] + d * MOTION_RESOLUTION/WB * np.tan(u))
        yaw1list[i+1] = pi_2_pi(yaw1list[i] + d * MOTION_RESOLUTION/LT * np.sin(yawlist[i]-yaw1list[i]))

    xind = int(round(xlist[-1]/c.xyreso))       #计算位姿属性离散坐标
    yind = int(round(ylist[-1]/c.xyreso))
    yawind = int(round(yawlist[-1]/c.yawreso))

    addedcost = 0.0     #计算路径代价
    if d > 0:
        direction = True
        addedcost += abs(arc_l)
    else:
        direction = False
        addedcost += abs(arc_l) * BACK_COST

    if direction != current.direction:
        addedcost += SB_COST

    addedcost += STEER_COST * abs(u)
    addedcost += STEER_CHANGE_COST * abs(current.steer - u)
    delta = np.array(yawlist) - np.array(yaw1list)
    addedcost += JACKKNIF_COST * np.sum(np.abs(np.array(rs_path.pi_2_pi(delta.tolist()))))

    cost = current.cost + addedcost

    directions = [direction for i in range(len(xlist))]
    node = Node(xind, yind, yawind, direction, xlist, ylist, yawlist, yaw1list,
                directions, u, cost, c_id)

    return node

def verify_index(node=None, c=None, ox=[], oy=[], inityaw1=0.0, kdtree=None):       #检测节点是否符合要求
    if (node.xind - c.minx) >= c.xw:        #检测是否超界
        return False
    elif (node.xind - c.minx) <= 0:
        return False

    if (node.yind - c.miny) >= c.yw:
        return False
    elif (node.yind - c.miny) <= 0:
        return False

    steps = (MOTION_RESOLUTION * np.array(node.directions)).tolist()
    yaw1 = trailerlib.calc_trailer_yaw_from_xyyaw(node.x, node.y, node.yaw, inityaw1, steps)
    ind = [i for i in range(0, len(node.x), SKIP_COLLISION_CHECK)]
    if not trailerlib.check_trailer_collision(ox, oy, (np.array(node.x)[ind]).tolist(), (np.array(node.y)[ind]).tolist(),       #碰撞检测
                                            (np.array(node.yaw)[ind]).tolist(), (np.array(yaw1)[ind]).tolist(),
                                          kdtree=kdtree):
        return False

    return True

def calc_hybrid_astar_path(sx=0.0, sy=0.0, syaw=0.0, syaw1=0.0, gx=0.0, gy=0.0, gyaw=0.0,       #计算Hybrid A*路径
                           gyaw1=0.0, ox=[], oy=[], xyreso=0.0, yawreso=0.0):
    syaw, gyaw = pi_2_pi(syaw), pi_2_pi(gyaw)
    oox, ooy = np.array(ox), np.array(oy)

    kdtree = KDTree(np.hstack((oox.reshape(-1, 1), ooy.reshape(-1, 1))), leaf_size=10)

    c = calc_config(ox, oy, xyreso, yawreso)
    nstart = Node(int(round(sx/xyreso)), int(round(sy/xyreso)), int(round(syaw/yawreso)), True,     #起始点
                  [sx], [sy], [syaw], [syaw1], [True], 0.0, 0.0, -1)
    ngoal = Node(int(round(gx/xyreso)), int(round(gy/xyreso)), int(round(gyaw/yawreso)), True,      #终止点
                 [gx], [gy], [gyaw], [gyaw1], [True], 0.0, 0.0, -1)

    h_dp = calc_holonomic_with_obstacle_heuristic(ngoal, ox, oy, xyreso)        #离线A*距离

    open, closed = {}, {}
    fnode = None
    open[calc_index(nstart, c)] = nstart
    pq = []
    heapq.heappush(pq, (calc_cost(nstart, h_dp, ngoal, c), calc_index(nstart, c)))

    u, d = calc_motion_inputs()     #扩展节点方式
    u = u.tolist()[0]
    d = d.tolist()[0]
    nmotion = len(u)

    j = 0
    while(1):
        print(j)
        if len(open) == 0:
            print("Error: Cannot find path, No open set")
            return []

        _, c_id = heapq.heappop(pq)
        current = open[c_id]

        del open[c_id]
        closed[c_id] = current

        isupdated, fpath = update_node_with_analystic_expantion(current, ngoal, c, ox, oy, kdtree, gyaw1)       #Reeds-Shepp扩展
        if isupdated:
            fnode = fpath
            break

        inityaw1 = current.yaw1[0]

        for i in range(nmotion):        #节点扩展
            node = calc_next_node(current, c_id, u[i], d[i], c)

            if not verify_index(node, c, ox, oy, inityaw1, kdtree):     #节点是否符合要求
                continue

            node_ind = calc_index(node, c)

            if node_ind in closed:
                continue
            if not (node_ind in open):
                open[node_ind] = node
                heapq.heappush(pq, (calc_cost(node, h_dp, ngoal, c), node_ind))
            else:
                if open[node_ind].cost > node.cost:
                    open[node_ind] = node
        j += 1

    print("final expand node:", len(open) + len(closed))

    path = get_final_path(closed, fnode, nstart, c)

    return path

def get_final_path(closed={}, ngoal=None, nstart=None, c=None):     #计算最终路径
    ngoal.x.reverse()
    rx = ngoal.x
    ngoal.y.reverse()
    ry = ngoal.y
    ngoal.yaw.reverse()
    ryaw = ngoal.yaw
    ngoal.yaw1.reverse()
    ryaw1 = ngoal.yaw1
    ngoal.directions.reverse()
    direction = ngoal.directions
    nid = ngoal.pind
    finalcost = ngoal.cost

    while (1):
        n = closed[nid]
        n.x.reverse()
        rx.extend(n.x)
        n.y.reverse()
        ry.extend(n.y)
        n.yaw.reverse()
        ryaw.extend(n.yaw)
        n.yaw1.reverse()
        ryaw1.extend(n.yaw1)
        n.directions.reverse()
        direction.extend(n.directions)
        nid = n.pind
        if is_same_grid(n, nstart):
            break

    rx.reverse()
    ry.reverse()
    ryaw.reverse()
    ryaw1.reverse()
    direction.reverse()

    direction[0] = direction[1]

    path = Path(rx, ry, ryaw, ryaw1, direction, finalcost)

    return path

def is_same_grid(node1=None, node2=None):       #检测两节点是否为同一节点
    if node1.xind != node2.xind:
        return False
    if node1.yind != node2.yind:
        return False
    if node1.yawind != node2.yawind:
        return False

    return True

if __name__ == "__main__":      #测试主函数
    print("start")

    sx = 14.0
    sy = 10.0
    syaw0 = np.radians(0.0)
    syaw1 = np.radians(0.0)

    gx, gy = 0.0, 0.0
    gyaw0 = np.radians(90.0)
    gyaw1 = np.radians(90.0)

    ox, oy = [], []

    for i in range(-25, 26):
        ox.append(float(i))
        oy.append(15.0)
    for i in range(-25, -3):
        ox.append(float(i))
        oy.append(4.0)
    for i in range(-15, 5):
        ox.append(-4.0)
        oy.append(float(i))
    for i in range(-15, 5):
        ox.append(4.0)
        oy.append(float(i))
    for i in range(4, 26):
        ox.append(float(i))
        oy.append(4.0)
    for i in range(-4, 5):
        ox.append(float(i))
        oy.append(-15.0)

    path = calc_hybrid_astar_path(sx, sy, syaw0, syaw1, gx, gy, gyaw0, gyaw1,
                                  ox, oy, XY_GRID_RESOLUTION, YAW_GRID_RESOLUTION)

    plt.plot(ox, oy, ".k")
    trailerlib.plot_trailer(sx, sy, syaw0, syaw1, 0.0)
    trailerlib.plot_trailer(gx, gy, gyaw0, gyaw1, 0.0)
    x = path.x
    y = path.y
    yaw = path.yaw
    yaw1 = path.yaw1
    direction = path.direction

    steer = 0.0
    for ii in range(len(x)):
        plt.cla()
        plt.plot(ox, oy, ".k")
        plt.plot(x, y, "-r", label="Hybrid A* path")

        if ii < len(x)-2:
            k = (yaw[ii+1] - yaw[ii]) / MOTION_RESOLUTION
            if not direction[ii]:
                k *= -1

            steer = np.arctan2(WB*k, 1.0)
        else:
            steer = 0.0

        trailerlib.plot_trailer(x[ii], y[ii], yaw[ii], yaw1[ii], steer)
        plt.grid(True)
        plt.axis("equal")
        plt.pause(0.0001)

    print("Done")
    plt.axis("equal")
    plt.show()
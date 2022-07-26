from sklearn.neighbors import KDTree
import numpy as np
import matplotlib.pyplot as plt

# Vehicle Parameter
WB = 3.7
LT = 8.0
W = 2.6
LF = 4.5
LB = 1.0
LTF = 1.0
LTB = 9.0
MAX_STEER = 0.6
TR = 0.5
TW = 1.0

# Collision Check
WBUBBLE_DIST = 3.5
WBUBBLE_R = 10.0
B = 4.45
C = 11.54
I = 8.55
VRX = [C, C, -B, -B, C]
VRY = [-I/2.0, I/2.0, I/2.0, -I/2.0, -I/2.0]

def rect_check(ix=0.0, iy=0.0, iyaw=0.0, ox=[], oy=[], vrx=[], vry=[]):     #矩形碰撞检测
    c = np.cos(-iyaw)
    s = np.sin(-iyaw)

    for (iox, ioy) in zip(ox, oy):
        tx = iox - ix
        ty = ioy - iy
        lx = (c*tx - s*ty)
        ly = (s*tx + c*ty)

        sumangle = 0.0
        for i in range(len(vrx)-1):
            x1 = vrx[i] - lx
            y1 = vry[i] - ly
            x2 = vrx[i+1] - lx
            y2 = vry[i+1] - ly
            d1 = np.sqrt(x1**2+y1**2)
            d2 = np.sqrt(x2**2+y2**2)
            theta1 = np.arctan2(y1, x1)
            tty = -np.sin(theta1) * x2 + np.cos(theta1) * y2
            tmp = (x1 * x2 + y1 * y2) / (d1 * d2)

            if tmp >= 1.0:
                tmp = 1.0
            elif tmp <= 0.0:
                tmp = 0.0

            if tty >= 0.0:
                sumangle += np.arccos(tmp)
            else:
                sumangle -= np.arccos(tmp)

        if sumangle >= np.pi:
            return False

    return True

def check_collision(x=[], y=[], yaw=[], kdtree=None, ox=[], oy=[],      #碰撞检测
                    wbd=0.0, wbr=0.0, vrx=[], vry=[]):
    for (ix, iy, iyaw) in zip(x, y, yaw):
        cx = ix + wbd * np.cos(iyaw)
        cy = iy + wbd * np.sin(iyaw)

        ids, _ = kdtree.query_radius(np.array([cx, cy]).reshape(1, -1), r=wbr,      #寻找离目标点坐标一定距离的障碍物坐标点
                                    return_distance=True, sort_results=True)
        ids = ids[0]
        if len(ids) == 0:
            continue

        oox = np.array(ox)[ids]     #障碍物x坐标
        ooy = np.array(oy)[ids]     #障碍物y坐标

        if not rect_check(ix, iy, iyaw, oox.tolist(), ooy.tolist(), vrx, vry):      #矩形碰撞检测
            return False

        #if not rect_check(ix, iy, iyaw, ox, oy, vrx, vry):
        #    return False

    return True

def calc_trailer_yaw_from_xyyaw(x=[], y=[], yaw=[], init_tyaw=0.0, steps=[]):       #计算拖车角度
    tyaw = [0.0] * len(x)
    tyaw[0] = init_tyaw

    for i in range(1, len(x)):
        tyaw[i] += tyaw[i-1] + steps[i-1] / LT * np.sin(yaw[i-1] - tyaw[i-1])

    return tyaw

def trailer_motion_model(x, y, yaw0, yaw1, D, d, L, delta):     #拖车运动学模型
    x += D * np.cos(yaw0)
    y += D * np.sin(yaw0)
    yaw0 += D / L * np.tan(delta)
    yaw1 += D / d * np.sin(yaw0 - yaw1)

    return x, y, yaw0, yaw1

def check_trailer_collision(ox=[], oy=[], x=[], y=[], yaw0=[], yaw1=[], kdtree=None):       #拖车碰撞检测
    oox = np.array(ox)
    ooy = np.array(oy)

    if kdtree == None:
        kdtree = KDTree(np.hstack((oox.reshape(-1, 1), ooy.reshape(-1, 1))), leaf_size=10)

    vrxt = [LTF, LTF, -LTB, -LTB, LTF]
    vryt = [-W/2.0, W/2.0, W/2.0, -W/2.0, -W/2.0]

    DT = (LTF + LTB) / 2.0 - LTB
    DTR = (LTF + LTB) / 2.0 + 0.3

    if not check_collision(x, y, yaw1, kdtree, ox, oy, DT, DTR, vrxt, vryt):        #拖车检测
        return False

    vrxf = [LF, LF, -LB, -LB, LF]
    vryf = [-W/2.0, W/2.0, W/2.0, -W/2.0, -W/2.0]

    DF = (LF + LB) / 2.0 - LB
    DFR = (LF + LB) / 2.0 + 0.3

    if not check_collision(x, y, yaw0, kdtree, ox, oy, DF, DFR, vrxf, vryf):        #前车检测
        return False

    for i in range(len(yaw0)):      #检测两车互相是否碰撞
        if abs(yaw0[i] - yaw1[i]) > np.pi/2.0:
            return False

    return True

def plot_trailer(x=0.0, y=0.0, yaw=0.0, yaw1=0.0, steer=0.0):
    truckcolor = "-k"

    LENGTH = LB + LF
    LENGTHt = LTB + LTF

    truckOutLine = np.array([[-LB, LENGTH-LB, LENGTH-LB, -LB, -LB],
                             [W/2, W/2, -W/2, -W/2, W/2]])
    trailerOutLine = np.array([[-LTB, LENGTHt-LTB, LENGTHt-LTB, -LTB, -LTB],
                               [W/2, W/2, -W/2, -W/2, W/2]])

    rr_wheel = np.array([[TR, -TR, -TR, TR, TR],
                         [-W/12.0+TW, -W/12.0+TW, W/12.0+TW, W/12.0+TW, -W/12.0+TW]])
    rl_wheel = np.array([[TR, -TR, -TR, TR, TR],
                         [-W/12.0-TW, -W/12.0-TW, W/12.0-TW, W/12.0-TW, -W/12.0-TW]])
    fr_wheel = np.array([[TR, -TR, -TR, TR, TR],
                         [-W/12.0+TW, -W/12.0+TW, W/12.0+TW, W/12.0+TW, -W/12.0+TW]])
    fl_wheel = np.array([[TR, -TR, -TR, TR, TR],
                         [-W/12.0-TW, -W/12.0-TW, W/12.0-TW, W/12.0-TW, -W/12.0-TW]])
    tr_wheel = np.array([[TR, -TR, -TR, TR, TR],
                         [-W/12.0+TW, -W/12.0+TW, W/12.0+TW, W/12.0+TW, -W/12.0+TW]])
    tl_wheel = np.array([[TR, -TR, -TR, TR, TR],
                         [-W/12.0-TW, -W/12.0-TW, W/12.0-TW, W/12.0-TW, -W/12.0-TW]])

    Rot1 = np.array([[np.cos(yaw), np.sin(yaw)],
                     [-np.sin(yaw), np.cos(yaw)]])
    Rot2 = np.array([[np.cos(steer), np.sin(steer)],
                     [-np.sin(steer), np.cos(steer)]])
    Rot3 = np.array([[np.cos(yaw1), np.sin(yaw1)],
                     [-np.sin(yaw1), np.cos(yaw1)]])

    fr_wheel = (fr_wheel.T.dot(Rot2)).T
    fl_wheel = (fl_wheel.T.dot(Rot2)).T
    fr_wheel[0, :] = fr_wheel[0, :] + WB
    fl_wheel[0, :] = fl_wheel[0, :] + WB
    fr_wheel = (fr_wheel.T.dot(Rot1)).T
    fl_wheel = (fl_wheel.T.dot(Rot1)).T

    tr_wheel[0, :] = tr_wheel[0, :] - LT
    tl_wheel[0, :] = tl_wheel[0, :] - LT
    tr_wheel = (tr_wheel.T.dot(Rot3)).T
    tl_wheel = (tl_wheel.T.dot(Rot3)).T

    truckOutLine = (truckOutLine.T.dot(Rot1)).T
    trailerOutLine = (trailerOutLine.T.dot(Rot3)).T
    rr_wheel = (rr_wheel.T.dot(Rot1)).T
    rl_wheel = (rl_wheel.T.dot(Rot1)).T

    truckOutLine[0, :] = truckOutLine[0, :] + x
    truckOutLine[1, :] = truckOutLine[1, :] + y
    trailerOutLine[0, :] = trailerOutLine[0, :] + x
    trailerOutLine[1, :] = trailerOutLine[1, :] + y
    fr_wheel[0, :] = fr_wheel[0, :] + x
    fr_wheel[1, :] = fr_wheel[1, :] + y
    rr_wheel[0, :] = rr_wheel[0, :] + x
    rr_wheel[1, :] = rr_wheel[1, :] + y
    fl_wheel[0, :] = fl_wheel[0, :] + x
    fl_wheel[1, :] = fl_wheel[1, :] + y
    rl_wheel[0, :] = rl_wheel[0, :] + x
    rl_wheel[1, :] = rl_wheel[1, :] + y

    tr_wheel[0, :] = tr_wheel[0, :] + x
    tr_wheel[1, :] = tr_wheel[1, :] + y
    tl_wheel[0, :] = tl_wheel[0, :] + x
    tl_wheel[1, :] = tl_wheel[1, :] + y

    plt.plot(truckOutLine[0, :], truckOutLine[1, :], truckcolor)
    plt.plot(trailerOutLine[0, :], trailerOutLine[1, :], truckcolor)
    plt.plot(fr_wheel[0, :], fr_wheel[1, :], truckcolor)
    plt.plot(rr_wheel[0, :], rr_wheel[1, :], truckcolor)
    plt.plot(fl_wheel[0, :], fl_wheel[1, :], truckcolor)
    plt.plot(rl_wheel[0, :], rl_wheel[1, :], truckcolor)

    plt.plot(tr_wheel[0, :], tr_wheel[1, :], truckcolor)
    plt.plot(tl_wheel[0, :], tl_wheel[1, :], truckcolor)
    plt.plot(x, y, "*")

if __name__ == "__main__":      #测试主函数
    x, y = 0.0, 0.0
    yaw0 = np.radians(90.0)
    yaw1 = np.radians(20.0)

    plot_trailer(x, y, yaw0, yaw1, np.radians(0.0))

    DF = (LF + LB) / 2.0 - LB
    DFR = (LF + LB) / 2.0 + 0.3

    DT = (LTF + LTB) / 2.0 - LTB
    DTR = (LTF + LTB) / 2.0 + 0.3

    plt.axis("equal")
    plt.show()
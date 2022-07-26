import matplotlib.pyplot as plt
import trailer_hybrid_a_star
import numpy as np

def show_animation(path, oox, ooy, sx, sy, syaw0, syaw1, gx, gy, gyaw0, gyaw1):     #画图
    plt.plot(oox, ooy, ".k")
    trailer_hybrid_a_star.trailerlib.plot_trailer(sx, sy, syaw0, syaw1, 0.0)
    trailer_hybrid_a_star.trailerlib.plot_trailer(gx, gy, gyaw0, gyaw1, 0.0)
    x = path.x
    y = path.y
    yaw = path.yaw
    yaw1 = path.yaw1
    direction = path.direction

    steer = 0.0
    for ii in range(len(x)):
        plt.cla()
        plt.plot(oox, ooy, ".k")
        plt.plot(x, y, "-r")

        if ii < len(x)-2:
            k = (yaw[ii+1] - yaw[ii]) / trailer_hybrid_a_star.MOTION_RESOLUTION
            if not direction[ii]:
                k *= -1

            steer = np.arctan2(trailer_hybrid_a_star.WB*k, 1.0)
        else:
            steer = 0.0

        trailer_hybrid_a_star.trailerlib.plot_trailer(x[ii], y[ii], yaw[ii], yaw1[ii], steer)
        plt.grid(True)
        plt.axis("equal")
        plt.pause(0.0001)

if __name__ == "__main__":      #测试主函数
    print("start")

    sx = -15.0
    sy = 8.0
    syaw0 = np.radians(180.0)
    syaw1 = np.radians(180.0)
    '''
    sx = 13.0
    sy = 10.0
    syaw0 = np.radians(0.0)
    syaw1 = np.radians(0.0)
    '''
    gx = 1.0
    gy = 3.0
    gyaw0 = np.radians(0.0)
    gyaw1 = np.radians(0.0)

    ox, oy = [], []

    for i in range(-25, 26):
        ox.append(float(i))
        oy.append(15.0)
    for i in range(-10, 11):
        ox.append(float(i))
        oy.append(0.0)
    for i in range(-25, -9):
        ox.append(float(i))
        oy.append(5.0)
    for i in range(10, 26):
        ox.append(float(i))
        oy.append(5.0)
    for i in range(6):
        ox.append(10.0)
        oy.append(float(i))
    for i in range(6):
        ox.append(-10.0)
        oy.append(float(i))

    oox = ox[:]
    ooy = oy[:]

    path = trailer_hybrid_a_star.calc_hybrid_astar_path(sx, sy, syaw0, syaw1, gx, gy, gyaw0, gyaw1, ox, oy,
                                                        trailer_hybrid_a_star.XY_GRID_RESOLUTION,
                                                        trailer_hybrid_a_star.YAW_GRID_RESOLUTION)
    while(1):
        show_animation(path, oox, ooy, sx, sx, syaw0, syaw1, gx, gy, gyaw0, gyaw1)

        print("Done")
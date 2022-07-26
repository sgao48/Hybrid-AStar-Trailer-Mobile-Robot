import numpy as np
import matplotlib.pyplot as plt

STEP_SIZE = 0.1
MAX_PATH_LENGTH = 1000.0

class Path:     #定义路径类
    def __init__(self, lengths=[], ctypes=[],
                 L=0.0, x=[], y=[], yaw=[], directions=[]):
        self.lengths = lengths
        self.ctypes = ctypes
        self.L = L
        self.x, self.y, self.yaw = x, y, yaw
        self.directions = directions
    def distance(self):
        return sum([abs(i) for i in self.lengths])

def pi_2_pi(iangle = 0.0):      #转换角至[-pi, pi]
    for angle in iangle:
        while angle > np.pi:
            angle -= 2.0 * np.pi
        while angle < -np.pi:
            angle += 2.0 * np.pi

    return iangle

def calc_shortest_path(sx = 0.0, sy = 0.0, syaw = 0.0,      #计算最短Reeds-Shepp路径
                       gx = 0.0, gy = 0.0, gyaw = 0.0,
                       maxc = 0.0, step_size = STEP_SIZE):
    paths = calc_paths(sx, sy, syaw, gx, gy, gyaw, maxc, step_size=step_size)
    minL = np.inf
    best_path_index = -1
    for i in range(len(paths)):
        if paths[i].L <= minL:
            minL = paths[i].L
            best_path_index = i

    return paths[best_path_index]

def calc_shortest_path_length(sx = 0.0, sy = 0.0, syaw = 0.0,       #计算最短路径长度
                              gx = 0.0, gy = 0.0, gyaw = 0.0,
                              maxc = 0.0, step_size = STEP_SIZE):
    q0 = [sx, sy, syaw]
    q1 = [gx, gy, gyaw]
    paths = generate_path(q0, q1, maxc)
    minL = np.inf
    for i in range(1, len(paths)+1):
        L = paths[i-1].L/maxc
        if L <= minL:
            minL = L

    return minL

def calc_paths(sx = 0.0, sy = 0.0, syaw = 0.0,          #计算起始点至目标点间所有Reeds-Shepp曲线
               gx = 0.0, gy = 0.0, gyaw = 0.0,
               maxc = 0.0, step_size = STEP_SIZE):
    q0 = [sx, sy, syaw]
    q1 = [gx, gy, gyaw]
    paths = generate_path(q0, q1, maxc)
    for path in paths:      #计算路径属性
        x, y, yaw, directions = generate_local_course(path.L, path.lengths,
                                                      path.ctypes, maxc, step_size*maxc)
        path.x = [np.cos(-q0[2])*ix + np.sin(-q0[2])*iy + q0[0] for (ix, iy) in zip(x, y)]
        path.y = [-np.sin(-q0[2])*ix + np.cos(-q0[2])*iy + q0[1] for (ix, iy) in zip(x, y)]
        path.yaw = pi_2_pi([iyaw+q0[2] for iyaw in yaw])
        path.directions = directions
        path.lengths = [l/maxc for l in path.lengths]
        path.L = path.L/maxc

    return paths

def get_label(path = Path()):       #标记前进或后退
    label = ''
    for (m, l) in zip(path.ctypes, path.lengths):
        label += m
        if l > 0.0:
            label += '+'
        else:
            label += '-'

    return label

def polar(x = 0.0, y = 0.0):        #转换极坐标
    r = np.sqrt(x**2+y**2)
    theta = np.arctan2(y, x)

    return r, theta

def mod2pi(x = 0.0):        #对2pi取模
    v = x % (2.0*np.pi)
    '''
    if v < -np.pi:
        v += 2.0 * np.pi
    elif v > np.pi:
        v -= 2.0 * np.pi
    '''
    if v < 0:
        v += 2.0*np.pi
    return v

def LSL(x = 0.0, y = 0.0, phi = 0.0):       #LSL模式
    u, t = polar(x-np.sin(phi), y-1.0+np.cos(phi))
    if t >= 0.0:
        v = mod2pi(phi-t)
        if v >= 0.0:
            return True, t, u, v

    return False, 0.0, 0.0, 0.0

def LSR(x = 0.0, y = 0.0, phi = 0.0):       #LSR模式
    u1, t1 = polar(x+np.sin(phi), y-1.0-np.cos(phi))
    u1 = u1**2
    if u1 >= 4.0:
        u = np.sqrt(u1-4.0)
        theta = np.arctan2(2.0, u)
        t = mod2pi(t1+theta)
        v = mod2pi(t-phi)
        if t >= 0.0 and v >= 0.0:
            return True, t, u, v

    return False, 0.0, 0.0, 0.0

def LRL(x = 0.0, y = 0.0, phi = 0.0):       #LRL模式
    u1, t1 = polar(x-np.sin(phi), y-1.0+np.cos(phi))
    if u1 <= 4.0:
        u = -2.0*np.arcsin(0.25*u1)
        t = mod2pi(t1 + 0.5*u + np.pi)
        v = mod2pi(phi - t + u)
        if t >= 0.0 and u <= 0.0:
            return True, t, u, v

    return False, 0.0, 0.0, 0.0

def set_path(paths = None, lengths = [], ctypes = []):      #添加路径
    path = Path(lengths, ctypes, 0.0)
    for tpath in paths:
        typeissame = (tpath.ctypes == path.ctypes)
        if typeissame:
            if np.sum(tpath.lengths) - np.sum(path.lengths) <= 0.01:   ### UPDATE
                return paths
    path.L = np.sum(np.array([abs(i) for i in lengths]))
    if path.L >= MAX_PATH_LENGTH:
        return paths
    paths.append(path)

    return paths

def SCS(x = 0.0, y = 0.0, phi = 0.0, paths=None):       #SCS模式
    flag, t, u, v = SLS(x, y, phi)
    if flag:
        paths = set_path(paths, [t, u, v], ['S', 'L', 'S'])
    flag, t, u, v = SLS(x, -y, -phi)
    if flag:
        paths = set_path(paths, [t, u, v], ['S', 'R', 'S'])

    return paths

def SLS(x = 0.0, y = 0.0, phi = 0.0):       #SLS模式
    phi = mod2pi(phi)
    if y > 0.0 and phi > 0.0 and phi < np.pi*0.99:
        xd = -y/np.tan(phi) + x
        t = xd - np.tan(phi/2.0)
        u = phi
        v = np.sqrt((x-xd)**2 + y**2) - np.tan(phi/2.0)
        return True, t, u, v
    elif y < 0.0 and phi > 0.0 and phi < np.pi*0.99:
        xd = -y/np.tan(phi) + x
        t = xd - np.tan(phi/2.0)
        u = phi
        v = -np.sqrt((x-xd)**2 + y**2) - np.tan(phi/2.0)
        return True, t, u, v

    return False, 0.0, 0.0, 0.0

def CSC(x = 0.0, y = 0.0, phi = 0.0, paths = []):       #CSC模式
    flag, t, u, v = LSL(x, y, phi)
    if flag:
        paths = set_path(paths, [t, u, v], ['L', 'S', 'L'])
    flag, t, u, v = LSL(-x, y, -phi)
    if flag:
        paths = set_path(paths, [-t, -u, -v], ['L', 'S', 'L'])
    flag, t, u, v = LSL(x, -y, -phi)
    if flag:
        paths = set_path(paths, [t, u, v], ['R', 'S', 'R'])
    flag, t, u, v = LSL(-x, -y, phi)
    if flag:
        paths = set_path(paths, [-t, -u, -v], ['R', 'S', 'R'])
    flag, t, u, v = LSR(x, y, phi)
    if flag:
        paths = set_path(paths, [t, u, v], ['L', 'S', 'R'])
    flag, t, u, v = LSR(-x, y, -phi)
    if flag:
        paths = set_path(paths, [-t, -u, -v], ['L', 'S', 'R'])
    flag, t, u, v = LSR(x, -y, -phi)
    if flag:
        paths = set_path(paths, [t, u, v], ['R', 'S', 'L'])
    flag, t, u, v = LSR(-x, -y, phi)
    if flag:
        paths = set_path(paths, [-t, -u, -v], ['R', 'S', 'L'])

    return paths

def CCC(x = 0.0, y = 0.0, phi = 0.0, paths = []):       #CCC模式
    flag, t, u, v = LRL(x, y, phi)
    if flag:
        paths = set_path(paths, [t, u, v], ['L', 'R', 'L'])
    flag, t, u, v = LRL(-x, y, -phi)
    if flag:
        paths = set_path(paths, [-t, -u, -v], ['L', 'R', 'L'])
    flag, t, u, v = LRL(x, -y, -phi)
    if flag:
        paths = set_path(paths, [t, u, v], ['R', 'L', 'R'])
    flag, t, u, v = LRL(-x, -y, phi)
    if flag:
        paths = set_path(paths, [-t, -u, -v], ['R', 'L', 'R'])

    xb = x*np.cos(phi) + y*np.sin(phi)
    yb = x*np.sin(phi) - y*np.cos(phi)

    flag, t, u, v = LRL(xb, yb, phi)
    if flag:
        paths = set_path(paths, [v, u, t], ['L', 'R', 'L'])
    flag, t, u, v = LRL(-xb, yb, -phi)
    if flag:
        paths = set_path(paths, [-v, -u, -t], ['L', 'R', 'L'])
    flag, t, u, v = LRL(xb, -yb, -phi)
    if flag:
        paths = set_path(paths, [v, u, t], ['R', 'L', 'R'])
    flag, t, u, v = LRL(-xb, -yb, phi)
    if flag:
        paths = set_path(paths, [-v, -u, -t], ['R', 'L', 'R'])

    return paths

def calc_tauOmega(u = 0.0, v = 0.0, xi = 0.0, eta = 0.0, phi = 0.0):        #计算角度tau和omega
    delta = mod2pi(u-v)
    A = np.sin(u) - np.sin(delta)
    B = np.cos(u) - np.cos(delta) - 1.0

    t1 = np.arctan2((eta*A - xi*B), (xi*A + eta*B))
    t2 = 2.0*(np.cos(delta) - np.cos(v) - np.cos(u)) + 3.0

    if t2 < 0:
        tau = mod2pi(t1 + np.pi)
    else:
        tau = mod2pi(t1)
    omega = mod2pi(tau - u + v - phi)

    return tau, omega

def LRLRn(x = 0.0, y = 0.0, phi = 0.0):     #LRLR负向
    xi = x + np.sin(phi)
    eta = y - 1.0 - np.cos(phi)
    rho = 0.25*(2.0 + np.sqrt(xi*xi + eta*eta))

    if rho <= 1.0:
        u = np.arccos(rho)
        t, v = calc_tauOmega(u, -u, xi, eta, phi)
        if t >= 0.0 and v <= 0.0:
            return True, t, u, v

    return False, 0.0, 0.0, 0.0

def LRLRp(x = 0.0, y = 0.0, phi = 0.0):     #LRLR正向
    xi = x + np.sin(phi)
    eta = y - 1.0 - np.cos(phi)
    rho = (20.0 - xi*xi - eta*eta)/16.0

    if rho >= 0.0 and rho <= 1.0:
        u = -np.arccos(rho)
        if u >= -0.5*np.pi:
            t, v = calc_tauOmega(u, u, xi, eta, phi)
            if t >= 0.0 and v >= 0.0:
                return True, t, u, v

    return False, 0.0, 0.0, 0.0

def CCCC(x = 0.0, y = 0.0, phi = 0.0, paths = []):      #CCCC模式
    flag, t, u, v = LRLRn(x, y, phi)
    if flag:
        paths = set_path(paths, [t, u, -u, v], ['L', 'R', 'L', 'R'])
    flag, t, u, v = LRLRn(-x, y, -phi)
    if flag:
        paths = set_path(paths, [-t, -u, u, -v], ['L', 'R', 'L', 'R'])
    flag, t, u, v = LRLRn(x, -y, -phi)
    if flag:
        paths = set_path(paths, [t, u, -u, v], ['R', 'L', 'R', 'L'])
    flag, t, u, v = LRLRn(-x, -y, phi)
    if flag:
        paths = set_path(paths, [-t, -u, u, -v], ['R', 'L', 'R', 'L'])
    flag, t, u, v = LRLRp(x, y, phi)
    if flag:
        paths = set_path(paths, [t, u, u, v], ['L', 'R', 'L', 'R'])
    flag, t, u, v = LRLRp(-x, y, -phi)
    if flag:
        paths = set_path(paths, [-t, -u, -u, -v], ['L', 'R', 'L', 'R'])
    flag, t, u, v = LRLRp(x, -y, -phi)
    if flag:
        paths = set_path(paths, [t, u, u, v], ['R', 'L', 'R', 'L'])
    flag, t, u, v = LRLRp(-x, -y, phi)
    if flag:
        paths = set_path(paths, [-t, -u, -u, -v], ['R', 'L', 'R', 'L'])

    return paths

def LRSR(x = 0.0, y = 0.0, phi = 0.0):      #LRSR模式
    xi = x + np.sin(phi)
    eta = y - 1.0 - np.cos(phi)
    rho, theta = polar(-eta, xi)

    if rho >= 2.0:
        t = theta
        u = 2.0 - rho
        v = mod2pi(t + 0.5*np.pi - phi)
        if t >= 0.0 and u <= 0.0 and v <= 0.0:
            return True, t, u, v

    return False, 0.0, 0.0, 0.0

def LRSL(x = 0.0, y = 0.0, phi = 0.0):      #LRSL模式
    xi = x - np.sin(phi)
    eta = y - 1.0 + np.cos(phi)
    rho, theta = polar(xi, eta)

    if rho >= 2.0:
        r = np.sqrt(rho*rho - 4.0)
        u = 2.0 - r
        t = mod2pi(theta + np.arctan2(-r, 2.0))
        v = mod2pi(phi - 0.5*np.pi - t)
        if t >= 0.0 and u <= 0.0 and v <= 0.0:
            return True, t, u, v

    return False, 0.0, 0.0, 0.0

def CCSC(x = 0.0, y = 0.0, phi = 0.0, paths = []):      #CCSC模式
    flag, t, u, v = LRSL(x, y, phi)
    if flag:
        paths = set_path(paths, [t, -0.5*np.pi, u, v], ['L', 'R', 'S', 'L'])
    flag, t, u, v = LRSL(-x, y, -phi)
    if flag:
        paths = set_path(paths, [-t, 0.5*np.pi, -u, -v], ['L', 'R', 'S', 'L'])
    flag, t, u, v = LRSL(x, -y, -phi)
    if flag:
        paths = set_path(paths, [t, -0.5 * np.pi, u, v], ['R', 'L', 'S', 'R'])
    flag, t, u, v = LRSL(-x, -y, phi)
    if flag:
        paths = set_path(paths, [-t, 0.5 * np.pi, -u, -v], ['R', 'L', 'S', 'R'])
    flag, t, u, v = LRSR(x, y, phi)
    if flag:
        paths = set_path(paths, [t, -0.5*np.pi, u, v], ['L', 'R', 'S', 'R'])
    flag, t, u, v = LRSR(-x, y, -phi)
    if flag:
        paths = set_path(paths, [-t, 0.5 * np.pi, -u, -v], ['L', 'R', 'S', 'R'])
    flag, t, u, v = LRSR(x, -y, -phi)
    if flag:
        paths = set_path(paths, [t, -0.5*np.pi, u, v], ['R', 'L', 'S', 'L'])
    flag, t, u, v = LRSR(-x, -y, phi)
    if flag:
        paths = set_path(paths, [-t, 0.5*np.pi, -u, -v], ['R', 'L', 'S', 'L'])

    xb = x*np.cos(phi) + y*np.sin(phi)
    yb = x*np.sin(phi) - y*np.cos(phi)

    flag, t, u, v = LRSL(-xb, yb, -phi)
    if flag:
        paths = set_path(paths, [-v, -u, 0.5*np.pi, -t], ['L', 'S', 'R', 'L'])
    flag, t, u, v = LRSL(xb, -yb, -phi)
    if flag:
        paths = set_path(paths, [v, u, -0.5*np.pi, t], ['R', 'S', 'L', 'R'])
    flag, t, u, v = LRSL(-xb, -yb, phi)
    if flag:
        paths = set_path(paths, [-v, -u, 0.5*np.pi, -t], ['R', 'S', 'L', 'R'])
    flag, t, u, v = LRSR(xb, yb, phi)
    if flag:
        paths = set_path(paths, [v, u, -0.5*np.pi, t], ['R', 'S', 'R', 'L'])
    flag, t, u, v = LRSR(-xb, yb, -phi)
    if flag:
        paths = set_path(paths, [-v, -u, 0.5*np.pi, -t], ['R', 'S', 'R', 'L'])
    flag, t, u, v = LRSR(xb, -yb, -phi)
    if flag:
        paths = set_path(paths, [v, u, -0.5*np.pi, t], ['L', 'S', 'L', 'R'])
    flag, t, u, v = LRSR(-xb, -yb, phi)
    if flag:
        paths = set_path(paths, [-v, -u, 0.5*np.pi, -t], ['L', 'S', 'L', 'R'])

    return paths

def LRSLR(x = 0.0, y = 0.0, phi = 0.0):     #LRSLR模式
    xi = x + np.sin(phi)
    eta = y - 1.0 - np.cos(phi)
    rho, theta = polar(xi, eta)

    if rho >= 2.0:
        u = 4.0 - np.sqrt(rho*rho - 4.0)
        if u <= 0.0:
            t = mod2pi(np.arctan2(((4.0-u)*xi - 2.0*eta), (-2.0*xi + (u-4.0)*eta)))
            v = mod2pi(t - phi)
            if t >= 0.0 and v >= 0.0:
                return True, t, u, v

    return False, 0.0, 0.0, 0.0

def CCSCC(x = 0.0, y = 0.0, phi = 0.0, paths = []):     #CCSCC模式
    flag, t, u, v = LRSLR(x, y, phi)
    if flag:
        paths = set_path(paths, [t, -0.5*np.pi, u, -0.5*np.pi, v], ['L', 'R', 'S', 'L', 'R'])
    flag, t, u, v = LRSLR(-x, y, -phi)
    if flag:
        paths = set_path(paths, [-t, 0.5*np.pi, -u, 0.5*np.pi, -v], ['L', 'R', 'S', 'L', 'R'])
    flag, t, u, v = LRSLR(x, -y, -phi)
    if flag:
        paths = set_path(paths, [t, -0.5*np.pi, u, -0.5*np.pi, v], ['R', 'L', 'S', 'R', 'L'])
    flag, t, u, v = LRSLR(-x, -y, phi)
    if flag:
        paths = set_path(paths, [-t, 0.5*np.pi, -u, 0.5*np.pi, -v], ['R', 'L', 'S', 'R', 'L'])

    return paths

def trunc(x = 0.0):     #向0圆整
    if x >= 0:
        return np.floor(x)
    else:
        return np.ceil(x)

def generate_local_course(L = 0.0, lengths = [], mode = [], maxc = 0.0, step_size = 0.0):       #生成局部解
    npoint = int(trunc(L/step_size)) + len(lengths) + 3

    px = [0.0] * npoint
    py = [0.0] * npoint
    pyaw = [0.0] * npoint
    directions = [0] * npoint
    ind = 2

    if lengths[0] > 0.0:
        directions[0] = 1
    else:
        directions[0] = -1

    if lengths[0] > 0.0:
        d = step_size
    else:
        d = -step_size

    pd = d
    ll = 0.0

    for (m, l, i) in zip(mode, lengths, [j for j in range(1, len(mode)+1)]):
        if l > 0.0:
            d = step_size
        else:
            d = -step_size

        ox, oy, oyaw = px[ind-1], py[ind-1], pyaw[ind-1]

        ind -= 1
        if i >= 2 and (lengths[i-2]*lengths[i-1]) > 0:
            pd = -d - ll
        else:
            pd = d - ll

        while abs(pd) <= abs(l):
            ind += 1
            px, py, pyaw, directions = interpolate(ind, pd, m, maxc,
                                                   ox, oy, oyaw,
                                                   px, py, pyaw, directions)
            pd += d

        ll = l - pd - d
        ind += 1
        px, py, pyaw, directions = interpolate(ind, l, m, maxc,
                                               ox, oy, oyaw,
                                               px, py, pyaw, directions)

    while px[-1] == 0.0:
        px.pop()
        py.pop()
        pyaw.pop()
        directions.pop()

    return px, py, pyaw, directions

def interpolate(ind = 0, l = 0.0, m = '', maxc = 0.0,       #内插
                ox = 0.0, oy = 0.0, oyaw = 0.0,
                px = [], py = [], pyaw = [], directions = []):
    if m == 'S':
        px[ind-1] = ox + l/maxc*np.cos(oyaw)
        py[ind-1] = oy + l/maxc*np.sin(oyaw)
        pyaw[ind-1] = oyaw
    else:
        ldx = np.sin(l)/maxc
        if m == 'L':
            ldy = (1.0 - np.cos(l))/maxc
        elif m == 'R':
            ldy = -(1.0 - np.cos(l))/maxc
        gdx = np.cos(-oyaw)*ldx + np.sin(-oyaw)*ldy
        gdy = -np.sin(-oyaw)*ldx + np.cos(-oyaw)*ldy
        px[ind-1] = ox + gdx
        py[ind-1] = oy + gdy

    if m == 'L':
        pyaw[ind-1] = oyaw + l
    elif m == 'R':
        pyaw[ind-1] = oyaw - l

    if l > 0.0:
        directions[ind-1] = 1
    else:
        directions[ind-1] = -1

    return px, py, pyaw, directions

def generate_path(q0 = [], q1 = [], maxc = 0.0):        #生成所有路径
    dx = q1[0] - q0[0]
    dy = q1[1] - q0[1]
    dth = q1[2] - q0[2]
    c = np.cos(q0[2])
    s = np.sin(q0[2])
    x = (c*dx + s*dy) * maxc
    y = (-s*dx + c*dy) * maxc

    paths = []
    paths = SCS(x, y, dth, paths)
    paths = CSC(x, y, dth, paths)
    paths = CCC(x, y, dth, paths)
    paths = CCCC(x, y, dth, paths)
    paths = CCSC(x, y, dth, paths)
    paths = CCSCC(x, y, dth, paths)

    return paths

def calc_curvature(x, y, yaw, directions):      #计算曲率
    c = []
    ds = []
    for i in range(2, len(x)):
        dxn = x[i-1] - x[i-2]
        dxp = x[i] - x[i-1]
        dyn = y[i-1] - y[i-2]
        dyp = y[i] - y[i-1]
        dn = np.sqrt(dxn**2 + dyn**2)
        dp = np.sqrt(dxp**2 + dyp**2)
        dx = 1.0/(dn+dp) * (dp/dn*dxn + dn/dp*dxp)
        ddx = 2.0/(dn+dp) * (dxp/dp-dxn/dn)
        dy = 1.0/(dn+dp) * (dp/dn*dyn + dn/dp*dyp)
        ddy = 2.0/(dn+dp) * (dyp/dp - dyn/dn)
        curvature = (ddy*dx - ddx*dy)/(dx**2 + dy**2)
        d = (dn + dp)/2.0

        if np.isnan(curvature):
            curvature = 0.0
        if directions[i-1] <= 0.0:
            curvature = -curvature
        if len(c) == 0:
            ds.append(d)
            c.append(curvature)
        ds.append(d)
        c.append(curvature)

    ds.append(ds[-1])
    c.append(c[-1])

    return c, ds

if __name__ == "__main__":      #测试主函数

    start_x = 20.2
    start_y = 10
    start_yaw = np.radians(0.0)
    end_x = 0.0
    end_y = 0.0
    end_yaw = np.radians(90.0)
    max_curvature = 0.185
    '''
    start_x = 3.0
    start_y = 10.0
    start_yaw = np.radians(40.0)
    end_x = 0.0
    end_y = 1.0
    end_yaw = np.radians(0.0)
    max_curvature = 0.1
    '''
    bpath = calc_shortest_path(start_x, start_y, start_yaw,
                    end_x, end_y, end_yaw, max_curvature)
    print(bpath.x)
    print(bpath.y)
    plt.plot(bpath.x, bpath.y)
    plt.plot(start_x, start_y, 'ro', end_x, end_y, 'rx')
    plt.show()
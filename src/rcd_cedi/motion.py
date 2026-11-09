"""Piecewise longitudinal / lateral motion and continuous constraints."""
import math
import numpy as np
from numba import njit
from .constants import EPS
from .geometry import _native_roots, _native_overlaps

@njit(cache=True)
def _native_motion(s, ax, ay, turn, t, cfg):
    speed = s[11]
    stop = speed / -ax if ax < 0 else math.inf
    half = math.sqrt(cfg[4] / abs(ay)) if turn and ay != 0 else math.inf
    u = min(t, stop)
    q = speed * u + 0.5 * ax * u * u
    if t < half:
        z = 0.5 * ay * t * t
        vz = ay * t
        az = ay
    elif t < 2 * half:
        dt = t - half
        z = 0.5 * ay * half ** 2 + ay * half * dt - 0.5 * ay * dt ** 2
        vz = ay * (half - dt)
        az = -ay
    else:
        z = math.copysign(cfg[4], ay)
        vz = 0.0
        az = 0.0
    v = max(0.0, speed + ax * t)
    ac = ax if t < stop else 0.0
    ex, ey = (s[9], s[10])
    return (s[1] + ex * q - ey * z, s[2] + ey * q + ex * z, ex * v - ey * vz, ey * v + ex * vz, ex * ac - ey * az, ey * ac + ex * az)


@njit(cache=True)
def _native_add_knots(values, count, s, ax, ay, turn, cfg):
    stop = s[11] / -ax if ax < 0 else math.inf
    half = math.sqrt(cfg[4] / abs(ay)) if turn and ay != 0 else math.inf
    for value in (stop, half, 2 * half):
        if EPS < value < cfg[0] - EPS:
            values[count] = value
            count += 1
    return count


@njit(cache=True)
def _native_contact(states, all_axes, all_radii, i, j, axi, ayi, turni, axj, ayj, turnj, cfg):
    cuts = np.empty(8)
    cuts[0] = 0.0
    cuts[1] = cfg[0]
    count = _native_add_knots(cuts, 2, states[i], axi, ayi, turni, cfg)
    count = _native_add_knots(cuts, count, states[j], axj, ayj, turnj, cfg)
    times = np.sort(cuts[:count])
    axes = all_axes[i, j]
    radii = all_radii[i, j]
    polys = np.empty((4, 3))
    events = np.empty(18)
    for seg in range(len(times) - 1):
        left, right = (times[seg], times[seg + 1])
        if right <= left:
            continue
        sa = _native_motion(states[i], axi, ayi, turni, left, cfg)
        sb = _native_motion(states[j], axj, ayj, turnj, left, cfg)
        ma = _native_motion(states[i], axi, ayi, turni, (left + right) / 2, cfg)
        mb = _native_motion(states[j], axj, ayj, turnj, (left + right) / 2, cfg)
        px, py = (sb[0] - sa[0], sb[1] - sa[1])
        vx, vy = (sb[2] - sa[2], sb[3] - sa[3])
        ax, ay = ((mb[4] - ma[4]) / 2, (mb[5] - ma[5]) / 2)
        width = right - left
        separated = False
        for k in range(4):
            c0 = axes[k, 0] * px + axes[k, 1] * py
            c1 = axes[k, 0] * vx + axes[k, 1] * vy
            c2 = axes[k, 0] * ax + axes[k, 1] * ay
            polys[k, 0] = c0
            polys[k, 1] = c1
            polys[k, 2] = c2
            end = c0 + c1 * width + c2 * width * width
            low = min(c0, end)
            high = max(c0, end)
            if abs(c2) > 1e-14:
                t = -c1 / (2 * c2)
                if 0 < t < width:
                    value = c0 + c1 * t + c2 * t * t
                    low = min(low, value)
                    high = max(high, value)
            if low > radii[k] + EPS or high < -radii[k] - EPS:
                separated = True
                break
        if separated:
            continue
        events[0] = 0.0
        events[1] = width
        n = 2
        for k in range(4):
            n = _native_roots(events, n, polys[k, 0] + radii[k], polys[k, 1], polys[k, 2], width)
            n = _native_roots(events, n, polys[k, 0] - radii[k], polys[k, 1], polys[k, 2], width)
        ev = np.sort(events[:n])
        for k in range(n):
            if _native_overlaps(polys, radii, ev[k]):
                return True
            if k + 1 < n and _native_overlaps(polys, radii, (ev[k] + ev[k + 1]) / 2):
                return True
    return False


@njit(cache=True)
def _native_road(s, ax, ay, turn, cfg):
    cuts = np.empty(5)
    cuts[0] = 0.0
    cuts[1] = cfg[0]
    count = _native_add_knots(cuts, 2, s, ax, ay, turn, cfg)
    times = np.sort(cuts[:count])
    radius = s[12]
    for k in range(len(times) - 1):
        left, right = (times[k], times[k + 1])
        if right <= left:
            continue
        st = _native_motion(s, ax, ay, turn, left, cfg)
        acc = _native_motion(s, ax, ay, turn, (left + right) / 2, cfg)[5]
        dt = right - left
        low = st[1]
        high = st[1]
        value = st[1] + st[3] * dt + 0.5 * acc * dt * dt
        low = min(low, value)
        high = max(high, value)
        if abs(acc / 2) > 1e-14:
            t = -st[3] / acc
            if 0 < t < dt:
                value = st[1] + st[3] * t + 0.5 * acc * t * t
                low = min(low, value)
                high = max(high, value)
        if low - radius < cfg[1] + cfg[3] - EPS or high + radius > cfg[2] - cfg[3] + EPS:
            return False
    return True


@njit(cache=True)
def _zone_overlap(polys,lower,upper,t):
    for k in range(len(lower)):
        value=polys[k,0]+polys[k,1]*t+polys[k,2]*t*t
        if value<lower[k]-EPS or value>upper[k]+EPS:return False
    return True


@njit(cache=True)
def _native_zone(state,ax,ay,turn,axes,lower,upper,origin,cfg):
    """Continuous SAT against a static convex work-zone polygon."""
    if len(lower) == 0:
        return False
    cuts=np.empty(5);cuts[0]=0.;cuts[1]=cfg[0]
    count=_native_add_knots(cuts,2,state,ax,ay,turn,cfg)
    times=np.sort(cuts[:count])
    size=len(lower)
    polys=np.empty((size,3));events=np.empty(2+4*size)
    for seg in range(len(times)-1):
        left,right=times[seg],times[seg+1]
        if right<=left:continue
        st=_native_motion(state,ax,ay,turn,left,cfg)
        mid=_native_motion(state,ax,ay,turn,(left+right)/2,cfg)
        width=right-left;separated=False
        for k in range(size):
            nx,ny=axes[k,0],axes[k,1]
            c0=nx*(st[0]-origin[0])+ny*(st[1]-origin[1])
            c1=nx*st[2]+ny*st[3]
            c2=(nx*mid[4]+ny*mid[5])/2
            polys[k,0]=c0;polys[k,1]=c1;polys[k,2]=c2
            end=c0+c1*width+c2*width*width
            lo=min(c0,end);hi=max(c0,end)
            if abs(c2)>1e-14:
                t=-c1/(2*c2)
                if 0<t<width:
                    value=c0+c1*t+c2*t*t
                    lo=min(lo,value);hi=max(hi,value)
            if lo>upper[k]+EPS or hi<lower[k]-EPS:
                separated=True;break
        if separated:continue
        events[0]=0.;events[1]=width;n=2
        for k in range(size):
            n=_native_roots(events,n,polys[k,0]-lower[k],polys[k,1],polys[k,2],width)
            n=_native_roots(events,n,polys[k,0]-upper[k],polys[k,1],polys[k,2],width)
        ev=np.sort(events[:n])
        for k in range(n):
            if _zone_overlap(polys,lower,upper,ev[k]):return True
            if k+1<n and _zone_overlap(polys,lower,upper,(ev[k]+ev[k+1])/2):return True
    return False

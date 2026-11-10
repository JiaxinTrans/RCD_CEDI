"""Retained-action and independent terminal-kinematics validation helpers."""
import math
from numba import njit
from .constants import EPS
from .motion import _native_motion
from . import rcd as rcd_model

@njit(cache=True)
def advance(states,ax,ay,responses,cfg):
    output=states.copy()
    for i in range(len(states)):
        s=_native_motion(states[i],ax if i==0 else -responses[i],
                              ay if i==0 else 0.,i==0,cfg[0],cfg)
        output[i,1]=s[0];output[i,2]=s[1];output[i,3]=s[2];output[i,4]=s[3]
        speed=math.hypot(s[2],s[3]);output[i,11]=speed
        if speed>EPS:output[i,9]=s[2]/speed;output[i,10]=s[3]/speed
        else:output[i,9]=states[i,5];output[i,10]=states[i,6]
    return output


def raw_rcd_audit(states,active,pair_only=False):
    """Original RCD audit; pair_only restricts its ET query to indices 0/1."""
    vehicles=[rcd_model.Vehicle(str(int(s[0])),(float(s[1]),float(s[2])),
        (float(s[3]),float(s[4])),float(2*s[7]),float(2*s[8]),
        math.degrees(math.atan2(s[6],s[5]))) for s in states]
    potential=[];reachable=[];blocked=[];queries=0;errors=0;seen=set()
    for i in range(len(states)):
        if pair_only and i!=0:continue
        if i!=0 and not active[i]:continue
        for j in range(len(states)):
            if pair_only and j!=1:continue
            if i==j or tuple(sorted((i,j))) in seen:continue
            seen.add(tuple(sorted((i,j))));queries+=1
            a,b=vehicles[i],vehicles[j];et=rcd_model.swept_contact(a,b)
            if et is None:continue
            record=dict(vehicle_i=a.name,vehicle_j=b.name,ET_s=float(et));potential.append(record)
            if et<=rcd_model.EPS:
                record['reason']='contact_at_endpoint';reachable.append(record);continue
            try:
                _,pairs=rcd_model.earliest_ab_pairs(a,b,et)
                rel=rcd_model.sub(a.velocity,b.velocity);speed=math.hypot(*rel)
                theta=rcd_model.mul(1/speed,rel);blockers=[]
                for k,c in enumerate(vehicles):
                    if k in (i,j):continue
                    if rcd_model.swept_contact(a,c)==0. or rcd_model.swept_contact(b,c)==0.:continue
                    if rcd_model.c_blocks_prepared_ab(b,c,theta,pairs,et,c.corners()):blockers.append(c.name)
                if blockers:record['blockers']=blockers;blocked.append(record)
                else:reachable.append(record)
            except (ValueError,RuntimeError,ZeroDivisionError):
                errors+=1;record['reason']='P2_geometry_error';reachable.append(record)
    return dict(queries=queries,geometric_contacts=potential,reachable_contacts=reachable,
                blocked_contacts=blocked,geometry_errors=errors)


def manual_sv_end(s, ax, ay, t, lane_width):
    speed=s.speed
    ex,ey=s.direction
    u=min(t,speed/(-ax)) if ax<0 else t
    q=speed*u+.5*ax*u*u
    v=max(0.,speed+ax*t)
    if ay==0.:
        z,vz=0.,0.
    else:
        half=math.sqrt(lane_width/abs(ay))
        if t<half:
            z=.5*ay*t*t;vz=ay*t
        elif t<2*half:
            dt=t-half;z=.5*ay*half*half+ay*half*dt-.5*ay*dt*dt;vz=ay*(half-dt)
        else:
            z=math.copysign(lane_width,ay);vz=0.
    return s.x+ex*q-ey*z,s.y+ey*q+ex*z,ex*v-ey*vz,ey*v+ex*vz

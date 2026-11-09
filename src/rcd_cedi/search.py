"""Nested finite action sets; ranking always uses base normalization scales."""
import math
from dataclasses import replace
from functools import lru_cache
import numpy as np
from .constants import G

@lru_cache(maxsize=16)
def _action_axes(ax_steps,ay_steps,accel_limit,brake_limit,lateral_limit):
    positive=np.linspace(0.,accel_limit,max(5,ax_steps//4))
    if ax_steps>241:
        positive=np.unique(np.r_[positive,np.linspace(0.,accel_limit,60)])
    xs=np.unique(np.r_[np.linspace(-brake_limit,0.,ax_steps),positive])
    ys=np.unique(np.r_[np.linspace(-lateral_limit,lateral_limit,ay_steps),0.])
    ux=xs/np.where(xs>0.,accel_limit,brake_limit)
    uy=ys/lateral_limit
    return xs,ys,ux,uy


def sorted_actions(actions,accel_limit,brake_limit,lateral_limit):
    actions=np.unique(np.asarray(actions,dtype=float).reshape(-1,2),axis=0)
    lower=np.hypot(actions[:,0]/np.where(actions[:,0]>0.,accel_limit,brake_limit),
                   actions[:,1]/lateral_limit)
    order=np.lexsort((actions[:,1],actions[:,0],lower))
    return np.ascontiguousarray(np.column_stack((lower,actions))[order])


@lru_cache(maxsize=16)
def _sector_candidates(ax_steps,ay_steps,accel_limit,brake_limit,lateral_limit,steps):
    """Ring spacing 1/steps; angular spacing scales with radius.

    This covers all four action quadrants, retains boundary actions, and
    preserves fine near-axis strips. It does not restrict maneuver directions
    using a heuristic about whether ego is in front of or behind its opponent.
    """
    rows=[np.zeros((1,2))]
    for k in range(1,math.ceil(math.sqrt(2.)*steps)+1):
        radius=min(k/steps,math.sqrt(2.))
        n=max(8,8*math.ceil(2.*math.pi*radius*steps/8))
        angles=np.arange(n)*(2.*math.pi/n)
        uv=radius*np.column_stack((np.cos(angles),np.sin(angles)))
        uv[np.abs(uv)<1e-14]=0.
        rows.append(np.clip(uv[np.max(np.abs(uv),axis=1)<=1.+1e-12],-1.,1.))
    angles=np.arange(steps*8)*(2.*math.pi/(steps*8))
    boundary=np.column_stack((np.cos(angles),np.sin(angles)))
    boundary/=np.max(np.abs(boundary),axis=1)[:,None]
    boundary[np.abs(boundary)<1e-14]=0.
    rows.append(boundary)
    uv=np.concatenate(rows)
    actions=uv*np.column_stack((np.where(uv[:,0]>0.,accel_limit,brake_limit),
                                np.full(len(uv),lateral_limit)))
    xs,ys,_,_=_action_axes(ax_steps,ay_steps,accel_limit,brake_limit,lateral_limit)
    zx,zy=np.searchsorted(xs,0.),np.searchsorted(ys,0.)
    near_x=np.unique(np.clip(zx+np.array([-2,-1,0,1,2]),0,len(xs)-1))
    near_y=np.unique(np.clip(zy+np.array([-8,-4,-2,-1,0,1,2,4,8]),0,len(ys)-1))
    x1,y1=np.meshgrid(xs,ys[near_y],indexing='ij')
    x2,y2=np.meshgrid(xs[near_x],ys,indexing='ij')
    strips=np.vstack((np.column_stack((x1.ravel(),y1.ravel())),
                      np.column_stack((x2.ravel(),y2.ravel()))))
    return sorted_actions(np.vstack((actions,strips)),accel_limit,brake_limit,lateral_limit)


def sector_candidates(settings):
    return _sector_candidates(settings.ax_steps,settings.ay_steps,
        settings.ax_accel_g*G,settings.ax_brake_g*G,settings.ay_g*G,settings.sector_steps)

def candidate_grids(settings, factors):
    """Nested expansion grids; preserve all earlier actions and normalizers."""
    controls = np.empty((0, 2))
    grids = []
    for factor in factors:
        expanded = replace(settings, ax_accel_g=settings.ax_accel_g*factor,
            ax_brake_g=settings.ax_brake_g*factor, ay_g=settings.ay_g*factor)
        controls = np.unique(np.vstack((controls, sector_candidates(expanded)[:, 1:3])), axis=0)
        grids.append(sorted_actions(controls, settings.ax_accel_g*G,
                        settings.ax_brake_g*G, settings.ay_g*G))
    return grids

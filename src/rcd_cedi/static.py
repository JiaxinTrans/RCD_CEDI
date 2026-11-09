"""Static convex obstacles, scene packing and initial-overlap checks."""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from .constants import EPS

@dataclass(frozen=True)
class StaticEnvironment:
    """Optional single static convex polygon; road bounds are in VehicleConfig.

    None explicitly means no static obstacle. All moving actors are rectangles.
    Vertices must be in cyclic order, without repeating the first vertex.
    """
    polygon: tuple[tuple[float, float], ...] | None = None

    def __post_init__(self):
        if self.polygon is None:
            return
        p = np.asarray(self.polygon, dtype=float)
        if p.ndim != 2 or p.shape[1] != 2 or len(p) < 3 or not np.isfinite(p).all():
            raise ValueError('polygon must contain at least three finite 2D vertices')
        edges = np.roll(p, -1, axis=0) - p
        if np.any(np.linalg.norm(edges, axis=1) <= EPS):
            raise ValueError('polygon has repeated adjacent vertices')
        # Every vertex must lie in the same half-plane of every edge.
        # This also rejects self-crossing vertex orders.
        sides = edges[:, 0, None] * (p[None, :, 1]-p[:, None, 1]) - edges[:, 1, None] * (p[None, :, 0]-p[:, None, 0])
        if not (np.all(sides >= -EPS) or np.all(sides <= EPS)):
            raise ValueError('polygon must be convex and cyclically ordered')
        area = abs(np.dot(p[:, 0], np.roll(p[:, 1], -1))-np.dot(p[:, 1], np.roll(p[:, 0], -1)))/2
        if area <= EPS:
            raise ValueError('polygon must have positive area')
        object.__setattr__(self, 'polygon', tuple(tuple(float(v) for v in row) for row in p))

    @classmethod
    def rectangle(cls, xmin, xmax, ymin, ymax):
        if not xmin < xmax or not ymin < ymax:
            raise ValueError('rectangle bounds must be increasing')
        return cls(((xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)))

    def as_array(self):
        return None if self.polygon is None else np.asarray(self.polygon, dtype=float)

# Compatibility for existing top-level imports: from rcd_cedi import Environment.
Environment = StaticEnvironment


def polygon_normals(poly):
    normals=[]
    for p,q in zip(poly,np.roll(poly,-1,axis=0)):
        dx,dy=q-p;length=math.hypot(dx,dy)
        if length<=EPS:continue
        n=(-dy/length,dx/length)
        if not any(abs(np.dot(n,old))>1-1e-12 for old in normals):normals.append(n)
    return normals


def pack_scene(ego,counter,scene,polygon,settings):
    others=[s for s in scene if s.vid not in (ego.vid,counter.vid)]
    others.sort(key=lambda s:(s.x<ego.x,abs(s.x-ego.x)))
    actors=[ego,counter]+others
    n=len(actors)
    states=np.empty((n,13));axes=np.empty((n,n,4,2));radii=np.empty((n,n,4))
    origin=np.zeros(2) if polygon is None else np.asarray(polygon[0],dtype=float)
    local_polygon=None if polygon is None else polygon-origin
    normals=[] if polygon is None else polygon_normals(polygon)
    k=0 if polygon is None else 2+len(normals)
    zaxes=np.empty((n,k,2));zlower=np.empty((n,k));zupper=np.empty((n,k))
    for i,s in enumerate(actors):
        states[i]=(s.vid,s.x,s.y,s.vx,s.vy,math.cos(s.heading),math.sin(s.heading),s.length/2,s.width/2,*s.direction,math.hypot(s.vx,s.vy),s.radius((0,1)))
        for j,o in enumerate(actors):
            axes[i,j]=list(s.basis)+list(o.basis)
            radii[i,j]=[s.radius(axis)+o.radius(axis)+settings.gap for axis in axes[i,j]]
        if polygon is None:
            continue
        zaxes[i]=list(s.basis)+normals
        projected=local_polygon@zaxes[i].T
        r=np.asarray([s.radius(axis)+settings.gap for axis in zaxes[i]])
        zlower[i]=projected.min(axis=0)-r
        zupper[i]=projected.max(axis=0)+r
    return actors,states,axes,radii,zaxes,zlower,zupper,origin


def initial_zone_overlap(state,polygon,gap):
    if polygon is None:
        return False
    origin=polygon[0]
    for axis in list(state.basis)+polygon_normals(polygon):
        projections=(polygon-origin)@np.asarray(axis)
        p=np.dot(np.asarray([state.x,state.y])-origin,axis)
        r=state.radius(axis)+gap
        if p<projections.min()-r-EPS or p>projections.max()+r+EPS:return False
    return True

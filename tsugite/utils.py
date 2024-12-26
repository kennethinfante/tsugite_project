# NOTE: the functions moved here are more of helper math/linear algebra functions
import numpy as np
import math

# selection.py
def angle_between_with_direction(v0, v1):
    v0 = v0 / np.linalg.norm(v0)
    v1 = v1 / np.linalg.norm(v1)
    angle = np.arctan2(np.linalg.det([v0,v1]),np.dot(v0,v1))
    return math.degrees(angle)

def unitize(v):
    uv = v/np.linalg.norm(v)
    return uv

def get_same_height_neighbors(hfield,inds):
    dim = len(hfield)
    val = hfield[tuple(inds[0])]
    new_inds = list(inds)
    for ind in inds:
        for ax in range(2):
            for dir in range(-1,2,2):
                ind2 = ind.copy()
                ind2[ax] += dir
                if np.all(ind2>=0) and np.all(ind2<dim):
                    val2 = hfield[tuple(ind2)]
                    if val2==val:
                        unique = True
                        for ind3 in new_inds:
                            if ind2[0]==ind3[0] and ind2[1]==ind3[1]:
                                unique = False
                                break
                        if unique: new_inds.append(ind2)
    if len(new_inds)>len(inds):
        new_inds = get_same_height_neighbors(hfield,new_inds)
    return new_inds

# misc
def depth(l):
    if isinstance(l, list):
        return 1 + max(depth(item) for item in l)
    else:
        return 0
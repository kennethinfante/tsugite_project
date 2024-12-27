# NOTE: the functions moved here are more of helper math/linear algebra functions
import numpy as np
import math

# joint_types
def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: return v
    else: return v / norm

def angle_between(vector_1, vector_2):
    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product)
    return angle

def rotate_vector_around_axis(vec=[3,5,0], axis=[4,4,1], theta=1.2): #example values
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    mat = np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                    [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                    [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])
    rotated_vec = np.dot(mat, vec)
    return rotated_vec

# unused
def filleted_points(pt,one_voxel,off_dist,ax,n):
    ##
    addx = (one_voxel[0]*2-1)*off_dist
    addy = (one_voxel[1]*2-1)*off_dist
    ###
    pt1 = pt.copy()
    add = [addx,-addy]
    add.insert(ax,0)
    pt1[0] += add[0]
    pt1[1] += add[1]
    pt1[2] += add[2]
    #
    pt2 = pt.copy()
    add = [-addx,addy]
    add.insert(ax,0)
    pt2[0] += add[0]
    pt2[1] += add[1]
    pt2[2] += add[2]
    #
    if n%2==1: pt1,pt2 = pt2,pt1
    return [pt1,pt2]

# unused
def get_outline(type,verts,lay_num,n):
    fdir = type.mesh.fab_directions[n]
    outline = []
    for rv in verts:
        ind = rv.ind.copy()
        ind.insert(type.sax,(type.dim-1)*(1-fdir)+(2*fdir-1)*lay_num)
        add = [0,0,0]
        add[type.sax] = 1-fdir
        i_pt = get_index(ind,add,type.dim)
        pt = get_vertex(i_pt,type.jverts[n],type.vertex_no_info)
        outline.append(MillVertex(pt))
    return outline

# unused
def is_additional_outer_corner(type,rv,ind,ax,n):
    outer_corner = False
    if rv.region_count==1 and rv.block_count==1:
        other_fixed_sides = type.fixed.sides.copy()
        other_fixed_sides.pop(n)
        for sides in other_fixed_sides:
            for side in sides:
                if side.ax==ax: continue
                axes = [0,0,0]
                axes[side.ax] = 1
                axes.pop(ax)
                oax = axes.index(1)
                not_oax = axes.index(0)
                # what is this odir?
                if rv.ind[oax]==odir*type.dim:
                    if rv.ind[not_oax]!=0 and rv.ind[not_oax]!=type.dim:
                        outer_corner = True
                        break
            if outer_corner: break
    return outer_corner

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


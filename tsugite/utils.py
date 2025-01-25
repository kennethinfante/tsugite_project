# NOTE: the functions moved here are more of helper math/linear algebra functions
import numpy as np
import math
import random
import copy

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: return v
    else: return v / norm

def unitize(v):
    uv = v/np.linalg.norm(v)
    return uv

def angle_between_vectors1(vector_1, vector_2, direction=False):
    unit_vector_1 = unitize(vector_1)
    unit_vector_2 = unitize(vector_2)
    v_dot_product = np.dot(unit_vector_1, unit_vector_2)

    if direction:
        angle = np.arctan2(np.linalg.det([unit_vector_1, unit_vector_2]), v_dot_product)
        return math.degrees(angle)
    else:
        angle = np.arccos(v_dot_product)
        return angle

def angle_between_vectors2(vector_1, vector_2, normal_vector=[]):
    unit_vector_1 = unitize(vector_1)
    unit_vector_2 = unitize(vector_2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product)
    cross = np.cross(unit_vector_1,unit_vector_2)
    if len(normal_vector)>0 and np.dot(normal_vector, cross)<0: angle = -angle
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

def matrix_from_height_fields(hfs,ax): ### duplicated function - also exists in Geometries
    dim = len(hfs[0])
    mat = np.zeros(shape=(dim,dim,dim))
    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                ind = [i,j]
                ind3d = ind.copy()
                ind3d.insert(ax,k)
                ind3d = tuple(ind3d)
                ind2d = tuple(ind)
                h = 0
                for n,hf in enumerate(hfs):
                    if k<hf[ind2d]: mat[ind3d]=n; break
                    else: mat[ind3d]=n+1
    mat = np.array(mat)
    return mat

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

def get_random_height_fields(dim,noc):
    hfs = []
    phf = np.zeros((dim,dim))
    for n in range(noc-1):
        hf = np.zeros((dim,dim))
        for i in range(dim):
            for j in range(dim): 
                hf[i,j]=random.randint(int(phf[i,j]),dim)
        hfs.append(hf)
        phf = copy.deepcopy(hf)
    return hfs

def get_diff_neighbors(mat2,inds,val):
    new_inds = list(inds)
    for ind in inds:
        for ax in range(2):
            for dir in range(-1,2,2):
                ind2 = ind.copy()
                ind2[ax] += dir
                if ind2[ax]>=0 and ind2[ax]<mat2.shape[ax]:
                    val2 = mat2[tuple(ind2)]
                    if val2==val or val2==-1: continue
                    unique = True
                    for ind3 in new_inds:
                        if ind2[0]==ind3[0] and ind2[1]==ind3[1]:
                            unique = False
                            break
                    if unique: new_inds.append(ind2)
    if len(new_inds)>len(inds):
        new_inds = get_diff_neighbors(mat2,new_inds,val)
    return new_inds

def set_starting_vert(verts):
    first_i = None
    second_i = None
    for i,rv in enumerate(verts):
        if rv.block_count>0:
            if rv.free_count>0: first_i=i
            else: second_i = i
    if first_i==None:
        first_i=second_i
    if first_i==None: first_i=0
    verts.insert(0,verts[first_i])
    verts.pop(first_i+1)
    return verts

def get_sublist_of_ordered_verts(verts):
    ord_verts = []

    # Start ordered vertices with the first item (simultaneously remove from main list)
    ord_verts.append(verts[0])
    verts.remove(verts[0])

    browse_num = len(verts)
    for i in range(browse_num):
        found_next = False
        #try all directions to look for next vertex
        for vax in range(2):
            for vdir in range(-1,2,2):
                # check if there is an available vertex
                next_ind = ord_verts[-1].ind.copy()
                next_ind[vax]+=vdir
                next_rv = None
                for rv in verts:
                    if rv.ind==next_ind:
                        if len(ord_verts)>1 and rv.ind==ord_verts[-2].ind: break # prevent going back
                        # check so that it is not crossing a blocked region etc
                        # 1) from point of view of previous point
                        p_neig = ord_verts[-1].neighbors
                        vaxval = int(0.5*(vdir+1))
                        nind0 = [0,0]
                        nind0[vax] = vaxval
                        nind1 = [1,1]
                        nind1[vax] = vaxval
                        ne0 = p_neig[nind0[0]][nind0[1]]
                        ne1 = p_neig[nind1[0]][nind1[1]]
                        if ne0!=1 and ne1!=1: continue # no block
                        if int(0.5*(ne0+1))==int(0.5*(ne1+1)): continue # trying to cross blocked material
                        # 2) from point of view of point currently tested
                        nind0 = [0,0]
                        nind0[vax] = 1-vaxval
                        nind1 = [1,1]
                        nind1[vax] = 1-vaxval
                        ne0 = rv.neighbors[nind0[0]][nind0[1]]
                        ne1 = rv.neighbors[nind1[0]][nind1[1]]
                        if ne0!=1 and ne1!=1: continue # no block
                        if int(0.5*(ne0+1))==int(0.5*(ne1+1)): continue # trying to cross blocked material
                        # If you made it here, you found the next vertex!
                        found_next=True
                        ord_verts.append(rv)
                        verts.remove(rv)
                        break
                if found_next: break
            if found_next: break
        if found_next: continue

    # check if outline is closed by ckecing if endpoint finds startpoint

    closed = False
    if len(ord_verts)>3: # needs to be at least 4 vertices to be able to close
        start_ind = np.array(ord_verts[0].ind.copy())
        end_ind = np.array(ord_verts[-1].ind.copy())
        diff_ind = start_ind-end_ind ###reverse?
        if len(np.argwhere(diff_ind==0))==1: #difference only in one axis
            vax = np.argwhere(diff_ind!=0)[0][0]
            if abs(diff_ind[vax])==1: #difference is only one step
                vdir = diff_ind[vax]
            # check so that it is not crossing a blocked region etc
                p_neig = ord_verts[-1].neighbors
                vaxval = int(0.5*(vdir+1))
                nind0 = [0,0]
                nind0[vax] = vaxval
                nind1 = [1,1]
                nind1[vax] = vaxval
                ne0 = p_neig[nind0[0]][nind0[1]]
                ne1 = p_neig[nind1[0]][nind1[1]]
                if ne0==1 or ne1==1:
                    if int(0.5*(ne0+1))!=int(0.5*(ne1+1)):
                        # If you made it here, you found the next vertex!
                        closed=True

    return ord_verts, verts, closed

def set_vector_length(vec,new_norm):
    norm = np.linalg.norm(vec)
    vec = vec/norm
    vec = new_norm*vec
    return vec

def get_vertex(index,verts,n):
    x = verts[n*index]
    y = verts[n*index+1]
    z = verts[n*index+2]
    return np.array([x,y,z])

def get_segment_proportions(outline):
    olen = 0
    slens = []
    sprops = []

    for i in range(1,len(outline)):
        ppt = outline[i-1].pt
        pt = outline[i].pt
        dist = np.linalg.norm(pt-ppt)
        slens.append(dist)
        olen+=dist

    olen2=0
    sprops.append(0.0)
    for slen in slens:
        olen2+=slen
        sprop = olen2/olen
        sprops.append(sprop)

    return sprops

def any_minus_one_neighbor(ind,lay_mat):
    bool = False
    for add0 in range(-1,1,1):
        temp = []
        temp2 = []
        for add1 in range(-1,1,1):
            # Define neighbor index to test
            nind = [ind[0]+add0,ind[1]+add1]
            # If test index is within bounds
            if np.all(np.array(nind)>=0) and nind[0]<lay_mat.shape[0] and nind[1]<lay_mat.shape[1]:
                # If the value is -1
                if lay_mat[tuple(nind)]==-1:
                    bool = True
                    break
    return bool

def get_neighbors_in_out(ind,reg_inds,lay_mat,org_lay_mat,n):
    # 0 = in region
    # 1 = outside region, block
    # 2 = outside region, free
    in_out = []
    values = []
    for add0 in range(-1,1,1):
        temp = []
        temp2 = []
        for add1 in range(-1,1,1):

            # Define neighbor index to test
            nind = [ind[0]+add0,ind[1]+add1]

            # FIND TYPE
            type = -1
            val = None
            # Check if this index is in the list of region-included indices
            for rind in reg_inds:
                if rind[0]==nind[0] and rind[1]==nind[1]:
                    type = 0 # in region
                    break
            if type!=0:
                # If there are out of bound indices they are free
                if np.any(np.array(nind)<0) or nind[0]>=lay_mat.shape[0] or nind[1]>=lay_mat.shape[1]:
                    type = 2 # free
                    val =-1
                elif lay_mat[tuple(nind)]<0:
                    type = 2 # free
                    val = -2
                else: type = 1 # blocked

            if val==None:
                val=org_lay_mat[tuple(nind)]

            temp.append(type)
            temp2.append(val)
        in_out.append(temp)
        values.append(temp2)
    return in_out, values

def face_neighbors(mat,ind,ax,n,fixed_sides):
    values = []
    dim = len(mat)
    for i in range(2):
        val = None
        ind2 = ind.copy()
        ind2[ax] = ind2[ax]-i
        ind2 = np.array(ind2)
        if np.all(ind2>=0) and np.all(ind2<dim):
            val = mat[tuple(ind2)]
        elif len(fixed_sides)>0:
            for fixed_side in fixed_sides:
                ind3 = np.delete(ind2,fixed_side.ax)
                if np.all(ind3>=0) and np.all(ind3<dim):
                    if ind2[fixed_side.ax]<0 and fixed_side.dir==0: val = n
                    elif ind2[fixed_side.ax]>=dim and fixed_side.dir==1: val = n
        values.append(val)
    values = np.array(values)
    count = np.count_nonzero(values==n)
    return count,values

def get_index(ind,add,dim):
    d = dim+1
    (i,j,k) = ind
    index = (i+add[0])*d*d + (j+add[1])*d + k+add[2]
    return index

def get_corner_indices(ax,n,dim):
    other_axes = np.array([0,1,2])
    other_axes = np.delete(other_axes,np.where(other_axes==ax))
    ind = np.array([0,0,0])
    ind[ax] = n*dim
    corner_indices = []
    for x in range(2):
        for y in range(2):
            add = np.array([0,0,0])
            add[other_axes[0]] = x*dim
            add[other_axes[1]] = y*dim
            corner_indices.append(get_index(ind,add,dim))
    return corner_indices
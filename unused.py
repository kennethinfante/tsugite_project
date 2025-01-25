
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

# unused
def get_top_corner_heights(mat,n,ax,dir):
    heights = []
    dim = len(mat)
    for i in range(2):
        i = i*(dim-1)
        temp = []
        for j in range(2):
            j = j*(dim-1)
            top_cor = 0
            for k in range(dim):
                ind = [i,j]
                ind.insert(ax,k)
                val = mat[tuple(ind)]
                if val==n: top_cor=k
            if dir==1: top_cor = dim-top_cor
            temp.append(top_cor)
        heights.append(temp)
    return heights

# unused
def get_same_neighbors(ind,fixed_sides,voxel_matrix,dim):
    neighbors = []
    val = voxel_matrix[tuple(ind)]
    for ax in range(3):
        for n in range(2):
            add = [0,0]
            add.insert(ax,2*n-1)
            add = np.array(add)
            ind2 = ind+add
            if (ind2[ax]<0 or ind2[ax]>=dim) and not FixedSide(ax,n).unique(fixed_sides): #and [ax,n] in fixed_sides:
                val2 = val
            elif np.all(ind2>=0) and np.all(ind2<dim):
                val2 = voxel_matrix[tuple(ind2)]
            else: val2=None
            if val==val2:
                neighbors.append([ax,n])
    return neighbors

# unused
def get_count(ind,neighbors,fixed_sides,voxel_matrix,dim):
    cnt = 0
    val = int(voxel_matrix[ind])
    vals2 = []
    for item in neighbors:
        i = ind[0]+item[0]
        j = ind[1]+item[1]
        k = ind[2]+item[2]
        ###
        val2 = None
        # Check fixed sides
        if (i<0 or i>=dim) and j>=0 and j<dim and k>=0 and k<dim:
            if i<0 and not FixedSide(0,0).unique(fixed_sides): #[0,0] in fixed_sides:
                val2 = val
            elif i>=dim and not FixedSide(0,1).unique(fixed_sides): #[0,1] in fixed_sides:
                val2 = val
        elif (j<0 or j>=dim) and i>=0 and i<dim and k>=0 and k<dim:
            if j<0 and not FixedSide(1,0).unique(fixed_sides): #[1,0] in fixed_sides:
                val2 = val
            elif j>=dim and not FixedSide(1,1).unique(fixed_sides): #[1,1] in fixed_sides:
                val2 = val
        elif (k<0 or k>=dim) and i>=0 and i<dim and j>=0 and j<dim:
            if k<0 and not FixedSide(2,0).unique(fixed_sides): #[2,0] in fixed_sides:
                val2 = val
            elif k>=dim and not FixedSide(2,1).unique(fixed_sides): #[2,1] in fixed_sides:
                val2 = val
        # Check neighbours
        elif np.all(np.array([i,j,k])>=0) and np.all(np.array([i,j,k])<dim):
            val2 = int(voxel_matrix[i,j,k])
        if val==val2: cnt = cnt+1
        vals2.append(val2)
    return cnt,vals2[2],vals2[0],vals2[1]

# unused
def get_next_same_axial_index(ind,ax,mat,dim):
    if ind[ax]<dim-1:
        val = mat[tuple(ind)]
        ind_next = ind.copy()
        ind_next[ax] += 1
        val_next = mat[tuple(ind_next)]
        if val==val_next:
            ind_next_next = get_next_same_axial_index(ind_next,ax,mat,dim)
            return ind_next_next
        else: return ind
    else: return ind
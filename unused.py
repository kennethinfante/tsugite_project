
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

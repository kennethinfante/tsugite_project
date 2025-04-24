import copy
import random

from buffer import Buffer
from evaluation import Evaluation
from fabrication import *
from geometries import Geometries
from fixed_side import FixedSides

import utils as Utils

# this remains here for now because it is using other classes

class Joint:
    def __init__(self, pwidget, fs=[], sax=2, dim=3, ang=0.0, td=[44.0, 44.0, 44.0], fspe=400, fspi=6000,
                 fabtol=0.15, fabdia=6.00, align_ax=0, fabext="gcode", incremental=False, hfs=[], finterp=True):
        self.pwidget=pwidget
        self.sax = sax
        self.fixed = FixedSides(self)
        self.noc = len(self.fixed.sides) #number of components
        self.dim = dim
        self.suggestions_on = True
        self.component_size = 0.275
        self.real_tim_dims = np.array(td)
        self.component_length = 0.5*self.component_size
        self.ratio = np.average(self.real_tim_dims)/self.component_size
        self.voxel_sizes = np.copy(self.real_tim_dims)/(self.ratio*self.dim)
        self.fab = Fabrication(self, tol=fabtol, dia=fabdia, ext=fabext, align_ax=align_ax, interp=finterp, spi=fspi, spe=fspe)
        self.vertex_no_info = 8
        self.ang = ang
        self.buff = Buffer(self) #initiating the buffer
        self.fixed.update_unblocked()
        self.vertices = self.create_and_buffer_vertices(milling_path=False) # create and buffer vertices
        self.mesh = Geometries(self, hfs=hfs)
        self.suggestions = []
        self.gals = []
        self.update_suggestions()
        self.combine_and_buffer_indices()
        self.gallary_start_index = -20
        self.incremental = incremental

    def _arrow_vertices(self):
        vertices = []
        r=g=b=0.0
        tx=ty=0.0
        vertices.extend([0,0,0, r,g,b, tx,ty]) # origin
        for ax in range(3):
            for dir in range(-1,2,2):
                #arrow base
                xyz = dir*self.pos_vecs[ax]*self.dim*0.4
                vertices.extend([xyz[0],xyz[1],xyz[2], r,g,b, tx,ty]) # end of line
                #arrow head
                for i in range(-1,2,2):
                    for j in range(-1,2,2):
                        other_axes = [0,1,2]
                        other_axes.pop(ax)
                        pos = dir*self.pos_vecs[ax]*self.dim*0.3
                        pos+= i*self.pos_vecs[other_axes[0]]*self.dim*0.025
                        pos+= j*self.pos_vecs[other_axes[1]]*self.dim*0.025
                        vertices.extend([pos[0],pos[1],pos[2], r,g,b, tx,ty]) # arrow head indices
        # Format
        vertices = np.array(vertices, dtype = np.float32) #converts to correct format
        return vertices

    def _layer_mat_from_cube(self, lay_num, n):
        mat = np.ndarray(shape=(self.dim,self.dim), dtype=int)
        fdir = self.mesh.fab_directions[n]
        for i in range(self.dim):
            for j in range(self.dim):
                ind = [i,j]
                zval = (self.dim-1)*(1-fdir)+(2*fdir-1)*lay_num
                ind.insert(self.sax,zval)
                mat[i][j]=int(self.mesh.voxel_matrix[tuple(ind)])
        return mat

    def _pad_layer_mat_with_fixed_sides(self, mat,n):
        pad_loc = [[0,0],[0,0]]
        pad_val = [[-1,-1],[-1,-1]]
        for n2 in range(len(self.fixed.sides)):
            for oside in self.fixed.sides[n2]:
                if oside.ax==self.sax: continue
                axes = [0,0,0]
                axes[oside.ax] = 1
                axes.pop(self.sax)
                oax = axes.index(1)
                pad_loc[oax][oside.dir] = 1
                pad_val[oax][oside.dir] = n2
        # If it is an angled joint, pad so that the edge of a joint located on an edge will be trimmed well
        #if abs(self.ang-90)>1 and len(self.fixed.sides[n])==1 and self.fixed.sides[n][0].ax!=self.sax:
        #    print("get here")
        #    ax = self.fixed.sides[n][0].ax
        #    dir = self.fixed.sides[n][0].dir
        #    odir = 1-dir
        #    axes = [0,0,0]
        #    axes[ax] = 1
        #    axes.pop(self.sax)
        #    oax = axes.index(1)
        #    pad_loc[oax][odir] = 1
        #    pad_val[oax][odir] = 9
        # Perform the padding
        pad_loc = tuple(map(tuple, pad_loc))
        pad_val = tuple(map(tuple, pad_val))
        mat = np.pad(mat, pad_loc, 'constant', constant_values=pad_val)

        # Handle corner cases
        mat = self._handle_corner_cases(mat)
        return mat, pad_loc

    def _handle_corner_cases(self, mat):
        # take care of -1 corners
        for fixed_sides_1 in self.fixed.sides:
            for fixed_sides_2 in self.fixed.sides:
                for side1 in fixed_sides_1:
                    if side1.ax==self.sax: continue
                    axes = [0,0,0]
                    axes[side1.ax] = 1
                    axes.pop(self.sax)
                    ax1 = axes.index(1)
                    for side2 in fixed_sides_2:
                        if side2.ax==self.sax: continue
                        axes = [0,0,0]
                        axes[side2.ax] = 1
                        axes.pop(self.sax)
                        ax2 = axes.index(1)
                        if ax1==ax2: continue
                        ind = [0,0]
                        ind[ax1] = side1.dir*(mat.shape[ax1]-1)
                        ind[ax2] = side2.dir*(mat.shape[ax2]-1)
                        mat[tuple(ind)] = -1
        return mat

    # def _rough_milling_path(self,rough_pixs,lay_num,n):
    #     mvertices = []
    #
    #     # Defines axes
    #     ax = self.sax # mill bit axis
    #     dir = self.mesh.fab_directions[n]
    #     axes = [0,1,2]
    #     axes.pop(ax)
    #     dir_ax = axes[0] # primary milling direction axis
    #     off_ax = axes[1] # milling offset axis
    #
    #     # Define fabrication parameters
    #
    #     no_lanes = 2+math.ceil(((self.real_tim_dims[axes[1]]/self.dim)-2*self.fab.dia)/self.fab.dia)
    #     lane_width = (self.voxel_sizes[axes[1]]-self.fab.vdia)/(no_lanes-1)
    #     ratio = np.linalg.norm(self.pos_vecs[axes[1]])/self.voxel_sizes[axes[1]]
    #     v_vrad = self.fab.vrad*ratio
    #     lane_width = lane_width*ratio
    #
    #     # create offset direction vectors
    #     dir_vec = Utils.normalize(self.pos_vecs[axes[0]])
    #     off_vec = Utils.normalize(self.pos_vecs[axes[1]])
    #
    #     # get top ones to cut out
    #     for pix in rough_pixs:
    #         mverts = []
    #         if pix.outside: continue
    #         if no_lanes<=2:
    #             if pix.neighbors[0][0]==1 and pix.neighbors[0][1]==1: continue
    #             elif pix.neighbors[1][0]==1 and pix.neighbors[1][1]==1: continue
    #         pix_end = pix
    #
    #         # check that there is no previous same
    #         nind = pix.ind_abs.copy()
    #         nind[dir_ax] -=1
    #         found = False
    #         for pix2 in rough_pixs:
    #             if pix2.outside: continue
    #             if pix2.ind_abs[0]==nind[0] and pix2.ind_abs[1]==nind[1]:
    #                 if pix.neighbors[1][0]==pix2.neighbors[1][0]:
    #                     if pix.neighbors[1][1]==pix2.neighbors[1][1]:
    #                         found = True
    #                         break
    #         if found: continue
    #
    #         # find next same
    #         for i in range(self.dim):
    #             nind = pix.ind_abs.copy()
    #             nind[0] +=i
    #             found = False
    #             for pix2 in rough_pixs:
    #                 if pix2.outside: continue
    #                 if pix2.ind_abs[0]==nind[0] and pix2.ind_abs[1]==nind[1]:
    #                     if pix.neighbors[1][0]==pix2.neighbors[1][0]:
    #                         if pix.neighbors[1][1]==pix2.neighbors[1][1]:
    #                             found = True
    #                             pix_end = pix2
    #                             break
    #             if found==False: break
    #
    #         # start
    #         ind = list(pix.ind_abs)
    #         ind.insert(ax,(self.dim-1)*(1-dir)+(2*dir-1)*lay_num) # 0 when n is 1, dim-1 when n is 0
    #         add = [0,0,0]
    #         add[ax] = 1-dir
    #         i_pt = Utils.get_index(ind,add,self.dim)
    #         pt1 = Utils.get_vertex(i_pt,self.jverts[n],self.vertex_no_info)
    #         #end
    #         ind = list(pix_end.ind_abs)
    #         ind.insert(ax,(self.dim-1)*(1-dir)+(2*dir-1)*lay_num) # 0 when n is 1, dim-1 when n is 0
    #         add = [0,0,0]
    #         add[ax] = 1-dir
    #         add[dir_ax] = 1
    #         i_pt = Utils.get_index(ind,add,self.dim)
    #         pt2 = Utils.get_vertex(i_pt,self.jverts[n],self.vertex_no_info)
    #
    #         ### REFINE THIS FUNCTION
    #         dir_add1 = pix.neighbors[dir_ax][0]*2.5*self.fab.vrad*dir_vec
    #         dir_add2 = -pix_end.neighbors[dir_ax][1]*2.5*self.fab.vrad*dir_vec
    #
    #         pt1 = pt1+v_vrad*off_vec+dir_add1
    #         pt2 = pt2+v_vrad*off_vec+dir_add2
    #         for i in range(no_lanes):
    #             # skip lane if on blocked side in off direction
    #             if pix.neighbors[1][0]==1 and i==0: continue
    #             elif pix.neighbors[1][1]==1 and i==no_lanes-1: continue
    #
    #             ptA = pt1+lane_width*off_vec*i
    #             ptB = pt2+lane_width*off_vec*i
    #             pts = [ptA,ptB]
    #             if i%2==1: pts.reverse()
    #             for pt in pts: mverts.append(MillVertex(pt))
    #         mvertices.append(mverts)
    #     return mvertices

    def _rough_milling_path(self, rough_pixs, lay_num, n):
        mvertices = []

        # Defines axes
        ax = self.sax # mill bit axis
        dir = self.mesh.fab_directions[n]
        axes = [0,1,2]
        axes.pop(ax)
        dir_ax = axes[0] # primary milling direction axis
        off_ax = axes[1] # milling offset axis

        # Define fabrication parameters
        no_lanes, lane_width, v_vrad = self._calculate_lane_parameters(axes)

        # create offset direction vectors
        dir_vec = Utils.normalize(self.pos_vecs[axes[0]])
        off_vec = Utils.normalize(self.pos_vecs[axes[1]])

        # Process each pixel
        for pix in rough_pixs:
            mverts = self._process_rough_pixel(pix, rough_pixs, dir_ax, no_lanes, v_vrad, lane_width, dir_vec, off_vec)
            if mverts:
                mvertices.append(mverts)

        return mvertices

    def _calculate_lane_parameters(self, axes):
        no_lanes = 2+math.ceil(((self.real_tim_dims[axes[1]]/self.dim)-2*self.fab.dia)/self.fab.dia)
        lane_width = (self.voxel_sizes[axes[1]]-self.fab.vdia)/(no_lanes-1)
        ratio = np.linalg.norm(self.pos_vecs[axes[1]])/self.voxel_sizes[axes[1]]
        v_vrad = self.fab.vrad*ratio
        lane_width = lane_width*ratio
        return no_lanes, lane_width, v_vrad

    def _process_rough_pixel(self, pix, rough_pixs, dir_ax, no_lanes, v_vrad, lane_width, dir_vec, off_vec):
        if pix.outside:
            return None

        if no_lanes <= 2:
            if pix.neighbors[0][0]==1 and pix.neighbors[0][1]==1:
                return None
            elif pix.neighbors[1][0]==1 and pix.neighbors[1][1]==1:
                return None

        # Check if there is a previous same pixel
        if self._has_previous_same_pixel(pix, rough_pixs, dir_ax):
            return None

        # Find the end pixel in the same row
        pix_end = self._find_end_pixel(pix, rough_pixs, dir_ax)

        # Calculate start and end points
        pt1, pt2 = self._calculate_rough_path_endpoints(pix, pix_end, dir_ax, v_vrad, dir_vec, off_vec)

        # Generate vertices for each lane
        return self._generate_lane_vertices(pt1, pt2, pix, pix_end, no_lanes, lane_width, off_vec, v_vrad)

    def _has_previous_same_pixel(self, pix, rough_pixs, dir_ax):
        nind = pix.ind_abs.copy()
        nind[dir_ax] -= 1
        for pix2 in rough_pixs:
            if pix2.outside:
                continue
            if (pix2.ind_abs[0] == nind[0] and
                pix2.ind_abs[1] == nind[1] and
                pix.neighbors[1][0] == pix2.neighbors[1][0] and
                pix.neighbors[1][1] == pix2.neighbors[1][1]):
                return True
        return False

    def _find_end_pixel(self, pix, rough_pixs, dir_ax):
        pix_end = pix
        for i in range(self.dim):
            nind = pix.ind_abs.copy()
            nind[0] += i
            found = False
            for pix2 in rough_pixs:
                if pix2.outside:
                    continue
                if (pix2.ind_abs[0] == nind[0] and
                    pix2.ind_abs[1] == nind[1] and
                    pix.neighbors[1][0] == pix2.neighbors[1][0] and
                    pix.neighbors[1][1] == pix2.neighbors[1][1]):
                    found = True
                    pix_end = pix2
                    break
            if not found:
                break
        return pix_end

    def _calculate_rough_path_endpoints(self, pix, pix_end, dir_ax, v_vrad, dir_vec, off_vec):
        # Calculate start point
        ind_start = list(pix.ind_abs)
        ind_start.insert(self.sax, (self.dim-1)*(1-self.mesh.fab_directions[0])+(2*self.mesh.fab_directions[0]-1)*0)
        add_start = [0,0,0]
        add_start[self.sax] = 1-self.mesh.fab_directions[0]
        i_pt_start = Utils.get_index(ind_start, add_start, self.dim)
        pt1 = Utils.get_vertex(i_pt_start, self.jverts[0], self.vertex_no_info)

        # Calculate end point
        ind_end = list(pix_end.ind_abs)
        ind_end.insert(self.sax, (self.dim-1)*(1-self.mesh.fab_directions[0])+(2*self.mesh.fab_directions[0]-1)*0)
        add_end = [0,0,0]
        add_end[self.sax] = 1-self.mesh.fab_directions[0]
        add_end[dir_ax] = 1
        i_pt_end = Utils.get_index(ind_end, add_end, self.dim)
        pt2 = Utils.get_vertex(i_pt_end, self.jverts[0], self.vertex_no_info)

        # Apply offsets
        dir_add1 = pix.neighbors[dir_ax][0]*2.5*self.fab.vrad*dir_vec
        dir_add2 = -pix_end.neighbors[dir_ax][1]*2.5*self.fab.vrad*dir_vec
        pt1 = pt1 + v_vrad*off_vec + dir_add1
        pt2 = pt2 + v_vrad*off_vec + dir_add2

        return pt1, pt2

    def _generate_lane_vertices(self, pt1, pt2, pix, pix_end, no_lanes, lane_width, off_vec, v_vrad):
        mverts = []
        for i in range(no_lanes):
            # Skip lane if on blocked side in off direction
            if pix.neighbors[1][0]==1 and i==0:
                continue
            elif pix.neighbors[1][1]==1 and i==no_lanes-1:
                continue

            ptA = pt1 + lane_width*off_vec*i
            ptB = pt2 + lane_width*off_vec*i
            pts = [ptA, ptB]
            if i%2==1:
                pts.reverse()
            for pt in pts:
                mverts.append(MillVertex(pt))
        return mverts

    # def _edge_milling_path(self,lay_num,n):
    #     mverts = []
    #
    #     if len(self.fixed.sides[n])==1 and self.fixed.sides[n][0].ax!=self.sax:
    #
    #         # ax dir of current fixed side
    #         ax = self.fixed.sides[n][0].ax
    #         dir = self.fixed.sides[n][0].dir
    #         # oax - axis perp. to component axis
    #         oax = [0,1,2]
    #         oax.remove(self.sax)
    #         oax.remove(ax)
    #         oax=oax[0]
    #         # fabrication direction
    #         fdir = self.mesh.fab_directions[n]
    #
    #         # check so that that part is not removed anyways...
    #         # i.e. if the whole bottom row in that direction is of other material
    #         ind = [0,0,0]
    #         ind[ax] = (1-dir)*(self.dim-1)
    #         ind[self.sax] = fdir*(self.dim-1)
    #         free=True
    #         for i in range(self.dim):
    #             ind[oax]=i
    #             val = self.mesh.voxel_matrix[tuple(ind)]
    #             if int(val)==n:
    #                 free=False
    #                 break
    #
    #         if not free:
    #             # define start (pt0) and end (pt1) points of edge
    #             ind = [0,0,0]
    #             add = [0,0,0]
    #             ind[ax] = (1-dir)*self.dim
    #             ind[self.sax] = self.dim*(1-fdir)+(2*fdir-1)*lay_num
    #             i_pt = Utils.get_index(ind,add,self.dim)
    #             pt0 = Utils.get_vertex(i_pt,self.jverts[n],self.vertex_no_info)
    #             ind[oax] = self.dim
    #             i_pt = Utils.get_index(ind,add,self.dim)
    #             pt1 = Utils.get_vertex(i_pt,self.jverts[n],self.vertex_no_info)
    #
    #             # offset edge line by radius of millingbit
    #             dir_vec = Utils.normalize(pt0-pt1)
    #             sax_vec = [0,0,0]
    #             sax_vec[self.sax] = 2*fdir-1
    #             off_vec = Utils.rotate_vector_around_axis(dir_vec, sax_vec, math.radians(90))
    #             off_vec = (2*dir-1)*self.fab.vrad*off_vec
    #             pt0 = pt0+off_vec
    #             pt1 = pt1+off_vec
    #
    #             # Write to mverts
    #             mverts = [MillVertex(pt0),MillVertex(pt1)]
    #
    #     return mverts

    def _edge_milling_path(self, lay_num, n):
        """Generate milling path for the edge of a component."""
        mverts = []

        # Only process if there's exactly one fixed side that's not on the sliding axis
        if not self._should_create_edge_path(n):
            return mverts

        # Get axes information
        ax, dir, oax, fdir = self._get_edge_axes_info(n)

        # Check if the edge needs to be milled
        if self._is_edge_already_removed(ax, dir, oax, fdir):
            return mverts

        # Calculate edge path points
        pt0, pt1 = self._calculate_edge_endpoints(ax, dir, oax, lay_num, fdir, n)

        # Apply offset to edge line
        pt0, pt1 = self._apply_edge_offset(pt0, pt1, dir, fdir)

        # Create milling vertices
        mverts = [MillVertex(pt0), MillVertex(pt1)]
        return mverts

    def _should_create_edge_path(self, n):
        """Check if an edge path should be created for this component."""
        return len(self.fixed.sides[n]) == 1 and self.fixed.sides[n][0].ax != self.sax

    def _get_edge_axes_info(self, n):
        """Get axis information for edge milling."""
        # Get axis and direction of current fixed side
        ax = self.fixed.sides[n][0].ax
        dir = self.fixed.sides[n][0].dir

        # Get axis perpendicular to component axis and sliding axis
        oax = [0, 1, 2]
        oax.remove(self.sax)
        oax.remove(ax)
        oax = oax[0]

        # Get fabrication direction
        fdir = self.mesh.fab_directions[n]

        return ax, dir, oax, fdir

    def _is_edge_already_removed(self, ax, dir, oax, fdir):
        """Check if the edge is already removed by other operations."""
        # Check if the whole bottom row in that direction is of other material
        ind = [0, 0, 0]
        ind[ax] = (1-dir) * (self.dim-1)
        ind[self.sax] = fdir * (self.dim-1)

        for i in range(self.dim):
            ind[oax] = i
            val = self.mesh.voxel_matrix[tuple(ind)]
            if int(val) == 0:  # If any voxel belongs to this component
                return False

        return True  # All voxels are of other material, so edge is already removed

    def _calculate_edge_endpoints(self, ax, dir, oax, lay_num, fdir, n):
        """Calculate the start and end points of the edge path."""
        # Define start point (pt0)
        ind = [0, 0, 0]
        add = [0, 0, 0]
        ind[ax] = (1-dir) * self.dim
        ind[self.sax] = self.dim * (1-fdir) + (2*fdir-1) * lay_num
        i_pt = Utils.get_index(ind, add, self.dim)
        pt0 = Utils.get_vertex(i_pt, self.jverts[n], self.vertex_no_info)

        # Define end point (pt1)
        ind[oax] = self.dim
        i_pt = Utils.get_index(ind, add, self.dim)
        pt1 = Utils.get_vertex(i_pt, self.jverts[n], self.vertex_no_info)

        return pt0, pt1

    def _apply_edge_offset(self, pt0, pt1, dir, fdir):
        """Offset the edge line by the radius of the milling bit."""
        # Calculate direction vector
        dir_vec = Utils.normalize(pt0 - pt1)

        # Calculate offset vector
        sax_vec = [0, 0, 0]
        sax_vec[self.sax] = 2 * fdir - 1
        off_vec = Utils.rotate_vector_around_axis(dir_vec, sax_vec, math.radians(90))
        off_vec = (2 * dir - 1) * self.fab.vrad * off_vec

        # Apply offset
        pt0 = pt0 + off_vec
        pt1 = pt1 + off_vec

        return pt0, pt1

    def _offset_verts(self,neighbor_vectors,neighbor_vectors_a,neighbor_vectors_b,verts,lay_num,n):
        outline = []
        corner_artifacts = []

        fdir = self.mesh.fab_directions[n]

        test_first=True
        for i,rv in enumerate(list(verts)): # browse each vertex in the outline

            # remove vertices with neighbor count 2 #OK
            if rv.region_count==2 and rv.block_count==2: continue # redundant
            if rv.block_count==0: continue                        # redundant
            if rv.ind[0]<0 or rv.ind[0]>self.dim: continue        # out of bounds
            if rv.ind[1]<0 or rv.ind[1]>self.dim: continue        # out of bounds

            # add vertex information #OK
            ind = rv.ind.copy()
            ind.insert(self.sax,(self.dim-1)*(1-fdir)+(2*fdir-1)*lay_num)
            add = [0,0,0]
            add[self.sax] = 1-fdir
            i_pt = Utils.get_index(ind,add,self.dim)
            pt = Utils.get_vertex(i_pt,self.jverts[n],self.vertex_no_info)

            # move vertex according to boundary condition <---needs to be updated
            off_vecs = []
            if rv.block_count==1:
                nind = tuple(np.argwhere(rv.neighbors==1)[0])
                off_vecs.append(-neighbor_vectors[nind])
            if rv.region_count==1 and rv.free_count!=3:
                nind = tuple(np.argwhere(rv.neighbors==0)[0])
                off_vecs.append(neighbor_vectors[nind])
                if np.any(rv.flat_neighbor_values==-2):
                    nind = tuple(np.argwhere(rv.neighbor_values==-2)[0])
                    off_vecs.append(neighbor_vectors[nind])

            off_vec = np.average(off_vecs,axis=0)
            # check if it is an outer corner that should be rounded
            rounded = False
            if rv.region_count==3: # outer corner, check if it should be rounded or not
                # check if this outer corner correspond to an inner corner of another material
                for n2 in range(self.noc):
                    if n2==n: continue
                    cnt = np.sum(rv.flat_neighbor_values==n2)
                    if cnt==3: rounded = True
                    elif cnt==2:
                        # Check if it is a diagonal
                        dia1 = rv.neighbor_values[0][0]==rv.neighbor_values[1][1]
                        dia2 = rv.neighbor_values[0][1]==rv.neighbor_values[1][0]
                        if dia1 or dia2:
                            rounded = True
            if rounded:
                nind = tuple(np.argwhere(rv.neighbors==1)[0])
                off_vec_a = -neighbor_vectors_a[nind]
                off_vec_b = -neighbor_vectors_b[nind]
                le2 = math.sqrt(math.pow(2*np.linalg.norm(off_vec_a+off_vec_b),2)-math.pow(2*self.fab.vrad,2))-np.linalg.norm(off_vec_a)
                off_vec_a2 = Utils.set_vector_length(off_vec_a,le2)
                off_vec_b2 = Utils.set_vector_length(off_vec_b,le2)

                # define end points and the center point of the arc
                pt1 = pt+off_vec_a-off_vec_b2
                pt2 = pt+off_vec_b-off_vec_a2
                pts = [pt1,pt2]
                ctr = pt-off_vec_a-off_vec_b # arc center


                # Reorder pt1 and pt2
                if len(outline)>0: # if it is not the first point in the outline
                    ppt = outline[-1].pt
                    v1 = pt1-ppt
                    v2 = pt2-ppt
                    ang1 = Utils.angle_between_vectors1(v1, off_vec_b) #should be 0 if order is already good
                    ang2 = Utils.angle_between_vectors1(v2, off_vec_b) #should be more than 0
                    if ang1>ang2: pts.reverse()
                outline.append(MillVertex(pts[0],is_arc=True,arc_ctr=ctr))
                outline.append(MillVertex(pts[1],is_arc=True,arc_ctr=ctr))

                # Extreme case where corner is very rounded and everything is not cut
                dist = np.linalg.norm(pt-ctr)
                if dist>self.fab.vdia and lay_num<self.dim-1:
                    artifact = []
                    v0 = self.fab.vdia * Utils.normalize(pt+off_vec-pts[0])
                    v1 = self.fab.vdia * Utils.normalize(pt+off_vec-pts[1])
                    vp = self.fab.vrad * Utils.normalize(pts[1]-pts[0])
                    pts3 =  [pts[0]-vp+v0,pt+2*off_vec,pts[1]+vp+v1]
                    while np.linalg.norm(pts3[2]-pts3[0])>self.fab.vdia:
                        pts3[0] += vp
                        pts3[1] += -off_vec
                        pts3[2] += -vp
                        for i in range(3): artifact.append(MillVertex(pts3[i]))
                        pts3.reverse()
                        vp = -vp
                    if len(artifact)>0:
                        corner_artifacts.append(artifact)

            else: # other corner
                pt = pt+off_vec
                outline.append(MillVertex(pt))
            if len(outline)>2 and outline[0].is_arc and test_first:
                # if the previous one was an arc but it was the first point of the outline,
                # so we couldn't verify the order of the points
                # we might need to retrospectively switch order of the arc points
                npt = outline[2].pt
                d1 = np.linalg.norm(outline[0].pt-npt)
                d2 = np.linalg.norm(outline[1].pt-npt)
                if d1<d2: outline[0],outline[1] = outline[1],outline[0]
                test_first=False

        return outline, corner_artifacts

    # def _get_layered_vertices(self, outline,n,lay_num,no_z,dep):
    #     verts = []
    #     mverts = []
    #
    #     r = g = b = tx = ty = 0.0
    #
    #     fdir = self.mesh.fab_directions[n]
    #     # add startpoint
    #     start_vert = [outline[0].x,outline[0].y,outline[0].z]
    #     safe_height = outline[0].pt[self.sax]-(2*fdir-1)*(lay_num*self.voxel_sizes[self.sax]+2*dep)
    #     start_vert[self.sax] = safe_height
    #     mverts.append(MillVertex(start_vert,is_tra=True))
    #     verts.extend([start_vert[0],start_vert[1],start_vert[2],r,g,b,tx,ty])
    #     if lay_num!=0:
    #         start_vert2 = [outline[0].x,outline[0].y,outline[0].z]
    #         safe_height2 = outline[0].pt[self.sax]-(2*fdir-1)*dep
    #         start_vert2[self.sax] = safe_height2
    #         mverts.append(MillVertex(start_vert2,is_tra=True))
    #         verts.extend([start_vert2[0],start_vert2[1],start_vert2[2],r,g,b,tx,ty])
    #
    #     # add layers with Z-height
    #     # set start number (one layer earlier if first layer)
    #     if lay_num==0: stn=0
    #     else: stn=1
    #     # set end number (one layer more if last layer and not sliding direction aligned component)
    #     if lay_num==self.dim-1 and self.sax!=self.fixed.sides[n][0].ax: enn=no_z+2
    #     else: enn=no_z+1
    #     if self.incremental:
    #         enn+=1
    #         seg_props = Utils.get_segment_proportions(outline)
    #     else: seg_props = [1.0]*len(outline)
    #     #calculate depth for incremental setting
    #
    #     for num in range(stn,enn):
    #         if self.incremental and num==enn-1: seg_props = [0.0]*len(outline)
    #         for i, (mv, sp) in enumerate(zip(outline,seg_props)):
    #             pt = [mv.x,mv.y,mv.z]
    #             pt[self.sax] += (2*fdir-1)*(num-1+sp)*dep
    #             if mv.is_arc:
    #                 ctr = [mv.arc_ctr[0],mv.arc_ctr[1],mv.arc_ctr[2]]
    #                 ctr[self.sax] += (2*fdir-1)*(num-1+sp)*dep
    #                 mverts.append(MillVertex(pt, is_arc=True, arc_ctr=ctr))
    #             else:
    #                 mverts.append(MillVertex(pt))
    #             if i>0:
    #                 pmv = outline[i-1]
    #             if i>0 and Utils.connected_arc(mv,pmv):
    #                 ppt = [pmv.x,pmv.y,pmv.z]
    #                 ppt[self.sax] += (2*fdir-1)*(num-1+sp)*dep
    #                 pctr = [pmv.arc_ctr[0],pmv.arc_ctr[1],pmv.arc_ctr[2]]
    #                 pctr[self.sax] += (2*fdir-1)*(num-1+sp)*dep
    #                 arc_pts = Utils.arc_points(ppt,pt,pctr,ctr,self.sax,math.radians(5))
    #                 for arc_pt in arc_pts: verts.extend([arc_pt[0],arc_pt[1],arc_pt[2],r,g,b,tx,ty])
    #             else:
    #                 verts.extend([pt[0],pt[1],pt[2],r,g,b,tx,ty])
    #         outline.reverse()
    #
    #     # add endpoint
    #     end_vert = [outline[0].x,outline[0].y,outline[0].z]
    #     end_vert[self.sax] = safe_height
    #     mverts.append(MillVertex(end_vert, is_tra=True))
    #     verts.extend([end_vert[0],end_vert[1],end_vert[2],r,g,b,tx,ty])
    #
    #     return verts,mverts

    def _get_layered_vertices(self, outline, n, lay_num, no_z, dep):
        verts = []
        mverts = []

        # Extract common calculations to a helper method
        fdir, safe_height = self._calculate_layer_parameters(outline, lay_num, n, dep)

        # Handle start points in a separate method
        verts, mverts = self._add_layer_start_points(outline, lay_num, fdir, safe_height, verts, mverts)

        # Handle z-layers in a separate method
        verts, mverts = self._add_z_layers(outline, lay_num, n, no_z, dep, fdir, verts, mverts)

        # Handle end point in a separate method
        verts, mverts = self._add_layer_end_point(outline, safe_height, verts, mverts)

        return verts, mverts

    def _calculate_layer_parameters(self, outline, lay_num, n, dep):
        """Calculate common parameters for layered vertices."""
        fdir = self.mesh.fab_directions[n]
        safe_height = outline[0].pt[self.sax] - (2*fdir-1) * (lay_num*self.voxel_sizes[self.sax] + 2*dep)
        return fdir, safe_height

    def _add_layer_start_points(self, outline, lay_num, fdir, safe_height, verts, mverts):
        """Add start points for the layer."""
        r = g = b = tx = ty = 0.0

        # Add initial safe height point
        start_vert = [outline[0].x, outline[0].y, outline[0].z]
        start_vert[self.sax] = safe_height
        mverts.append(MillVertex(start_vert, is_tra=True))
        verts.extend([start_vert[0], start_vert[1], start_vert[2], r, g, b, tx, ty])

        # Add intermediate point for non-first layers
        if lay_num != 0:
            start_vert2 = [outline[0].x, outline[0].y, outline[0].z]
            safe_height2 = outline[0].pt[self.sax] - (2*fdir-1) * dep
            start_vert2[self.sax] = safe_height2
            mverts.append(MillVertex(start_vert2, is_tra=True))
            verts.extend([start_vert2[0], start_vert2[1], start_vert2[2], r, g, b, tx, ty])

        return verts, mverts

    def _add_z_layers(self, outline, lay_num, n, no_z, dep, fdir, verts, mverts):
        """Add z-layers for the milling path."""
        r = g = b = tx = ty = 0.0

        # Determine start and end numbers
        stn = 0 if lay_num == 0 else 1
        enn = self._calculate_end_layer_number(lay_num, n, no_z)

        # Handle incremental setting
        seg_props = self._calculate_segment_properties(outline, enn)

        # Process each z-layer
        for num in range(stn, enn):
            if self.incremental and num == enn - 1:
                seg_props = [0.0] * len(outline)

            # Process each point in the outline
            for i, (mv, sp) in enumerate(zip(outline, seg_props)):
                verts, mverts = self._process_z_layer_point(mv, sp, i, num, dep, fdir, outline, verts, mverts)

            outline.reverse()

        return verts, mverts

    def _calculate_end_layer_number(self, lay_num, n, no_z):
        """Calculate the end layer number based on conditions."""
        if lay_num == self.dim - 1 and self.sax != self.fixed.sides[n][0].ax:
            enn = no_z + 2
        else:
            enn = no_z + 1

        if self.incremental:
            enn += 1

        return enn

    def _calculate_segment_properties(self, outline, enn):
        """Calculate segment properties for incremental setting."""
        if self.incremental:
            return Utils.get_segment_proportions(outline)
        else:
            return [1.0] * len(outline)

    def _process_z_layer_point(self, mv, sp, i, num, dep, fdir, outline, verts, mverts):
        """Process a single point in a z-layer."""
        r = g = b = tx = ty = 0.0

        # Calculate point position
        pt = [mv.x, mv.y, mv.z]
        pt[self.sax] += (2*fdir-1) * (num-1+sp) * dep

        # Handle arc points
        if mv.is_arc:
            ctr = [mv.arc_ctr[0], mv.arc_ctr[1], mv.arc_ctr[2]]
            ctr[self.sax] += (2*fdir-1) * (num-1+sp) * dep
            mverts.append(MillVertex(pt, is_arc=True, arc_ctr=ctr))
        else:
            mverts.append(MillVertex(pt))

        # Handle connected arcs
        if i > 0:
            pmv = outline[i-1]
            if Utils.connected_arc(mv, pmv):
                verts = self._add_arc_points(mv, pmv, i, num, sp, dep, fdir, verts)
            else:
                verts.extend([pt[0], pt[1], pt[2], r, g, b, tx, ty])
        else:
            verts.extend([pt[0], pt[1], pt[2], r, g, b, tx, ty])

        return verts, mverts

    def _add_arc_points(self, mv, pmv, i, num, sp, dep, fdir, verts):
        """Add points for an arc segment."""
        r = g = b = tx = ty = 0.0

        # Calculate previous point and centers
        ppt = [pmv.x, pmv.y, pmv.z]
        ppt[self.sax] += (2*fdir-1) * (num-1+sp) * dep

        pctr = [pmv.arc_ctr[0], pmv.arc_ctr[1], pmv.arc_ctr[2]]
        pctr[self.sax] += (2*fdir-1) * (num-1+sp) * dep

        ctr = [mv.arc_ctr[0], mv.arc_ctr[1], mv.arc_ctr[2]]
        ctr[self.sax] += (2*fdir-1) * (num-1+sp) * dep

        pt = [mv.x, mv.y, mv.z]
        pt[self.sax] += (2*fdir-1) * (num-1+sp) * dep

        # Generate arc points
        arc_pts = Utils.arc_points(ppt, pt, pctr, ctr, self.sax, math.radians(5))
        for arc_pt in arc_pts:
            verts.extend([arc_pt[0], arc_pt[1], arc_pt[2], r, g, b, tx, ty])

        return verts

    def _add_layer_end_point(self, outline, safe_height, verts, mverts):
        """Add end point for the layer."""
        r = g = b = tx = ty = 0.0

        end_vert = [outline[0].x, outline[0].y, outline[0].z]
        end_vert[self.sax] = safe_height
        mverts.append(MillVertex(end_vert, is_tra=True))
        verts.extend([end_vert[0], end_vert[1], end_vert[2], r, g, b, tx, ty])

        return verts, mverts

    def _get_milling_end_points(self,n,last_z):
        verts = []
        mverts = []

        r = g = b = tx = ty = 0.0

        fdir = self.mesh.fab_directions[n]

        origin_vert = [0,0,0]
        origin_vert[self.sax] = last_z

        extra_zheight = 15/self.ratio
        above_origin_vert = [0,0,0]
        above_origin_vert[self.sax] = last_z-(2*fdir-1)*extra_zheight

        mverts.append(MillVertex(origin_vert, is_tra=True))
        mverts.append(MillVertex(above_origin_vert, is_tra=True))
        verts.extend([origin_vert[0],origin_vert[1],origin_vert[2],r,g,b,tx,ty])
        verts.extend([above_origin_vert[0],above_origin_vert[1],above_origin_vert[2],r,g,b,tx,ty])

        return verts,mverts

    # def _milling_path_vertices(self, n):
    #
    #     vertices = []
    #     milling_vertices = []
    #
    #
    #     min_vox_size = np.min(self.voxel_sizes)
    #     # Check that the milling bit is not too large for the voxel size
    #     if np.min(self.voxel_sizes)<self.fab.vdia: print("Could not generate milling path. The milling bit is too large.")
    #
    #     # Calculate depth constants
    #     no_z = int(self.ratio*self.voxel_sizes[self.sax]/self.fab.dep)
    #     dep = self.voxel_sizes[self.sax]/no_z
    #
    #     # Defines axes and vectors
    #     fdir = self.mesh.fab_directions[n]
    #     axes = [0,1,2]
    #     axes.pop(self.sax)
    #     dir_ax = axes[0] # primary milling direction axis
    #     off_ax = axes[1] # milling offset axis
    #     ### new for oblique angles ### neighbor vectors
    #     le = self.fab.vrad/math.cos(abs(math.radians(-self.ang)))
    #     dir_vec = le*self.pos_vecs[axes[0]]/np.linalg.norm(self.pos_vecs[axes[0]])
    #     off_vec = le*self.pos_vecs[axes[1]]/np.linalg.norm(self.pos_vecs[axes[1]])
    #     neighbor_vectors = []
    #     neighbor_vectors_a = []
    #     neighbor_vectors_b = []
    #     for x in range(-1,2,2):
    #         temp = []
    #         tempa = []
    #         tempb = []
    #         for y in range(-1,2,2):
    #             temp.append(x*dir_vec+y*off_vec)
    #             tempa.append(x*dir_vec)
    #             tempb.append(y*off_vec)
    #         neighbor_vectors.append(temp)
    #         neighbor_vectors_a.append(tempa)
    #         neighbor_vectors_b.append(tempb)
    #     neighbor_vectors = np.array(neighbor_vectors)
    #     neighbor_vectors_a = np.array(neighbor_vectors_a)
    #     neighbor_vectors_b = np.array(neighbor_vectors_b)
    #
    #     # Browse layers
    #     for lay_num in range(self.dim):
    #
    #         # Create a 2D matrix of current layer
    #         lay_mat = self._layer_mat_from_cube(lay_num, n) #OK
    #
    #         # Pad 2d matrix with fixed sides
    #         lay_mat,pad_loc = self._pad_layer_mat_with_fixed_sides(lay_mat, n) #OK
    #         org_lay_mat = copy.deepcopy(lay_mat) #OK
    #
    #         # Get/browse regions
    #         for reg_num in range(self.dim*self.dim):
    #
    #             # Get indices of a region
    #             inds = np.argwhere((lay_mat!=-1) & (lay_mat!=n)) #OK
    #             if len(inds)==0: break #OK
    #             reg_inds = Utils.get_diff_neighbors(lay_mat,[inds[0]],n) #OK
    #
    #             #If oblique joint, create path to trim edge
    #             edge_path = []
    #             if abs(self.ang)>1: edge_path = self._edge_milling_path(lay_num, n)
    #             if len(edge_path)>0:
    #                 verts,mverts = self._get_layered_vertices(edge_path,n,lay_num,no_z,dep)
    #                 vertices.extend(verts)
    #                 milling_vertices.extend(mverts)
    #
    #             # Anaylize which voxels needs to be roguhly cut initially
    #             # 1. Add all open voxels in the region
    #             rough_inds = []
    #             for ind in reg_inds:
    #                 rough_inds.append(RoughPixel(ind, lay_mat, pad_loc,self.dim,n)) #should be same...
    #             # 2. Produce rough milling paths
    #             rough_paths = self._rough_milling_path(rough_inds,lay_num,n)
    #             for rough_path in rough_paths:
    #                 if len(rough_path)>0:
    #                     verts,mverts = self._get_layered_vertices(rough_path,n,lay_num,no_z,dep)
    #                     vertices.extend(verts)
    #                     milling_vertices.extend(mverts)
    #
    #             # Overwrite detected regin in original matrix
    #             for reg_ind in reg_inds: lay_mat[tuple(reg_ind)]=n #OK
    #
    #             # Make a list of all edge vertices of the outline of the region
    #             reg_verts = Utils.get_region_outline_vertices(reg_inds,lay_mat,org_lay_mat,pad_loc,n) #OK
    #
    #             # Order the vertices to create an outline
    #             for isl_num in range(10):
    #                 reg_ord_verts = []
    #                 if len(reg_verts)==0: break
    #
    #                 #Make sure first item in region vertices is on blocked/free corner, or blocked
    #                 reg_verts = Utils.set_starting_vert(reg_verts) #OK
    #
    #                 #Get a sequence of ordered vertices
    #                 reg_ord_verts,reg_verts,closed = Utils.get_sublist_of_ordered_verts(reg_verts) #OK
    #
    #                 # Make outline of ordered vertices (for dedugging only!!!!!!!)
    #                 # if len(reg_ord_verts)>1: outline = Utils.get_outline(joint_self,reg_ord_verts,lay_num,n)
    #
    #                 # Offset vertices according to boundary condition (and remove if redundant)
    #                 outline,corner_artifacts = self._offset_verts(neighbor_vectors,neighbor_vectors_a,neighbor_vectors_b,reg_ord_verts,lay_num,n) #<----needs to be updated for oblique angles!!!!!<---
    #
    #                 # Get z height and extend vertices to global list
    #                 if len(reg_ord_verts)>1 and len(outline)>0:
    #                     if closed: outline.append(MillVertex(outline[0].pt))
    #                     verts,mverts = self._get_layered_vertices(outline,n,lay_num,no_z,dep)
    #                     vertices.extend(verts)
    #                     milling_vertices.extend(mverts)
    #
    #                 if len(corner_artifacts)>0:
    #                     for artifact in corner_artifacts:
    #                         verts,mverts = self._get_layered_vertices(artifact,n,lay_num,no_z,dep)
    #                         vertices.extend(verts)
    #                         milling_vertices.extend(mverts)
    #
    #     # Add end point - check first if it is empty
    #     if milling_vertices:
    #         end_verts, end_mverts = self._get_milling_end_points(n,milling_vertices[-1].pt[self.sax])
    #         vertices.extend(end_verts)
    #         milling_vertices.extend(end_mverts)
    #
    #     # Format and return
    #     vertices = np.array(vertices, dtype = np.float32)
    #
    #     return vertices, milling_vertices

    def _milling_path_vertices(self, n):
        """Generate vertices for milling paths for component n."""
        vertices = []
        milling_vertices = []

        # Validate milling bit size
        if not self._validate_milling_bit_size():
            return np.array([], dtype=np.float32), []

        # Calculate depth parameters
        depth_params = self._calculate_depth_parameters()
        no_z, dep = depth_params['no_z'], depth_params['dep']

        # Calculate neighbor vectors for offset calculations
        neighbor_vectors, neighbor_vectors_a, neighbor_vectors_b = self._calculate_neighbor_vectors(n)

        # Process each layer
        for lay_num in range(self.dim):
            # Create and prepare layer matrices
            lay_mat, pad_loc, org_lay_mat = self._prepare_layer_matrices(lay_num, n)

            # Process regions in the layer
            self._process_regions_in_layer(lay_mat, org_lay_mat, pad_loc, lay_num, n, no_z, dep,
                                          neighbor_vectors, neighbor_vectors_a, neighbor_vectors_b,
                                          vertices, milling_vertices)

        # Add end point if there are any milling vertices
        if milling_vertices:
            end_verts, end_mverts = self._get_milling_end_points(n, milling_vertices[-1].pt[self.sax])
            vertices.extend(end_verts)
            milling_vertices.extend(end_mverts)

        # Format and return
        return np.array(vertices, dtype=np.float32), milling_vertices

    def _validate_milling_bit_size(self):
        """Check if the milling bit is not too large for the voxel size."""
        if np.min(self.voxel_sizes) < self.fab.vdia:
            print("Could not generate milling path. The milling bit is too large.")
            return False
        return True

    def _calculate_depth_parameters(self):
        """Calculate depth-related parameters for milling."""
        no_z = int(self.ratio * self.voxel_sizes[self.sax] / self.fab.dep)
        dep = self.voxel_sizes[self.sax] / no_z
        return {'no_z': no_z, 'dep': dep}

    def _calculate_neighbor_vectors(self, n):
        """Calculate neighbor vectors for offset calculations."""
        # Define axes and vectors
        axes = [0, 1, 2]
        axes.pop(self.sax)
        dir_ax, off_ax = axes[0], axes[1]  # primary milling direction axis, milling offset axis

        # Calculate length and direction vectors
        le = self.fab.vrad / math.cos(abs(math.radians(-self.ang)))
        dir_vec = le * self.pos_vecs[axes[0]] / np.linalg.norm(self.pos_vecs[axes[0]])
        off_vec = le * self.pos_vecs[axes[1]] / np.linalg.norm(self.pos_vecs[axes[1]])

        # Initialize neighbor vectors arrays
        neighbor_vectors = []
        neighbor_vectors_a = []
        neighbor_vectors_b = []

        # Calculate neighbor vectors in all directions
        for x in range(-1, 2, 2):
            temp, tempa, tempb = [], [], []
            for y in range(-1, 2, 2):
                temp.append(x * dir_vec + y * off_vec)
                tempa.append(x * dir_vec)
                tempb.append(y * off_vec)
            neighbor_vectors.append(temp)
            neighbor_vectors_a.append(tempa)
            neighbor_vectors_b.append(tempb)

        return np.array(neighbor_vectors), np.array(neighbor_vectors_a), np.array(neighbor_vectors_b)

    def _prepare_layer_matrices(self, lay_num, n):
        """Create and prepare matrices for the current layer."""
        # Create a 2D matrix of current layer
        lay_mat = self._layer_mat_from_cube(lay_num, n)

        # Pad 2d matrix with fixed sides
        lay_mat, pad_loc = self._pad_layer_mat_with_fixed_sides(lay_mat, n)
        org_lay_mat = copy.deepcopy(lay_mat)

        return lay_mat, pad_loc, org_lay_mat

    def _process_regions_in_layer(self, lay_mat, org_lay_mat, pad_loc, lay_num, n, no_z, dep,
                                 neighbor_vectors, neighbor_vectors_a, neighbor_vectors_b,
                                 vertices, milling_vertices):
        """Process all regions in the current layer."""
        # Get/browse regions
        for reg_num in range(self.dim * self.dim):
            # Get indices of a region
            inds = np.argwhere((lay_mat != -1) & (lay_mat != n))
            if len(inds) == 0:
                break

            # Get all connected indices in the region
            reg_inds = Utils.get_diff_neighbors(lay_mat, [inds[0]], n)

            # Process edge paths for oblique joints
            self._process_edge_paths(lay_num, n, no_z, dep, vertices, milling_vertices)

            # Process rough milling paths
            self._process_rough_milling(reg_inds, lay_mat, pad_loc, lay_num, n, no_z, dep,
                                       vertices, milling_vertices)

            # Mark processed region in the matrix
            for reg_ind in reg_inds:
                lay_mat[tuple(reg_ind)] = n

            # Process region outlines
            self._process_region_outlines(reg_inds, lay_mat, org_lay_mat, pad_loc, lay_num, n, no_z, dep,
                                         neighbor_vectors, neighbor_vectors_a, neighbor_vectors_b,
                                         vertices, milling_vertices)

    def _process_edge_paths(self, lay_num, n, no_z, dep, vertices, milling_vertices):
        """Process edge paths for oblique joints."""
        if abs(self.ang) > 1:
            edge_path = self._edge_milling_path(lay_num, n)
            if len(edge_path) > 0:
                verts, mverts = self._get_layered_vertices(edge_path, n, lay_num, no_z, dep)
                vertices.extend(verts)
                milling_vertices.extend(mverts)

    def _process_rough_milling(self, reg_inds, lay_mat, pad_loc, lay_num, n, no_z, dep,
                              vertices, milling_vertices):
        """Process rough milling paths for the region."""
        # Create rough pixels for all indices in the region
        rough_inds = [RoughPixel(ind, lay_mat, pad_loc, self.dim, n) for ind in reg_inds]

        # Generate rough milling paths
        rough_paths = self._rough_milling_path(rough_inds, lay_num, n)

        # Add vertices for each rough path
        for rough_path in rough_paths:
            if len(rough_path) > 0:
                verts, mverts = self._get_layered_vertices(rough_path, n, lay_num, no_z, dep)
                vertices.extend(verts)
                milling_vertices.extend(mverts)

    def _process_region_outlines(self, reg_inds, lay_mat, org_lay_mat, pad_loc, lay_num, n, no_z, dep,
                                neighbor_vectors, neighbor_vectors_a, neighbor_vectors_b,
                                vertices, milling_vertices):
        """Process outlines of regions in the layer."""
        # Get region outline vertices
        reg_verts = Utils.get_region_outline_vertices(reg_inds, lay_mat, org_lay_mat, pad_loc, n)

        # Process each island in the region (up to 10 islands)
        for isl_num in range(10):
            if len(reg_verts) == 0:
                break

            # Set starting vertex and get ordered vertices
            reg_verts = Utils.set_starting_vert(reg_verts)
            reg_ord_verts, reg_verts, closed = Utils.get_sublist_of_ordered_verts(reg_verts)

            # Skip if not enough vertices
            if len(reg_ord_verts) <= 1:
                continue

            # Offset vertices according to boundary conditions
            outline, corner_artifacts = self._offset_verts(
                neighbor_vectors, neighbor_vectors_a, neighbor_vectors_b,
                reg_ord_verts, lay_num, n
            )

            # Skip if no outline
            if len(outline) == 0:
                continue

            # Close the outline if needed and add vertices
            if closed:
                outline.append(MillVertex(outline[0].pt))

            verts, mverts = self._get_layered_vertices(outline, n, lay_num, no_z, dep)
            vertices.extend(verts)
            milling_vertices.extend(mverts)

            # Process corner artifacts
            for artifact in corner_artifacts:
                verts, mverts = self._get_layered_vertices(artifact, n, lay_num, no_z, dep)
                vertices.extend(verts)
                milling_vertices.extend(mverts)

    def create_and_buffer_vertices(self, milling_path=False):
        """Create and buffer vertices for joint visualization and milling paths."""
        # Initialize vertex arrays
        self.jverts = []
        self.everts = []
        self.mverts = []
        self.gcodeverts = []

        # Create joint vertices for each axis
        for ax in range(3):
            self.jverts.append(self.create_joint_vertices(ax))

        # Create milling path vertices if requested
        if milling_path:
            self._create_milling_path_vertices()

        # Create arrow vertices for visualization
        va = self._arrow_vertices()

        # Combine all vertices
        self._combine_vertices(va, milling_path)

        # Buffer the vertices
        self.buff.buffer_vertices()

    def _create_milling_path_vertices(self):
        """Create vertices for milling paths."""
        for n in range(self.noc):
            mvs, gvs = self._milling_path_vertices(n)
            self.mverts.append(mvs)
            self.gcodeverts.append(gvs)

    def _combine_vertices(self, arrow_vertices, milling_path):
        """Combine all vertex arrays into a single array."""
        # Combine joint vertices
        jverts = np.concatenate(self.jverts)

        # Add milling vertices if available
        if milling_path and len(self.mverts) > 0 and len(self.mverts[0]) > 0:
            mverts = np.concatenate(self.mverts)
            self.vertices = np.concatenate([jverts, arrow_vertices, mverts])

            # Calculate milling start indices
            self._calculate_milling_start_indices()
        else:
            self.vertices = np.concatenate([jverts, arrow_vertices])

        # Store vertex counts
        self.vn = int(len(self.jverts[0])/8)
        self.van = int(len(arrow_vertices)/8)

    def _calculate_milling_start_indices(self):
        """Calculate start indices for milling vertices."""
        self.m_start = []
        mst = 3*self.vn + self.van
        for n in range(self.noc):
            self.m_start.append(mst)
            mst += int(len(self.mverts[n])/8)

    def create_joint_vertices(self, ax):
        vertices = []
        r = g = b = 0.0
        # Create vectors - one for each of the 3 axis
        vx = np.array([1.0,0,0])*self.voxel_sizes[0]
        vy = np.array([0,1.0,0])*self.voxel_sizes[1]
        vz = np.array([0,0,1.0])*self.voxel_sizes[2]
        self.pos_vecs = [vx,vy,vz]
        # If it is possible to rotate the geometry, rotate position vectors
        if self.rot:
            non_sax = [0,1,2]
            non_sax.remove(self.sax)
            for i,ax in enumerate(non_sax):
                theta = math.radians(0.5*self.ang)
                if i%2==1: theta = -theta
                self.pos_vecs[ax] = Utils.rotate_vector_around_axis(self.pos_vecs[ax],self.pos_vecs[self.sax],theta)
                self.pos_vecs[ax] = self.pos_vecs[ax]/math.cos(math.radians(abs(self.ang)))
        # Add all vertices of the dim*dim*dim voxel cube
        for i in range(self.dim+1):
            for j in range(self.dim+1):
                for k in range(self.dim+1):
                    # position coordinates
                    ivec = (i-0.5*self.dim)*self.pos_vecs[0]
                    jvec = (j-0.5*self.dim)*self.pos_vecs[1]
                    kvec = (k-0.5*self.dim)*self.pos_vecs[2]
                    pos = ivec+jvec+kvec
                    x,y,z = pos
                    # texture coordinates
                    tex_coords = [i,j,k] ##################################################################
                    tex_coords.pop(ax)
                    tx = tex_coords[0]/self.dim
                    ty = tex_coords[1]/self.dim
                    # extend list of vertices
                    vertices.extend([x,y,z,r,g,b,tx,ty])
        # Calculate extra length for angled components
        extra_len=0
        if self.ang!=0.0 and self.rot:
            extra_len = 0.1*self.component_size*math.tan(math.radians(abs(self.ang)))
        # Add component base vertices
        for ax in range(3):
            if ax==self.sax: extra_l=0
            else: extra_l=extra_len
            for dir in range(-1,2,2):
                for step in range(3):
                    if step==0: step=1
                    else: step+=0.5+extra_len
                    axvec = dir*step*(self.component_size+extra_l)*self.pos_vecs[ax]/np.linalg.norm(self.pos_vecs[ax])
                    for x in range(2):
                        for y in range(2):
                            other_vecs = copy.deepcopy(self.pos_vecs)
                            other_vecs.pop(ax)
                            if ax!=self.sax and self.rot and step!=0.5:
                                #cvec = copy.deep(self.pos_vecs[ax])
                                xvec = (x-0.5)*self.dim*other_vecs[0]#+cvec
                                yvec = (y-0.5)*self.dim*other_vecs[1]#-cvec
                            else:
                                xvec = (x-0.5)*self.dim*other_vecs[0]
                                yvec = (y-0.5)*self.dim*other_vecs[1]
                            pos = axvec+xvec+yvec
                            # texture coordinates
                            tex_coords = [x,y]
                            tx = tex_coords[0]
                            ty = tex_coords[1]
                            # extend list of vertices
                            vertices.extend([pos[0],pos[1],pos[2],r,g,b,tx,ty])
        # Format
        vertices = np.array(vertices, dtype = np.float32) #converts to correct format
        return vertices

    def combine_and_buffer_indices(self, milling_path=False):
        self.update_suggestions()
        self.mesh.create_indices(milling_path=milling_path)
        glo_off = len(self.mesh.indices) # global offset
        for i in range(len(self.suggestions)):
            self.suggestions[i].create_indices(glo_off=glo_off, milling_path=False)
            glo_off+=len(self.suggestions[i].indices)
        for i in range(len(self.gals)):
            self.gals[i].create_indices(glo_off=glo_off,milling_path=False)
            glo_off+=len(self.gals[i].indices)
        indices = []
        indices.extend(self.mesh.indices)
        for mesh in self.suggestions: indices.extend(mesh.indices)
        for mesh in self.gals: indices.extend(mesh.indices)
        self.indices = np.array(indices, dtype=np.uint32)
        Buffer.buffer_indices(self.buff)

    def update_sliding_direction(self,sax):
        blocked = False
        for i,sides in enumerate(self.fixed.sides):
            for side in sides:
                if side.ax==sax:
                    if side.dir==0 and i==0: continue
                    if side.dir==1 and i==self.noc-1: continue
                    blocked = True
        if blocked:
            return False, "This sliding direction is blocked"
        else:
            self.sax = sax
            self.fixed.update_unblocked()
            self.create_and_buffer_vertices(milling_path=False)
            self.mesh.update_voxel_matrix_from_height_fields()
            for mesh in self.suggestions: mesh.update_voxel_matrix_from_height_fields()
            self.combine_and_buffer_indices()
            return True, ''

    def update_dimension(self,add):
        self.dim+=add
        self.voxel_sizes = np.copy(self.real_tim_dims)/(self.ratio*self.dim)
        self.create_and_buffer_vertices(milling_path=False)
        self.mesh.randomize_height_fields()

    def update_angle(self,ang):
        self.ang = ang
        self.create_and_buffer_vertices(milling_path=False)

    def update_timber_width_and_height(self,inds,val,milling_path=False):
        for i in inds: self.real_tim_dims[i]=val
        self.ratio = np.average(self.real_tim_dims)/self.component_size
        self.voxel_sizes = np.copy(self.real_tim_dims)/(self.ratio*self.dim)
        self.fab.vdia = self.fab.dia/self.ratio
        self.fab.vrad = self.fab.rad/self.ratio
        self.fab.vtol = self.fab.tol/self.ratio
        self.create_and_buffer_vertices(milling_path)

    def update_number_of_components(self,new_noc):
        if new_noc!=self.noc:
            # Increasing number of components
            if new_noc>self.noc:
                if len(self.fixed.unblocked)>=(new_noc-self.noc):
                    for i in range(new_noc-self.noc):
                        random_i = random.randint(0,len(self.fixed.unblocked)-1)
                        if self.fixed.sides[-1][0].ax==self.sax: # last component is aligned with the sliding axis
                            self.fixed.sides.insert(-1,[self.fixed.unblocked[random_i]])
                        else:
                            self.fixed.sides.append([self.fixed.unblocked[random_i]])
                        #also consider if it is aligned and should be the first one in line... rare though...
                        self.fixed.update_unblocked()
                    self.noc = new_noc
            # Decreasing number of components
            elif new_noc<self.noc:
                for i in range(self.noc-new_noc):
                    self.fixed.sides.pop()
                self.noc = new_noc
            # Rebuffer
            self.fixed.update_unblocked()
            self.create_and_buffer_vertices(milling_path=False)
            self.mesh.randomize_height_fields()

    def update_component_position(self,new_sides,n):
        self.fixed.sides[n] = new_sides
        self.fixed.update_unblocked()
        self.create_and_buffer_vertices(milling_path=False)
        self.mesh.update_voxel_matrix_from_height_fields()
        self.combine_and_buffer_indices()

    def reset(self, fs=None, sax=2, dim=3, ang=90., td=[44.0,44.0,44.0], incremental=False, align_ax=0, fabdia=6.0, fabtol=0.15, finterp=True, fabrot=0.0, fabext="gcode", hfs=[], fspe=400, fspi=600):
        self.fixed = FixedSides(self,fs=fs)
        self.noc=len(self.fixed.sides)
        self.sax=sax
        self.dim=dim
        self.ang=ang
        self.real_tim_dims = np.array(td)
        self.ratio = np.average(self.real_tim_dims)/self.component_size
        self.voxel_sizes = np.copy(self.real_tim_dims)/(self.ratio*self.dim)
        self.fab.tol=fabtol
        self.fab.real_dia = fabdia
        self.fab.rad = 0.5*self.fab.real_dia-self.fab.tol
        self.fab.dia = 2*self.fab.rad
        self.fab.vdia = self.fab.dia/self.ratio
        self.fab.vrad = self.fab.rad/self.ratio
        self.fab.vtol = self.fab.tol/self.ratio
        self.fab.speed = fspe
        self.fab.spindlespeed = fspi
        self.fab.extra_rot_deg=fabrot
        self.fab.ext=fabext
        self.fab.align_ax=align_ax
        self.fab.interp = finterp
        self.incremental=incremental
        self.mesh = Geometries(self, hfs=hfs)
        self.fixed.update_unblocked()
        self.create_and_buffer_vertices(milling_path=False)
        self.combine_and_buffer_indices()

    def _produce_suggestions(self, hfs):
        valid_suggestions = []
        for i in range(len(hfs)):
            for j in range(self.dim):
                for k in range(self.dim):
                    for add in range(-1,2,2):
                        sugg_hfs = copy.deepcopy(hfs)
                        sugg_hfs[i][j][k]+=add
                        val = sugg_hfs[i][j][k]
                        if val>=0 and val<=self.dim:
                            sugg_voxmat = Utils.matrix_from_height_fields(sugg_hfs,self.sax)
                            sugg_eval = Evaluation(sugg_voxmat,self,mainmesh=False)
                            if sugg_eval.valid:
                                valid_suggestions.append(sugg_hfs)
                                if len(valid_suggestions)==4: break
        return valid_suggestions

    def update_suggestions(self):
        self.suggestions = [] # clear list of suggestions
        if self.suggestions_on:
            sugg_hfs = []
            if not self.mesh.eval.valid:
                sugg_hfs = self._produce_suggestions(self.mesh.height_fields)
                for i in range(len(sugg_hfs)): self.suggestions.append(Geometries(self, mainmesh=False, hfs=sugg_hfs[i]))

    def init_gallery(self,start_index):
        self.gallary_start_index = start_index
        self.gals = []
        self.suggestions = []
        # Folder
        location = os.path.abspath(os.getcwd())
        location = location.split(os.sep)
        location.pop()
        location = os.sep.join(location)
        location += os.sep+"search_results"+os.sep+"noc_"+str(self.noc)+os.sep+"dim_"+str(self.dim)+os.sep+"fs_"
        for i in range(len(self.fixed.sides)):
            for fs in self.fixed.sides[i]:
                location+=str(fs.ax)+str(fs.dir)
            if i!=len(self.fixed.sides)-1: location+=("_")
        location+=os.sep+"allvalid"
        maxi = len(os.listdir(location))-1
        for i in range(20):
            if (i+start_index)>maxi: break
            try:
                hfs = np.load(location+os.sep+"height_fields_"+str(start_index+i)+".npy")
                self.gals.append(Geometries(self,mainmesh=False,hfs=hfs))
            except:
                abc = 0

    def save(self,filename="joint.tsu"):

        """
        Meaning of abbreviations:
        SAX: sliding axis           (0-2)   (the sliding axis, not all possible sliding directions) (refer to Figure 3d of the paper)
        NOT: number of timbers      (2-6)   (refer to Figure 3e of the paper)
        RES: voxel resolution       (2-5)   (2-->[2,2,2], 3-->[3,3,3] and so on. Non-uniform resolution such as [2,3,4] is not possible currently) (refer to Figure 3f of the paper)
        ANG: angle of intersection          (refer to Figure 27a of the paper)
        TDX: timber timension in x-axis (mm)
        TDY: timber timension in y-axis (mm)
        TDZ: timber timension in z-axis (mm) (TDX, TDY, and TDZ does not have to be equal. Refer for Figure 27b of the paper)
        DIA: diameter of the milling bit
        TOL: tolerance
        SPE: speed of the milling bit
        SPI: spindle speed
        INC: incremental            (T/F)   Option for the layering of the milling path to avoid "downcuts"
        FIN: interpolation of arcs  (T/F)   Milling path true arcs or divided into many points (depending on milling machine)
        ALN: align                          Axis to align the timber element with during fabrication
        EXT: extension ("gcode"/"sbp"/"nc") File format for the milling machine. Roland machine: nc. Shopbot machine: sbp
        FSS: fixed sides                    Fixed sides of the cube are connected to the timber (non-fixed sides are free/open)
        HFS: height fields                  Voxel geometry described by height fields of size res*res
        """

        #Initiate
        file = open(filename,"w")

        # Joint properties
        file.write("SAX "+str(self.sax)+"\n")
        file.write("NOT "+str(self.noc)+"\n")
        file.write("RES "+str(self.dim)+"\n")
        file.write("ANG "+str(self.ang)+"\n")
        file.write("TDX "+str(self.real_tim_dims[0])+"\n")
        file.write("TDY "+str(self.real_tim_dims[1])+"\n")
        file.write("TDZ "+str(self.real_tim_dims[2])+"\n")
        file.write("DIA "+str(self.fab.real_dia)+"\n")
        file.write("TOL "+str(self.fab.tol)+"\n")
        file.write("SPE "+str(self.fab.speed)+"\n")
        file.write("SPI "+str(self.fab.spindlespeed)+"\n")
        file.write("INC "+str(self.incremental)+"\n")
        file.write("FIN "+str(self.fab.interp)+"\n")
        file.write("ALN "+str(self.fab.align_ax)+"\n")
        file.write("EXT "+self.fab.ext+"\n")

        # Fixed sides
        file.write("FSS ")
        for n in range(len(self.fixed.sides)):
            for i in range(len(self.fixed.sides[n])):
                file.write(str(int(self.fixed.sides[n][i].ax))+",")
                file.write(str(int(self.fixed.sides[n][i].dir)))
                if i!=len(self.fixed.sides[n])-1: file.write(".")
            if n!=len(self.fixed.sides)-1: file.write(":")

        # Joint geometry
        file.write("\nHFS \n")
        for n in range(len(self.mesh.height_fields)):
            for i in range(len(self.mesh.height_fields[n])):
                for j in range(len(self.mesh.height_fields[n][i])):
                    file.write(str(int(self.mesh.height_fields[n][i][j])))
                    if j!=len(self.mesh.height_fields[n][i])-1: file.write(",")
                if i!=len(self.mesh.height_fields[n])-1: file.write(":")
            if n!=len(self.mesh.height_fields)-1: file.write("\n")

        #Finalize
        print("Saved",filename)
        file.close()

    def open(self,filename="joint.tsu"):

        # Open
        file = open(filename,"r")

        # Default values
        sax = self.sax
        noc = self.noc
        dim = self.dim
        ang = self.ang
        dx, dy, dz = self.real_tim_dims
        dia = self.fab.real_dia
        tol = self.fab.tol
        spe = self.fab.speed
        spi = self.fab.spindlespeed
        inc = self.incremental
        aln = self.fab.align_ax
        ext = self.fab.ext
        fs = self.fixed.sides
        fin = self.fab.interp

        # Read
        hfs = []
        hfi = 999
        for i,line in enumerate(file.readlines()):
            items = line.split( )
            if items[0]=="SAX": sax = int(items[1])
            elif items[0]=="NOT": noc = int(items[1])
            elif items[0]=="RES": dim = int(items[1])
            elif items[0]=="ANG": ang = float(items[1])
            elif items[0]=="TDX": dx = float(items[1])
            elif items[0]=="TDY": dy = float(items[1])
            elif items[0]=="TDZ": dz = float(items[1])
            elif items[0]=="DIA": dia = float(items[1])
            elif items[0]=="TOL": tol = float(items[1])
            elif items[0]=="SPE": spe = float(items[1])
            elif items[0]=="SPI": spi = float(items[1])
            elif items[0]=="INC":
                if items[1]=="True": inc = True
                else: inc = False
            elif items[0]=="FIN":
                if items[1]=="True": fin = True
                else: fin = False
            elif items[0]=="ALN": aln = float(items[1])
            elif items[0]=="EXT": ext = items[1]
            elif items[0]=="FSS": fs = FixedSides(self,side_str=items[1]).sides
            elif items[0]=="HFS": hfi = i
            elif i>hfi:
                hf = []
                for row in line.split(":"):
                    temp = []
                    for item in row.split(","): temp.append(int(float(item)))
                    hf.append(temp)
                hfs.append(hf)
        hfs = np.array(hfs)

        # Reinitiate
        self.reset(fs=fs, sax=sax, dim=dim, ang=ang, td=[dx,dy,dz], fabdia=dia, fabtol=tol, align_ax=aln, finterp=fin, incremental=inc, fabext=ext, hfs=hfs, fspe=spe, fspi=spi)

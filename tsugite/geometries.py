import random
import copy
import os

import numpy as np
import OpenGL.GL as GL  # imports start with GL

from selection import Selection
from evaluation import Evaluation
from buffer import ElementProperties
from misc import FixedSide

import utils as Utils

class Geometries:
    def __init__(self,parent,mainmesh=True,hfs=[]):
        self.mainmesh = mainmesh
        self.parent = parent
        self.fab_directions = [0,1] #Initiate list of fabrication directions
        for i in range(1,self.parent.noc-1): self.fab_directions.insert(1,1)
        if len(hfs)==0: self.height_fields = Utils.get_random_height_fields(self.parent.dim,self.parent.noc) #Initiate a random joint geometry
        else: self.height_fields = hfs
        if self.mainmesh: self.select = Selection(self)
        self.voxel_matrix_from_height_fields(first=True)

    def voxel_matrix_from_height_fields(self, first=False):
        vox_mat = Utils.matrix_from_height_fields(self.height_fields,self.parent.sax)
        self.voxel_matrix = vox_mat
        if self.mainmesh:
            self.eval = Evaluation(self.voxel_matrix,self.parent)
            self.fab_directions = self.eval.fab_directions
        if self.mainmesh and not first:
            self.parent.update_suggestions()

    def _joint_line_indices(self,all_indices,n,offset,global_offset=0):
        fixed_sides = self.parent.fixed.sides[n]
        d = self.parent.dim+1
        indices = []
        for i in range(d):
            for j in range(d):
                for k in range(d):
                    ind = [i,j,k]
                    for ax in range(3):
                        if ind[ax]==self.parent.dim: continue
                        cnt,vals = self._line_neighbors(ind,ax,n)
                        diagonal = False
                        if vals[0]==vals[3] or vals[1]==vals[2]: diagonal = True
                        if cnt==1 or cnt==3 or (cnt==2 and diagonal):
                            add = [0,0,0]
                            add[ax] = 1
                            start_i = Utils.get_index(ind,[0,0,0],self.parent.dim)
                            end_i = Utils.get_index(ind,add,self.parent.dim)
                            indices.extend([start_i,end_i])
        #Outline of component base
        start = d*d*d
        for side in fixed_sides:
            a1,b1,c1,d1 = Utils.get_corner_indices(side.ax,side.dir,self.parent.dim)
            step = 2
            if len(self.parent.fixed.sides[n])==2: step = 1
            off = 24*side.ax+12*side.dir+4*step
            a0,b0,c0,d0 = start+off,start+off+1,start+off+2,start+off+3
            indices.extend([a0,b0, b0,d0, d0,c0, c0,a0])
            indices.extend([a0,a1, b0,b1, c0,c1, d0,d1])
        # Format
        indices = np.array(indices, dtype=np.uint32)
        indices = indices + offset
        # Store
        indices_prop = ElementProperties(GL.GL_LINES, len(indices), len(all_indices)+global_offset, n)
        all_indices = np.concatenate([all_indices, indices])
        # Return
        return indices_prop, all_indices

    def _line_neighbors(self,ind,ax,n):
        values = []
        for i in range(-1,1):
            for j in range(-1,1):
                val = None
                add = [i,j]
                add.insert(ax,0)
                ind2 = np.array(ind)+np.array(add)
                if np.all(ind2>=0) and np.all(ind2<self.parent.dim):
                    val = self.voxel_matrix[tuple(ind2)]
                else:
                    for n2 in range(self.parent.noc):
                        for side in self.parent.fixed.sides[n2]:
                            ind3 = np.delete(ind2,side.ax)
                            if np.all(ind3>=0) and np.all(ind3<self.parent.dim):
                                if ind2[side.ax]<0 and side.dir==0: val = n2
                                elif ind2[side.ax]>=self.parent.dim and side.dir==1: val = n2
                values.append(val)
        values = np.array(values)
        count = np.count_nonzero(values==n)
        return count,values

    def _chess_line_indices(self,all_indices,chess_verts,n,offset):
        indices = []
        for vert in chess_verts:
            add = [0,0,0]
            st = Utils.get_index(vert,add,self.parent.dim)
            add[self.parent.sax]=1
            en = Utils.get_index(vert,add,self.parent.dim)
            indices.extend([st,en])
        # Format
        indices = np.array(indices, dtype=np.uint32)
        indices = indices + offset
        # Store
        indices_prop = ElementProperties(GL.GL_LINES, len(indices), len(all_indices), n)
        all_indices = np.concatenate([all_indices, indices])
        # Return
        return indices_prop, all_indices

    def _break_line_indices(self,all_indices,break_inds,n,offset):
        indices = []
        for ind3d in break_inds:
            add = [0,0,0]
            ind = Utils.get_index(ind3d,add,self.parent.dim)
            indices.append(ind)
        # Format
        indices = np.array(indices, dtype=np.uint32)
        indices = indices + offset
        # Store
        indices_prop = ElementProperties(GL.GL_LINES, len(indices), len(all_indices), n)
        all_indices = np.concatenate([all_indices, indices])
        # Return
        return indices_prop, all_indices

    def _open_line_indices(self,all_indices,n,offset):
        indices = []
        dirs = [0,1]
        if n==0: dirs=[0]
        elif n==self.parent.noc-1: dirs=[1]
        d = self.parent.dim+1
        start = d*d*d
        for dir in dirs:
            a1,b1,c1,d1 = Utils.get_corner_indices(self.parent.sax,dir,self.parent.dim)
            off = 24*self.parent.sax+12*(1-dir)
            a0,b0,c0,d0 = start+off,start+off+1,start+off+2,start+off+3
            indices.extend([a0,a1, b0,b1, c0,c1, d0,d1])
        # Format
        indices = np.array(indices, dtype=np.uint32)
        indices = indices + offset
        # Store
        indices_prop = ElementProperties(GL.GL_LINES, len(indices), len(all_indices), n)
        all_indices = np.concatenate([all_indices, indices])
        # Return
        return indices_prop, all_indices

    def _arrow_indices(self,all_indices,slide_dirs,n,offset):
        line_indices = []
        face_indices = []
        for ax,dir in slide_dirs:
            #arrow line
            start = 1+10*ax+5*dir
            line_indices.extend([0, start])
            #arrow head (triangles)
            face_indices.extend([start+1, start+2, start+4])
            face_indices.extend([start+1, start+4, start+3])
            face_indices.extend([start+1, start+2, start])
            face_indices.extend([start+2, start+4, start])
            face_indices.extend([start+3, start+4, start])
            face_indices.extend([start+1, start+3, start])
        # Format
        line_indices = np.array(line_indices, dtype=np.uint32)
        line_indices = line_indices + offset
        face_indices = np.array(face_indices, dtype=np.uint32)
        face_indices = face_indices + offset
        # Store
        line_indices_prop = ElementProperties(GL.GL_LINES, len(line_indices), len(all_indices), n)
        all_indices = np.concatenate([all_indices, line_indices])
        face_indices_prop = ElementProperties(GL.GL_TRIANGLES, len(face_indices), len(all_indices), n)
        all_indices = np.concatenate([all_indices, face_indices])
        # Return
        return line_indices_prop, face_indices_prop, all_indices

    def _joint_face_indices(self,all_indices,mat,fixed_sides,n,offset,global_offset=0):
        # Make indices of faces for drawing method GL_QUADS
        # 1. Faces of joint
        indices = []
        indices_ends = []
        d = self.parent.dim+1
        indices = []
        for i in range(d):
            for j in range(d):
                for k in range(d):
                    ind = [i,j,k]
                    for ax in range(3):
                        test_ind = np.array([i,j,k])
                        test_ind = np.delete(test_ind,ax)
                        if np.any(test_ind==self.parent.dim): continue
                        cnt,vals = Utils.face_neighbors(mat,ind,ax,n,fixed_sides)
                        if cnt==1:
                            for x in range(2):
                                for y in range(2):
                                    add = [x,abs(y-x)]
                                    add.insert(ax,0)
                                    index = Utils.get_index(ind,add,self.parent.dim)
                                    if len(fixed_sides)>0:
                                        if fixed_sides[0].ax==ax:
                                            indices_ends.append(index)
                                        else:
                                            indices.append(index)
                                    else: indices.append(index)

            # 2. Faces of component base
            d = self.parent.dim+1
            start = d*d*d
            if len(fixed_sides)>0:
                for side in fixed_sides:
                    a1,b1,c1,d1 = Utils.get_corner_indices(side.ax,side.dir,self.parent.dim)
                    step = 2
                    if len(self.parent.fixed.sides[n])==2: step = 1
                    off = 24*side.ax+12*side.dir+4*step
                    a0,b0,c0,d0 = start+off,start+off+1,start+off+2,start+off+3
                    # Add component side to indices
                    indices_ends.extend([a0,b0,d0,c0]) #bottom face
                    indices.extend([a0,b0,b1,a1]) #side face 1
                    indices.extend([b0,d0,d1,b1]) #side face 2
                    indices.extend([d0,c0,c1,d1]) #side face 3
                    indices.extend([c0,a0,a1,c1]) ##side face 4
        # Format
        indices = np.array(indices, dtype=np.uint32)
        indices = indices + offset
        indices_ends = np.array(indices_ends, dtype=np.uint32)
        indices_ends = indices_ends + offset
        # Store
        indices_prop = ElementProperties(GL.GL_QUADS, len(indices), len(all_indices)+global_offset, n)
        if len(all_indices)>0: all_indices = np.concatenate([all_indices, indices])
        else: all_indices = indices
        indices_ends_prop = ElementProperties(GL.GL_QUADS, len(indices_ends), len(all_indices)+global_offset, n)
        all_indices = np.concatenate([all_indices, indices_ends])
        indices_all_prop = ElementProperties(GL.GL_QUADS, len(indices)+len(indices_ends), indices_prop.start_index, n)
        # Return
        return indices_prop, indices_ends_prop, indices_all_prop, all_indices

    def _joint_area_face_indices(self,all_indices,mat,area_faces,n):
        # Make indices of faces for drawing method GL_QUADS
        # 1. Faces of joint
        indices = []
        indices_ends = []
        d = self.parent.dim+1
        indices = []
        for i in range(d):
            for j in range(d):
                for k in range(d):
                    ind = [i,j,k]
                    for ax in range(3):
                        offset = ax*self.parent.vn
                        test_ind = np.array([i,j,k])
                        test_ind = np.delete(test_ind,ax)
                        if np.any(test_ind==self.parent.dim): continue
                        cnt,vals = Utils.face_neighbors(mat,ind,ax,n,self.parent.fixed.sides[n])
                        if cnt==1:
                            for x in range(2):
                                for y in range(2):
                                    add = [x,abs(y-x)]
                                    add.insert(ax,0)
                                    index = Utils.get_index(ind,add,self.parent.dim)
                                    if [ax,ind] in area_faces: indices.append(index+offset)
                                    else: indices_ends.append(index+offset)
        # 2. Faces of component base
        d = self.parent.dim+1
        start = d*d*d
        if len(self.parent.fixed.sides[n])>0:
            for side in self.parent.fixed.sides[n]:
                offset = side.ax*self.parent.vn
                a1,b1,c1,d1 = Utils.get_corner_indices(side.ax,side.dir,self.parent.dim)
                step = 2
                if len(self.parent.fixed.sides[n])==2: step = 1
                off = 24*side.ax+12*side.dir+4*step
                a0,b0,c0,d0 = start+off,start+off+1,start+off+2,start+off+3
                # Add component side to indices
                indices_ends.extend([a0+offset,b0+offset,d0+offset,c0+offset]) #bottom face
                indices_ends.extend([a0+offset,b0+offset,b1+offset,a1+offset]) #side face 1
                indices_ends.extend([b0+offset,d0+offset,d1+offset,b1+offset]) #side face 2
                indices_ends.extend([d0+offset,c0+offset,c1+offset,d1+offset]) #side face 3
                indices_ends.extend([c0+offset,a0+offset,a1+offset,c1+offset]) ##side face 4
        # Format
        indices = np.array(indices, dtype=np.uint32)
        #indices = indices + offset
        indices_ends = np.array(indices_ends, dtype=np.uint32)
        #indices_ends = indices_ends + offset
        # Store
        indices_prop = ElementProperties(GL.GL_QUADS, len(indices), len(all_indices), n)
        if len(all_indices)>0: all_indices = np.concatenate([all_indices, indices])
        else: all_indices = indices
        indices_ends_prop = ElementProperties(GL.GL_QUADS, len(indices_ends), len(all_indices), n)
        all_indices = np.concatenate([all_indices, indices_ends])
        # Return
        return indices_prop, indices_ends_prop, all_indices

    def _joint_top_face_indices(self,all_indices,n,noc,offset):
        # Make indices of faces for drawing method GL_QUADS
        # Set face direction
        if n==0: sdirs = [0]
        elif n==noc-1: sdirs = [1]
        else: sdirs = [0,1]
        # 1. Faces of joint
        indices = []
        indices_tops = []
        indices = []
        sax = self.parent.sax
        for ax in range(3):
            for i in range(self.parent.dim):
                for j in range(self.parent.dim):
                    top_face_indices_cnt=0
                    for k in range(self.parent.dim+1):
                        if sdirs[0]==0: k = self.parent.dim-k
                        ind = [i,j]
                        ind.insert(ax,k)
                        # count number of neigbors (0, 1, or 2)
                        cnt,vals = Utils.face_neighbors(self.voxel_matrix,ind,ax,n,self.parent.fixed.sides[n])
                        on_free_base = False
                        # add base if edge component
                        if ax==sax and ax!=self.parent.fixed.sides[n][0].ax and len(sdirs)==1:
                            base = sdirs[0]*self.parent.dim
                            if ind[ax]==base: on_free_base=True
                        if cnt==1 or on_free_base:
                            for x in range(2):
                                for y in range(2):
                                    add = [x,abs(y-x)]
                                    add.insert(ax,0)
                                    index = Utils.get_index(ind,add,self.parent.dim)
                                    if ax==sax and top_face_indices_cnt<4*len(sdirs):
                                        indices_tops.append(index)
                                        top_face_indices_cnt+=1
                                    #elif not on_free_base: indices.append(index)
                                    else: indices.append(index)
                    if top_face_indices_cnt<4*len(sdirs) and ax==sax:
                        neg_i = -offset-1
                        for k in range(4*len(sdirs)-top_face_indices_cnt):
                            indices_tops.append(neg_i)
        # 2. Faces of component base
        d = self.parent.dim+1
        start = d*d*d
        for side in self.parent.fixed.sides[n]:
            a1,b1,c1,d1 = Utils.get_corner_indices(side.ax,side.dir,self.parent.dim)
            step = 2
            if len(self.parent.fixed.sides[n])==2: step = 1
            off = 24*side.ax+12*side.dir+4*step
            a0,b0,c0,d0 = start+off,start+off+1,start+off+2,start+off+3
            # Add component side to indices
            indices.extend([a0,b0,d0,c0]) #bottom face
            indices.extend([a0,b0,b1,a1]) #side face 1
            indices.extend([b0,d0,d1,b1]) #side face 2
            indices.extend([d0,c0,c1,d1]) #side face 3
            indices.extend([c0,a0,a1,c1]) ##side face 4
        # Format
        indices = np.array(indices, dtype=np.int32)
        indices = indices + offset
        indices_tops = np.array(indices_tops, dtype=np.int32)
        indices_tops = indices_tops + offset
        # Store
        indices_prop = ElementProperties(GL.GL_QUADS, len(indices), len(all_indices), n)
        all_indices = np.concatenate([all_indices, indices])
        indices_tops_prop = ElementProperties(GL.GL_QUADS, len(indices_tops), len(all_indices), n)
        all_indices = np.concatenate([all_indices, indices_tops])
        # Return
        return indices_prop, indices_tops_prop, all_indices

    def _joint_selected_top_line_indices(self,select,all_indices):
        # Make indices of lines for drawing method GL_LINES
        n = select.n
        dir = select.dir
        offset = n*self.parent.vn
        sax = self.parent.sax
        h = self.height_fields[n-dir][tuple(select.faces[0])]
        # 1. Outline of selected top faces of joint
        indices = []
        for face in select.faces:
            ind = [int(face[0]),int(face[1])]
            ind.insert(sax,h)
            other_axes = [0,1,2]
            other_axes.pop(sax)
            for i in range(2):
                ax = other_axes[i]
                for j in range(2):
                    # Check neighboring faces
                    nface = face.copy()
                    nface[i] += 2*j-1
                    nface = np.array(nface, dtype=np.uint32)
                    if np.all(nface>=0) and np.all(nface<self.parent.dim):
                        unique = True
                        for face2 in select.faces:
                            if nface[0]==face2[0] and nface[1]==face2[1]:
                                unique = False
                                break
                        if not unique: continue
                    for k in range(2):
                        add = [k,k,k]
                        add[ax] = j
                        add[sax] = 0
                        index = Utils.get_index(ind,add,self.parent.dim)
                        indices.append(index)
        # Format
        indices = np.array(indices, dtype=np.uint32)
        indices = indices + offset
        # Store
        indices_prop = ElementProperties(GL.GL_LINES, len(indices), len(all_indices), n)
        all_indices = np.concatenate([all_indices, indices])
        # Return
        return indices_prop, all_indices

    def _component_outline_indices(self,all_indices,fixed_sides,n,offset):
        d = self.parent.dim+1
        indices = []
        start = d*d*d
        #Outline of component base
        #1) Base of first fixed side
        ax = fixed_sides[0].ax
        dir = fixed_sides[0].dir
        step = 2
        if len(fixed_sides)==2: step = 1
        off = 24*ax+12*dir+4*step
        a0,b0,c0,d0 = start+off,start+off+1,start+off+2,start+off+3
        #2) Base of first fixed side OR top of component
        if len(fixed_sides)==2:
            ax = fixed_sides[1].ax
            dir = fixed_sides[1].dir
            off = 24*ax+12*dir+4*step
            a1,b1,c1,d1 = start+off,start+off+1,start+off+2,start+off+3
        else:
            a1,b1,c1,d1 = Utils.get_corner_indices(ax,1-dir,self.parent.dim)
        # append list of indices
        indices.extend([a0,b0, b0,d0, d0,c0, c0,a0])
        indices.extend([a0,a1, b0,b1, c0,c1, d0,d1])
        indices.extend([a1,b1, b1,d1, d1,c1, c1,a1])
        # Format
        indices = np.array(indices, dtype=np.uint32)
        indices = indices + offset
        # Store
        indices_prop = ElementProperties(GL.GL_LINES, len(indices), len(all_indices), n)
        all_indices = np.concatenate([all_indices, indices])
        # Return
        return indices_prop, all_indices
    
    def _milling_path_indices(self,all_indices,count,start,n):
        indices = []
        for i in range(count):
            indices.append(int(start+i))
        # Format
        indices = np.array(indices, dtype=np.uint32)
        # Store
        indices_prop = ElementProperties(GL.GL_LINE_STRIP, len(indices), len(all_indices), n)
        all_indices = np.concatenate([all_indices, indices])
        # Return
        return indices_prop, all_indices

    def create_indices(self, glo_off=0, milling_path=False):
        #shared lists
        all_inds = []
        self.indices_fall = []
        self.indices_lns = []
        #suggestion geometries
        if not self.mainmesh: # for suggestions and gallery - just show basic geometry - no feedback - global offset necessary
            for n in range(self.parent.noc):
                ax = self.parent.fixed.sides[n][0].ax
                nend,end,all,all_inds = self._joint_face_indices(all_inds,
                        self.voxel_matrix,self.parent.fixed.sides[n],n,ax*self.parent.vn,global_offset=glo_off)
                lns,all_inds = self._joint_line_indices(all_inds,n,ax*self.parent.vn,global_offset=glo_off)
                self.indices_fall.append(all)
                self.indices_lns.append(lns)
        #current geometry (main including feedback)
        else:
            self.indices_fend=[]
            self.indices_not_fend=[]
            self.indices_fcon = []
            self.indices_not_fcon = []
            self.indices_fbrk = []
            self.indices_not_fbrk = []
            self.indices_open_lines = []
            self.indices_not_fbridge = []
            self.indices_ffric = []
            self.indices_not_ffric = []
            self.indices_fcont = []
            self.indices_not_fcont = []
            self.indices_arrows = []
            self.indices_fpick_top = []
            self.indices_fpick_not_top = []
            self.outline_selected_faces = None
            self.outline_selected_component = None
            self.indices_chess_lines = []
            self.indices_breakable_lines = []
            self.indices_milling_path = []
            for n in range(self.parent.noc):
                ax = self.parent.fixed.sides[n][0].ax
                #Faces
                nend,end,con,all_inds = self._joint_face_indices(all_inds,
                        self.eval.voxel_matrix_connected,self.parent.fixed.sides[n],n,ax*self.parent.vn)
                if not self.eval.connected[n]:
                    fne,fe,uncon,all_inds = self._joint_face_indices(all_inds,self.eval.voxel_matrix_unconnected,[],n,ax*self.parent.vn)
                    self.indices_not_fcon.append(uncon)
                    all = ElementProperties(GL.GL_QUADS, con.count+uncon.count, con.start_index, n)
                else:
                    self.indices_not_fcon.append(None)
                    all = con

                #breakable and not breakable faces
                fne,fe,brk_faces,all_inds = self._joint_face_indices(all_inds,self.eval.breakable_voxmat,[],n,ax*self.parent.vn)
                fne,fe,not_brk_faces,all_inds = self._joint_face_indices(all_inds,self.eval.non_breakable_voxmat,self.parent.fixed.sides[n],n,n*self.parent.vn)

                if not self.eval.bridged[n]:
                    unbris = []
                    for m in range(2):
                        fne,fe,unbri,all_inds = self._joint_face_indices(all_inds,self.eval.voxel_matrices_unbridged[n][m],[self.parent.fixed.sides[n][m]],n,n*self.parent.vn)
                        unbris.append(unbri)
                else: unbris = None

                # Friction ad contact faces
                fric,nfric,all_inds = self._joint_area_face_indices(all_inds, self.voxel_matrix, self.eval.friction_faces[n], n)
                cont,ncont,all_inds = self._joint_area_face_indices(all_inds, self.voxel_matrix, self.eval.contact_faces[n], n)

                #picking faces
                faces_pick_not_tops, faces_pick_tops, all_inds = self._joint_top_face_indices(all_inds,n,self.parent.noc,ax*self.parent.vn)

                #Lines
                lns,all_inds = self._joint_line_indices(all_inds,n,ax*self.parent.vn)

                # Chessboard feedback lines
                if self.eval.checker[n]:
                    chess,all_inds = self._chess_line_indices(all_inds,self.eval.checker_vertices[n],n,ax*self.parent.vn)
                else: chess = []
                # Breakable lines
                if self.eval.breakable:
                    break_lns, all_inds = self._break_line_indices(all_inds,self.eval.breakable_outline_inds[n],n,ax*self.parent.vn)

                # Opening lines
                open,all_inds = self._open_line_indices(all_inds,n,ax*self.parent.vn)
                self.indices_open_lines.append(open)

                #arrows
                larr, farr, all_inds = self._arrow_indices(all_inds,self.eval.slides[n],n,3*self.parent.vn)
                arrows = [larr,farr]

                if milling_path and len(self.parent.mverts[0])>0:
                    mill,all_inds = self._milling_path_indices(all_inds,int(len(self.parent.mverts[n])/8),self.parent.m_start[n],n)

                # Append lists
                self.indices_fend.append(end)
                self.indices_not_fend.append(nend)
                self.indices_fcon.append(con)
                self.indices_fall.append(all)
                self.indices_lns.append(lns)
                self.indices_not_fbridge.append(unbris)
                self.indices_arrows.append(arrows)
                self.indices_fpick_top.append(faces_pick_tops)
                self.indices_fpick_not_top.append(faces_pick_not_tops)
                self.indices_chess_lines.append(chess)
                if self.eval.breakable:
                    self.indices_breakable_lines.append(break_lns)
                    self.indices_fbrk.append(brk_faces)
                    self.indices_not_fbrk.append(not_brk_faces)
                if milling_path and len(self.parent.mverts[0])>0:
                    self.indices_milling_path.append(mill)
                self.indices_ffric.append(fric)
                self.indices_not_ffric.append(nfric)
                self.indices_fcont.append(cont)
                self.indices_not_fcont.append(ncont)

            #outline of selected faces
            if self.select.state==2:
                self.outline_selected_faces, all_inds = self._joint_selected_top_line_indices(self.select,all_inds)

            if self.select.n!=None and self.select.new_fixed_sides_for_display!=None:
                self.outline_selected_component, all_inds = self._component_outline_indices(all_inds,self.select.new_fixed_sides_for_display,self.select.n,self.select.n*self.parent.vn)
        self.indices = all_inds

    def randomize_height_fields(self):
        self.height_fields = Utils.get_random_height_fields(self.parent.dim,self.parent.noc)
        self.voxel_matrix_from_height_fields()
        self.parent.combine_and_buffer_indices()

    def clear_height_fields(self):
        self.height_fields = []
        for n in range(self.parent.noc-1):
            hf = np.zeros((self.parent.dim,self.parent.dim))
            self.height_fields.append(hf)
        self.voxel_matrix_from_height_fields()
        self.parent.combine_and_buffer_indices()

    def load_search_results(self,index=-1):
        # Folder
        location = os.path.abspath(os.getcwd())
        location = location.split(os.sep)
        location.pop()
        location = os.sep.join(location)
        location += os.sep+"search_results"+os.sep+"noc_"+str(self.parent.noc)+os.sep+"dim_"+str(self.parent.dim)+os.sep+"fs_"
        for i in range(len(self.parent.fixed.sides)):
            for fs in self.parent.fixed.sides[i]:
                location+=str(fs[0])+str(fs[1])
            if i!=len(self.parent.fixed.sides)-1: location+=("_")
        location+=os.sep+"allvalid"
        print("Trying to load geometry from",location)
        maxi = len(os.listdir(location))-1
        if index==-1: index=random.randint(0,maxi)
        self.height_fields = np.load(location+os.sep+"height_fields_"+str(index)+".npy")
        self.fab_directions = []
        for i in range(self.parent.noc):
            if i==0: self.fab_directions.append(0)
            else: self.fab_directions.append(1)
        self.voxel_matrix_from_height_fields()
        self.parent.combine_and_buffer_indices()

    def edit_height_fields(self,faces,h,n,dir):
        for ind in faces:
            self.height_fields[n-dir][tuple(ind)] = h
            if dir==0: # If editing top
                # If new height is higher than following hf, update to same height
                for i in range(n-dir+1,self.parent.noc-1):
                    h2 = self.height_fields[i][tuple(ind)]
                    if h>h2: self.height_fields[i][tuple(ind)]=h
            if dir==1: # If editing bottom
                # If new height is lower than previous hf, update to same height
                for i in range(0,n-dir):
                    h2 = self.height_fields[i][tuple(ind)]
                    if h<h2: self.height_fields[i][tuple(ind)]=h
        self.voxel_matrix_from_height_fields()
        self.parent.combine_and_buffer_indices()

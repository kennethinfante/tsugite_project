import copy
import math

import numpy as np
import pyrr

from misc import FixedSide

import utils as Utils

class Selection:
    def __init__(self, pgeom):
        self.state = -1 #-1: nothing, 0: hovered, 1: adding, 2: pulling, 10: timber hovered, 12: timber pulled
        self.sugg_state = -1 #-1: nothing, 0: hovering first, 1: hovering secong, and so on.
        self.gallstate = -1
        self.pgeom = pgeom
        self.n = self.x = self.y = None
        self.refresh = False
        self.shift = False
        self.faces = []
        self.new_fixed_sides_for_display = None
        self.val=0

    def update_pick(self,x,y,n,dir):
        self.n = n
        self.x = x
        self.y = y
        self.dir = dir
        if self.x!=None and self.y!=None:
            if self.shift:
                self.faces = Utils.get_same_height_neighbors(self.pgeom.height_fields[n - dir], [np.array([self.x, self.y])])
            else: self.faces = [np.array([self.x,self.y])]

    def start_pull(self,mouse_pos):
        self.state=2
        self.start_pos = np.array([mouse_pos[0],-mouse_pos[1]])
        self.start_height = self.pgeom.height_fields[self.n - self.dir][self.x][self.y]
        self.pgeom.pjoint.combine_and_buffer_indices() # for selection area

    def end_pull(self):
        if self.val!=0: self.pgeom.edit_height_fields(self.faces, self.current_height, self.n, self.dir)
        self.state=-1
        self.refresh = True

    def edit(self,mouse_pos,screen_xrot,screen_yrot,w=1600,h=1600):
        self.current_pos = np.array([mouse_pos[0],-mouse_pos[1]])
        self.current_height = self.start_height
        ## Mouse vector
        mouse_vec = np.array(self.current_pos-self.start_pos)
        mouse_vec = mouse_vec.astype(float)
        mouse_vec[0] = 2*mouse_vec[0]/w
        mouse_vec[1] = 2*mouse_vec[1]/h
        ## Sliding direction vector
        sdir_vec = [0,0,0]
        sdir_vec = np.copy(self.pgeom.pjoint.pos_vecs[self.pgeom.pjoint.sax])  #<-new
        rot_x = pyrr.Matrix33.from_x_rotation(screen_xrot)
        rot_y = pyrr.Matrix33.from_y_rotation(screen_yrot)
        sdir_vec = np.dot(sdir_vec,rot_x*rot_y)
        sdir_vec = np.delete(sdir_vec,2) # delete Z-value
        ## Calculate angle between mouse vector and sliding direction vector
        cosang = np.dot(mouse_vec, sdir_vec) # Negative / positive depending on direction
        val = int(np.linalg.norm(mouse_vec)/np.linalg.norm(sdir_vec)+0.5)
        if cosang!=None and cosang<0: val = -val
        if self.start_height + val>self.pgeom.pjoint.dim: val = self.pgeom.pjoint.dim - self.start_height
        elif self.start_height+val<0: val = -self.start_height
        self.current_height = self.start_height + val
        self.val = int(val)

    def start_move(self,mouse_pos, h=1600):
        self.state=12
        self.start_pos = np.array([mouse_pos[0],h-mouse_pos[1]])
        self.new_fixed_sides = self.pgeom.pjoint.fixed.sides[self.n]
        self.new_fixed_sides_for_display = self.pgeom.pjoint.fixed.sides[self.n]
        self.pgeom.pjoint.combine_and_buffer_indices # for move preview outline

    def end_move(self):
        self.pgeom.pjoint.update_component_position(self.new_fixed_sides, self.n)
        self.state=-1
        self.new_fixed_sides_for_display = None

    def move(self,mouse_pos,screen_xrot,screen_yrot,w=1600,h=1600): # actually move OR rotate
        sax = self.pgeom.pjoint.sax
        noc = self.pgeom.pjoint.noc
        self.new_fixed_sides = copy.deepcopy(self.pgeom.pjoint.fixed.sides[self.n])
        self.new_fixed_sides_for_display = copy.deepcopy(self.pgeom.pjoint.fixed.sides[self.n])
        self.current_pos = np.array([mouse_pos[0],h-mouse_pos[1]])
        ## Mouse vector
        mouse_vec = np.array(self.current_pos-self.start_pos)
        mouse_vec = mouse_vec.astype(float)
        mouse_vec[0] = 2*mouse_vec[0]/w
        mouse_vec[1] = 2*mouse_vec[1]/h
        ## Check that the move distance is above some threshold
        move_dist = np.linalg.norm(mouse_vec)
        if move_dist>0.01:
            ## Get component direction vector
            comp_ax = self.pgeom.pjoint.fixed.sides[self.n][0].ax # component axis
            comp_dir = self.pgeom.pjoint.fixed.sides[self.n][0].dir
            comp_len = 2.5*(2*comp_dir-1)*self.pgeom.pjoint.component_size
            comp_vec = comp_len * Utils.unitize(self.pgeom.pjoint.pos_vecs[comp_ax])
            ## Flatten vector to screen
            rot_x = pyrr.Matrix33.from_x_rotation(screen_xrot)
            rot_y = pyrr.Matrix33.from_y_rotation(screen_yrot)
            comp_vec = np.dot(comp_vec,rot_x*rot_y)
            comp_vec = np.delete(comp_vec,2) # delete Z-value
            ## Calculate angle between mouse vector and component vector
            ang = Utils.angle_between_vectors1(mouse_vec, comp_vec, direction=True)
            oax = None
            absang = abs(ang)%180
            if absang>45 and absang<135: # Timber rotation mode
                # Check plane of rotating by checking which axis the vector is more aligned to
                other_axes = [0,1,2]
                other_axes.pop(comp_ax)
                # The axis that is flatter to the scren will be processed
                maxlen = 0
                for i in range(len(other_axes)):
                    other_vec = [0,0,0]
                    other_vec[other_axes[i]]=1
                    ## Flatten vector to screen
                    other_vec = np.dot(other_vec,rot_x*rot_y)
                    other_vec = np.delete(other_vec,2) # delete Z-value
                    ## Check length
                    other_length = np.linalg.norm(other_vec)
                    if other_length>maxlen:
                        maxlen = other_length
                        oax = other_axes[i]
                # check rotation direction
                clockwise = True
                if ang<0: clockwise = False
                #screen_direction
                lax = [0,1,2]
                lax.remove(comp_ax)
                lax.remove(oax)
                lax = lax[0]
                screen_dir = 1
                screen_vec = self.pgeom.pjoint.pos_vecs[lax]
                screen_vec = np.dot(screen_vec,rot_x*rot_y)
                if screen_vec[2]<0: screen_dir=-1
                ###
                self.new_fixed_sides_for_display = []
                for i in range(len(self.pgeom.pjoint.fixed.sides[self.n])):
                    ndir = self.pgeom.pjoint.fixed.sides[self.n][i].dir
                    ordered = False
                    if comp_ax<oax and oax-comp_ax==1: ordered=True
                    elif oax<comp_ax and comp_ax-oax==2: ordered=True
                    if (clockwise and not ordered) or (not clockwise and ordered):
                        ndir=1-ndir
                    if screen_dir>0: ndir=1-ndir
                    side = FixedSide(oax,ndir)
                    self.new_fixed_sides_for_display.append(side)
                    if side.ax==sax and side.dir==0 and self.n!=0: blocked=True; break
                    if side.ax==sax and side.dir==1 and self.n!=noc-1: blocked=True; break
            else: # Timber moveing mode
                length_ratio = np.linalg.norm(mouse_vec)/np.linalg.norm(comp_vec)
                side_num = len(self.pgeom.pjoint.fixed.sides[self.n])
                if side_num==1 and absang>135: #currently L
                    if length_ratio<0.5: # moved just a bit, L to T
                        self.new_fixed_sides_for_display = [FixedSide(comp_ax,0),FixedSide(comp_ax,1)]
                    elif length_ratio<2.0: # moved a lot, L to other L
                        self.new_fixed_sides_for_display = [FixedSide(comp_ax,1-comp_dir)]
                elif side_num==2: # currently T
                    if absang>135: self.new_fixed_sides_for_display = [FixedSide(comp_ax,1)] # positive direction
                    else: self.new_fixed_sides_for_display = [FixedSide(comp_ax,0)] # negative direction                            self.new_fixed_sides_for_display = [FixedSide(comp_ax,0)]
            # check if the direction is blocked
            blocked = False
            for side in self.new_fixed_sides_for_display:
                if side.unique(self.pgeom.pjoint.fixed.sides[self.n]):
                    if side.unique(self.pgeom.pjoint.fixed.unblocked):
                        blocked=True
            if blocked:
                all_same = True
                for side in self.new_fixed_sides_for_display:
                    if side.unique(self.pgeom.pjoint.fixed.sides[self.n]):
                        all_same=False
                if all_same: blocked = False
            if not blocked: self.new_fixed_sides = self.new_fixed_sides_for_display
        if not np.equal(self.pgeom.pjoint.fixed.sides[self.n], np.array(self.new_fixed_sides_for_display)).all():
            self.pgeom.pjoint.combine_and_buffer_indices()# for move/rotate preview outline # can't you show this by tansformation instead?

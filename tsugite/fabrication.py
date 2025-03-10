import math
import os
import numpy as np

import utils as Utils

class RoughPixel:
    def __init__(self,ind,mat,pad_loc,dim,n):
        self.ind = ind
        self.ind_abs = ind.copy()
        self.ind_abs[0] -= pad_loc[0][0]
        self.ind_abs[1] -= pad_loc[1][0]
        self.outside = False
        if self.ind_abs[0]<0 or self.ind_abs[0]>=dim:
            self.outside = True
        elif self.ind_abs[1]<0 or self.ind_abs[1]>=dim:
            self.outside = True
        self.neighbors = []
        # Region or free=0
        # Blocked=1
        for ax in range(2):
            temp = []
            for dir in range(-1,2,2):
                nind = self.ind.copy()
                nind[ax] += dir
                type = 0
                if nind[0]>=0 and nind[0]<mat.shape[0] and nind[1]>=0 and nind[1]<mat.shape[1]:
                    val = mat[tuple(nind)]
                    if val==n: type = 1
                temp.append(type)
            self.neighbors.append(temp)
        self.flat_neighbors = [x for sublist in self.neighbors for x in sublist]

class MillVertex:
    def __init__(self,pt,is_tra=False,is_arc=False,arc_ctr=np.array([0,0,0])):
        self.pt = np.array(pt)
        self.x = pt[0]
        self.y = pt[1]
        self.z = pt[2]
        self.is_arc = is_arc
        self.arc_ctr = np.array(arc_ctr)
        self.is_tra = is_tra # is traversing, gcode_mode G0 (max speed) (otherwise G1)

    def scale_and_swap(self,ax,dir,ratio,unit_scale,real_tim_dims,coords,d,n):
        
        #print("1", self.x,self.y,self.z)

        #sawp and scale
        xyz = [unit_scale*ratio*self.x, unit_scale*ratio*self.y, unit_scale*ratio*self.z]
        if ax==2: xyz[1] = -xyz[1]
        xyz = xyz[coords[0]],xyz[coords[1]],xyz[coords[2]]
        self.x,self.y,self.z = xyz[0],xyz[1],xyz[2]

        #print("2",self.x,self.y,self.z)
        
        #move z down, flip if component b
        self.z = -(2*dir-1)*self.z-0.5*unit_scale*real_tim_dims[ax]
        self.y = -(2*dir-1)*self.y
        self.pt = np.array([self.x,self.y,self.z])
        self.pos = np.array([self.x,self.y,self.z],dtype=np.float64)
        self.xstr = str(round(self.x,d))
        self.ystr = str(round(self.y,d))
        self.zstr = str(round(self.z,d))
        ##
        if self.is_arc:
            self.arc_ctr = [unit_scale*ratio*self.arc_ctr[0], unit_scale*ratio*self.arc_ctr[1], unit_scale*ratio*self.arc_ctr[2]] 
            if ax==2: self.arc_ctr[1] = -self.arc_ctr[1]
            self.arc_ctr = [self.arc_ctr[coords[0]],self.arc_ctr[coords[1]],self.arc_ctr[coords[2]]]
            self.arc_ctr[2] = -(2*dir-1)*self.arc_ctr[2]-0.5*unit_scale*real_tim_dims[ax]
            self.arc_ctr[1] = -(2*dir-1)*self.arc_ctr[1]
            self.arc_ctr = np.array(self.arc_ctr)

    def rotate(self,ang,d):
        self.pt = np.array([self.x,self.y,self.z])
        self.pt = Utils.rotate_vector_around_axis(self.pt, [0,0,1], ang)
        self.x = self.pt[0]
        self.y = self.pt[1]
        self.z = self.pt[2]
        self.pos = np.array([self.x,self.y,self.z],dtype=np.float64)
        self.xstr = str(round(self.x,d))
        self.ystr = str(round(self.y,d))
        self.zstr = str(round(self.z,d))
        ##
        if self.is_arc:
            self.arc_ctr = Utils.rotate_vector_around_axis(self.arc_ctr, [0,0,1], ang)
            self.arc_ctr = np.array(self.arc_ctr)

class Fabrication:
    def __init__(self,pjoint,tol=0.15,dia=6.00,ext="gcode",align_ax=0,interp=True, spe=400, spi=6000):
        self.pjoint = pjoint
        self.real_dia = dia #milling bit radius in mm
        self.tol = tol #0.10 #tolerance in mm
        self.rad = 0.5*self.real_dia-self.tol
        self.dia = 2*self.rad
        self.unit_scale = 1.0
        #if ext=='sbp': self.unit_scale=1/25.4 #inches
        self.vdia = self.dia/self.pjoint.ratio
        self.vrad = self.rad/self.pjoint.ratio
        self.vtol = self.tol/self.pjoint.ratio
        self.dep = 1.5 #milling depth in mm
        self.align_ax = align_ax
        self.ext = ext
        self.interp=interp
        self.speed = spe
        self.spindlespeed = spi
    
    def update_extension(self,ext):
        self.ext = ext
        self.unit_scale = 1.0
        #if self.ext=='sbp': self.unit_scale=1/25.4 #inches
        #print(self.ext,self.unit_scale)


    def export_gcode(self,filename_tsu=os.getcwd()+os.sep+"joint.tsu"):
        print(self.ext)
        # make sure that the z axis of the gcode is facing up
        fax = self.pjoint.sax
        coords = [0,1]
        coords.insert(fax,2)
        #
        d = 5 # =precision / no of decimals to write
        names = ["A","B","C","D","E","F"]
        for n in range(self.pjoint.noc):
            fdir = self.pjoint.mesh.fab_directions[n]
            comp_ax = self.pjoint.fixed.sides[n][0].ax
            comp_dir = self.pjoint.fixed.sides[n][0].dir # component direction
            comp_vec = self.pjoint.pos_vecs[comp_ax]
            if comp_dir==0 and comp_ax!=self.pjoint.sax: comp_vec=-comp_vec
            comp_vec = np.array([comp_vec[coords[0]],comp_vec[coords[1]],comp_vec[coords[2]]])
            comp_vec = comp_vec/np.linalg.norm(comp_vec) #unitize
            zax = np.array([0,0,1])
            aax = [0,0,0]
            aax[int(self.align_ax/2)] = 2*(self.align_ax%2)-1
            #aax = rotate_vector_around_axis(aax, axis=zax, theta=math.radians(self.extra_rot_deg))
            rot_ang = Utils.angle_between_vectors2(aax,comp_vec,normal_vector=zax)
            if fdir==0: rot_ang=-rot_ang
            #
            file_name = filename_tsu[:-4] + "_"+names[n]+"."+self.ext
            file = open(file_name,"w")
            if self.ext=="gcode" or self.ext=="nc":
                ###initialization .goce and .nc
                file.write("%\n")
                file.write("G90 (Absolute [G91 is incremental])\n")
                file.write("G17 (set XY plane for circle path)\n")
                file.write("G94 (set unit/minute)\n")
                file.write("G21 (set unit[mm])\n")
                spistr = str(int(self.spindlespeed))
                file.write("S"+spistr+" (Spindle "+spistr+"rpm)\n")
                file.write("M3 (spindle start)\n")
                file.write("G54\n")
                spestr=str(int(self.speed))
                file.write("F"+spestr+" (Feed "+spestr+"mm/min)\n")
            elif self.ext=="sbp":
                file.write("'%\n")
                #file.write("VN, 2\n")
                file.write("SA\n") # Set to Absolute Distances
                #file.write("TR 6000\n\n") #?
                file.write("TR,18000\n") # from VUILD
                file.write("C6\n")
                file.write("PAUSE 2\n")
                file.write("'\n")
                #file.write("SO 1,1\n") #Set Output Switch ON
                file.write("MS,6.67,6.67\n\n") # Move Speed Set
            else:
                print("Unknown extension:", self.ext)

            ###content
            for i,mv in enumerate(self.pjoint.gcodeverts[n]):
                mv.scale_and_swap(fax, fdir, self.pjoint.ratio, self.unit_scale, self.pjoint.real_tim_dims, coords, d, n)
                if comp_ax!=fax: mv.rotate(rot_ang,d)
                if i>0: pmv = self.pjoint.gcodeverts[n][i - 1] #pmv=previous mill vertex
                # check segment angle
                arc = False
                clockwise = False
                if i>0 and Utils.connected_arc(mv,pmv):
                    arc = True
                    vec1 = mv.pt-mv.arc_ctr
                    vec1 = vec1/np.linalg.norm(vec1)
                    zvec = np.array([0,0,1])
                    xvec = np.cross(vec1,zvec)
                    vec2 = pmv.pt-mv.arc_ctr
                    vec2 = vec2/np.linalg.norm(vec2)
                    diff_ang = Utils.angle_between_vectors2(xvec,vec2)
                    if diff_ang>0.5*math.pi: clockwise = True

                #write to file
                if self.ext=="gcode" or self.ext=="nc":
                    if arc and self.interp:
                        if clockwise: file.write("G2")
                        else: file.write("G3")
                        file.write(" R"+str(round(self.dia,d))+" X"+mv.xstr+" Y"+mv.ystr)
                        if mv.z!=pmv.z: file.write(" Z"+mv.zstr)
                        file.write("\n")
                    elif arc and not self.interp:
                        pts = Utils.arc_points(pmv.pt,mv.pt,pmv.arc_ctr,mv.arc_ctr,2,math.radians(1))
                        for pt in pts:
                            file.write("G1")
                            file.write(" X"+str(round(pt[0],3))+" Y"+str(round(pt[1],3)))
                            if mv.z!=pmv.z: file.write(" Z"+str(round(pt[2],3)))
                            file.write("\n")
                    elif i==0 or mv.x!=pmv.x or mv.y!=pmv.y or mv.z!=pmv.z:
                        if mv.is_tra: file.write("G0")
                        else: file.write("G1")
                        if i==0 or mv.x!=pmv.x: file.write(" X"+mv.xstr)
                        if i==0 or mv.y!=pmv.y: file.write(" Y"+mv.ystr)
                        if i==0 or mv.z!=pmv.z: file.write(" Z"+mv.zstr)
                        file.write("\n")
                        
                elif self.ext=="sbp":
                    if arc and mv.z==pmv.z:
                        file.write("CG,"+str(round(2*self.dia*self.unit_scale,d))+","+mv.xstr+","+mv.ystr+",,,T,")
                        if clockwise: file.write("1\n")
                        else: file.write("-1\n")
                    elif arc and mv.z!=pmv.z:
                        pts = Utils.arc_points(pmv.pt,mv.pt,pmv.arc_ctr,mv.arc_ctr,2,math.radians(1))
                        for pt in pts:
                            file.write("M3,"+str(round(pt[0],3))+","+str(round(pt[1],3))+","+str(round(pt[2],3))+"\n")
                    elif i==0 or mv.x!=pmv.x or mv.y!=pmv.y or mv.z!=pmv.z:
                        if mv.is_tra: file.write("J3,")
                        else: file.write("M3,")
                        if i==0 or mv.x!=pmv.x: file.write(mv.xstr+",")
                        else: file.write(" ,")
                        if i==0 or mv.y!=pmv.y: file.write(mv.ystr+",")
                        else: file.write(" ,")
                        if i==0 or mv.z!=pmv.z: file.write(mv.zstr+"\n")
                        else: file.write(" \n")
            #end
            if self.ext=="gcode" or self.ext=="nc":
                file.write("M5 (Spindle stop)\n")
                file.write("M2 (end of program)\n")
                file.write("M30 (delete sd file)\n")
                file.write("%\n")
            elif self.ext=="sbp":
                file.write("SO 1,0\n")
                file.write("END\n")
                file.write("'%\n")

            print("Exported",file_name)
            file.close()

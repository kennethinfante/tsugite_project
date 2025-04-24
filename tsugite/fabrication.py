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
        self.pt: np.ndarray = np.array(pt)
        self.x = pt[0]
        self.y = pt[1]
        self.z = pt[2]
        self.is_arc: bool = is_arc
        self.arc_ctr: Optional[np.ndarray] = np.array(arc_ctr)
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


    # def export_gcode(self,filename_tsu=os.getcwd()+os.sep+"joint.tsu"):
    #     print(self.ext)
    #     # make sure that the z axis of the gcode is facing up
    #     fax = self.pjoint.sax
    #     coords = [0,1]
    #     coords.insert(fax,2)
    #     #
    #     d = 5 # =precision / no of decimals to write
    #     names = ["A","B","C","D","E","F"]
    #     for n in range(self.pjoint.noc):
    #         fdir = self.pjoint.mesh.fab_directions[n]
    #         comp_ax = self.pjoint.fixed.sides[n][0].ax
    #         comp_dir = self.pjoint.fixed.sides[n][0].dir # component direction
    #         comp_vec = self.pjoint.pos_vecs[comp_ax]
    #         if comp_dir==0 and comp_ax!=self.pjoint.sax: comp_vec=-comp_vec
    #         comp_vec = np.array([comp_vec[coords[0]],comp_vec[coords[1]],comp_vec[coords[2]]])
    #         comp_vec = comp_vec/np.linalg.norm(comp_vec) #unitize
    #         zax = np.array([0,0,1])
    #         aax = [0,0,0]
    #         aax[int(self.align_ax/2)] = 2*(self.align_ax%2)-1
    #         #aax = rotate_vector_around_axis(aax, axis=zax, theta=math.radians(self.extra_rot_deg))
    #         rot_ang = Utils.angle_between_vectors2(aax,comp_vec,normal_vector=zax)
    #         if fdir==0: rot_ang=-rot_ang
    #         #
    #         file_name = filename_tsu[:-4] + "_"+names[n]+"."+self.ext
    #         file = open(file_name,"w")
    #         if self.ext=="gcode" or self.ext=="nc":
    #             ###initialization .goce and .nc
    #             file.write("%\n")
    #             file.write("G90 (Absolute [G91 is incremental])\n")
    #             file.write("G17 (set XY plane for circle path)\n")
    #             file.write("G94 (set unit/minute)\n")
    #             file.write("G21 (set unit[mm])\n")
    #             spistr = str(int(self.spindlespeed))
    #             file.write("S"+spistr+" (Spindle "+spistr+"rpm)\n")
    #             file.write("M3 (spindle start)\n")
    #             file.write("G54\n")
    #             spestr=str(int(self.speed))
    #             file.write("F"+spestr+" (Feed "+spestr+"mm/min)\n")
    #         elif self.ext=="sbp":
    #             file.write("'%\n")
    #             #file.write("VN, 2\n")
    #             file.write("SA\n") # Set to Absolute Distances
    #             #file.write("TR 6000\n\n") #?
    #             file.write("TR,18000\n") # from VUILD
    #             file.write("C6\n")
    #             file.write("PAUSE 2\n")
    #             file.write("'\n")
    #             #file.write("SO 1,1\n") #Set Output Switch ON
    #             file.write("MS,6.67,6.67\n\n") # Move Speed Set
    #         else:
    #             print("Unknown extension:", self.ext)
    #
    #         ###content
    #         for i,mv in enumerate(self.pjoint.gcodeverts[n]):
    #             mv.scale_and_swap(fax, fdir, self.pjoint.ratio, self.unit_scale, self.pjoint.real_tim_dims, coords, d, n)
    #             if comp_ax!=fax: mv.rotate(rot_ang,d)
    #             if i>0: pmv = self.pjoint.gcodeverts[n][i - 1] #pmv=previous mill vertex
    #             # check segment angle
    #             arc = False
    #             clockwise = False
    #             if i>0 and Utils.connected_arc(mv,pmv):
    #                 arc = True
    #                 vec1 = mv.pt-mv.arc_ctr
    #                 vec1 = vec1/np.linalg.norm(vec1)
    #                 zvec = np.array([0,0,1])
    #                 xvec = np.cross(vec1,zvec)
    #                 vec2 = pmv.pt-mv.arc_ctr
    #                 vec2 = vec2/np.linalg.norm(vec2)
    #                 diff_ang = Utils.angle_between_vectors2(xvec,vec2)
    #                 if diff_ang>0.5*math.pi: clockwise = True
    #
    #             #write to file
    #             if self.ext=="gcode" or self.ext=="nc":
    #                 if arc and self.interp:
    #                     if clockwise: file.write("G2")
    #                     else: file.write("G3")
    #                     file.write(" R"+str(round(self.dia,d))+" X"+mv.xstr+" Y"+mv.ystr)
    #                     if mv.z!=pmv.z: file.write(" Z"+mv.zstr)
    #                     file.write("\n")
    #                 elif arc and not self.interp:
    #                     pts = Utils.arc_points(pmv.pt,mv.pt,pmv.arc_ctr,mv.arc_ctr,2,math.radians(1))
    #                     for pt in pts:
    #                         file.write("G1")
    #                         file.write(" X"+str(round(pt[0],3))+" Y"+str(round(pt[1],3)))
    #                         if mv.z!=pmv.z: file.write(" Z"+str(round(pt[2],3)))
    #                         file.write("\n")
    #                 elif i==0 or mv.x!=pmv.x or mv.y!=pmv.y or mv.z!=pmv.z:
    #                     if mv.is_tra: file.write("G0")
    #                     else: file.write("G1")
    #                     if i==0 or mv.x!=pmv.x: file.write(" X"+mv.xstr)
    #                     if i==0 or mv.y!=pmv.y: file.write(" Y"+mv.ystr)
    #                     if i==0 or mv.z!=pmv.z: file.write(" Z"+mv.zstr)
    #                     file.write("\n")
    #
    #             elif self.ext=="sbp":
    #                 if arc and mv.z==pmv.z:
    #                     file.write("CG,"+str(round(2*self.dia*self.unit_scale,d))+","+mv.xstr+","+mv.ystr+",,,T,")
    #                     if clockwise: file.write("1\n")
    #                     else: file.write("-1\n")
    #                 elif arc and mv.z!=pmv.z:
    #                     pts = Utils.arc_points(pmv.pt,mv.pt,pmv.arc_ctr,mv.arc_ctr,2,math.radians(1))
    #                     for pt in pts:
    #                         file.write("M3,"+str(round(pt[0],3))+","+str(round(pt[1],3))+","+str(round(pt[2],3))+"\n")
    #                 elif i==0 or mv.x!=pmv.x or mv.y!=pmv.y or mv.z!=pmv.z:
    #                     if mv.is_tra: file.write("J3,")
    #                     else: file.write("M3,")
    #                     if i==0 or mv.x!=pmv.x: file.write(mv.xstr+",")
    #                     else: file.write(" ,")
    #                     if i==0 or mv.y!=pmv.y: file.write(mv.ystr+",")
    #                     else: file.write(" ,")
    #                     if i==0 or mv.z!=pmv.z: file.write(mv.zstr+"\n")
    #                     else: file.write(" \n")
    #         #end
    #         if self.ext=="gcode" or self.ext=="nc":
    #             file.write("M5 (Spindle stop)\n")
    #             file.write("M2 (end of program)\n")
    #             file.write("M30 (delete sd file)\n")
    #             file.write("%\n")
    #         elif self.ext=="sbp":
    #             file.write("SO 1,0\n")
    #             file.write("END\n")
    #             file.write("'%\n")
    #
    #         print("Exported",file_name)
    #         file.close()

    def export_gcode(self, filename_tsu=os.getcwd()+os.sep+"joint.tsu"):
        """Export fabrication instructions to G-code or ShopBot files."""
        print(self.ext)

        # Calculate common parameters
        fax = self.pjoint.sax
        coords = [0, 1]
        coords.insert(fax, 2)

        d = 5  # precision / no of decimals to write
        names = ["A", "B", "C", "D", "E", "F"]

        # Process each component
        for n in range(self.pjoint.noc):
            # Calculate component parameters
            comp_params = self._calculate_component_parameters(n, fax, coords)

            # Create output file
            file_name = filename_tsu[:-4] + "_" + names[n] + "." + self.ext
            with open(file_name, "w") as file:
                # Write file header
                self._write_file_header(file)

                # Write toolpath instructions
                self._write_toolpath_instructions(file, n, fax, comp_params, coords, d)

                # Write file footer
                self._write_file_footer(file)

            print("Exported", file_name)

    def _calculate_component_parameters(self, n, fax, coords):
        """Calculate parameters for a component."""
        fdir = self.pjoint.mesh.fab_directions[n]
        comp_ax = self.pjoint.fixed.sides[n][0].ax
        comp_dir = self.pjoint.fixed.sides[n][0].dir  # component direction

        # Calculate component vector
        comp_vec = self.pjoint.pos_vecs[comp_ax]
        if comp_dir == 0 and comp_ax != self.pjoint.sax:
            comp_vec = -comp_vec
        comp_vec = np.array([comp_vec[coords[0]], comp_vec[coords[1]], comp_vec[coords[2]]])
        comp_vec = comp_vec / np.linalg.norm(comp_vec)  # unitize

        # Calculate rotation angle
        zax = np.array([0, 0, 1])
        aax = [0, 0, 0]
        aax[int(self.align_ax/2)] = 2*(self.align_ax%2)-1
        rot_ang = Utils.angle_between_vectors2(aax, comp_vec, normal_vector=zax)
        if fdir == 0:
            rot_ang = -rot_ang

        return {
            'fdir': fdir,
            'comp_ax': comp_ax,
            'rot_ang': rot_ang
        }

    def _write_file_header(self, file):
        """Write the header section of the output file."""
        if self.ext == "gcode" or self.ext == "nc":
            self._write_gcode_header(file)
        elif self.ext == "sbp":
            self._write_sbp_header(file)
        else:
            print("Unknown extension:", self.ext)

    def _write_gcode_header(self, file):
        """Write G-code file header."""
        file.write("%\n")
        file.write("G90 (Absolute [G91 is incremental])\n")
        file.write("G17 (set XY plane for circle path)\n")
        file.write("G94 (set unit/minute)\n")
        file.write("G21 (set unit[mm])\n")
        spistr = str(int(self.spindlespeed))
        file.write("S" + spistr + " (Spindle " + spistr + "rpm)\n")
        file.write("M3 (spindle start)\n")
        file.write("G54\n")
        spestr = str(int(self.speed))
        file.write("F" + spestr + " (Feed " + spestr + "mm/min)\n")

    def _write_sbp_header(self, file):
        """Write ShopBot file header."""
        file.write("'%\n")
        file.write("SA\n")  # Set to Absolute Distances
        file.write("TR,18000\n")  # from VUILD
        file.write("C6\n")
        file.write("PAUSE 2\n")
        file.write("'\n")
        file.write("MS,6.67,6.67\n\n")  # Move Speed Set

    def _write_file_footer(self, file):
        """Write the footer section of the output file."""
        if self.ext == "gcode" or self.ext == "nc":
            file.write("M5 (Spindle stop)\n")
            file.write("M2 (end of program)\n")
            file.write("M30 (delete sd file)\n")
            file.write("%\n")
        elif self.ext == "sbp":
            file.write("SO 1,0\n")
            file.write("END\n")
            file.write("'%\n")

    def _write_toolpath_instructions(self, file, n, fax, comp_params, coords, d):
        """Write toolpath instructions to the file."""
        for i, mv in enumerate(self.pjoint.gcodeverts[n]):
            # Scale and transform the vertex
            mv.scale_and_swap(fax, comp_params['fdir'], self.pjoint.ratio,
                              self.unit_scale, self.pjoint.real_tim_dims, coords, d, n)

            # Rotate if needed
            if comp_params['comp_ax'] != fax:
                mv.rotate(comp_params['rot_ang'], d)

            # Get previous vertex if available
            pmv = self.pjoint.gcodeverts[n][i - 1] if i > 0 else None

            # Check if this is an arc segment
            arc_params = self._check_for_arc(mv, pmv, i)

            # Write the appropriate instruction
            self._write_instruction(file, mv, pmv, i, arc_params)

    def _check_for_arc(self, mv, pmv, i):
        """Check if the current segment is an arc and determine its parameters."""
        if i == 0 or pmv is None:
            return {'is_arc': False}

        if Utils.connected_arc(mv, pmv):
            # Calculate arc direction
            vec1 = mv.pt - mv.arc_ctr
            vec1 = vec1 / np.linalg.norm(vec1)
            zvec = np.array([0, 0, 1])
            xvec = np.cross(vec1, zvec)
            vec2 = pmv.pt - mv.arc_ctr
            vec2 = vec2 / np.linalg.norm(vec2)
            diff_ang = Utils.angle_between_vectors2(xvec, vec2)
            clockwise = diff_ang > 0.5 * math.pi

            return {
                'is_arc': True,
                'clockwise': clockwise
            }

        return {'is_arc': False}

    def _write_instruction(self, file, mv, pmv, i, arc_params):
        """Write a single instruction to the file."""
        if self.ext == "gcode" or self.ext == "nc":
            self._write_gcode_instruction(file, mv, pmv, i, arc_params)
        elif self.ext == "sbp":
            self._write_sbp_instruction(file, mv, pmv, i, arc_params)

    def _write_gcode_instruction(self, file, mv, pmv, i, arc_params):
        """Write a G-code instruction."""
        if arc_params['is_arc'] and self.interp:
            # Write interpolated arc
            cmd = "G2" if arc_params['clockwise'] else "G3"
            file.write(f"{cmd} R{round(self.dia, 5)} X{mv.xstr} Y{mv.ystr}")
            if i > 0 and mv.z != pmv.z:
                file.write(f" Z{mv.zstr}")
            file.write("\n")
        elif arc_params['is_arc'] and not self.interp:
            # Write non-interpolated arc as a series of linear moves
            pts = Utils.arc_points(pmv.pt, mv.pt, pmv.arc_ctr, mv.arc_ctr, 2, math.radians(1))
            for pt in pts:
                file.write(f"G1 X{round(pt[0], 3)} Y{round(pt[1], 3)}")
                if i > 0 and mv.z != pmv.z:
                    file.write(f" Z{round(pt[2], 3)}")
                file.write("\n")
        elif i == 0 or mv.x != pmv.x or mv.y != pmv.y or mv.z != pmv.z:
            # Write linear move
            cmd = "G0" if mv.is_tra else "G1"
            file.write(cmd)
            if i == 0 or mv.x != pmv.x:
                file.write(f" X{mv.xstr}")
            if i == 0 or mv.y != pmv.y:
                file.write(f" Y{mv.ystr}")
            if i == 0 or mv.z != pmv.z:
                file.write(f" Z{mv.zstr}")
            file.write("\n")

    def _write_sbp_instruction(self, file, mv, pmv, i, arc_params):
        """Write a ShopBot instruction."""
        if arc_params['is_arc'] and mv.z == pmv.z:
            # Write arc in same Z plane
            file.write(f"CG,{round(2*self.dia*self.unit_scale, 5)},{mv.xstr},{mv.ystr},,,T,")
            file.write("1\n" if arc_params['clockwise'] else "-1\n")
        elif arc_params['is_arc'] and mv.z != pmv.z:
            # Write arc with Z change as a series of linear moves
            pts = Utils.arc_points(pmv.pt, mv.pt, pmv.arc_ctr, mv.arc_ctr, 2, math.radians(1))
            for pt in pts:
                file.write(f"M3,{round(pt[0], 3)},{round(pt[1], 3)},{round(pt[2], 3)}\n")
        elif i == 0 or mv.x != pmv.x or mv.y != pmv.y or mv.z != pmv.z:
            # Write linear move
            cmd = "J3," if mv.is_tra else "M3,"
            file.write(cmd)
            if i == 0 or mv.x != pmv.x:
                file.write(f"{mv.xstr},")
            else:
                file.write(" ,")
            if i == 0 or mv.y != pmv.y:
                file.write(f"{mv.ystr},")
            else:
                file.write(" ,")
            if i == 0 or mv.z != pmv.z:
                file.write(f"{mv.zstr}\n")
            else:
                file.write(" \n")


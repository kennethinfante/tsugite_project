import math
import os
import numpy as np

import utils_ as Utils

class RoughPixel:

    def __init__(self, ind, mat, pad_loc, dim, n):
        """Initialize a RoughPixel for rough milling path generation."""
        # Store indices and calculate absolute indices
        self.ind = ind
        self.ind_abs = self._calculate_absolute_indices(ind, pad_loc)

        # Check if pixel is outside the valid region
        self.outside = self._is_outside_region(dim)

        # Calculate neighbor information
        self.neighbors = self._calculate_neighbors(mat, n)
        self.flat_neighbors = [x for sublist in self.neighbors for x in sublist]

    def _calculate_absolute_indices(self, ind, pad_loc):
        """Calculate absolute indices by removing padding."""
        ind_abs = ind.copy()
        ind_abs[0] -= pad_loc[0][0]
        ind_abs[1] -= pad_loc[1][0]
        return ind_abs

    def _is_outside_region(self, dim):
        """Check if pixel is outside the valid region."""
        if self.ind_abs[0] < 0 or self.ind_abs[0] >= dim:
            return True
        elif self.ind_abs[1] < 0 or self.ind_abs[1] >= dim:
            return True
        return False

    def _calculate_neighbors(self, mat, n):
        """Calculate neighbor information for the pixel."""
        neighbors = []

        # Check neighbors in each axis direction
        for ax in range(2):
            temp = []
            for dir in range(-1, 2, 2):
                # Calculate neighbor index
                nind = self.ind.copy()
                nind[ax] += dir

                # Check if neighbor is valid and blocked
                type = 0  # Default: region or free
                if self._is_valid_index(nind, mat.shape):
                    val = mat[tuple(nind)]
                    if val == n:
                        type = 1  # Blocked

                temp.append(type)
            neighbors.append(temp)

        return neighbors

    def _is_valid_index(self, ind, shape):
        """Check if index is within matrix bounds."""
        return (0 <= ind[0] < shape[0]) and (0 <= ind[1] < shape[1])

class MillVertex:
    def __init__(self,pt,is_tra=False,is_arc=False,arc_ctr=np.array([0,0,0])):
        self.pt: np.ndarray = np.array(pt)
        self.x = pt[0]
        self.y = pt[1]
        self.z = pt[2]
        self.is_arc: bool = is_arc
        self.arc_ctr: Optional[np.ndarray] = np.array(arc_ctr)
        self.is_tra = is_tra # is traversing, gcode_mode G0 (max speed) (otherwise G1)

    def scale_and_swap(self, ax, dir, ratio, unit_scale, real_tim_dims, coords, d, n):
        """Scale and transform vertex coordinates for fabrication."""
        # Scale coordinates
        xyz = self._scale_coordinates(ratio, unit_scale)

        # Apply axis-specific transformations
        xyz = self._apply_axis_transformations(xyz, ax, coords)

        # Update vertex properties
        self._update_vertex_properties(xyz, ax, dir, unit_scale, real_tim_dims, d)

        # Handle arc center if this is an arc
        if self.is_arc:
            self._transform_arc_center(ax, dir, ratio, unit_scale, real_tim_dims, coords)

    def _scale_coordinates(self, ratio, unit_scale):
        """Scale the coordinates by ratio and unit scale."""
        return [
            unit_scale * ratio * self.x,
            unit_scale * ratio * self.y,
            unit_scale * ratio * self.z
        ]

    def _apply_axis_transformations(self, xyz, ax, coords):
        """Apply axis-specific transformations to coordinates."""
        # Flip Y coordinate if Z axis
        if ax == 2:
            xyz[1] = -xyz[1]

        # Reorder coordinates according to the coordinate mapping
        return xyz[coords[0]], xyz[coords[1]], xyz[coords[2]]

    def _update_vertex_properties(self, xyz, ax, dir, unit_scale, real_tim_dims, d):
        """Update vertex properties with transformed coordinates."""
        self.x, self.y, self.z = xyz[0], xyz[1], xyz[2]

        # Apply Z-axis adjustments and Y-axis flipping
        self.z = -(2*dir-1) * self.z - 0.5 * unit_scale * real_tim_dims[ax]
        self.y = -(2*dir-1) * self.y

        # Update point arrays and string representations
        self.pt = np.array([self.x, self.y, self.z])
        self.pos = np.array([self.x, self.y, self.z], dtype=np.float64)
        self.xstr = str(round(self.x, d))
        self.ystr = str(round(self.y, d))
        self.zstr = str(round(self.z, d))

    def _transform_arc_center(self, ax, dir, ratio, unit_scale, real_tim_dims, coords):
        """Transform arc center coordinates if this is an arc vertex."""
        # Scale arc center coordinates
        self.arc_ctr = [
            unit_scale * ratio * self.arc_ctr[0],
            unit_scale * ratio * self.arc_ctr[1],
            unit_scale * ratio * self.arc_ctr[2]
        ]

        # Apply axis-specific transformations
        if ax == 2:
            self.arc_ctr[1] = -self.arc_ctr[1]

        # Reorder coordinates
        self.arc_ctr = [
            self.arc_ctr[coords[0]],
            self.arc_ctr[coords[1]],
            self.arc_ctr[coords[2]]
        ]

        # Apply Z-axis adjustments and Y-axis flipping
        self.arc_ctr[2] = -(2*dir-1) * self.arc_ctr[2] - 0.5 * unit_scale * real_tim_dims[ax]
        self.arc_ctr[1] = -(2*dir-1) * self.arc_ctr[1]

        # Convert to numpy array
        self.arc_ctr = np.array(self.arc_ctr)

    def rotate(self, ang, d):
        """Rotate vertex around Z axis by the given angle."""
        # Rotate point coordinates
        self._rotate_point(ang, d)

        # Rotate arc center if this is an arc
        if self.is_arc:
            self._rotate_arc_center(ang)

    def _rotate_point(self, ang, d):
        """Rotate the point coordinates around Z axis."""
        self.pt = np.array([self.x, self.y, self.z])
        self.pt = Utils.rotate_vector_around_axis(self.pt, [0, 0, 1], ang)

        # Update coordinates and string representations
        self.x = self.pt[0]
        self.y = self.pt[1]
        self.z = self.pt[2]
        self.pos = np.array([self.x, self.y, self.z], dtype=np.float64)
        self.xstr = str(round(self.x, d))
        self.ystr = str(round(self.y, d))
        self.zstr = str(round(self.z, d))

    def _rotate_arc_center(self, ang):
        """Rotate the arc center coordinates around Z axis."""
        self.arc_ctr = Utils.rotate_vector_around_axis(self.arc_ctr, [0, 0, 1], ang)
        self.arc_ctr = np.array(self.arc_ctr)

class Fabrication:

    def __init__(self, pjoint, tol=0.15, dia=6.00, ext="gcode", align_ax=0, interp=True, spe=400, spi=6000):
        """Initialize fabrication parameters."""
        # Store joint reference and basic parameters
        self.pjoint = pjoint
        self.ext = ext
        self.align_ax = align_ax
        self.interp = interp
        self.speed = spe
        self.spindlespeed = spi

        # Initialize tool parameters
        self._init_tool_parameters(tol, dia)

        # Initialize unit scale
        self.unit_scale = 1.0
        self.dep = 1.5  # milling depth in mm

    def _init_tool_parameters(self, tol, dia):
        """Initialize tool-related parameters."""
        # Store basic tool parameters
        self.real_dia = dia  # milling bit diameter in mm
        self.tol = tol  # tolerance in mm

        # Calculate derived tool parameters
        self.rad = 0.5 * self.real_dia - self.tol
        self.dia = 2 * self.rad

        # Calculate virtual (scaled) tool parameters
        self.vdia = self.dia / self.pjoint.ratio
        self.vrad = self.rad / self.pjoint.ratio
        self.vtol = self.tol / self.pjoint.ratio

    def update_extension(self,ext):
        self.ext = ext
        self.unit_scale = 1.0
        #if self.ext=='sbp': self.unit_scale=1/25.4 #inches
        #print(self.ext,self.unit_scale)


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
        rot_ang = Utils.angle_between_vectors(aax, comp_vec, normal_vector=zax)
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
            diff_ang = Utils.angle_between_vectors(xvec, vec2)
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


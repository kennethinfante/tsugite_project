import random
import os

import numpy as np
import OpenGL.GL as GL  # imports start with GL

from selection import Selection
from evaluation import Evaluation
from buffer import ElementProperties

import utils_ as Utils

class Geometries:
    def __init__(self, pjoint, mainmesh=True, hfs=[]):
        self.mainmesh = mainmesh
        self.pjoint = pjoint
        self.fab_directions = [0,1] #Initiate list of fabrication directions
        for i in range(1, self.pjoint.noc - 1): self.fab_directions.insert(1, 1)
        if len(hfs)==0: self.height_fields = Utils.get_random_height_fields(self.pjoint.dim, self.pjoint.noc) #Initiate a random joint geometry
        else: self.height_fields = hfs
        if self.mainmesh: self.select = Selection(self)
        self.update_voxel_matrix_from_height_fields(first=True)

    def _process_indices(self, indices, all_indices, element_type, n, offset=0, global_offset=0):
        """Process and format indices, creating element properties."""
        indices = np.array(indices, dtype=np.uint32)
        indices = indices + offset

        indices_prop = ElementProperties(element_type, len(indices), len(all_indices) + global_offset, n)
        all_indices = np.concatenate([all_indices, indices]) if len(all_indices) > 0 else indices

        return indices_prop, all_indices

    def update_voxel_matrix_from_height_fields(self, first=False):
        vox_mat = Utils.matrix_from_height_fields(self.height_fields, self.pjoint.sax)
        self.voxel_matrix = vox_mat
        if self.mainmesh:
            self.eval = Evaluation(self.voxel_matrix, self.pjoint)
            self.fab_directions = self.eval.fab_directions
        if self.mainmesh and not first:
            self.pjoint.update_suggestions()

    def _joint_line_indices(self, all_indices, n, offset, global_offset=0):
        fixed_sides = self.pjoint.fixed.sides[n]
        indices = []

        # Add joint outline indices
        indices.extend(self._get_joint_outline_indices(n))

        # Add component base outline indices
        indices.extend(self._get_component_base_outline_indices(fixed_sides, n))

        # Process and return
        return self._process_indices(indices, all_indices, GL.GL_LINES, n, offset, global_offset)

    def _get_joint_outline_indices(self, n):
        """Extract indices for the joint outline."""
        indices = []
        d = self.pjoint.dim + 1

        for i in range(d):
            for j in range(d):
                for k in range(d):
                    ind = [i, j, k]
                    for ax in range(3):
                        if ind[ax] == self.pjoint.dim:
                            continue

                        cnt, vals = self._get_line_neighbor_values(ind, ax, n)
                        diagonal = (vals[0] == vals[3] or vals[1] == vals[2])

                        if cnt == 1 or cnt == 3 or (cnt == 2 and diagonal):
                            add = [0, 0, 0]
                            add[ax] = 1
                            start_i = Utils.get_index(ind, [0, 0, 0], self.pjoint.dim)
                            end_i = Utils.get_index(ind, add, self.pjoint.dim)
                            indices.extend([start_i, end_i])

        return indices

    def _get_component_base_outline_indices(self, fixed_sides, n):
        """Extract indices for the component base outline."""
        indices = []
        d = self.pjoint.dim + 1
        start = d * d * d

        for side in fixed_sides:
            a1, b1, c1, d1 = Utils.get_corner_indices(side.ax, side.dir, self.pjoint.dim)
            step = 2 if len(self.pjoint.fixed.sides[n]) != 2 else 1
            off = 24 * side.ax + 12 * side.dir + 4 * step
            a0, b0, c0, d0 = start + off, start + off + 1, start + off + 2, start + off + 3
            indices.extend([a0, b0, b0, d0, d0, c0, c0, a0])
            indices.extend([a0, a1, b0, b1, c0, c1, d0, d1])

        return indices

    def _get_line_neighbor_values(self, ind, ax, n):
        values = []
        for i in range(-1,1):
            for j in range(-1,1):
                val = None
                add = [i,j]
                add.insert(ax,0)
                ind2 = np.array(ind)+np.array(add)
                if np.all(ind2>=0) and np.all(ind2<self.pjoint.dim):
                    val = self.voxel_matrix[tuple(ind2)]
                else:
                    for n2 in range(self.pjoint.noc):
                        for side in self.pjoint.fixed.sides[n2]:
                            ind3 = np.delete(ind2,side.ax)
                            if np.all(ind3>=0) and np.all(ind3<self.pjoint.dim):
                                if ind2[side.ax]<0 and side.dir==0: val = n2
                                elif ind2[side.ax]>=self.pjoint.dim and side.dir==1: val = n2
                values.append(val)
        values = np.array(values)
        count = np.count_nonzero(values==n)
        return count,values

    def _chess_line_indices(self, all_indices, chess_verts, n, offset):
        """Generate indices for chess pattern feedback lines."""
        indices = []
        for vert in chess_verts:
            add = [0, 0, 0]
            st = Utils.get_index(vert, add, self.pjoint.dim)
            add[self.pjoint.sax] = 1
            en = Utils.get_index(vert, add, self.pjoint.dim)
            indices.extend([st, en])

        return self._process_indices(indices, all_indices, GL.GL_LINES, n, offset)

    def _break_line_indices(self, all_indices, break_inds, n, offset):
        """Generate indices for breakable outline lines."""
        indices = []
        for ind3d in break_inds:
            add = [0, 0, 0]
            ind = Utils.get_index(ind3d, add, self.pjoint.dim)
            indices.append(ind)

        return self._process_indices(indices, all_indices, GL.GL_LINES, n, offset)

    def _open_line_indices(self, all_indices, n, offset):
        """Generate indices for open lines at component ends."""
        indices = []

        # Determine which directions to process based on component position
        dirs = [0, 1]
        if n == 0:
            dirs = [0]
        elif n == self.pjoint.noc - 1:
            dirs = [1]

        d = self.pjoint.dim + 1
        start = d * d * d

        for dir in dirs:
            a1, b1, c1, d1 = Utils.get_corner_indices(self.pjoint.sax, dir, self.pjoint.dim)
            off = 24 * self.pjoint.sax + 12 * (1 - dir)
            a0, b0, c0, d0 = start + off, start + off + 1, start + off + 2, start + off + 3
            indices.extend([a0, a1, b0, b1, c0, c1, d0, d1])

        return self._process_indices(indices, all_indices, GL.GL_LINES, n, offset)

    def _arrow_indices(self, all_indices, slide_dirs, n, offset):
        """Generate indices for direction arrows."""
        line_indices = self._get_arrow_line_indices(slide_dirs)
        face_indices = self._get_arrow_face_indices(slide_dirs)

        # Process line indices
        line_indices_prop, all_indices = self._process_indices(
            line_indices, all_indices, GL.GL_LINES, n, offset)

        # Process face indices
        face_indices_prop, all_indices = self._process_indices(
            face_indices, all_indices, GL.GL_TRIANGLES, n, offset)

        return line_indices_prop, face_indices_prop, all_indices

    def _get_arrow_line_indices(self, slide_dirs):
        """Generate line indices for arrows."""
        indices = []
        for ax, dir in slide_dirs:
            start = 1 + 10 * ax + 5 * dir
            indices.extend([0, start])
        return indices

    def _get_arrow_face_indices(self, slide_dirs):
        """Generate face indices for arrow heads."""
        indices = []
        for ax, dir in slide_dirs:
            start = 1 + 10 * ax + 5 * dir
            # Arrow head triangles
            indices.extend([
                start + 1, start + 2, start + 4,
                start + 1, start + 4, start + 3,
                start + 1, start + 2, start,
                start + 2, start + 4, start,
                start + 3, start + 4, start,
                start + 1, start + 3, start
            ])
        return indices

    # def _extract_joint_faces(self, mat, fixed_sides, n):
    #     """Extract indices for joint faces.
    #
    #     Args:
    #         mat: The voxel matrix to extract faces from
    #         fixed_sides: List of fixed sides for this component
    #         n: Component index
    #
    #     Returns:
    #         tuple: (indices, indices_ends) Lists of indices for regular faces and end faces
    #     """
    #     indices = []
    #     indices_ends = []
    #     d = self.pjoint.dim + 1
    #
    #     for i in range(d):
    #         for j in range(d):
    #             for k in range(d):
    #                 ind = [i, j, k]
    #                 for ax in range(3):
    #                     test_ind = np.array([i, j, k])
    #                     test_ind = np.delete(test_ind, ax)
    #                     if np.any(test_ind == self.pjoint.dim):
    #                         continue
    #
    #                     cnt, vals = Utils.face_neighbors(mat, ind, ax, n, fixed_sides)
    #                     if cnt == 1:
    #                         for x in range(2):
    #                             for y in range(2):
    #                                 add = [x, abs(y-x)]
    #                                 add.insert(ax, 0)
    #                                 index = Utils.get_index(ind, add, self.pjoint.dim)
    #
    #                                 if len(fixed_sides) > 0:
    #                                     if fixed_sides[0].ax == ax:
    #                                         indices_ends.append(index)
    #                                     else:
    #                                         indices.append(index)
    #                                 else:
    #                                     indices.append(index)
    #
    #     return indices, indices_ends

    def _extract_joint_faces(self, mat, fixed_sides, n):
        """Extract indices for joint faces."""
        indices = []
        indices_ends = []
        d = self.pjoint.dim + 1

        # Extract face indices
        for i in range(d):
            for j in range(d):
                for k in range(d):
                    self._process_voxel_faces(mat, [i, j, k], fixed_sides, n, indices, indices_ends)

        return indices, indices_ends

    def _process_voxel_faces(self, mat, ind, fixed_sides, n, indices, indices_ends):
        """Process faces for a single voxel position."""
        for ax in range(3):
            test_ind = np.array(ind)
            test_ind = np.delete(test_ind, ax)
            if np.any(test_ind == self.pjoint.dim):
                continue

            cnt, vals = Utils.face_neighbors(mat, ind, ax, n, fixed_sides)
            if cnt == 1:
                self._add_face_indices(ind, ax, fixed_sides, indices, indices_ends)

    def _add_face_indices(self, ind, ax, fixed_sides, indices, indices_ends):
        """Add face indices for a specific position and axis."""
        for x in range(2):
            for y in range(2):
                add = [x, abs(y-x)]
                add.insert(ax, 0)
                index = Utils.get_index(ind, add, self.pjoint.dim)

                if len(fixed_sides) > 0 and fixed_sides[0].ax == ax:
                    indices_ends.append(index)
                else:
                    indices.append(index)

    def _extract_component_base_faces(self, fixed_sides, n):
        """Extract indices for component base faces.

        Args:
            fixed_sides: List of fixed sides for this component
            n: Component index

        Returns:
            tuple: (indices, indices_ends) Lists of indices for regular faces and end faces
        """
        indices = []
        indices_ends = []
        d = self.pjoint.dim + 1
        start = d * d * d

        if len(fixed_sides) > 0:
            for side in fixed_sides:
                a1, b1, c1, d1 = Utils.get_corner_indices(side.ax, side.dir, self.pjoint.dim)
                step = 2
                if len(self.pjoint.fixed.sides[n]) == 2:
                    step = 1

                off = 24 * side.ax + 12 * side.dir + 4 * step
                a0, b0, c0, d0 = start + off, start + off + 1, start + off + 2, start + off + 3

                # Add component side to indices
                indices_ends.extend([a0, b0, d0, c0])  # bottom face
                indices.extend([a0, b0, b1, a1])       # side face 1
                indices.extend([b0, d0, d1, b1])       # side face 2
                indices.extend([d0, c0, c1, d1])       # side face 3
                indices.extend([c0, a0, a1, c1])       # side face 4

        return indices, indices_ends

    def generate_joint_face_indices(self, all_indices, mat, fixed_sides, n, offset, global_offset=0):
        """Generate indices for joint faces.

        Args:
            all_indices: Existing indices list to append to
            mat: The voxel matrix to extract faces from
            fixed_sides: List of fixed sides for this component
            n: Component index
            offset: Offset to apply to indices
            global_offset: Global offset for indices

        Returns:
            tuple: (indices_prop, indices_ends_prop, indices_all_prop, all_indices)
        """
        # Extract joint faces and component base faces
        joint_indices, joint_indices_ends = self._extract_joint_faces(mat, fixed_sides, n)
        base_indices, base_indices_ends = self._extract_component_base_faces(fixed_sides, n)

        # Combine indices
        indices = joint_indices + base_indices
        indices_ends = joint_indices_ends + base_indices_ends

        # Format indices
        indices = np.array(indices, dtype=np.uint32)
        indices = indices + offset
        indices_ends = np.array(indices_ends, dtype=np.uint32)
        indices_ends = indices_ends + offset

        # Store indices
        indices_prop = ElementProperties(GL.GL_QUADS, len(indices), len(all_indices) + global_offset, n)
        if len(all_indices) > 0:
            all_indices = np.concatenate([all_indices, indices])
        else:
            all_indices = indices

        indices_ends_prop = ElementProperties(GL.GL_QUADS, len(indices_ends), len(all_indices) + global_offset, n)
        all_indices = np.concatenate([all_indices, indices_ends])

        indices_all_prop = ElementProperties(GL.GL_QUADS, len(indices) + len(indices_ends), indices_prop.start_index, n)

        # Return properties and updated indices
        return indices_prop, indices_ends_prop, indices_all_prop, all_indices

    def _joint_area_face_indices(self, all_indices, mat, area_faces, n):
        """Generate indices for joint area faces (friction/contact)."""
        # Extract joint faces
        indices, indices_ends = self._extract_area_joint_faces(mat, area_faces, n)

        # Extract component base faces
        base_indices, base_indices_ends = self._extract_area_component_base_faces(n)

        # Combine indices
        indices.extend(base_indices)
        indices_ends.extend(base_indices_ends)

        # Process indices
        indices_prop, all_indices = self._process_indices(
            indices, all_indices, GL.GL_QUADS, n)
        indices_ends_prop, all_indices = self._process_indices(
            indices_ends, all_indices, GL.GL_QUADS, n)

        return indices_prop, indices_ends_prop, all_indices

    # def _extract_area_joint_faces(self, mat, area_faces, n):
    #     """Extract joint faces for area visualization."""
    #     indices = []
    #     indices_ends = []
    #     d = self.pjoint.dim + 1
    #
    #     for i in range(d):
    #         for j in range(d):
    #             for k in range(d):
    #                 ind = [i, j, k]
    #                 for ax in range(3):
    #                     offset = ax * self.pjoint.vn
    #                     test_ind = np.array([i, j, k])
    #                     test_ind = np.delete(test_ind, ax)
    #                     if np.any(test_ind == self.pjoint.dim):
    #                         continue
    #
    #                     cnt, vals = Utils.face_neighbors(mat, ind, ax, n, self.pjoint.fixed.sides[n])
    #                     if cnt == 1:
    #                         for x in range(2):
    #                             for y in range(2):
    #                                 add = [x, abs(y-x)]
    #                                 add.insert(ax, 0)
    #                                 index = Utils.get_index(ind, add, self.pjoint.dim)
    #                                 if [ax, ind] in area_faces:
    #                                     indices.append(index + offset)
    #                                 else:
    #                                     indices_ends.append(index + offset)
    #
    #     return indices, indices_ends

    def _extract_area_joint_faces(self, mat, area_faces, n):
        """Extract joint faces for area visualization."""
        indices = []
        indices_ends = []
        d = self.pjoint.dim + 1

        for i in range(d):
            for j in range(d):
                for k in range(d):
                    self._process_area_voxel_faces(mat, [i, j, k], area_faces, n, indices, indices_ends)

        return indices, indices_ends

    def _process_area_voxel_faces(self, mat, ind, area_faces, n, indices, indices_ends):
        """Process area faces for a single voxel position."""
        for ax in range(3):
            offset = ax * self.pjoint.vn
            test_ind = np.array(ind)
            test_ind = np.delete(test_ind, ax)
            if np.any(test_ind == self.pjoint.dim):
                continue

            cnt, vals = Utils.face_neighbors(mat, ind, ax, n, self.pjoint.fixed.sides[n])
            if cnt == 1:
                self._add_area_face_indices(ind, ax, area_faces, offset, indices, indices_ends)

    def _add_area_face_indices(self, ind, ax, area_faces, offset, indices, indices_ends):
        """Add area face indices for a specific position and axis."""
        for x in range(2):
            for y in range(2):
                add = [x, abs(y-x)]
                add.insert(ax, 0)
                index = Utils.get_index(ind, add, self.pjoint.dim)

                if [ax, ind] in area_faces:
                    indices.append(index + offset)
                else:
                    indices_ends.append(index + offset)

    def _extract_area_component_base_faces(self, n):
        """Extract component base faces for area visualization."""
        indices = []
        indices_ends = []
        d = self.pjoint.dim + 1
        start = d * d * d

        if len(self.pjoint.fixed.sides[n]) > 0:
            for side in self.pjoint.fixed.sides[n]:
                offset = side.ax * self.pjoint.vn
                a1, b1, c1, d1 = Utils.get_corner_indices(side.ax, side.dir, self.pjoint.dim)
                step = 2
                if len(self.pjoint.fixed.sides[n]) == 2:
                    step = 1

                off = 24 * side.ax + 12 * side.dir + 4 * step
                a0, b0, c0, d0 = start + off, start + off + 1, start + off + 2, start + off + 3

                # Add component side to indices
                indices_ends.extend([a0 + offset, b0 + offset, d0 + offset, c0 + offset])  # bottom face
                indices_ends.extend([a0 + offset, b0 + offset, b1 + offset, a1 + offset])  # side face 1
                indices_ends.extend([b0 + offset, d0 + offset, d1 + offset, b1 + offset])  # side face 2
                indices_ends.extend([d0 + offset, c0 + offset, c1 + offset, d1 + offset])  # side face 3
                indices_ends.extend([c0 + offset, a0 + offset, a1 + offset, c1 + offset])  # side face 4

        return indices, indices_ends

    def _joint_top_face_indices(self, all_indices, n, noc, offset):
        """Generate indices for joint top faces for picking."""
        # Determine face directions based on component position
        if n == 0:
            sdirs = [0]
        elif n == noc - 1:
            sdirs = [1]
        else:
            sdirs = [0, 1]

        # Extract joint faces
        indices, indices_tops = self._extract_top_joint_faces(n, sdirs, offset)

        # Extract component base faces
        base_indices = self._extract_top_component_base_faces(n)

        # Combine indices
        indices.extend(base_indices)

        # Process indices
        indices_prop, all_indices = self._process_indices(
            indices, all_indices, GL.GL_QUADS, n, offset)
        indices_tops_prop, all_indices = self._process_indices(
            indices_tops, all_indices, GL.GL_QUADS, n, offset)

        # Return properties and updated indices
        return indices_prop, indices_tops_prop, all_indices

    # def _extract_top_joint_faces(self, n, sdirs, offset):
    #     """Extract joint top faces for picking."""
    #     indices = []
    #     indices_tops = []
    #     sax = self.pjoint.sax
    #
    #     for ax in range(3):
    #         for i in range(self.pjoint.dim):
    #             for j in range(self.pjoint.dim):
    #                 top_face_indices_cnt = 0
    #                 for k in range(self.pjoint.dim + 1):
    #                     if sdirs[0] == 0:
    #                         k = self.pjoint.dim - k
    #
    #                     ind = [i, j]
    #                     ind.insert(ax, k)
    #
    #                     # Count number of neighbors (0, 1, or 2)
    #                     cnt, vals = Utils.face_neighbors(self.voxel_matrix, ind, ax, n, self.pjoint.fixed.sides[n])
    #                     on_free_base = False
    #
    #                     # Add base if edge component
    #                     if ax == sax and ax != self.pjoint.fixed.sides[n][0].ax and len(sdirs) == 1:
    #                         base = sdirs[0] * self.pjoint.dim
    #                         if ind[ax] == base:
    #                             on_free_base = True
    #
    #                     if cnt == 1 or on_free_base:
    #                         for x in range(2):
    #                             for y in range(2):
    #                                 add = [x, abs(y-x)]
    #                                 add.insert(ax, 0)
    #                                 index = Utils.get_index(ind, add, self.pjoint.dim)
    #
    #                                 # Determine if this is a top face for picking
    #                                 if ax == sax and top_face_indices_cnt < 4 * len(sdirs):
    #                                     indices_tops.append(index)
    #                                     top_face_indices_cnt += 1
    #                                 else:
    #                                     indices.append(index)
    #
    #                 # Add padding indices if needed
    #                 if top_face_indices_cnt < 4 * len(sdirs) and ax == sax:
    #                     neg_i = -offset - 1
    #                     for k in range(4 * len(sdirs) - top_face_indices_cnt):
    #                         indices_tops.append(neg_i)
    #
    #     return indices, indices_tops

    def _extract_top_joint_faces(self, n, sdirs, offset):
        """Extract joint top faces for picking."""
        indices = []
        indices_tops = []
        sax = self.pjoint.sax

        for ax in range(3):
            for i in range(self.pjoint.dim):
                for j in range(self.pjoint.dim):
                    self._process_top_faces_for_position(n, sdirs, ax, i, j, sax, indices, indices_tops, offset)

        return indices, indices_tops

    def _process_top_faces_for_position(self, n, sdirs, ax, i, j, sax, indices, indices_tops, offset):
        """Process top faces for a specific position."""
        top_face_indices_cnt = 0

        for k in range(self.pjoint.dim + 1):
            k_value = self.pjoint.dim - k if sdirs[0] == 0 else k

            ind = [i, j]
            ind.insert(ax, k_value)

            # Check if this is a visible face
            if self._is_visible_face(ind, ax, n, sax, sdirs):
                top_face_indices_cnt = self._add_top_face_indices(
                    ind, ax, sax, top_face_indices_cnt, len(sdirs), indices, indices_tops)

        # Add padding indices if needed
        if top_face_indices_cnt < 4 * len(sdirs) and ax == sax:
            self._add_padding_indices(top_face_indices_cnt, len(sdirs), indices_tops, -offset - 1)

    def _is_visible_face(self, ind, ax, n, sax, sdirs):
        """Check if a face is visible and should be rendered."""
        cnt, vals = Utils.face_neighbors(self.voxel_matrix, ind, ax, n, self.pjoint.fixed.sides[n])
        on_free_base = False

        # Check if on free base
        if ax == sax and ax != self.pjoint.fixed.sides[n][0].ax and len(sdirs) == 1:
            base = sdirs[0] * self.pjoint.dim
            if ind[ax] == base:
                on_free_base = True

        return cnt == 1 or on_free_base

    def _add_top_face_indices(self, ind, ax, sax, count, sdirs_len, indices, indices_tops):
        """Add indices for a top face."""
        for x in range(2):
            for y in range(2):
                add = [x, abs(y-x)]
                add.insert(ax, 0)
                index = Utils.get_index(ind, add, self.pjoint.dim)

                # Determine if this is a top face for picking
                if ax == sax and count < 4 * sdirs_len:
                    indices_tops.append(index)
                    count += 1
                else:
                    indices.append(index)

        return count

    def _add_padding_indices(self, count, sdirs_len, indices_tops, padding_value):
        """Add padding indices to ensure consistent array size."""
        for k in range(4 * sdirs_len - count):
            indices_tops.append(padding_value)

    def _extract_top_component_base_faces(self, n):
        """Extract component base faces for top face picking."""
        indices = []
        d = self.pjoint.dim + 1
        start = d * d * d

        for side in self.pjoint.fixed.sides[n]:
            a1, b1, c1, d1 = Utils.get_corner_indices(side.ax, side.dir, self.pjoint.dim)
            step = 2
            if len(self.pjoint.fixed.sides[n]) == 2:
                step = 1

            off = 24 * side.ax + 12 * side.dir + 4 * step
            a0, b0, c0, d0 = start + off, start + off + 1, start + off + 2, start + off + 3

            # Add component side faces to indices
            indices.extend([a0, b0, d0, c0])  # bottom face
            indices.extend([a0, b0, b1, a1])  # side face 1
            indices.extend([b0, d0, d1, b1])  # side face 2
            indices.extend([d0, c0, c1, d1])  # side face 3
            indices.extend([c0, a0, a1, c1])  # side face 4

        return indices

    # def _joint_selected_top_line_indices(self,select,all_indices):
    #     # Make indices of lines for drawing method GL_LINES
    #     n = select.n
    #     dir = select.dir
    #     offset = n*self.pjoint.vn
    #     sax = self.pjoint.sax
    #     h = self.height_fields[n-dir][tuple(select.faces[0])]
    #     # 1. Outline of selected top faces of joint
    #     indices = []
    #     for face in select.faces:
    #         ind = [int(face[0]),int(face[1])]
    #         ind.insert(sax,h)
    #         other_axes = [0,1,2]
    #         other_axes.pop(sax)
    #         for i in range(2):
    #             ax = other_axes[i]
    #             for j in range(2):
    #                 # Check neighboring faces
    #                 nface = face.copy()
    #                 nface[i] += 2*j-1
    #                 nface = np.array(nface, dtype=np.uint32)
    #                 if np.all(nface>=0) and np.all(nface<self.pjoint.dim):
    #                     unique = True
    #                     for face2 in select.faces:
    #                         if nface[0]==face2[0] and nface[1]==face2[1]:
    #                             unique = False
    #                             break
    #                     if not unique: continue
    #                 for k in range(2):
    #                     add = [k,k,k]
    #                     add[ax] = j
    #                     add[sax] = 0
    #                     index = Utils.get_index(ind, add, self.pjoint.dim)
    #                     indices.append(index)
    #     # Format
    #     indices = np.array(indices, dtype=np.uint32)
    #     indices = indices + offset
    #     # Store
    #     indices_prop = ElementProperties(GL.GL_LINES, len(indices), len(all_indices), n)
    #     all_indices = np.concatenate([all_indices, indices])
    #     # Return
    #     return indices_prop, all_indices

    def _joint_selected_top_line_indices(self, select, all_indices):
        """Make indices of lines for drawing selected top faces."""
        n = select.n
        dir = select.dir
        offset = n * self.pjoint.vn
        sax = self.pjoint.sax
        h = self.height_fields[n-dir][tuple(select.faces[0])]

        # Collect line indices for selected faces
        indices = self._collect_selected_face_outline_indices(select, sax, h)

        # Format and store indices
        indices_prop, all_indices = self._process_indices(
            indices, all_indices, GL.GL_LINES, n, offset)

        return indices_prop, all_indices

    def _collect_selected_face_outline_indices(self, select, sax, h):
        """Collect outline indices for selected faces."""
        indices = []

        for face in select.faces:
            ind = [int(face[0]), int(face[1])]
            ind.insert(sax, h)

            # Get axes perpendicular to sliding axis
            other_axes = [0, 1, 2]
            other_axes.pop(sax)

            # Check each perpendicular axis
            for i in range(2):
                ax = other_axes[i]
                self._add_face_edge_indices(face, ind, i, ax, sax, select.faces, indices)

        return indices

    def _add_face_edge_indices(self, face, ind, i, ax, sax, all_faces, indices):
        """Add edge indices for a face if the edge is on the boundary."""
        for j in range(2):
            # Check neighboring faces
            nface = face.copy()
            nface[i] += 2*j-1
            nface = np.array(nface, dtype=np.uint32)

            if not self._is_valid_neighboring_face(nface, all_faces):
                # Add edge indices
                for k in range(2):
                    add = [k, k, k]
                    add[ax] = j
                    add[sax] = 0
                    index = Utils.get_index(ind, add, self.pjoint.dim)
                    indices.append(index)

    def _is_valid_neighboring_face(self, nface, all_faces):
        """Check if a neighboring face is valid and in the selection."""
        # Check if face is within bounds
        if not (np.all(nface >= 0) and np.all(nface < self.pjoint.dim)):
            return False

        # Check if face is in the selection
        for face2 in all_faces:
            if nface[0] == face2[0] and nface[1] == face2[1]:
                return True

        return False

    # def _component_outline_indices(self,all_indices,fixed_sides,n,offset):
    #     d = self.pjoint.dim + 1
    #     indices = []
    #     start = d*d*d
    #     #Outline of component base
    #     #1) Base of first fixed side
    #     ax = fixed_sides[0].ax
    #     dir = fixed_sides[0].dir
    #     step = 2
    #     if len(fixed_sides)==2: step = 1
    #     off = 24*ax+12*dir+4*step
    #     a0,b0,c0,d0 = start+off,start+off+1,start+off+2,start+off+3
    #     #2) Base of first fixed side OR top of component
    #     if len(fixed_sides)==2:
    #         ax = fixed_sides[1].ax
    #         dir = fixed_sides[1].dir
    #         off = 24*ax+12*dir+4*step
    #         a1,b1,c1,d1 = start+off,start+off+1,start+off+2,start+off+3
    #     else:
    #         a1,b1,c1,d1 = Utils.get_corner_indices(ax, 1 - dir, self.pjoint.dim)
    #     # append list of indices
    #     indices.extend([a0,b0, b0,d0, d0,c0, c0,a0])
    #     indices.extend([a0,a1, b0,b1, c0,c1, d0,d1])
    #     indices.extend([a1,b1, b1,d1, d1,c1, c1,a1])
    #     # Format
    #     indices = np.array(indices, dtype=np.uint32)
    #     indices = indices + offset
    #     # Store
    #     indices_prop = ElementProperties(GL.GL_LINES, len(indices), len(all_indices), n)
    #     all_indices = np.concatenate([all_indices, indices])
    #     # Return
    #     return indices_prop, all_indices

    def _component_outline_indices(self, all_indices, fixed_sides, n, offset):
        """Generate indices for component outline."""
        d = self.pjoint.dim + 1
        indices = []
        start = d * d * d

        # Get base corners
        base_corners = self._get_component_base_corners(fixed_sides, start)

        # Get top corners (either from second fixed side or calculated)
        top_corners = self._get_component_top_corners(fixed_sides, start)

        # Add outline edges
        indices = self._add_component_outline_edges(base_corners, top_corners)

        # Format and store indices
        indices_prop, all_indices = self._process_indices(
            indices, all_indices, GL.GL_LINES, n, offset)

        return indices_prop, all_indices

    def _get_component_base_corners(self, fixed_sides, start):
        """Get the corner indices for the component base."""
        ax = fixed_sides[0].ax
        dir = fixed_sides[0].dir
        step = 2 if len(fixed_sides) == 1 else 1

        off = 24 * ax + 12 * dir + 4 * step
        return [start + off, start + off + 1, start + off + 2, start + off + 3]

    def _get_component_top_corners(self, fixed_sides, start):
        """Get the corner indices for the component top."""
        if len(fixed_sides) == 2:
            # Use second fixed side
            ax = fixed_sides[1].ax
            dir = fixed_sides[1].dir
            step = 1

            off = 24 * ax + 12 * dir + 4 * step
            return [start + off, start + off + 1, start + off + 2, start + off + 3]
        else:
            # Calculate from first fixed side
            ax = fixed_sides[0].ax
            dir = fixed_sides[0].dir
            return Utils.get_corner_indices(ax, 1 - dir, self.pjoint.dim)

    def _add_component_outline_edges(self, base, top):
        """Add edges for component outline."""
        indices = []

        # Base edges (bottom face)
        indices.extend([base[0], base[1], base[1], base[3], base[3], base[2], base[2], base[0]])

        # Vertical edges connecting base to top
        indices.extend([base[0], top[0], base[1], top[1], base[2], top[2], base[3], top[3]])

        # Top edges (top face)
        indices.extend([top[0], top[1], top[1], top[3], top[3], top[2], top[2], top[0]])

        return indices


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
        """Create all indices for rendering the joint geometry."""
        # Shared lists
        all_inds = []
        self.indices_fall = []
        self.indices_lns = []

        if not self.mainmesh:
            # For suggestions and gallery - just show basic geometry - no feedback
            all_inds = self._create_suggestion_indices(all_inds, glo_off)
        else:
            # Current geometry (main including feedback)
            all_inds = self._create_main_mesh_indices(all_inds, milling_path)

        self.indices = all_inds

    def _create_suggestion_indices(self, all_inds, glo_off=0):
        """Create indices for suggestion geometries (simplified version)."""
        for n in range(self.pjoint.noc):
            ax = self.pjoint.fixed.sides[n][0].ax
            offset = ax * self.pjoint.vn

            # Generate face indices
            nend, end, all_faces, all_inds = self.generate_joint_face_indices(
                all_inds, self.voxel_matrix, self.pjoint.fixed.sides[n],
                n, offset, global_offset=glo_off
            )

            # Generate line indices
            lns, all_inds = self._joint_line_indices(
                all_inds, n, offset, global_offset=glo_off
            )

            # Store indices
            self.indices_fall.append(all_faces)
            self.indices_lns.append(lns)

        return all_inds

    def _create_main_mesh_indices(self, all_inds, milling_path=False):
        """Create indices for the main mesh with all feedback visualizations."""
        # Initialize all index lists
        self._initialize_main_mesh_index_lists()

        # Create component-specific indices
        for n in range(self.pjoint.noc):
            all_inds = self._create_component_indices(n, all_inds, milling_path)

        # Create selection-related indices
        all_inds = self._create_selection_indices(all_inds)

        return all_inds

    def _initialize_main_mesh_index_lists(self):
        """Initialize all the index lists used for the main mesh."""
        self.indices_fend = []
        self.indices_not_fend = []
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

    def _create_component_indices(self, n, all_inds, milling_path=False):
        """Create indices for a specific component."""
        ax = self.pjoint.fixed.sides[n][0].ax
        offset = ax * self.pjoint.vn

        # Generate face indices
        nend, end, con, all_inds = self.generate_joint_face_indices(
            all_inds, self.eval.voxel_matrix_connected,
            self.pjoint.fixed.sides[n], n, offset
        )

        # Handle connected/disconnected components
        if not self.eval.connected[n]:
            fne, fe, uncon, all_inds = self.generate_joint_face_indices(
                all_inds, self.eval.voxel_matrix_unconnected, [], n, offset
            )
            self.indices_not_fcon.append(uncon)
            all_faces = ElementProperties(GL.GL_QUADS, con.count + uncon.count, con.start_index, n)
        else:
            self.indices_not_fcon.append(None)
            all_faces = con

        # Generate breakable and non-breakable face indices
        fne, fe, brk_faces, all_inds = self.generate_joint_face_indices(
            all_inds, self.eval.breakable_voxmat, [], n, n * self.pjoint.vn
        )
        fne, fe, not_brk_faces, all_inds = self.generate_joint_face_indices(
            all_inds, self.eval.non_breakable_voxmat, self.pjoint.fixed.sides[n],
            n, n * self.pjoint.vn
        )

        # Handle unbridged components
        if not self.eval.bridged[n]:
            unbris = []
            for m in range(2):
                fne, fe, unbri, all_inds = self.generate_joint_face_indices(
                    all_inds, self.eval.voxel_matrices_unbridged[n][m],
                    [self.pjoint.fixed.sides[n][m]], n, n * self.pjoint.vn
                )
                unbris.append(unbri)
        else:
            unbris = None

        # Generate friction and contact face indices
        fric, nfric, all_inds = self._joint_area_face_indices(
            all_inds, self.voxel_matrix, self.eval.friction_faces[n], n
        )
        cont, ncont, all_inds = self._joint_area_face_indices(
            all_inds, self.voxel_matrix, self.eval.contact_faces[n], n
        )

        # Generate picking face indices
        faces_pick_not_tops, faces_pick_tops, all_inds = self._joint_top_face_indices(
            all_inds, n, self.pjoint.noc, offset
        )

        # Generate line indices
        lns, all_inds = self._joint_line_indices(all_inds, n, offset)

        # Generate chessboard feedback line indices
        if self.eval.checker[n]:
            chess, all_inds = self._chess_line_indices(
                all_inds, self.eval.checker_vertices[n], n, offset
            )
        else:
            chess = []

        # Generate breakable line indices
        if self.eval.breakable:
            break_lns, all_inds = self._break_line_indices(
                all_inds, self.eval.breakable_outline_inds[n], n, offset
            )

        # Generate opening line indices
        open_lines, all_inds = self._open_line_indices(all_inds, n, offset)
        self.indices_open_lines.append(open_lines)

        # Generate arrow indices
        larr, farr, all_inds = self._arrow_indices(
            all_inds, self.eval.slides[n], n, 3 * self.pjoint.vn
        )
        arrows = [larr, farr]

        # Generate milling path indices if needed
        if milling_path and len(self.pjoint.mverts[0]) > 0:
            mill, all_inds = self._milling_path_indices(
                all_inds, int(len(self.pjoint.mverts[n]) / 8),
                self.pjoint.m_start[n], n
            )

        # Store all indices
        self.indices_fend.append(end)
        self.indices_not_fend.append(nend)
        self.indices_fcon.append(con)
        self.indices_fall.append(all_faces)
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

        if milling_path and len(self.pjoint.mverts[0]) > 0:
            self.indices_milling_path.append(mill)

        self.indices_ffric.append(fric)
        self.indices_not_ffric.append(nfric)
        self.indices_fcont.append(cont)
        self.indices_not_fcont.append(ncont)

        return all_inds

    def _create_selection_indices(self, all_inds):
        """Create indices for selection visualization."""
        # Generate outline of selected faces
        if self.select.state == 2:
            self.outline_selected_faces, all_inds = self._joint_selected_top_line_indices(
                self.select, all_inds
            )

        # Generate outline of selected component
        if self.select.n is not None and self.select.new_fixed_sides_for_display is not None:
            self.outline_selected_component, all_inds = self._component_outline_indices(
                all_inds, self.select.new_fixed_sides_for_display,
                self.select.n, self.select.n * self.pjoint.vn
            )

        return all_inds

    def randomize_height_fields(self):
        self.height_fields = Utils.get_random_height_fields(self.pjoint.dim, self.pjoint.noc)
        self.update_voxel_matrix_from_height_fields()
        self.pjoint.combine_and_buffer_indices()

    def clear_height_fields(self):
        self.height_fields = []
        for n in range(self.pjoint.noc - 1):
            hf = np.zeros((self.pjoint.dim, self.pjoint.dim))
            self.height_fields.append(hf)
        self.update_voxel_matrix_from_height_fields()
        self.pjoint.combine_and_buffer_indices()

    def load_search_results(self,index=-1):
        # Folder
        location = os.path.abspath(os.getcwd())
        location = location.split(os.sep)
        location.pop()
        location = os.sep.join(location)
        location += os.sep +"search_results" + os.sep +"noc_" + str(self.pjoint.noc) + os.sep + "dim_" + str(self.pjoint.dim) + os.sep + "fs_"
        for i in range(len(self.pjoint.fixed.sides)):
            for fs in self.pjoint.fixed.sides[i]:
                location+=str(fs[0])+str(fs[1])
            if i!=len(self.pjoint.fixed.sides)-1: location+=("_")
        location+=os.sep+"allvalid"
        print("Trying to load geometry from",location)
        maxi = len(os.listdir(location))-1
        if index==-1: index=random.randint(0,maxi)
        self.height_fields = np.load(location+os.sep+"height_fields_"+str(index)+".npy")
        self.fab_directions = []
        for i in range(self.pjoint.noc):
            if i==0: self.fab_directions.append(0)
            else: self.fab_directions.append(1)
        self.update_voxel_matrix_from_height_fields()
        self.pjoint.combine_and_buffer_indices()

    # def edit_height_fields(self,faces,h,n,dir):
    #     for ind in faces:
    #         self.height_fields[n-dir][tuple(ind)] = h
    #         if dir==0: # If editing top
    #             # If new height is higher than following hf, update to same height
    #             for i in range(n-dir+1, self.pjoint.noc - 1):
    #                 h2 = self.height_fields[i][tuple(ind)]
    #                 if h>h2: self.height_fields[i][tuple(ind)]=h
    #         if dir==1: # If editing bottom
    #             # If new height is lower than previous hf, update to same height
    #             for i in range(0,n-dir):
    #                 h2 = self.height_fields[i][tuple(ind)]
    #                 if h<h2: self.height_fields[i][tuple(ind)]=h
    #     self.update_voxel_matrix_from_height_fields()
    #     self.pjoint.combine_and_buffer_indices()

    def edit_height_fields(self, faces, h, n, dir):
        """Edit height fields for the specified faces.

        Args:
            faces: List of face indices to edit
            h: New height value
            n: Component index
            dir: Direction (0 for top, 1 for bottom)
        """
        # Update the specified height fields
        for ind in faces:
            self.height_fields[n-dir][tuple(ind)] = h

        # Propagate changes to maintain consistency
        if dir == 0:  # If editing top
            self._propagate_height_up(faces, h, n-dir)
        else:  # If editing bottom
            self._propagate_height_down(faces, h, n-dir)

        # Update the voxel matrix and buffer indices
        self.update_voxel_matrix_from_height_fields()
        self.pjoint.combine_and_buffer_indices()

    def _propagate_height_up(self, faces, h, start_index):
        """Propagate height changes upward to maintain consistency."""
        # If new height is higher than following hf, update to same height
        for ind in faces:
            for i in range(start_index + 1, self.pjoint.noc - 1):
                h2 = self.height_fields[i][tuple(ind)]
                if h > h2:
                    self.height_fields[i][tuple(ind)] = h

    def _propagate_height_down(self, faces, h, start_index):
        """Propagate height changes downward to maintain consistency."""
        # If new height is lower than previous hf, update to same height
        for ind in faces:
            for i in range(0, start_index):
                h2 = self.height_fields[i][tuple(ind)]
                if h < h2:
                    self.height_fields[i][tuple(ind)] = h

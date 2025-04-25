import copy
import math

import numpy as np
import pyrr

from fixed_side import FixedSide

import utils_ as Utils

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

    def update_pick(self, x, y, n, dir):
        self.n = n
        self.x = x
        self.y = y
        self.dir = dir

        if self.x is not None and self.y is not None:
            if self.shift:
                self.faces = Utils.get_same_height_neighbors(
                    self.pgeom.height_fields[n - dir],
                    [np.array([self.x, self.y])]
                )
            else:
                self.faces = [np.array([self.x, self.y])]

    def start_pull(self, mouse_pos):
        self.state = 2
        self.start_pos = np.array([mouse_pos[0], -mouse_pos[1]])
        self.start_height = self.pgeom.height_fields[self.n - self.dir][self.x][self.y]
        self.pgeom.pjoint.combine_and_buffer_indices()  # for selection area

    def end_pull(self):
        if self.val != 0:
            self.pgeom.edit_height_fields(self.faces, self.current_height, self.n, self.dir)
        self.state = -1
        self.refresh = True

    def edit(self, mouse_pos, screen_xrot, screen_yrot, w=1600, h=1600):
        self.current_pos = np.array([mouse_pos[0], -mouse_pos[1]])
        self.current_height = self.start_height

        # Calculate mouse vector and sliding direction
        mouse_vec = self._calculate_edit_mouse_vector(w, h)
        sdir_vec = self._calculate_sliding_direction_vector(screen_xrot, screen_yrot)

        # Calculate value change based on vectors
        val = self._calculate_height_change(mouse_vec, sdir_vec)

        # Apply constraints to the value
        val = self._constrain_height_value(val)

        self.current_height = self.start_height + val
        self.val = int(val)

    def _calculate_edit_mouse_vector(self, w, h):
        mouse_vec = np.array(self.current_pos - self.start_pos)
        mouse_vec = mouse_vec.astype(float)
        mouse_vec[0] = 2 * mouse_vec[0] / w
        mouse_vec[1] = 2 * mouse_vec[1] / h
        return mouse_vec

    def _calculate_sliding_direction_vector(self, screen_xrot, screen_yrot):
        sdir_vec = np.copy(self.pgeom.pjoint.pos_vecs[self.pgeom.pjoint.sax])
        rot_x = pyrr.Matrix33.from_x_rotation(screen_xrot)
        rot_y = pyrr.Matrix33.from_y_rotation(screen_yrot)
        sdir_vec = np.dot(sdir_vec, rot_x * rot_y)
        sdir_vec = np.delete(sdir_vec, 2)  # delete Z-value
        return sdir_vec

    def _calculate_height_change(self, mouse_vec, sdir_vec):
        cosang = np.dot(mouse_vec, sdir_vec)  # Negative/positive depending on direction
        val = int(np.linalg.norm(mouse_vec) / np.linalg.norm(sdir_vec) + 0.5)
        if cosang is not None and cosang < 0:
            val = -val
        return val

    def _constrain_height_value(self, val):
        if self.start_height + val > self.pgeom.pjoint.dim:
            val = self.pgeom.pjoint.dim - self.start_height
        elif self.start_height + val < 0:
            val = -self.start_height
        return val

    def start_move(self, mouse_pos, h=1600):
        self.state = 12
        self.start_pos = np.array([mouse_pos[0], h - mouse_pos[1]])
        self.new_fixed_sides = self.pgeom.pjoint.fixed.sides[self.n]
        self.new_fixed_sides_for_display = self.pgeom.pjoint.fixed.sides[self.n]
        self.pgeom.pjoint.combine_and_buffer_indices()  # for move preview outline

    def end_move(self):
        self.pgeom.pjoint.update_component_position(self.new_fixed_sides, self.n)
        self.state = -1
        self.new_fixed_sides_for_display = None

    def move(self, mouse_pos, screen_xrot, screen_yrot, w=1600, h=1600):
        """Handle component movement or rotation based on mouse position."""
        # Initialize component data
        sax = self.pgeom.pjoint.sax
        noc = self.pgeom.pjoint.noc
        self.new_fixed_sides = copy.deepcopy(self.pgeom.pjoint.fixed.sides[self.n])
        self.new_fixed_sides_for_display = copy.deepcopy(self.pgeom.pjoint.fixed.sides[self.n])

        # Calculate mouse movement
        self.current_pos = np.array([mouse_pos[0], h - mouse_pos[1]])
        mouse_vec = self._calculate_mouse_vector(w, h)

        # Only process if movement is significant
        move_dist = np.linalg.norm(mouse_vec)
        if move_dist > 0.01:
            # Get component direction data
            comp_ax, comp_dir, comp_vec = self._get_component_vector(screen_xrot, screen_yrot)

            # Calculate angle between mouse vector and component vector
            ang = Utils.angle_between_vectors1(mouse_vec, comp_vec, direction=True)
            absang = abs(ang) % 180

            # Determine if we're rotating or moving
            if self._is_rotation_mode(absang):
                self._handle_rotation(ang, comp_ax, screen_xrot, screen_yrot, sax)
            else:
                self._handle_movement(absang, comp_ax, comp_dir, comp_vec, mouse_vec)

            # Check if the new position is blocked
            if not self._is_position_blocked():
                self.new_fixed_sides = self.new_fixed_sides_for_display

        # Update display if needed
        self._update_display_if_changed()

    def _calculate_mouse_vector(self, w, h):
        """Calculate normalized mouse movement vector."""
        mouse_vec = np.array(self.current_pos - self.start_pos)
        mouse_vec = mouse_vec.astype(float)
        mouse_vec[0] = 2 * mouse_vec[0] / w
        mouse_vec[1] = 2 * mouse_vec[1] / h
        return mouse_vec

    def _get_component_vector(self, screen_xrot, screen_yrot):
        """Get component axis, direction and screen-space vector."""
        comp_ax = self.pgeom.pjoint.fixed.sides[self.n][0].ax  # component axis
        comp_dir = self.pgeom.pjoint.fixed.sides[self.n][0].dir
        comp_len = 2.5 * (2 * comp_dir - 1) * self.pgeom.pjoint.component_size
        comp_vec = comp_len * Utils.unitize(self.pgeom.pjoint.pos_vecs[comp_ax])

        # Flatten vector to screen
        rot_x = pyrr.Matrix33.from_x_rotation(screen_xrot)
        rot_y = pyrr.Matrix33.from_y_rotation(screen_yrot)
        comp_vec = np.dot(comp_vec, rot_x * rot_y)
        comp_vec = np.delete(comp_vec, 2)  # delete Z-value

        return comp_ax, comp_dir, comp_vec

    def _is_rotation_mode(self, absang):
        """Determine if we're in rotation mode based on angle."""
        return 45 < absang < 135

    def _handle_rotation(self, ang, comp_ax, screen_xrot, screen_yrot, sax):
        """Handle timber rotation logic."""
        # Find rotation axis (the axis most aligned with screen)
        oax = self._find_rotation_axis(comp_ax, screen_xrot, screen_yrot)

        # Determine rotation direction
        clockwise = ang >= 0

        # Get screen direction for correct rotation visualization
        screen_dir = self._get_screen_direction(comp_ax, oax, screen_xrot, screen_yrot)

        # Calculate new fixed sides based on rotation
        self.new_fixed_sides_for_display = self._calculate_rotated_sides(
            comp_ax, oax, clockwise, screen_dir, sax
        )

    def _find_rotation_axis(self, comp_ax, screen_xrot, screen_yrot):
        """Find the axis to rotate around (the one most aligned with screen)."""
        other_axes = [0, 1, 2]
        other_axes.pop(comp_ax)

        rot_x = pyrr.Matrix33.from_x_rotation(screen_xrot)
        rot_y = pyrr.Matrix33.from_y_rotation(screen_yrot)

        # The axis that is flatter to the screen will be processed
        maxlen = 0
        oax = None

        for i in range(len(other_axes)):
            other_vec = [0, 0, 0]
            other_vec[other_axes[i]] = 1

            # Flatten vector to screen
            other_vec = np.dot(other_vec, rot_x * rot_y)
            other_vec = np.delete(other_vec, 2)  # delete Z-value

            # Check length
            other_length = np.linalg.norm(other_vec)
            if other_length > maxlen:
                maxlen = other_length
                oax = other_axes[i]

        return oax

    def _get_screen_direction(self, comp_ax, oax, screen_xrot, screen_yrot):
        """Get screen direction for correct rotation visualization."""
        lax = [0, 1, 2]
        lax.remove(comp_ax)
        lax.remove(oax)
        lax = lax[0]

        rot_x = pyrr.Matrix33.from_x_rotation(screen_xrot)
        rot_y = pyrr.Matrix33.from_y_rotation(screen_yrot)

        screen_dir = 1
        screen_vec = self.pgeom.pjoint.pos_vecs[lax]
        screen_vec = np.dot(screen_vec, rot_x * rot_y)

        if screen_vec[2] < 0:
            screen_dir = -1

        return screen_dir

    def _calculate_rotated_sides(self, comp_ax, oax, clockwise, screen_dir, sax):
        """Calculate new fixed sides after rotation."""
        new_sides = []
        blocked = False
        noc = self.pgeom.pjoint.noc

        for i in range(len(self.pgeom.pjoint.fixed.sides[self.n])):
            ndir = self.pgeom.pjoint.fixed.sides[self.n][i].dir

            # Determine if axes are in ordered sequence
            ordered = False
            if comp_ax < oax and oax - comp_ax == 1:
                ordered = True
            elif oax < comp_ax and comp_ax - oax == 2:
                ordered = True

            # Adjust direction based on rotation
            if (clockwise and not ordered) or (not clockwise and ordered):
                ndir = 1 - ndir

            # Adjust for screen direction
            if screen_dir > 0:
                ndir = 1 - ndir

            side = FixedSide(oax, ndir)
            new_sides.append(side)

            # Check if rotation is blocked by sliding axis constraints
            if side.ax == sax and side.dir == 0 and self.n != 0:
                blocked = True
                break
            if side.ax == sax and side.dir == 1 and self.n != noc - 1:
                blocked = True
                break

        return new_sides if not blocked else self.pgeom.pjoint.fixed.sides[self.n]

    def _handle_movement(self, absang, comp_ax, comp_dir, comp_vec, mouse_vec):
        """Handle timber movement logic."""
        length_ratio = np.linalg.norm(mouse_vec) / np.linalg.norm(comp_vec)
        side_num = len(self.pgeom.pjoint.fixed.sides[self.n])

        if side_num == 1 and absang > 135:  # Currently L
            self._handle_L_movement(comp_ax, comp_dir, length_ratio)
        elif side_num == 2:  # Currently T
            self._handle_T_movement(comp_ax, absang)

    def _handle_L_movement(self, comp_ax, comp_dir, length_ratio):
        """Handle movement for L-shaped configuration."""
        if length_ratio < 0.5:  # Moved just a bit, L to T
            self.new_fixed_sides_for_display = [
                FixedSide(comp_ax, 0),
                FixedSide(comp_ax, 1)
            ]
        elif length_ratio < 2.0:  # Moved a lot, L to other L
            self.new_fixed_sides_for_display = [
                FixedSide(comp_ax, 1 - comp_dir)
            ]

    def _handle_T_movement(self, comp_ax, absang):
        """Handle movement for T-shaped configuration."""
        if absang > 135:
            self.new_fixed_sides_for_display = [FixedSide(comp_ax, 1)]  # Positive direction
        else:
            self.new_fixed_sides_for_display = [FixedSide(comp_ax, 0)]  # Negative direction

    def _is_position_blocked(self):
        """Check if the new position is blocked."""
        blocked = False

        for side in self.new_fixed_sides_for_display:
            if side.unique(self.pgeom.pjoint.fixed.sides[self.n]):
                if side.unique(self.pgeom.pjoint.fixed.unblocked):
                    blocked = True

        # If all sides are the same, it's not blocked
        if blocked:
            all_same = True
            for side in self.new_fixed_sides_for_display:
                if side.unique(self.pgeom.pjoint.fixed.sides[self.n]):
                    all_same = False
            if all_same:
                blocked = False

        return blocked

    def _update_display_if_changed(self):
        """Update display if fixed sides have changed."""
        if not np.equal(
            self.pgeom.pjoint.fixed.sides[self.n],
            np.array(self.new_fixed_sides_for_display)
        ).all():
            # Update buffer for move/rotate preview outline
            self.pgeom.pjoint.combine_and_buffer_indices()

import numpy as np
import copy

import utils_ as Utils

class Evaluation:
    def __init__(self, voxel_matrix, joint_, mainmesh=True):
        self.mainmesh = mainmesh
        self.valid = True
        self.slides = []
        self.number_of_slides = []
        self.interlock = False
        self.interlocks = []
        self.connected = []
        self.bridged = []
        self.breakable = []
        self.checker = []
        self.checker_vertices = []
        self.fab_direction_ok = []
        self.voxel_matrix_connected = None
        self.voxel_matrix_unconnected = None
        self.voxel_matrices_unbridged = []
        self.breakable_outline_inds = []
        self.breakable_voxel_inds = []
        self.sliding_depths = []
        self.friction_nums = []
        self.friction_faces = []
        self.contact_nums = []
        self.contact_faces = []
        self.fab_directions = self.update(voxel_matrix, joint_)

    # def update(self,voxel_matrix,joint_):
    #     self.voxel_matrix_with_sides = Utils.add_fixed_sides(voxel_matrix, joint_.fixed.sides)
    #
    #     # Voxel connection and bridgeing
    #     self.connected = []
    #     self.bridged = []
    #     self.voxel_matrices_unbridged = []
    #     for n in range(joint_.noc):
    #         self.connected.append(Utils.is_connected(self.voxel_matrix_with_sides,n))
    #         self.bridged.append(True)
    #         self.voxel_matrices_unbridged.append(None)
    #     self.voxel_matrix_connected = voxel_matrix.copy()
    #     self.voxel_matrix_unconnected = None
    #
    #     self.seperate_unconnected(voxel_matrix,joint_.fixed.sides,joint_.dim)
    #
    #     # Bridging
    #     voxel_matrix_connected_with_sides = Utils.add_fixed_sides(self.voxel_matrix_connected, joint_.fixed.sides)
    #     for n in range(joint_.noc):
    #         self.bridged[n] = Utils.is_connected(voxel_matrix_connected_with_sides,n)
    #         if not self.bridged[n]:
    #             voxel_matrix_unbridged_1, voxel_matrix_unbridged_2 = self.seperate_unbridged(voxel_matrix,joint_.fixed.sides,joint_.dim,n)
    #             self.voxel_matrices_unbridged[n] = [voxel_matrix_unbridged_1, voxel_matrix_unbridged_2]
    #
    #     # Fabricatability by direction constraint
    #     self.fab_direction_ok = []
    #     fab_directions = list(range(joint_.noc))
    #     for n in range(joint_.noc):
    #         if n==0 or n==joint_.noc-1:
    #             self.fab_direction_ok.append(True)
    #             if n==0: fab_directions[n]=0
    #             else: fab_directions[n]=1
    #         else:
    #             fab_ok,fab_dir = Utils.is_fab_direction_ok(voxel_matrix,joint_.sax,n)
    #             fab_directions[n] = fab_dir
    #             self.fab_direction_ok.append(fab_ok)
    #
    #     # Chessboard
    #     self.checker = []
    #     self.checker_vertices = []
    #     for n in range(joint_.noc):
    #         check,verts = Utils.get_chessboard_vertics(voxel_matrix,joint_.sax,joint_.noc,n)
    #         self.checker.append(check)
    #         self.checker_vertices.append(verts)
    #
    #     # Sliding directions
    #     self.slides,self.number_of_slides = Utils.get_sliding_directions(self.voxel_matrix_with_sides,joint_.noc)
    #     self.interlock = True
    #     for n in range(joint_.noc):
    #         if (n==0 or n==joint_.noc-1):
    #             if self.number_of_slides[n]<=1:
    #                 self.interlocks.append(True)
    #             else:
    #                 self.interlocks.append(False)
    #                 self.interlock=False
    #         else:
    #             if self.number_of_slides[n]==0:
    #                 self.interlocks.append(True)
    #             else:
    #                 self.interlocks.append(False)
    #                 self.interlock=False
    #
    #     # Friction
    #     self.friction_nums = []
    #     self.friction_faces = []
    #     self.contact_nums = []
    #     self.contact_faces = []
    #     for n in range(joint_.noc):
    #         friction,ffaces,contact,cfaces, = Utils.get_friction_and_contact_areas(voxel_matrix,self.slides[n],joint_.fixed.sides,n)
    #         self.friction_nums.append(friction)
    #         self.friction_faces.append(ffaces)
    #         self.contact_nums.append(contact)
    #         self.contact_faces.append(cfaces)
    #
    #
    #     # Grain direction
    #     for n in range(joint_.noc):
    #         brk,brk_oinds,brk_vinds = Utils.get_breakable_voxels(voxel_matrix,joint_.fixed.sides[n],joint_.sax,n)
    #         self.breakable.append(brk)
    #         self.breakable_outline_inds.append(brk_oinds)
    #         self.breakable_voxel_inds.append(brk_vinds)
    #     self.non_breakable_voxmat, self.breakable_voxmat = self.seperate_voxel_matrix(voxel_matrix,self.breakable_voxel_inds)
    #
    #     if not self.interlock or not all(self.connected) or not all(self.bridged):
    #         self.valid=False
    #     elif any(self.breakable) or any(self.checker) or not all(self.fab_direction_ok):
    #         self.valid=False
    #
    #     """
    #     # Sliding depth
    #     sliding_depths = [3,3,3]
    #     open_mat = np.copy(self.voxel_matrix_with_sides)
    #     for depth in range(4):
    #         slds,nos = get_sliding_directions(open_mat,noc)
    #         for n in range(noc):
    #             if sliding_depths[n]!=3: continue
    #             if n==0 or n==noc-1:
    #                 if nos[n]>1: sliding_depths[n]=depth
    #             else:
    #                 if nos[n]>0: sliding_depths[n]=depth
    #         open_mat = open_matrix(open_mat,sax,noc)
    #     self.slide_depths = sliding_depths
    #     self.slide_depth_product = np.prod(np.array(sliding_depths))
    #     print(self.slide_depths,self.slide_depth_product)
    #     """
    #     return fab_directions

    def update(self, voxel_matrix, joint_):
        self.voxel_matrix_with_sides = Utils.add_fixed_sides(voxel_matrix, joint_.fixed.sides)

        # Process connectivity and bridging
        self._process_connectivity_and_bridging(voxel_matrix, joint_)

        # Process fabrication directions
        fab_directions = self._process_fabrication_directions(voxel_matrix, joint_)

        # Process chessboard pattern
        self._process_chessboard(voxel_matrix, joint_)

        # Process sliding directions and interlocking
        self._process_sliding_and_interlocking(joint_)

        # Process friction and contact areas
        self._process_friction_and_contact(voxel_matrix, joint_)

        # Process grain direction and breakability
        self._process_grain_direction(voxel_matrix, joint_)

        # Validate the joint
        self._validate_joint()

        return fab_directions

    def _process_connectivity_and_bridging(self, voxel_matrix, joint_):
        # Initialize connectivity and bridging arrays
        self.connected = []
        self.bridged = []
        self.voxel_matrices_unbridged = []

        # Check connectivity for each component
        for n in range(joint_.noc):
            self.connected.append(Utils.is_connected(self.voxel_matrix_with_sides, n))
            self.bridged.append(True)
            self.voxel_matrices_unbridged.append(None)

        # Initialize connected and unconnected matrices
        self.voxel_matrix_connected = voxel_matrix.copy()
        self.voxel_matrix_unconnected = None

        # Separate unconnected voxels
        self.separate_unconnected(voxel_matrix, joint_.fixed.sides, joint_.dim)

        # Check bridging for each component
        voxel_matrix_connected_with_sides = Utils.add_fixed_sides(self.voxel_matrix_connected, joint_.fixed.sides)
        for n in range(joint_.noc):
            self.bridged[n] = Utils.is_connected(voxel_matrix_connected_with_sides, n)
            if not self.bridged[n]:
                voxel_matrix_unbridged_1, voxel_matrix_unbridged_2 = self.separate_unbridged(
                    voxel_matrix, joint_.fixed.sides, joint_.dim, n
                )
                self.voxel_matrices_unbridged[n] = [voxel_matrix_unbridged_1, voxel_matrix_unbridged_2]

    def _process_fabrication_directions(self, voxel_matrix, joint_):
        # Initialize fabrication direction arrays
        self.fab_direction_ok = []
        fab_directions = list(range(joint_.noc))

        # Check fabrication direction for each component
        for n in range(joint_.noc):
            if n == 0 or n == joint_.noc - 1:
                self.fab_direction_ok.append(True)
                fab_directions[n] = 0 if n == 0 else 1
            else:
                fab_ok, fab_dir = Utils.is_fab_direction_ok(voxel_matrix, joint_.sax, n)
                fab_directions[n] = fab_dir
                self.fab_direction_ok.append(fab_ok)

        return fab_directions

    def _process_chessboard(self, voxel_matrix, joint_):
        # Initialize chessboard arrays
        self.checker = []
        self.checker_vertices = []

        # Check chessboard pattern for each component
        for n in range(joint_.noc):
            check, verts = Utils.get_chessboard_vertics(voxel_matrix, joint_.sax, joint_.noc, n)
            self.checker.append(check)
            self.checker_vertices.append(verts)

    def _process_sliding_and_interlocking(self, joint_):
        # Get sliding directions for all components
        self.slides, self.number_of_slides = Utils.get_sliding_directions(self.voxel_matrix_with_sides, joint_.noc)

        # Initialize interlocking
        self.interlock = True
        self.interlocks = []

        # Check interlocking for each component
        for n in range(joint_.noc):
            if n == 0 or n == joint_.noc - 1:
                # End components should have exactly one sliding direction
                if self.number_of_slides[n] <= 1:
                    self.interlocks.append(True)
                else:
                    self.interlocks.append(False)
                    self.interlock = False
            else:
                # Middle components should have no sliding directions
                if self.number_of_slides[n] == 0:
                    self.interlocks.append(True)
                else:
                    self.interlocks.append(False)
                    self.interlock = False

    def _process_friction_and_contact(self, voxel_matrix, joint_):
        # Initialize friction and contact arrays
        self.friction_nums = []
        self.friction_faces = []
        self.contact_nums = []
        self.contact_faces = []

        # Calculate friction and contact areas for each component
        for n in range(joint_.noc):
            friction, ffaces, contact, cfaces = Utils.get_friction_and_contact_areas(
                voxel_matrix, self.slides[n], joint_.fixed.sides, n
            )
            self.friction_nums.append(friction)
            self.friction_faces.append(ffaces)
            self.contact_nums.append(contact)
            self.contact_faces.append(cfaces)

    def _process_grain_direction(self, voxel_matrix, joint_):
        # Initialize breakability arrays
        self.breakable = []
        self.breakable_outline_inds = []
        self.breakable_voxel_inds = []

        # Check breakability for each component
        for n in range(joint_.noc):
            brk, brk_oinds, brk_vinds = Utils.get_breakable_voxels(
                voxel_matrix, joint_.fixed.sides[n], joint_.sax, n
            )
            self.breakable.append(brk)
            self.breakable_outline_inds.append(brk_oinds)
            self.breakable_voxel_inds.append(brk_vinds)

        # Separate breakable and non-breakable voxels
        self.non_breakable_voxmat, self.breakable_voxmat = self.separate_voxel_matrix(
            voxel_matrix, self.breakable_voxel_inds
        )

    def _validate_joint(self):
        # Check if the joint is valid based on all criteria
        if not self.interlock or not all(self.connected) or not all(self.bridged):
            self.valid = False
        elif any(self.breakable) or any(self.checker) or not all(self.fab_direction_ok):
            self.valid = False
        else:
            self.valid = True

    def separate_unconnected(self, voxel_matrix, fixed_sides, dim):
        """
        Separate voxels into connected and unconnected matrices based on their connection to fixed sides.

        Args:
            voxel_matrix: The original voxel matrix
            fixed_sides: The fixed sides for each component
            dim: The dimension of the voxel matrix
        """
        # Initialize matrices with -1 (empty space)
        connected_mat = np.zeros((dim, dim, dim)) - 1
        unconnected_mat = np.zeros((dim, dim, dim)) - 1

        # Iterate through all voxels
        for i in range(dim):
            for j in range(dim):
                for k in range(dim):
                    ind = [i, j, k]
                    val = voxel_matrix[tuple(ind)]

                    # Skip empty voxels (value < 0)
                    if val < 0:
                        continue

                    # Check if voxel is connected to its fixed side
                    connected = Utils.is_connected_to_fixed_side(
                        np.array([ind]), voxel_matrix, fixed_sides[int(val)]
                    )

                    # Assign voxel to appropriate matrix
                    if connected:
                        connected_mat[tuple(ind)] = val
                    else:
                        unconnected_mat[tuple(ind)] = val

        # Store the results
        self.voxel_matrix_connected = connected_mat
        self.voxel_matrix_unconnected = unconnected_mat

    def separate_voxel_matrix(self, voxmat, inds):
        """
        Separate a voxel matrix into two matrices based on specified indices.

        Args:
            voxmat: The original voxel matrix
            inds: List of indices to separate

        Returns:
            voxmat_a: Matrix with specified indices removed
            voxmat_b: Matrix with only the specified indices
        """
        dim = len(voxmat)

        # Create a deep copy for the first matrix
        voxmat_a = copy.deepcopy(voxmat)

        # Initialize the second matrix with -1 (empty space)
        voxmat_b = np.zeros((dim, dim, dim)) - 1

        # Move voxels from matrix A to matrix B based on indices
        for n in range(len(inds)):
            for ind in inds[n]:
                ind = tuple(ind)
                val = voxmat[ind]

                # Remove from matrix A and add to matrix B
                voxmat_a[ind] = -1
                voxmat_b[ind] = val

        return voxmat_a, voxmat_b

    def separate_unbridged(self, voxel_matrix, fixed_sides, dim, n):
        """
        Separate voxels of a specific component into two matrices based on their connection
        to each of the two fixed sides.

        Args:
            voxel_matrix: The original voxel matrix
            fixed_sides: The fixed sides for each component
            dim: The dimension of the voxel matrix
            n: The component number to process

        Returns:
            unbridged_1: Matrix with voxels connected to the first fixed side
            unbridged_2: Matrix with voxels connected to the second fixed side
        """
        # Initialize matrices with -1 (empty space)
        unbridged_1 = np.zeros((dim, dim, dim)) - 1
        unbridged_2 = np.zeros((dim, dim, dim)) - 1

        # Iterate through all voxels
        for i in range(dim):
            for j in range(dim):
                for k in range(dim):
                    ind = [i, j, k]
                    val = voxel_matrix[tuple(ind)]

                    # Skip voxels that don't belong to the specified component
                    if val != n:
                        continue

                    # Check connection to each fixed side
                    conn_1 = Utils.is_connected_to_fixed_side(
                        np.array([ind]), voxel_matrix, [fixed_sides[n][0]]
                    )
                    conn_2 = Utils.is_connected_to_fixed_side(
                        np.array([ind]), voxel_matrix, [fixed_sides[n][1]]
                    )

                    # Assign voxel to appropriate matrix
                    if conn_1:
                        unbridged_1[tuple(ind)] = val
                    if conn_2:
                        unbridged_2[tuple(ind)] = val

        return unbridged_1, unbridged_2

# class EvaluationOne:
#     def __init__(self,voxel_matrix,fixed_sides,sax,noc,level,last):
#
#         # Initiate metrics
#         self.connected_and_bridged = True
#         self.other_connected_and_bridged = True
#         self.nocheck = True
#         self.interlock = True
#         self.nofragile = True
#         self.valid = False
#
#         # Add fixed sides to voxel matrix, get dimension
#         self.voxel_matrix_with_sides = Utils.add_fixed_sides(voxel_matrix, fixed_sides)
#         dim = len(voxel_matrix)
#
#         #Connectivity and bridging
#         self.connected_and_bridged = Utils.is_connected(self.voxel_matrix_with_sides,level)
#         if not self.connected_and_bridged: return
#
#         # Other connectivity and bridging
#         if not last:
#             other_level = 0
#             if level==0: other_level = 1
#             special_voxmat_with_sides = Utils.add_fixed_sides(voxel_matrix, fixed_sides, 10)
#             self.other_connected_and_bridged = Utils.is_potentially_connected(special_voxmat_with_sides,dim,noc,level)
#             if not self.other_connected_and_bridged: return
#
#         # Checkerboard
#         if last:
#             check,verts = Utils.get_chessboard_vertics(voxel_matrix,sax,noc,level)
#             if check: self.nocheck=False
#             if not self.nocheck: return
#
#         # Slidability
#         self.slides,self.number_of_slides = Utils.get_sliding_directions_of_one_timber(self.voxel_matrix_with_sides,level)
#         if level==0 or level==noc-1:
#             if self.number_of_slides!=1: self.interlock=False
#         else:
#             if self.number_of_slides!=0: self.interlock=False
#         if not self.interlock: return
#
#         # Durability
#         brk,brk_inds = Utils.get_breakable_voxels(voxel_matrix,fixed_sides[level],sax,level)
#         if brk: self.nofragile = False
#         if not self.nofragile: return
#
#         self.valid=True

class EvaluationOne:
    def __init__(self, voxel_matrix, fixed_sides, sax, noc, level, last):
        # Initialize metrics
        self._initialize_metrics()

        # Add fixed sides to voxel matrix
        self.voxel_matrix_with_sides = Utils.add_fixed_sides(voxel_matrix, fixed_sides)
        dim = len(voxel_matrix)

        # Check connectivity and bridging
        if not self._check_connectivity(level):
            return

        # Check other connectivity and bridging if not the last component
        if not last and not self._check_other_connectivity(voxel_matrix, fixed_sides, dim, noc, level):
            return

        # Check for checkerboard pattern if it's the last component
        if last and not self._check_checkerboard(voxel_matrix, sax, noc, level):
            return

        # Check slidability and interlocking
        if not self._check_slidability(level, noc):
            return

        # Check durability
        if not self._check_durability(voxel_matrix, fixed_sides, sax, level):
            return

        # If all checks pass, the joint is valid
        self.valid = True

    def _initialize_metrics(self):
        self.connected_and_bridged = True
        self.other_connected_and_bridged = True
        self.nocheck = True
        self.interlock = True
        self.nofragile = True
        self.valid = False

    def _check_connectivity(self, level):
        self.connected_and_bridged = Utils.is_connected(self.voxel_matrix_with_sides, level)
        return self.connected_and_bridged

    def _check_other_connectivity(self, voxel_matrix, fixed_sides, dim, noc, level):
        other_level = 1 if level == 0 else 0
        special_voxmat_with_sides = Utils.add_fixed_sides(voxel_matrix, fixed_sides, 10)
        self.other_connected_and_bridged = Utils.is_potentially_connected(
            special_voxmat_with_sides, dim, noc, level
        )
        return self.other_connected_and_bridged

    def _check_checkerboard(self, voxel_matrix, sax, noc, level):
        check, verts = Utils.get_chessboard_vertics(voxel_matrix, sax, noc, level)
        if check:
            self.nocheck = False
        return self.nocheck

    def _check_slidability(self, level, noc):
        self.slides, self.number_of_slides = Utils.get_sliding_directions_of_one_timber(
            self.voxel_matrix_with_sides, level
        )

        if level == 0 or level == noc - 1:
            # End components should have exactly one sliding direction
            if self.number_of_slides != 1:
                self.interlock = False
        else:
            # Middle components should have no sliding directions
            if self.number_of_slides != 0:
                self.interlock = False

        return self.interlock

    def _check_durability(self, voxel_matrix, fixed_sides, sax, level):
        brk, brk_inds = Utils.get_breakable_voxels(voxel_matrix, fixed_sides[level], sax, level)
        if brk:
            self.nofragile = False
        return self.nofragile

# class EvaluationSlides:
#     def __init__(self,voxel_matrix,fixed_sides,sax,noc):
#         voxel_matrix_with_sides = Utils.add_fixed_sides(voxel_matrix, fixed_sides)
#         # Sliding depth
#         sliding_depths = [3,3,3]
#         open_mat = np.copy(voxel_matrix_with_sides)
#         for depth in range(4):
#             slds,nos = Utils.get_sliding_directions(open_mat,noc)
#             for n in range(noc):
#                 if sliding_depths[n]!=3: continue
#                 if n==0 or n==noc-1:
#                     if nos[n]>1: sliding_depths[n]=depth
#                 else:
#                     if nos[n]>0: sliding_depths[n]=depth
#             open_mat = Utils.open_matrix(open_mat,sax,noc)
#         self.slide_depths = sliding_depths
#         #self.slide_depths_sorted = sliding_depths
#         #self.slide_depth_product = np.prod(np.array(sliding_depths))

class EvaluationSlides:
    def __init__(self, voxel_matrix, fixed_sides, sax, noc):
        voxel_matrix_with_sides = Utils.add_fixed_sides(voxel_matrix, fixed_sides)
        self.slide_depths = self._calculate_sliding_depths(voxel_matrix_with_sides, sax, noc)

    def _calculate_sliding_depths(self, voxel_matrix_with_sides, sax, noc):
        sliding_depths = [3, 3, 3]
        open_mat = np.copy(voxel_matrix_with_sides)

        for depth in range(4):
            slds, nos = Utils.get_sliding_directions(open_mat, noc)

            for n in range(noc):
                if sliding_depths[n] != 3:
                    continue

                if n == 0 or n == noc - 1:
                    # End components
                    if nos[n] > 1:
                        sliding_depths[n] = depth
                else:
                    # Middle components
                    if nos[n] > 0:
                        sliding_depths[n] = depth

            open_mat = Utils.open_matrix(open_mat, sax, noc)

        return sliding_depths

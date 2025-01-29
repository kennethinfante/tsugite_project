import numpy as np
import copy

from utils import RegionVertex
from misc import FixedSide

import utils as Utils

class Evaluation:
    def __init__(self,voxel_matrix,type,mainmesh=True):
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
        self.fab_directions = self.update(voxel_matrix,type)

    def update(self,voxel_matrix,type):
        self.voxel_matrix_with_sides = Utils.add_fixed_sides(voxel_matrix, type.fixed.sides)

        # Voxel connection and bridgeing
        self.connected = []
        self.bridged = []
        self.voxel_matrices_unbridged = []
        for n in range(type.noc):
            self.connected.append(Utils.is_connected(self.voxel_matrix_with_sides,n))
            self.bridged.append(True)
            self.voxel_matrices_unbridged.append(None)
        self.voxel_matrix_connected = voxel_matrix.copy()
        self.voxel_matrix_unconnected = None

        self.seperate_unconnected(voxel_matrix,type.fixed.sides,type.dim)

        # Bridging
        voxel_matrix_connected_with_sides = Utils.add_fixed_sides(self.voxel_matrix_connected, type.fixed.sides)
        for n in range(type.noc):
            self.bridged[n] = Utils.is_connected(voxel_matrix_connected_with_sides,n)
            if not self.bridged[n]:
                voxel_matrix_unbridged_1, voxel_matrix_unbridged_2 = self.seperate_unbridged(voxel_matrix,type.fixed.sides,type.dim,n)
                self.voxel_matrices_unbridged[n] = [voxel_matrix_unbridged_1, voxel_matrix_unbridged_2]

        # Fabricatability by direction constraint
        self.fab_direction_ok = []
        fab_directions = list(range(type.noc))
        for n in range(type.noc):
            if n==0 or n==type.noc-1:
                self.fab_direction_ok.append(True)
                if n==0: fab_directions[n]=0
                else: fab_directions[n]=1
            else:
                fab_ok,fab_dir = Utils.is_fab_direction_ok(voxel_matrix,type.sax,n)
                fab_directions[n] = fab_dir
                self.fab_direction_ok.append(fab_ok)

        # Chessboard
        self.checker = []
        self.checker_vertices = []
        for n in range(type.noc):
            check,verts = Utils.get_chessboard_vertics(voxel_matrix,type.sax,type.noc,n)
            self.checker.append(check)
            self.checker_vertices.append(verts)

        # Sliding directions
        self.slides,self.number_of_slides = Utils.get_sliding_directions(self.voxel_matrix_with_sides,type.noc)
        self.interlock = True
        for n in range(type.noc):
            if (n==0 or n==type.noc-1):
                if self.number_of_slides[n]<=1:
                    self.interlocks.append(True)
                else:
                    self.interlocks.append(False)
                    self.interlock=False
            else:
                if self.number_of_slides[n]==0:
                    self.interlocks.append(True)
                else:
                    self.interlocks.append(False)
                    self.interlock=False

        # Friction
        self.friction_nums = []
        self.friction_faces = []
        self.contact_nums = []
        self.contact_faces = []
        for n in range(type.noc):
            friction,ffaces,contact,cfaces, = Utils.get_friction_and_contact_areas(voxel_matrix,self.slides[n],type.fixed.sides,n)
            self.friction_nums.append(friction)
            self.friction_faces.append(ffaces)
            self.contact_nums.append(contact)
            self.contact_faces.append(cfaces)


        # Grain direction
        for n in range(type.noc):
            brk,brk_oinds,brk_vinds = Utils.get_breakable_voxels(voxel_matrix,type.fixed.sides[n],type.sax,n)
            self.breakable.append(brk)
            self.breakable_outline_inds.append(brk_oinds)
            self.breakable_voxel_inds.append(brk_vinds)
        self.non_breakable_voxmat, self.breakable_voxmat = self.seperate_voxel_matrix(voxel_matrix,self.breakable_voxel_inds)

        if not self.interlock or not all(self.connected) or not all(self.bridged):
            self.valid=False
        elif any(self.breakable) or any(self.checker) or not all(self.fab_direction_ok):
            self.valid=False

        """
        # Sliding depth
        sliding_depths = [3,3,3]
        open_mat = np.copy(self.voxel_matrix_with_sides)
        for depth in range(4):
            slds,nos = get_sliding_directions(open_mat,noc)
            for n in range(noc):
                if sliding_depths[n]!=3: continue
                if n==0 or n==noc-1:
                    if nos[n]>1: sliding_depths[n]=depth
                else:
                    if nos[n]>0: sliding_depths[n]=depth
            open_mat = open_matrix(open_mat,sax,noc)
        self.slide_depths = sliding_depths
        self.slide_depth_product = np.prod(np.array(sliding_depths))
        print(self.slide_depths,self.slide_depth_product)
        """
        return fab_directions

    def seperate_unconnected(self,voxel_matrix,fixed_sides,dim):
        connected_mat = np.zeros((dim,dim,dim))-1
        unconnected_mat = np.zeros((dim,dim,dim))-1
        for i in range(dim):
            for j in range(dim):
                for k in range(dim):
                    connected = False
                    ind = [i,j,k]
                    val = voxel_matrix[tuple(ind)]
                    connected = Utils.is_connected_to_fixed_side(np.array([ind]),voxel_matrix,fixed_sides[int(val)])
                    if connected: connected_mat[tuple(ind)] = val
                    else: unconnected_mat[tuple(ind)] = val
        self.voxel_matrix_connected = connected_mat
        self.voxel_matrix_unconnected = unconnected_mat

    def seperate_voxel_matrix(self,voxmat,inds):
        dim = len(voxmat)
        voxmat_a = copy.deepcopy(voxmat)
        voxmat_b = np.zeros((dim,dim,dim))-1
        for n in range(len(inds)):
            for ind in inds[n]:
                ind = tuple(ind)
                val = voxmat[ind]
                voxmat_a[ind] = -1
                voxmat_b[ind] = val
        return voxmat_a,voxmat_b

    def seperate_unbridged(self,voxel_matrix,fixed_sides,dim,n):
        unbridged_1 = np.zeros((dim,dim,dim))-1
        unbridged_2 = np.zeros((dim,dim,dim))-1
        for i in range(dim):
            for j in range(dim):
                for k in range(dim):
                    ind = [i,j,k]
                    val = voxel_matrix[tuple(ind)]
                    if val!=n: continue
                    conn_1 = Utils.is_connected_to_fixed_side(np.array([ind]),voxel_matrix,[fixed_sides[n][0]])
                    conn_2 = Utils.is_connected_to_fixed_side(np.array([ind]),voxel_matrix,[fixed_sides[n][1]])
                    if conn_1: unbridged_1[tuple(ind)] = val
                    if conn_2: unbridged_2[tuple(ind)] = val
        return unbridged_1, unbridged_2

class EvaluationOne:
    def __init__(self,voxel_matrix,fixed_sides,sax,noc,level,last):

        # Initiate metrics
        self.connected_and_bridged = True
        self.other_connected_and_bridged = True
        self.nocheck = True
        self.interlock = True
        self.nofragile = True
        self.valid = False

        # Add fixed sides to voxel matrix, get dimension
        self.voxel_matrix_with_sides = Utils.add_fixed_sides(voxel_matrix, fixed_sides)
        dim = len(voxel_matrix)

        #Connectivity and bridging
        self.connected_and_bridged = Utils.is_connected(self.voxel_matrix_with_sides,level)
        if not self.connected_and_bridged: return

        # Other connectivity and bridging
        if not last:
            other_level = 0
            if level==0: other_level = 1
            special_voxmat_with_sides = Utils.add_fixed_sides(voxel_matrix, fixed_sides, 10)
            self.other_connected_and_bridged = Utils.is_potentially_connected(special_voxmat_with_sides,dim,noc,level)
            if not self.other_connected_and_bridged: return

        # Checkerboard
        if last:
            check,verts = Utils.get_chessboard_vertics(voxel_matrix,sax,noc,level)
            if check: self.nocheck=False
            if not self.nocheck: return

        # Slidability
        self.slides,self.number_of_slides = Utils.get_sliding_directions_of_one_timber(self.voxel_matrix_with_sides,level)
        if level==0 or level==noc-1:
            if self.number_of_slides!=1: self.interlock=False
        else:
            if self.number_of_slides!=0: self.interlock=False
        if not self.interlock: return

        # Durability
        brk,brk_inds = Utils.get_breakable_voxels(voxel_matrix,fixed_sides[level],sax,level)
        if brk: self.nofragile = False
        if not self.nofragile: return

        self.valid=True

class EvaluationSlides:
    def __init__(self,voxel_matrix,fixed_sides,sax,noc):
        voxel_matrix_with_sides = Utils.add_fixed_sides(voxel_matrix, fixed_sides)
        # Sliding depth
        sliding_depths = [3,3,3]
        open_mat = np.copy(voxel_matrix_with_sides)
        for depth in range(4):
            slds,nos = Utils.get_sliding_directions(open_mat,noc)
            for n in range(noc):
                if sliding_depths[n]!=3: continue
                if n==0 or n==noc-1:
                    if nos[n]>1: sliding_depths[n]=depth
                else:
                    if nos[n]>0: sliding_depths[n]=depth
            open_mat = Utils.open_matrix(open_mat,sax,noc)
        self.slide_depths = sliding_depths
        #self.slide_depths_sorted = sliding_depths
        #self.slide_depth_product = np.prod(np.array(sliding_depths))

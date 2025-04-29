import numpy as np
from numpy import ndarray, linalg
import math
import random
import copy
from typing import List, Tuple, Any

from fixed_side import FixedSide
from fabrication import MillVertex

class RegionVertex:
    def __init__(self, ind: List[int], abs_ind: List[int], neighbors: ndarray,
                 neighbor_values: ndarray, dia: bool = False, minus_one_neighbor: bool = False):
        self.ind: List[int] = ind
        self.i: int = ind[0]
        self.j: int = ind[1]
        self.neighbors: ndarray = neighbors
        self.flat_neighbors: ndarray = self.neighbors.flatten()
        self.region_count: int = np.sum(self.flat_neighbors == 0)
        self.block_count: int = np.sum(self.flat_neighbors == 1)
        self.free_count: int = np.sum(self.flat_neighbors == 2)
        ##
        self.minus_one_neighbor: bool = minus_one_neighbor
        ##
        self.dia: bool = dia
        ##
        self.neighbor_values: ndarray = np.array(neighbor_values)
        self.flat_neighbor_values: ndarray = self.neighbor_values.flatten()

def normalize(v: ndarray) -> np.ndarray:
    norm = linalg.norm(v)
    if norm == 0: return v
    else: return v / norm

def angle_between_vectors(vector_1: ndarray, vector_2: ndarray,
                          normal_vector: List[float] = None,
                          return_degrees: bool = False,
                          signed: bool = False) -> float:
    """Calculate the angle between two vectors.

    Args:
        vector_1: First vector
        vector_2: Second vector
        normal_vector: Optional reference vector for determining sign direction
        return_degrees: If True, returns angle in degrees, otherwise in radians
        signed: If True, returns signed angle (direction matters)

    Returns:
        Angle between vectors in radians (or degrees if return_degrees is True)
    """
    unit_vector_1 = normalize(vector_1)
    unit_vector_2 = normalize(vector_2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)

    # Clamp dot product to avoid numerical errors
    dot_product = max(min(dot_product, 1.0), -1.0)

    if signed:
        # Calculate signed angle using arctan2
        angle = np.arctan2(linalg.det([unit_vector_1, unit_vector_2]), dot_product)
    else:
        # Calculate unsigned angle using arccos
        angle = np.arccos(dot_product)

        # Apply sign based on normal vector if provided
        if normal_vector is not None:
            cross = np.cross(unit_vector_1, unit_vector_2)
            if np.dot(normal_vector, cross) < 0:
                angle = -angle

    # Convert to degrees if requested
    if return_degrees:
        angle = math.degrees(angle)

    return angle

# def angle_between_vectors1(vector_1: ndarray, vector_2: ndarray, direction: bool = False) -> float:
#     unit_vector_1 = normalize(vector_1)
#     unit_vector_2 = normalize(vector_2)
#     v_dot_product = np.dot(unit_vector_1, unit_vector_2)
#
#     if direction:
#         angle = np.arctan2(linalg.det([unit_vector_1, unit_vector_2]), v_dot_product)
#         return math.degrees(angle)
#     else:
#         angle = np.arccos(v_dot_product)
#         return angle
#
# def angle_between_vectors2(vector_1: ndarray, vector_2: ndarray,
#                            normal_vector: List[float] = []) -> float:
#     unit_vector_1 = normalize(vector_1)
#     unit_vector_2 = normalize(vector_2)
#     dot_product = np.dot(unit_vector_1, unit_vector_2)
#     angle = np.arccos(dot_product)
#     cross = np.cross(unit_vector_1, unit_vector_2)
#     if len(normal_vector) > 0 and np.dot(normal_vector, cross) < 0: angle = -angle
#     return angle

def rotate_vector_around_axis(vec: List[float] = [3, 5, 0],
                              axis: List[float] = [4, 4, 1],
                              theta: float = 1.2) -> np.ndarray:  # example values
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    mat = np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                    [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                    [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])
    rotated_vec = np.dot(mat, vec)
    return rotated_vec

def matrix_from_height_fields(hfs: List[np.ndarray], ax: int) -> np.ndarray:
    dim = len(hfs[0])
    mat = np.zeros(shape=(dim, dim, dim))

    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                ind = [i, j]
                ind3d = _insert_at_axis(ind, ax, k)
                ind2d = tuple(ind)

                mat[tuple(ind3d)] = _determine_material_from_heights(k, hfs, ind2d)

    return mat

def _insert_at_axis(ind: List[int], ax: int, val: int) -> List[int]:
    """Insert a value at the specified axis position."""
    ind3d = ind.copy()
    ind3d.insert(ax, val)
    return ind3d

def _determine_material_from_heights(k: int, hfs: List[np.ndarray], ind2d: Tuple[int, ...]) -> int:
    """Determine material type based on height fields."""
    for n, hf in enumerate(hfs):
        if k < hf[ind2d]:
            return n
    return len(hfs)  # If above all height fields, n+1


def get_same_height_neighbors(hfield: ndarray, inds: List[List[int]]) -> List[List[int]]:
    dim = len(hfield)
    val = hfield[tuple(inds[0])]
    new_inds = list(inds)

    for ind in inds:
        for ax in range(2):
            for dir in range(-1, 2, 2):
                ind2 = _get_neighbor_index(ind, ax, dir)

                if np.all(dim > ind2 >= 0) and hfield[tuple(ind2)] == val:
                    if _is_unique_index(ind2, new_inds):
                        new_inds.append(ind2)

    if len(new_inds) > len(inds):
        new_inds = get_same_height_neighbors(hfield, new_inds)

    return new_inds

def _is_unique_index(ind: List[int], inds: List[List[int]]) -> bool:
    """Check if index is not already in the list."""
    for existing_ind in inds:
        if ind[0] == existing_ind[0] and ind[1] == existing_ind[1]:
            return False
    return True

def get_random_height_fields(dim: int, noc: int) -> List[np.ndarray]:
    hfs = []
    phf = np.zeros((dim, dim))
    for n in range(noc - 1):
        hf = np.zeros((dim, dim))
        for i in range(dim):
            for j in range(dim):
                hf[i, j] = random.randint(int(phf[i, j]), dim)
        hfs.append(hf)
        phf = copy.deepcopy(hf)
    return hfs

def get_diff_neighbors(mat2: ndarray, inds: List[List[int]], val: int) -> List[List[int]]:
    new_inds = list(inds)
    for ind in inds:
        for ax in range(2):
            for dir in range(-1, 2, 2):
                ind2 = ind.copy()
                ind2[ax] += dir
                if (mat2.shape[ax] > ind2[ax] >= 0):
                    val2 = mat2[tuple(ind2)]
                    if val2 == val or val2 == -1: continue
                    unique = True
                    for ind3 in new_inds:
                        if ind2[0] == ind3[0] and ind2[1] == ind3[1]:
                            unique = False
                            break
                    if unique: new_inds.append(ind2)
    if len(new_inds) > len(inds):
        new_inds = get_diff_neighbors(mat2, new_inds, val)
    return new_inds

def set_starting_vert(verts: List[RegionVertex]) -> List[RegionVertex]:
    first_i = _find_best_starting_vertex(verts)

    # Move the selected vertex to the front
    verts.insert(0, verts[first_i])
    verts.pop(first_i + 1)

    return verts

def _find_best_starting_vertex(verts: List[RegionVertex]) -> int:
    """Find the best vertex to start with based on block and free counts."""
    first_i = None
    second_i = None

    for i, rv in enumerate(verts):
        if rv.block_count > 0:
            if rv.free_count > 0:
                first_i = i
            else:
                second_i = i

    # Prioritize vertices with both block and free counts
    if first_i is None:
        first_i = second_i

    # Default to first vertex if no suitable vertex found
    if first_i is None:
        first_i = 0

    return first_i


def get_outline(type: Any, verts: List[RegionVertex], lay_num: int, n: int) -> List[MillVertex]:
    fdir = type.mesh.fab_directions[n]
    outline = []

    for rv in verts:
        ind = _get_3d_vertex_index(rv.ind, type.sax, type.dim, fdir, lay_num)
        add = _get_direction_vector(type.sax, fdir)

        i_pt = get_index(ind, add, type.dim)
        pt = get_vertex(i_pt, type.jverts[n], type.vertex_no_info)

        outline.append(MillVertex(pt))

    return outline

def _get_3d_vertex_index(ind: List[int], sax: int, dim: int, fdir: int, lay_num: int) -> List[int]:
    """Get the 3D index for a vertex in the outline."""
    ind3d = ind.copy()
    ind3d.insert(sax, (dim - 1) * (1 - fdir) + (2 * fdir - 1) * lay_num)
    return ind3d

def _get_direction_vector(sax: int, fdir: int) -> List[int]:
    """Get the direction vector based on the sliding axis and fabrication direction."""
    add = [0, 0, 0]
    add[sax] = 1 - fdir
    return add

def set_vector_length(vec: ndarray, new_norm: float) -> np.ndarray:
    norm = linalg.norm(vec)
    vec = vec / norm
    vec = new_norm * vec
    return vec

def get_vertex(index: int, verts: ndarray, n: int) -> np.ndarray:
    x = verts[n * index]
    y = verts[n * index + 1]
    z = verts[n * index + 2]
    return np.array([x, y, z])

def get_segment_proportions(outline: List[MillVertex]) -> List[float]:
    segment_lengths = _calculate_segment_lengths(outline)
    total_length = sum(segment_lengths)

    return _calculate_proportions(segment_lengths, total_length)

def _calculate_segment_lengths(outline: List[MillVertex]) -> List[float]:
    """Calculate the length of each segment in the outline."""
    lengths = []

    for i in range(1, len(outline)):
        prev_point = outline[i-1].pt
        current_point = outline[i].pt
        distance = linalg.norm(current_point - prev_point)
        lengths.append(distance)

    return lengths

def _calculate_proportions(lengths: List[float], total_length: float) -> List[float]:
    """Calculate the proportional position of each vertex along the outline."""
    proportions = [0.0]  # Start with 0
    cumulative_length = 0

    for length in lengths:
        cumulative_length += length
        proportions.append(cumulative_length / total_length)

    return proportions


def has_minus_one_neighbor(ind: List[int], lay_mat: ndarray) -> bool:
    for add0 in range(-1, 1, 1):
        for add1 in range(-1, 1, 1):
            nind = [ind[0] + add0, ind[1] + add1]

            if _is_valid_index_in_matrix(nind, lay_mat) and lay_mat[tuple(nind)] == -1:
                return True

    return False

def _is_valid_index_in_matrix(ind: List[int], matrix: ndarray) -> bool:
    """Check if index is within matrix bounds."""
    return (np.all(np.array(ind) >= 0) and
            ind[0] < matrix.shape[0] and
            ind[1] < matrix.shape[1])


def get_neighbors_in_out(ind: List[int], reg_inds: List[List[int]], lay_mat: ndarray,
                         org_lay_mat: ndarray, n: int) -> Tuple[List[List[int]], List[List[int]]]:
    in_out = []
    values = []

    for add0 in range(-1, 1, 1):
        in_out_row = []
        values_row = []

        for add1 in range(-1, 1, 1):
            nind = [ind[0] + add0, ind[1] + add1]

            type_val = _determine_neighbor_type(nind, reg_inds, lay_mat)
            value = _get_neighbor_value(nind, org_lay_mat, type_val)

            in_out_row.append(type_val)
            values_row.append(value)

        in_out.append(in_out_row)
        values.append(values_row)

    return in_out, values

def _determine_neighbor_type(nind: List[int], reg_inds: List[List[int]], lay_mat: ndarray) -> int:
    """Determine the type of a neighbor cell.
    0 = in region
    1 = outside region, block
    2 = outside region, free
    """
    # Check if this index is in the list of region-included indices
    for rind in reg_inds:
        if rind[0] == nind[0] and rind[1] == nind[1]:
            return 0  # in region

    # If out of bounds or negative value, it's free
    if (np.any(np.array(nind) < 0) or
        nind[0] >= lay_mat.shape[0] or
        nind[1] >= lay_mat.shape[1] or
        lay_mat[tuple(nind)] < 0):
        return 2  # free

    return 1  # blocked

def _get_neighbor_value(nind: List[int], org_lay_mat: ndarray, type_val: int) -> int:
    """Get the value of a neighbor cell."""
    if type_val == 2:  # free
        if (np.any(np.array(nind) < 0) or
            nind[0] >= org_lay_mat.shape[0] or
            nind[1] >= org_lay_mat.shape[1]):
            return -1
        else:
            return -2

    return org_lay_mat[tuple(nind)]


def face_neighbors(mat: ndarray, ind: List[int], ax: int, n: int,
                   fixed_sides: List[FixedSide]) -> Tuple[int, np.ndarray]:
    values = []
    dim = len(mat)

    for i in range(2):
        ind2 = ind.copy()
        ind2[ax] = ind2[ax] - i

        val = _get_face_neighbor_value(ind2, mat, fixed_sides, ax, n, dim)
        values.append(val)

    values = np.array(values)
    count = np.count_nonzero(values == n)

    return count, values

def _get_face_neighbor_value(ind: List[int], mat: ndarray, fixed_sides: List[FixedSide],
                            ax: int, n: int, dim: int) -> int:
    """Get the value of a face neighbor, considering fixed sides."""
    ind = np.array(ind)

    if np.all(ind >= 0) and np.all(ind < dim):
        return mat[tuple(ind)]

    # Check fixed sides
    if len(fixed_sides) > 0:
        for fixed_side in fixed_sides:
            ind_without_ax = np.delete(ind, fixed_side.ax)

            if np.all(ind_without_ax >= 0) and np.all(ind_without_ax < dim):
                if ind[fixed_side.ax] < 0 and fixed_side.dir == 0:
                    return n
                elif ind[fixed_side.ax] >= dim and fixed_side.dir == 1:
                    return n

    return None  # No valid value found

def get_index(ind: List[int], add: List[int], dim: int) -> int:
    d = dim+1
    (i, j, k) = ind
    index = (i+add[0])*d*d + (j+add[1])*d + k+add[2]
    return index


def get_corner_indices(ax: int, n: int, dim: int) -> List[int]:
    other_axes = _get_perpendicular_axes(ax)
    base_index = _get_base_corner_index(ax, n, dim)

    return _generate_all_corners(base_index, other_axes, dim)

def _get_base_corner_index(ax: int, n: int, dim: int) -> np.ndarray:
    """Get the base corner index along the specified axis."""
    ind = np.array([0, 0, 0])
    ind[ax] = n * dim
    return ind

def _generate_all_corners(base_index: np.ndarray, other_axes: List[int], dim: int) -> List[int]:
    """Generate all corner indices from the base corner."""
    corner_indices = []

    for x in range(2):
        for y in range(2):
            add = np.array([0, 0, 0])
            add[other_axes[0]] = x * dim
            add[other_axes[1]] = y * dim
            corner_indices.append(get_index(base_index, add, dim))

    return corner_indices

def connected_arc(mv0: MillVertex, mv1: MillVertex) -> bool:
    """Check if two mill vertices form a connected arc."""
    return (mv0.is_arc and
            mv1.is_arc and
            mv0.arc_ctr[0] == mv1.arc_ctr[0] and
            mv0.arc_ctr[1] == mv1.arc_ctr[1])


def arc_points(st: List[float], en: List[float], ctr0: List[float], ctr1: List[float],
               ax: int, astep: float) -> List[np.ndarray]:
    pts = []

    # Convert to numpy arrays
    st, en = np.array(st), np.array(en)
    ctr0, ctr1 = np.array(ctr0), np.array(ctr1)

    # Calculate vectors and angle
    v0, v1 = st - ctr0, en - ctr1
    angle = angle_between_vectors(v0, v1)
    z_diff = en[ax] - st[ax]
    # Calculate steps
    step_info = _calculate_arc_steps(angle, astep, z_diff)
    cnt, astep, zstep = step_info

    # Calculate axis vector for rotation
    ax_vec = np.cross(v0, v1)

    # Generate points along the arc
    for i in range(1, cnt + 1):
        rvec = rotate_vector_around_axis(v0, ax_vec, astep * i)
        zvec = [0, 0, 0]
        zvec[ax] = zstep * i
        pts.append(ctr0 + rvec + np.array(zvec))

    return pts

def _calculate_arc_steps(angle: float, astep: float, z_diff: float) -> Tuple[int, float, float]:
    """Calculate number of steps, angle step, and z step for an arc."""
    cnt = int(0.5 + angle / astep)

    if cnt > 0:
        astep = angle / cnt
        zstep = z_diff / cnt
    else:
        astep = 0
        zstep = 0

    return cnt, astep, zstep

def is_connected(mat: ndarray, n: int) -> bool:
    """Check if all voxels with value n are connected."""
    all_same_count = np.count_nonzero(mat == n)

    if all_same_count == 0:
        return False

    # Find one voxel with value n
    start_index = tuple(np.argwhere(mat == n)[0])

    # Get all connected voxels with the same value
    connected_indices = get_all_same_connected(mat, [start_index])
    connected_count = len(connected_indices)

    return connected_count == all_same_count


# def get_sliding_directions(mat: ndarray, noc: int) -> Tuple[List[List[List[int]]], List[int]]:
#     """Get possible sliding directions for each component."""
#     sliding_directions = []
#     number_of_sliding_directions = []
#
#     for n in range(noc):
#         component_directions = []
#
#         for ax in range(3):
#             for dir in range(2):
#                 if _check_sliding_direction(mat, ax, dir, n):
#                     component_directions.append([ax, dir])
#
#         sliding_directions.append(component_directions)
#         number_of_sliding_directions.append(len(component_directions))
#
#     return sliding_directions, number_of_sliding_directions
#
# def get_sliding_directions_of_one_timber(mat: ndarray, level: int) -> Tuple[List[List[int]], int]:
#     """Get possible sliding directions for a specific timber."""
#     sliding_directions = []
#
#     for ax in range(3):
#         for dir in range(2):
#             if _check_sliding_direction(mat, ax, dir, level):
#                 sliding_directions.append([ax, dir])
#
#     number_of_sliding_directions = len(sliding_directions)
#
#     return sliding_directions, number_of_sliding_directions

def get_sliding_directions(mat: ndarray, noc: int) -> Tuple[List[List[List[int]]], List[int]]:
    """Get possible sliding directions for each component."""
    sliding_dirs = []
    number_of_sliding_dirs = []

    for n in range(noc):

        component_dirs, len_component_dirs = get_sliding_directions_of_one_timber(mat, n)

        sliding_dirs.append(component_dirs)
        number_of_sliding_dirs.append(len_component_dirs)

    return sliding_dirs, number_of_sliding_dirs

def get_sliding_directions_of_one_timber(mat: ndarray, level: int) -> Tuple[List[List[int]], int]:
    """Get possible sliding directions for a specific timber."""
    component_dirs = []

    for ax in range(3):
        for dir in range(2):
            if _check_sliding_direction(mat, ax, dir, level):
                component_dirs.append([ax, dir])

    len_component_dirs = len(component_dirs)

    return component_dirs, len_component_dirs

def _check_sliding_direction(mat: ndarray, ax: int, dir: int, n: int) -> bool:
    """Check if a component can slide in a specific direction."""
    oax = [0, 1, 2]
    oax.remove(ax)

    for i in range(mat.shape[oax[0]]):
        for j in range(mat.shape[oax[1]]):
            first_same = False

            for k in range(mat.shape[ax]):
                if dir == 0:
                    k = mat.shape[ax] - k - 1

                ind = [i, j]
                ind.insert(ax, k)
                val = mat[tuple(ind)]

                if val == n:
                    first_same = True
                    continue
                elif first_same and val != -1:
                    return False

    return True

def get_neighbors(mat: ndarray, ind: Tuple[int, ...]) -> Tuple[List[Tuple[int, ...]], np.ndarray]:
    indices = []
    values = []
    for m in range(len(ind)):   # For each direction (x,y)
        for n in range(2):      # go up and down one step
            n = 2*n-1             # -1,1
            ind0 = list(ind)
            ind0[m] = ind[m]+n
            ind0 = tuple(ind0)
            if ind0[m] >= 0 and ind0[m] < mat.shape[m]:
                indices.append(ind0)
                values.append(int(mat[ind0]))
    return indices, np.array(values)


def get_all_same_connected(mat: ndarray, indices: List[Tuple[int, ...]]) -> List[Tuple[int, ...]]:
    start_n = len(indices)
    val = int(mat[indices[0]])

    all_same_neighbors = _find_same_value_neighbors(mat, indices, val)
    indices.extend(all_same_neighbors)

    if len(indices) > 0:
        indices = _unique_indices(indices)
        if len(indices) > start_n:
            indices = get_all_same_connected(mat, indices)

    return indices

def _find_same_value_neighbors(mat: ndarray, indices: List[Tuple[int, ...]], val: int) -> List[Tuple[int, ...]]:
    """Find neighbors with the same value."""
    all_same_neighbors = []

    for ind in indices:
        n_indices, n_values = get_neighbors(mat, ind)
        for n_ind, n_val in zip(n_indices, n_values):
            if n_val == val:
                all_same_neighbors.append(n_ind)

    return all_same_neighbors

def _unique_indices(indices: List[Tuple[int, ...]]) -> List[Tuple[int, ...]]:
    """Get unique indices from a list."""
    indices = np.unique(indices, axis=0)
    return [tuple(ind) for ind in indices]

def _check_neighbor_crossing(neighbors: ndarray, vax: int, vaxval: int) -> bool:
    """Check if a vertex can be crossed based on its neighbors.

    Args:
        neighbors: 2x2 array of neighbor values
        vax: Vertex axis (0 or 1)
        vaxval: Vertex axis value (0 or 1)

    Returns:
        True if crossing is valid, False otherwise
    """
    # Define neighbor indices to check
    nind0 = [0, 0]  # First neighbor to check
    nind0[vax] = vaxval
    nind1 = [1, 1]  # Second neighbor to check
    nind1[vax] = vaxval

    # Get neighbor values
    ne0 = neighbors[nind0[0]][nind0[1]]
    ne1 = neighbors[nind1[0]][nind1[1]]

    # Must have at least one block
    if ne0 != 1 and ne1 != 1:
        return False

    # Cannot cross blocked material
    if int(0.5 * (ne0 + 1)) == int(0.5 * (ne1 + 1)):
        return False

    return True

def _find_next_vertex(ord_verts: List[RegionVertex], verts: List[RegionVertex]):
    """Find the next vertex in an outline sequence.

    Args:
        ord_verts: The current ordered list of vertices in the outline
        verts: The remaining unordered vertices to choose from

    Returns:
        Tuple containing (success_flag, next_vertex)
    """
    # Get the last vertex in the ordered list to find its neighbor
    last_vertex = ord_verts[-1]

    # Try each possible axis (0=x, 1=y) and direction (-1 or 1)
    for vax in range(2):  # vax = vertex axis (0 or 1)
        for vdir in range(-1, 2, 2):  # vdir = vertex direction (-1 or 1)
            # Calculate the expected position of the next vertex
            next_ind = last_vertex.ind.copy()
            next_ind[vax] += vdir

            # Find a vertex at that position
            for rv in verts:  # rv = region vertex
                if rv.ind != next_ind:
                    continue

                # Prevent going back to the previous vertex
                if len(ord_verts) > 1 and rv.ind == ord_verts[-2].ind:
                    continue

                # Convert direction (-1,1) to index (0,1)
                vaxval = int(0.5 * (vdir + 1))

                # Check crossing conditions from the last vertex's perspective
                if not _check_neighbor_crossing(last_vertex.neighbors, vax, vaxval):
                    continue

                # Check crossing conditions from the candidate vertex's perspective
                if not _check_neighbor_crossing(rv.neighbors, vax, 1 - vaxval):  # Opposite side
                    continue

                # If you made it here, you found the next vertex!
                return True, rv

    # No valid next vertex found
    return False, None

def get_ordered_outline(verts: List[RegionVertex]) -> Tuple[List[RegionVertex], List[RegionVertex]]:
    """Order vertices to form an outline.

    Args:
        verts: List of vertices to order

    Returns:
        Tuple containing (ordered_vertices, remaining_vertices)
    """
    if not verts:
        return [], []

    # Start with the first vertex
    remaining_verts = verts.copy()
    ord_verts = [remaining_verts.pop(0)]

    # Continue adding vertices until no more can be added
    while remaining_verts:
        found_next, next_vertex = _find_next_vertex(ord_verts, remaining_verts)
        if found_next:
            ord_verts.append(next_vertex)
            remaining_verts.remove(next_vertex)
        else:
            break

    return ord_verts, remaining_verts

def get_sublist_of_ordered_verts(verts: List[RegionVertex]) -> Tuple[List[RegionVertex], List[RegionVertex], bool]:
    """Get a sublist of ordered vertices and check if the outline is closed.

    Args:
        verts: List of vertices to order

    Returns:
        Tuple containing (ordered_vertices, remaining_vertices, is_closed)
    """
    # Get ordered vertices
    ord_verts, remaining_verts = get_ordered_outline(verts)

    # Check if outline is closed
    closed = False

    # Need at least 4 vertices to form a closed outline
    if len(ord_verts) > 3:
        # Get first and last vertices
        start_ind = np.array(ord_verts[0].ind.copy())
        end_ind = np.array(ord_verts[-1].ind.copy())

        # Calculate difference between first and last vertex
        diff_ind = start_ind - end_ind

        # Check if they differ in only one axis
        if len(np.argwhere(diff_ind == 0)) == 1:
            # Get the axis where they differ
            vax = np.argwhere(diff_ind != 0)[0][0]

            # Check if the difference is only one step
            if abs(diff_ind[vax]) == 1:
                # Get the direction
                vdir = diff_ind[vax]

                # Check crossing conditions
                p_neig = ord_verts[-1].neighbors
                vaxval = int(0.5 * (vdir + 1))

                # Define neighbor indices to check
                nind0 = [0, 0]
                nind0[vax] = vaxval
                nind1 = [1, 1]
                nind1[vax] = vaxval

                # Get neighbor values
                ne0 = p_neig[nind0[0]][nind0[1]]
                ne1 = p_neig[nind1[0]][nind1[1]]

                # Check if there's at least one block and we're not crossing blocked material
                if (ne0 == 1 or ne1 == 1) and (int(0.5 * (ne0 + 1)) != int(0.5 * (ne1 + 1))):
                    closed = True

    return ord_verts, remaining_verts, closed

def is_connected_to_fixed_side(indices: ndarray, mat: ndarray,
                              fixed_sides: List[FixedSide]) -> bool:
    if _is_directly_connected_to_fixed_side(indices, mat, fixed_sides):
        return True

    neighbors = get_indices_of_same_neighbors(indices, mat)

    if len(neighbors) > 0:
        new_indices = _combine_indices(indices, neighbors)
        if len(new_indices) > len(indices):
            return is_connected_to_fixed_side(new_indices, mat, fixed_sides)

    return False

def _is_directly_connected_to_fixed_side(indices: ndarray, mat: ndarray,
                                        fixed_sides: List[FixedSide]) -> bool:
    """Check if any index is directly connected to a fixed side."""
    val = mat[tuple(indices[0])]
    d = len(mat)

    for ind in indices:
        for side in fixed_sides:
            if ind[side.ax] == 0 and side.dir == 0:
                return True
            elif ind[side.ax] == d - 1 and side.dir == 1:
                return True

    return False

def get_indices_of_same_neighbors(indices: List[List[int]], mat: ndarray) -> np.ndarray:
    d = len(mat)
    val = mat[tuple(indices[0])]
    neighbors = []

    for ind in indices:
        for ax in range(3):
            for dir in range(2):
                dir = 2 * dir - 1
                ind2 = _get_neighbor_index(ind, ax, dir)

                if (d > ind2[ax] >= 0) and mat[tuple(ind2)] == val:
                    neighbors.append(ind2)

    if neighbors:
        return _unique_neighbors(neighbors)
    return np.array([])

def _get_neighbor_index(ind: List[int], ax: int, dir: int) -> List[int]:
    """Get index of neighbor in specified direction."""
    ind2 = ind.copy()
    ind2[ax] = ind2[ax] + dir
    return ind2

def _unique_neighbors(neighbors: List[List[int]]) -> np.ndarray:
    """Get unique neighbors."""
    neighbors = np.array(neighbors)
    return np.unique(neighbors, axis=0)

def _combine_indices(indices: ndarray, neighbors: ndarray) -> ndarray:
    """Combine and uniquify two sets of indices."""
    new_indices = np.concatenate([indices, neighbors])
    return np.unique(new_indices, axis=0)

def _get_region_outline(reg_inds: List[List[int]], lay_mat: ndarray,
                       fixed_neighbors: List[bool], n: int) -> List[RegionVertex]:
    # also duplicate vertices on diagonal
    reg_verts = []
    for i in range(lay_mat.shape[0] + 1):
        for j in range(lay_mat.shape[1] + 1):
            ind = [i, j]
            neigbors, neighbor_values = _get_neighbors_2d(ind, reg_inds, lay_mat, n)
            neigbors = np.array(neigbors)
            if np.any(neigbors.flatten() == 0) and not np.all(neigbors.flatten() == 0): # some but not all region neighbors
                dia1 = neigbors[0][1] == neigbors[1][0]
                dia2 = neigbors[0][0] == neigbors[1][1]
                if np.sum(neigbors.flatten() == 0) == 2 and np.sum(neigbors.flatten() == 1) == 2 and dia1 and dia2: # diagonal detected
                    other_indices = np.argwhere(neigbors == 0)
                    for oind in other_indices:
                        oneigbors = copy.deepcopy(neigbors)
                        oneigbors[tuple(oind)] = 1
                        oneigbors = np.array(oneigbors)
                        reg_verts.append(RegionVertex(ind, ind, oneigbors, neighbor_values, dia=True))
                else: # normal situation
                    reg_verts.append(RegionVertex(ind, ind, neigbors, neighbor_values))
    return reg_verts

def _get_neighbors_2d(ind: List[int], reg_inds: List[List[int]],
                     lay_mat: ndarray, n: int) -> Tuple[List[List[int]], List[List[int]]]:
    # 0 = in region
    # 1 = outside region, block
    # 2 = outside region, free
    in_out = []
    values = []
    for add0 in range(-1, 1, 1):
        temp = []
        temp2 = []
        for add1 in range(-1, 1, 1):

            # Define neighbor index to test
            nind = [ind[0] + add0, ind[1] + add1]

            # FIND TYPE
            type = -1
            val = None
            # Check if this index is in the list of region-included indices
            for rind in reg_inds:
                if rind[0] == nind[0] and rind[1] == nind[1]:
                    type = 0  # in region
                    break
            if type != 0:
                # If there are out of bound indices they are free
                if np.any(np.array(nind) < 0) or nind[0] >= lay_mat.shape[0] or nind[1] >= lay_mat.shape[1]:
                    type = 1  # free
                    val = -1
                elif lay_mat[tuple(nind)] < 0:
                    type = 1  # free
                else: type = 1  # blocked

            if val is None:
                val = lay_mat[tuple(nind)]

            temp.append(type)
            temp2.append(val)
        in_out.append(temp)
        values.append(temp2)
    return in_out, values

def _is_connected_to_fixed_side_2d(inds: List[List[int]], fixed_sides: List[FixedSide],
                                  ax: int, dim: int) -> bool:
    connected = False
    for side in fixed_sides:
        fax2d = [0, 0, 0]
        fax2d[side.ax] = 1
        fax2d.pop(ax)
        fax2d = fax2d.index(1)
        for ind in inds:
            if ind[fax2d] == side.dir * (dim - 1):
                connected = True
                break
        if connected: break
    return connected

def _get_same_neighbors_2d(mat2: ndarray, inds: List[List[int]], val: int) -> List[List[int]]:
    new_inds = list(inds)
    for ind in inds:
        for ax in range(2):
            for dir in range(-1, 2, 2):
                ind2 = ind.copy()
                ind2[ax] += dir
                if ind2[ax] >= 0 and ind2[ax] < mat2.shape[ax]:
                    val2 = mat2[tuple(ind2)]
                    if val2 != val: continue
                    unique = True
                    for ind3 in new_inds:
                        if ind2[0] == ind3[0] and ind2[1] == ind3[1]:
                            unique = False
                            break
                    if unique: new_inds.append(ind2)
    if len(new_inds) > len(inds):
        new_inds = _get_same_neighbors_2d(mat2, new_inds, val)
    return new_inds

def _layer_mat(mat3d: ndarray, ax: int, dim: int, lay_num: int) -> np.ndarray:
    mat2d = np.ndarray(shape=(dim, dim), dtype=int)
    for i in range(dim):
        for j in range(dim):
            ind = [i, j]
            ind.insert(ax, lay_num)
            mat2d[i][j] = int(mat3d[tuple(ind)])
    return mat2d


def get_breakable_voxels(mat: ndarray, fixed_sides: List[List[FixedSide]],
                        sax: int, n: int) -> Tuple[bool, List[List[int]], List[List[int]]]:
    breakable = False
    outline_indices = []
    voxel_indices = []
    dim = len(mat)
    gax = fixed_sides[0].ax  # grain axis

    if gax != sax:  # if grain direction does not equal to the sliding direction
        paxes = _get_perpendicular_axes(gax)

        for pax in paxes:
            potentially_fragile_reg_inds = _find_potentially_fragile_regions(mat, fixed_sides, pax, dim, n)

            for lay_num in range(dim):
                lay_mat = _layer_mat(mat, pax, dim, lay_num)

                for reg_inds in potentially_fragile_reg_inds[lay_num]:
                    fixed_neighbors = _check_fixed_neighbors(mat, reg_inds, potentially_fragile_reg_inds,
                                                           lay_num, pax, dim, n)

                    if fixed_neighbors[0] == False or fixed_neighbors[1] == False:
                        breakable = True

                        # Add voxel indices
                        voxel_indices.extend(_get_3d_indices(reg_inds, pax, lay_num))

                        # Get and order region outline
                        outline = _get_region_outline(reg_inds, lay_mat, fixed_neighbors, n)
                        outline, _ = get_ordered_outline(outline)
                        outline.append(outline[0])

                        # Add outline indices
                        outline_indices.extend(_get_outline_indices(outline, pax, lay_num))

    return breakable, outline_indices, voxel_indices

def _get_perpendicular_axes(axis: int) -> List[int]:
    """Get axes perpendicular to the given axis."""
    paxes = [0, 1, 2]
    paxes.remove(axis)
    return paxes

def _find_potentially_fragile_regions(mat: ndarray, fixed_sides: List[List[FixedSide]], pax: int, dim: int, n: int) -> List[List[List[List[int]]]]:
    """Find potentially fragile regions in each layer."""
    potentially_fragile_reg_inds = []

    for lay_num in range(dim):
        temp = []
        lay_mat = _layer_mat(mat, pax, dim, lay_num)

        for reg_num in range(dim * dim):  # region number
            # Get indices of a region
            inds = np.argwhere((lay_mat != -1) & (lay_mat == n))
            if len(inds) == 0:
                break

            reg_inds = _get_same_neighbors_2d(lay_mat, [inds[0]], n)

            # Check if any item in this region is connected to a fixed side
            fixed = _is_connected_to_fixed_side_2d(reg_inds, fixed_sides, pax, dim)

            if not fixed:
                temp.append(reg_inds)

            # Overwrite detected region in original matrix
            for reg_ind in reg_inds:
                lay_mat[tuple(reg_ind)] = -1

        potentially_fragile_reg_inds.append(temp)

    return potentially_fragile_reg_inds

def _check_fixed_neighbors(mat: ndarray, reg_inds: List[List[int]],
                          potentially_fragile_reg_inds: List[List[List[List[int]]]],
                          lay_num: int, pax: int, dim: int, n: int) -> List[bool]:
    """Check if region has fixed neighbors in both directions."""
    fixed_neighbors = [False, False]

    for reg_ind in reg_inds:
        # get 3d index
        ind3d = list(reg_ind.copy())
        ind3d.insert(pax, lay_num)

        for dir in range(-1, 2, 2):  # -1/1
            # check neighbor in direction
            ind3d_dir = ind3d.copy()
            ind3d_dir[pax] += dir

            if 0 <= ind3d_dir[pax] < dim:
                # Is there any material at all?
                val = mat[tuple(ind3d_dir)]

                if val == n:  # There is material
                    # Is this material in the list of potentially fragile or not?
                    if not _is_in_fragile_region(ind3d_dir, potentially_fragile_reg_inds,
                                               lay_num + dir, pax):
                        fixed_neighbors[int((dir + 1) / 2)] = True

    return fixed_neighbors

def _is_in_fragile_region(ind3d: List[int], potentially_fragile_reg_inds: List[List[List[List[int]]]],
                         layer: int, pax: int) -> bool:
    """Check if a 3D index is in a potentially fragile region."""
    if layer < 0 or layer >= len(potentially_fragile_reg_inds):
        return False

    ind2d = ind3d.copy()
    ind2d.pop(pax)

    for dir_reg_inds in potentially_fragile_reg_inds[layer]:
        for dir_ind in dir_reg_inds:
            if dir_ind[0] == ind2d[0] and dir_ind[1] == ind2d[1]:
                return True

    return False

def _get_3d_indices(reg_inds: List[List[int]], pax: int, lay_num: int) -> List[List[int]]:
    """Convert 2D region indices to 3D voxel indices."""
    indices = []
    for ind in reg_inds:
        ind3d = list(ind.copy())
        ind3d.insert(pax, lay_num)
        indices.append(ind3d)
    return indices

def _get_outline_indices(outline: List[RegionVertex], pax: int, lay_num: int) -> List[List[int]]:
    """Get 3D indices for outline vertices."""
    indices = []
    for dir in range(0, 2):
        for i in range(len(outline) - 1):
            for j in range(2):
                oind = outline[i + j].ind.copy()
                oind.insert(pax, lay_num)
                oind[pax] += dir
                indices.append(oind)
    return indices


def get_friction_and_contact_areas(mat: ndarray, slides: List[List[int]],
                                  fixed_sides: List[List[FixedSide]], n: int) -> Tuple[int, List[List], int, List[List]]:
    if not slides:
        return -1, [], -1, []

    friction = 0
    contact = 0
    ffaces = []
    cfaces = []

    other_fixed_sides = _get_other_fixed_sides(fixed_sides, n)
    friction_axes = _get_friction_axes(slides)

    # Check neighbors for voxels
    indices = np.argwhere(mat == n)
    for ind in indices:
        for ax in range(3):
            for dir in range(2):
                nind = ind.copy()
                nind[ax] += 2 * dir - 1

                if _is_contact_face(nind, mat, other_fixed_sides, ax, dir, n):
                    contact += 1
                    find = ind.copy()
                    find[ax] += dir
                    cfaces.append([ax, list(find)])

                    if ax in friction_axes:
                        friction += 1
                        ffaces.append([ax, list(find)])

    # Check neighbors for fixed sides
    _check_fixed_side_contacts(mat, fixed_sides[n], friction_axes, n, contact, cfaces, friction, ffaces)

    return friction, ffaces, contact, cfaces

def _get_other_fixed_sides(fixed_sides: List[List[FixedSide]], n: int) -> List[FixedSide]:
    """Get fixed sides from other components."""
    other_fixed_sides = []
    for n2 in range(len(fixed_sides)):
        if n == n2:
            continue
        other_fixed_sides.extend(fixed_sides[n2])
    return other_fixed_sides

def _get_friction_axes(slides: List[List[int]]) -> List[int]:
    """Get axes that contribute to friction."""
    friction_axes = [0, 1, 2]
    bad_axes = [item[0] for item in slides]
    return [x for x in friction_axes if x not in bad_axes]

def _is_contact_face(nind: ndarray, mat: ndarray, other_fixed_sides: List[FixedSide],
                    ax: int, dir: int, n: int) -> bool:
    """Check if a face is a contact face."""
    if nind[ax] < 0:
        return FixedSide(ax, 0).unique(other_fixed_sides)
    elif nind[ax] >= len(mat):
        return FixedSide(ax, 1).unique(other_fixed_sides)
    else:
        return mat[tuple(nind)] != n

def _check_fixed_side_contacts(mat: ndarray, fixed_sides: List[FixedSide], friction_axes: List[int],
                              n: int, contact: int, cfaces: List[List], friction: int, ffaces: List[List]) -> None:
    """Check contacts for fixed sides."""
    dim = len(mat)
    for side in fixed_sides:
        for i in range(dim):
            for j in range(dim):
                nind = [i, j]
                axind = side.dir * (dim - 1)
                nind.insert(side.ax, axind)

                if mat[tuple(nind)] != n:  # neighboring another timber
                    contact += 1
                    find = nind.copy()
                    find[side.ax] += side.dir
                    cfaces.append([side.ax, list(find)])

                    if side.ax in friction_axes:
                        friction += 1
                        ffaces.append([side.ax, list(find)])

def add_fixed_sides(mat: ndarray, fixed_sides: List[List[FixedSide]], add: int = 0) -> np.ndarray:
    """Add fixed sides to the matrix."""
    dim = len(mat)

    # Prepare padding configuration
    pad_loc, pad_val = _prepare_padding_config(fixed_sides, add)

    # Pad the matrix
    mat = np.pad(mat, pad_loc, 'constant', constant_values=pad_val)

    # Handle corner cases
    mat = _handle_corners(mat, fixed_sides, dim)

    return mat

def _prepare_padding_config(fixed_sides: List[List[FixedSide]], add: int) -> Tuple[tuple, tuple]:
    """Prepare padding configuration for fixed sides."""
    pad_loc = [[0, 0], [0, 0], [0, 0]]
    pad_val = [[-1, -1], [-1, -1], [-1, -1]]

    for n in range(len(fixed_sides)):
        for side in fixed_sides[n]:
            pad_loc[side.ax][side.dir] = 1
            pad_val[side.ax][side.dir] = n + add

    return tuple(map(tuple, pad_loc)), tuple(map(tuple, pad_val))

def _handle_corners(mat: ndarray, fixed_sides: List[List[FixedSide]], dim: int) -> ndarray:
    """Handle corner cases when adding fixed sides."""
    for fixed_sides_1 in fixed_sides:
        for fixed_sides_2 in fixed_sides:
            for side in fixed_sides_1:
                for side2 in fixed_sides_2:
                    if side.ax == side2.ax:
                        continue

                    for i in range(dim + 2):
                        ind = [i, i, i]
                        ind[side.ax] = side.dir * (mat.shape[side.ax] - 1)
                        ind[side2.ax] = side2.dir * (mat.shape[side2.ax] - 1)

                        try:
                            mat[tuple(ind)] = -1
                        except:
                            pass  # Index out of bounds, skip

    return mat

def get_chessboard_vertices(mat: ndarray, ax: int, noc: int, n: int) -> Tuple[bool, List[List[int]]]:
    chess = False
    dim = len(mat)
    verts = []

    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                ind3d = [i, j, k]
                ind2d = _get_2d_index(ind3d, ax)

                if ind2d[0] < 1 or ind2d[1] < 1:
                    continue

                neighbors = _get_neighbor_values(mat, ind3d, ax)
                flat_neighbors = np.array(neighbors).flatten()

                if _is_chessboard_pattern(flat_neighbors, neighbors, n):
                    chess = True
                    verts.append(ind3d)

    return chess, verts

def _get_2d_index(ind3d: List[int], ax: int) -> List[int]:
    """Extract 2D index from 3D index by removing the specified axis."""
    ind2d = ind3d.copy()
    ind2d.pop(ax)
    return ind2d

def _get_neighbor_values(mat: ndarray, ind3d: List[int], ax: int) -> List[List[ndarray]]:
    """Get values of neighboring cells in a 2x2 grid."""
    ind2d = _get_2d_index(ind3d, ax)
    neighbors = []

    for x in range(-1, 1, 1):
        temp = []
        for y in range(-1, 1, 1):
            nind = ind2d.copy()
            nind[0] += x
            nind[1] += y
            nind3d = nind.copy()
            nind3d.insert(ax, ind3d[ax])
            val = mat[tuple(nind3d)]
            temp.append(val)
        neighbors.append(temp)

    return neighbors

def _is_chessboard_pattern(flat_neighbors: List[int], neighbors, n: int) -> bool:
    """Check if the neighbors form a chessboard pattern."""
    cnt = np.sum(flat_neighbors == n)
    if cnt == 2:
        # check diagonal
        if neighbors[0][1] == neighbors[1][0] and neighbors[0][0] == neighbors[1][1]:
            return True

def is_fab_direction_ok(mat: ndarray, ax: int, n: int) -> Tuple[bool, int]:
    fab_dir = 1
    dim = len(mat)
    for dir in range(2):
        is_ok = True
        for i in range(dim):
            for j in range(dim):
                found_first_same = False
                for k in range(dim):
                    if dir == 0: k = dim - k - 1
                    ind = [i, j]
                    ind.insert(ax, k)
                    val = mat[tuple(ind)]
                    if val == n: found_first_same = True
                    elif found_first_same: is_ok = False; break
                if not is_ok: break
            if not is_ok: break
        if is_ok:
            fab_dir = dir
            break
    return is_ok, fab_dir

def open_matrix(mat: ndarray, sax: int, noc: int) -> np.ndarray:
    # Pad matrix by correct number of rows top and bottom
    dim = len(mat)
    pad_loc = [[0, 0], [0, 0], [0, 0]]
    pad_loc[sax] = [0, noc - 1]
    pad_val = [[-1, -1], [-1, -1], [-1, -1]]
    pad_loc = tuple(map(tuple, pad_loc))
    pad_val = tuple(map(tuple, pad_val))
    mat = np.pad(mat, pad_loc, 'constant', constant_values=pad_val)

    # Move integers one step at the time
    for i in range(noc - 1, 0, -1):
        inds = np.argwhere(mat == i)
        for ind in inds:
            mat[tuple(ind)] = -1
        for ind in inds:
            ind[sax] += i
            mat[tuple(ind)] = i
    return mat

def flood_all_nonneg(mat: ndarray, floodval: int) -> np.ndarray:
    inds = np.argwhere(mat == floodval)
    start_len = len(inds)
    for ind in inds:
        for ax in range(3):
            for dir in range(-1, 2, 2):
                # define neighbor index
                ind2 = np.copy(ind)
                ind2[ax] += dir
                # within bounds?
                if ind2[ax] < 0: continue
                if ind2[ax] >= mat.shape[ax]: continue
                # relevant value?
                val = mat[tuple(ind2)]
                if val < 0 or val == floodval: continue
                # overwrite
                mat[tuple(ind2)] = floodval
    end_len = len(np.argwhere(mat == floodval))
    if end_len > start_len:
        mat = flood_all_nonneg(mat, floodval)
    return mat

def _check_full_connectivity(mat: ndarray, floodval: int) -> bool:
    """Check if all components are connected."""
    # Recursively add all positive neighbors
    mat_conn = flood_all_nonneg(mat, floodval)

    # Get the count of all uncovered voxels
    uncovered_inds = np.argwhere((mat_conn != floodval) & (mat_conn >= 0))

    return len(uncovered_inds) == 0

def _check_fixed_side_connectivity(mat: ndarray, level: int, noc: int, floodval: int) -> bool:
    """Check if there are enough voxels connecting to each fixed side."""
    for n in range(noc):
        if n == level:
            continue

        mat_conn = np.copy(mat)
        mat_conn[mat_conn == n + 10] = floodval

        for n2 in range(noc):
            if n2 == level or n2 == n:
                continue
            mat_conn[mat_conn == n2 + 10] = -1

        start_len = len(np.argwhere(mat_conn == floodval))

        # Recursively add all positive neighbors
        mat_conn = flood_all_nonneg(mat_conn, floodval)

        end_len = len(np.argwhere(mat_conn == floodval))

        if end_len - start_len < 3:
            return False

    return True

def _check_no_bridging(mat: ndarray, level: int, noc: int, dim: int, floodval: int) -> bool:
    """Check that there's no bridging between fixed sides."""
    for n in range(noc):
        if n == level:
            continue

        inds = np.argwhere(mat == n + 10)

        if len(inds) > dim * dim * dim:  # i.e. if there are more than 1 fixed side
            mat_conn = np.copy(mat)
            mat_conn[tuple(inds[0])] = floodval

            for n2 in range(noc):
                if n2 == level or n2 == n:
                    continue
                mat_conn[mat_conn == n2 + 10] = -1

            # Recursively add all positive neighbors
            mat_conn = flood_all_nonneg(mat_conn, floodval)

            for ind in inds:
                if mat_conn[tuple(ind)] != floodval:
                    return False

    return True

def is_potentially_connected(mat: ndarray, dim: int, noc: int, level: int) -> bool:
    """Check if the joint is potentially connected."""
    # Prepare matrix
    mat_copy = np.copy(mat)
    mat_copy[mat_copy == level] = -1
    mat_copy[mat_copy == level + 10] = -1

    floodval = 99

    # Prepare flood starting points
    for n in range(noc):
        if n != level:
            mat_copy[mat_copy == n + 10] = floodval

    # Check full connectivity
    if not _check_full_connectivity(mat_copy, floodval):
        return False

    # Check fixed side connectivity
    if not _check_fixed_side_connectivity(mat_copy, level, noc, floodval):
        return False

    # Check no bridging
    if not _check_no_bridging(mat_copy, level, noc, dim, floodval):
        return False

    return True

def get_region_outline_vertices(reg_inds: List[List[int]], lay_mat: ndarray,
                               org_lay_mat: ndarray, pad_loc: List[List[int]],
                               n: int) -> List[RegionVertex]:
    # also duplicate vertices on diagonal
    reg_verts = []
    for i in range(lay_mat.shape[0] + 1):
        for j in range(lay_mat.shape[1] + 1):
            ind = [i, j]
            neigbors, neighbor_values = get_neighbors_in_out(ind, reg_inds, lay_mat, org_lay_mat, n)
            neigbors = np.array(neigbors)
            abs_ind = ind.copy()
            ind[0] -= pad_loc[0][0]
            ind[1] -= pad_loc[1][0]
            if np.any(neigbors.flatten() == 0) and not np.all(neigbors.flatten() == 0):  # some but not all region neighbors
                dia1 = neigbors[0][1] == neigbors[1][0]
                dia2 = neigbors[0][0] == neigbors[1][1]
                if np.sum(neigbors.flatten() == 0) == 2 and np.sum(neigbors.flatten() == 1) == 2 and dia1 and dia2:  # diagonal detected
                    other_indices = np.argwhere(neigbors == 0)
                    for oind in other_indices:
                        oneigbors = copy.deepcopy(neigbors)
                        oneigbors[tuple(oind)] = 1
                        oneigbors = np.array(oneigbors)
                        reg_verts.append(RegionVertex(ind, abs_ind, oneigbors, neighbor_values, dia=True))
                else:  # normal situation
                    if has_minus_one_neighbor(ind, lay_mat): mon = True
                    else: mon = False
                    reg_verts.append(RegionVertex(ind, abs_ind, neigbors, neighbor_values, minus_one_neighbor=mon))
    return reg_verts

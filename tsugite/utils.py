# NOTE: the functions moved here are more of helper math/linear algebra functions
import numpy as np
import math

class Utils:

    @staticmethod
    def normalize(v):
        norm = np.linalg.norm(v)
        if norm == 0: return v
        else: return v / norm

    @staticmethod
    def unitize(v):
        uv = v/np.linalg.norm(v)
        return uv

    @staticmethod
    def angle_between(vector_1, vector_2, direction=False):
        unit_vector_1 = Utils.unitize(vector_1)
        unit_vector_2 = Utils.unitize(vector_2)
        v_dot_product = np.dot(unit_vector_1, unit_vector_2)

        if direction:
            angle = np.arctan2(np.linalg.det([unit_vector_1, unit_vector_2]), v_dot_product)
            return math.degrees(angle)
        else:
            angle = np.arccos(v_dot_product)
            return angle
        
    @staticmethod
    def rotate_vector_around_axis(vec=[3,5,0], axis=[4,4,1], theta=1.2): #example values
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
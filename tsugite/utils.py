# NOTE: the functions moved here are more of helper math/linear algebra functions
import numpy as np
import math

class Utils:

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
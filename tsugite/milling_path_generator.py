# Create a new file: milling_path_generator.py
from typing import List, Tuple, Any, Optional
import numpy as np
import math
import copy
import utils as Utils
from fabrication import MillVertex  # Create this class to hold milling vertex data

class MillingPathGenerator:
    """Handles the generation of milling paths for fabrication."""

    def __init__(self, joint):
        self.joint = joint
        self.sax = joint.sax
        self.dim = joint.dim
        self.voxel_sizes = joint.voxel_sizes
        self.pos_vecs = joint.pos_vecs
        self.fab = joint.fab
        self.jverts = joint.jverts
        self.vertex_no_info = joint.vertex_no_info
        self.fixed = joint.fixed
        self.incremental = joint.incremental
        self.ratio = joint.ratio

    def generate_milling_paths(self, n):
        """Generate milling path vertices for component n."""
        vertices = []
        milling_vertices = []

        # Calculate depth constants
        min_vox_size = np.min(self.voxel_sizes)
        if min_vox_size < self.fab.vdia:
            print("Could not generate milling path. The milling bit is too large.")
            return np.array([], dtype=np.float32), []

        no_z, dep = self._calculate_depth_parameters()
        neighbor_vectors, neighbor_vectors_a, neighbor_vectors_b = self._calculate_neighbor_vectors()

        # Process each layer
        for lay_num in range(self.dim):
            vertices, milling_vertices = self._process_layer(
                lay_num, n, no_z, dep,
                neighbor_vectors, neighbor_vectors_a, neighbor_vectors_b,
                vertices, milling_vertices
            )

        # Add end point if there are milling vertices
        if milling_vertices:
            end_verts, end_mverts = self._get_milling_end_points(n, milling_vertices[-1].pt[self.sax])
            vertices.extend(end_verts)
            milling_vertices.extend(end_mverts)

        return np.array(vertices, dtype=np.float32), milling_vertices

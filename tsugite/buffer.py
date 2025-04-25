import OpenGL.GL as GL  # imports start with GL
import numpy as np
from PIL import Image
from ctypes import c_void_p as buffer_offset

import sys

class ElementProperties:
    def __init__(self, draw_type, count, start_index, n):
        self.draw_type = draw_type
        self.count = count
        self.start_index = start_index
        self.n = n

class Buffer:
    def __init__(self, pjoint):
        self.pjoint = pjoint
        self.VBO = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.VBO)

        self.EBO = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.EBO)

        self.vertex_no_info = 8

        # Load textures
        self._load_textures()

    def _load_textures(self):
        """Load texture images from files."""
        # Load end grain texture
        image = Image.open("textures/end_grain.jpg")
        self.img_data = np.array(list(image.getdata()), np.uint8)

        # Load friction area texture
        image = Image.open("textures/friction_area.jpg")
        self.img_data_fric = np.array(list(image.getdata()), np.uint8)

        # Load contact area texture
        image = Image.open("textures/contact_area.jpg")
        self.img_data_cont = np.array(list(image.getdata()), np.uint8)

    def buffer_vertices(self):
        """Buffer vertex data to GPU and set up vertex attributes and textures."""
        try:
            self._buffer_vertex_data()
            self._setup_vertex_attributes()
            self._setup_textures()
        except Exception as e:
            print(f"ERROR IN ARRAY BUFFER WRAPPER: {str(e)}")

    def _buffer_vertex_data(self):
        """Buffer vertex data to the GPU."""
        # 6 bytes to avoid buffer overflow
        cnt = 6 * len(self.pjoint.vertices)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, cnt, self.pjoint.vertices, GL.GL_DYNAMIC_DRAW)

    def _setup_vertex_attributes(self):
        """Configure vertex attribute pointers."""
        # Calculate stride and offsets
        stride = int(8 * 32 / 8)  # 8 floats * 4 bytes per float
        color_offset = int(3 * 32 / 8)  # Skip 3 position values
        texture_offset = int(6 * 32 / 8)  # Skip 3 position + 3 color values

        # Position attribute (location 0)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, stride, buffer_offset(0))
        GL.glEnableVertexAttribArray(0)

        # Color attribute (location 1)
        GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, GL.GL_FALSE, stride, buffer_offset(color_offset))
        GL.glEnableVertexAttribArray(1)

        # Texture coordinate attribute (location 2)
        GL.glVertexAttribPointer(2, 2, GL.GL_FLOAT, GL.GL_FALSE, stride, buffer_offset(texture_offset))
        GL.glEnableVertexAttribArray(2)

    def _setup_textures(self):
        """Set up texture parameters and load texture data to GPU."""
        # Generate texture objects
        GL.glGenTextures(3)

        # Set texture parameters (applies to currently bound texture)
        self._set_texture_parameters()

        # Set up end grain texture (texture unit 0)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB, 400, 400, 0, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, self.img_data)

        # Set up friction area texture (texture unit 1)
        GL.glActiveTexture(GL.GL_TEXTURE1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 1)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB, 400, 400, 0, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, self.img_data_fric)

        # Set up contact area texture (texture unit 2)
        GL.glActiveTexture(GL.GL_TEXTURE2)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 2)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB, 400, 400, 0, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, self.img_data_cont)

    def _set_texture_parameters(self):
        """Set common texture parameters."""
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_REPEAT)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)

    def buffer_indices(self):
        """Buffer index data to the GPU."""
        try:
            # 4 bytes per index (uint32)
            cnt = 4 * len(self.pjoint.indices)
            GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, cnt, self.pjoint.indices, GL.GL_DYNAMIC_DRAW)
        except Exception as e:
            print(f"ERROR IN ELEMENT BUFFER WRAPPER: {str(e)}")

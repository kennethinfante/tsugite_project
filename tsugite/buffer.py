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
    def __init__(self,parent):
        self.parent = parent
        self.VBO = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.VBO)

        self.EBO = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.EBO)
        
        self.vertex_no_info = 8

        image = Image.open("textures/end_grain.jpg")
        self.img_data = np.array(list(image.getdata()), np.uint8)
        
        image = Image.open("textures/friction_area.jpg")
        
        self.img_data_fric = np.array(list(image.getdata()), np.uint8)
        image = Image.open("textures/contact_area.jpg")
        
        self.img_data_cont = np.array(list(image.getdata()), np.uint8)

    def buffer_vertices(self):
        try:
            # 6 bytes to avoid buffer overflow
            cnt = 6*len(self.parent.vertices)

            # print("parent", sys.getsizeof(self.parent.vertices[10]))
            # print("parent dtype", self.parent.vertices.dtype)
            # print("len", len(self.parent.vertices)) #3512
            # print("6 * len", cnt) #21072
            # print("nbytes", self.parent.vertices.nbytes) #14048, because 4 bytes each (32 bits/8),

            # STREAM - The data store contents will be modified once and used at most a few times.
            # STATIC - The data store contents will be modified once and used many times.
            # DYNAMIC - The data store contents will be modified repeatedly and used many times.
            GL.glBufferData(GL.GL_ARRAY_BUFFER, cnt, self.parent.vertices, GL.GL_DYNAMIC_DRAW)
        except:
            print("--------------------------ERROR IN ARRAY BUFFER WRAPPER -------------------------------------")

        # for each array of len 8, do the ff
        stride = int(8*32/8)
        color_offset = int(3*32/8) # first 3 values are for position
        texture_offset = int(6*32/8) # first 3 for position, next 3 for color
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, stride, buffer_offset(0)) #position
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, GL.GL_FALSE, stride, buffer_offset(color_offset)) #color
        GL.glEnableVertexAttribArray(1)
        GL.glVertexAttribPointer(2, 2, GL.GL_FLOAT, GL.GL_FALSE, stride, buffer_offset(texture_offset)) #texture
        GL.glEnableVertexAttribArray(2)
        GL.glGenTextures(3)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_REPEAT)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB, 400, 400, 0, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, self.img_data)
        GL.glActiveTexture(GL.GL_TEXTURE1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 1)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB, 400, 400, 0, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, self.img_data_fric)
        GL.glActiveTexture(GL.GL_TEXTURE2)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 2)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB, 400, 400, 0, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, self.img_data_cont)

    def buffer_indices(self):
        # 4 bytes each number
        cnt = 4*len(self.parent.indices)
        GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, cnt, self.parent.indices, GL.GL_DYNAMIC_DRAW)

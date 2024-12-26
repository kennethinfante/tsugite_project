import OpenGL.GL as GL  # imports start with GL
import numpy as np
from PIL import Image
from ctypes import c_void_p as buffer_offset

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
            cnt = 6*len(self.parent.vertices)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, cnt, self.parent.vertices, GL.GL_DYNAMIC_DRAW)
        except:
            print("--------------------------ERROR IN ARRAY BUFFER WRAPPER -------------------------------------")

        # vertex attribute pointers
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 32, buffer_offset(0)) #position
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, GL.GL_FALSE, 32, buffer_offset(12)) #color
        GL.glEnableVertexAttribArray(1)
        GL.glVertexAttribPointer(2, 2, GL.GL_FLOAT, GL.GL_FALSE, 32, buffer_offset(24)) #texture
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
        cnt = 4*len(self.parent.indices)
        GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, cnt, self.parent.indices, GL.GL_DYNAMIC_DRAW)

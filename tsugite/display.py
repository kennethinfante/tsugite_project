import numpy as np
import pyrr
from ctypes import c_void_p as buffer_offset

import OpenGL.GL as GL  # imports start with GL
import OpenGL.GL.shaders as GLSH
from OpenGL.GLU import gluPerspective

from buffer import ElementProperties
from view_settings import ViewSettings

class Display:
    def __init__(self, parent, joint):
        self.parent = parent
        self.joint = joint
        self.view = ViewSettings()
        self.create_color_shaders()
        self.create_texture_shaders()

    # def resize( w, h ):
    #     width = w
    #     height = h
    #     aspect = w/h
    #     # glViewport( 0, 0, width, height )
    #     print("resize here")
    #     glViewport( 0, 0, height*aspect, height )



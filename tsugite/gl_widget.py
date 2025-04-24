import time
from math import tan, pi

import PyQt5.QtWidgets as qtw
import PyQt5.QtGui as qtg
import PyQt5.QtCore as qtc
import PyQt5.QtOpenGL as qgl

import OpenGL.GL as GL  # imports start with GL

from joint import Joint
from geometries import Geometries
from display import Display

class GLWidget(qgl.QGLWidget):
    def __init__(self, main_window=None, *args):
        super().__init__(main_window, *args)

        self.parent = main_window
        # self.setMinimumSize(800, 800)
        self.setMouseTracking(True)
        self.click_time = time.time()
        self.x = 0
        self.y = 0

    def print_system_info(self):
        vendor = GL.glGetString(GL.GL_VENDOR).decode('utf-8')
        renderer = GL.glGetString(GL.GL_RENDERER).decode('utf-8')
        opengl = GL.glGetString(GL.GL_VERSION).decode('utf-8')
        glsl = GL.glGetString(GL.GL_SHADING_LANGUAGE_VERSION).decode('utf-8')

        result = ''.join(['Vendor: ', vendor, '\n',
                          'Renderer: ', renderer, '\n',
                          'OpenGL version supported: ', opengl, '\n',
                          'GLSL version supported: ', glsl])
        print(result)

    def gl_settings(self):
        # self.qglClearColor(qtg.QColor(255, 255, 255))
        GL.glClearColor(255, 255, 255, 1)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthFunc(GL.GL_LESS)
        # the shapes are basically behind the white background
        # if you enabled face culling, they will not show
        # GL.glEnable(GL.GL_CULL_FACE)

    def clear(self):
        # color it white for better visibility
        GL.glClearColor(255, 255, 255, 1)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT | GL.GL_STENCIL_BUFFER_BIT)

    def initializeGL(self):
        self.print_system_info()
        self.gl_settings()

        sax = self.parent.cmb_sliding_axis.currentIndex() # string x, y, z
        dim = self.parent.spb_voxel_res.value() # int [2:5]
        ang = self.parent.spb_angle.value() # int [-80: 80]
        dx = self.parent.spb_xdim.value() # float [10:150]
        dy = self.parent.spb_ydim.value() # float [10:150]
        dz = self.parent.spb_zdim.value() # float [10:150]
        dia = self.parent.spb_milling_diam.value() # float [1:50]
        tol = self.parent.spb_tolerances.value() # float [0.15, 5]
        spe = self.parent.spb_milling_speed.value() # int [100, 1000]
        spi = self.parent.spb_spindle_speed.value() # int [1000, 10000]
        aax = self.parent.cmb_alignment_axis.currentIndex() # str x-, y-, x+, y+
        inc = self.parent.chk_increm_depth.isChecked() # bool
        fin = self.parent.chk_arc_interp.isChecked() # bool

        if self.parent.rdo_gcode.isChecked(): ext = "gcode"
        elif self.parent.rdo_nc.isChecked(): ext = "nc"
        elif self.parent.rdo_sbp.isChecked(): ext = "sbp"
        else: ext = "gcode"

        # joint and display objects are related to OpenGL hence initialized here
        # instead of the __init__
        self.joint = Joint(self, fs=[[[2, 0]], [[2, 1]]], sax=sax, dim=dim, ang=ang, td=[dx, dy, dz], fabtol=tol, fabdia=dia, fspe=spe, fspi=spi, fabext=ext, align_ax=aax, incremental=inc, finterp=fin)
        self.display = Display(self, self.joint)

    def resizeGL(self, w, h):
        # remove the calls to matrixmode because the programmable pipeline is used
        self.width = w
        self.height = h
        self.wstep = int(0.5+w/5)
        self.hstep = int(0.5+h/4)

    def paintGL(self):
        self.clear()

        # technically not needed because it is part of fixed pipeline
        # https://stackoverflow.com/questions/21112570/opengl-changing-from-fixed-functions-to-programmable-pipeline
        GL.glLoadIdentity()

        self.display.update()

        GL.glViewport(0, 0, self.width - self.wstep, self.height)

        # Color picking / editing
        # Pick faces -1: nothing, 0: hovered, 1: adding, 2: pulling

        # Draw back buffer colors
        if not self.joint.mesh.select.state == 2 and not self.joint.mesh.select.state == 12:
            self.display.pick(self.x, self.y, self.height)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT | GL.GL_STENCIL_BUFFER_BIT)
        elif self.joint.mesh.select.state == 2:  # Edit joint geometry
            self.joint.mesh.select.edit([self.x, self.y], self.display.view.xrot, self.display.view.yrot, w=self.width,
                                        h=self.height)
        elif self.joint.mesh.select.state == 12:  # Edit timber orientation/position
            self.joint.mesh.select.move([self.x, self.y], self.display.view.xrot, self.display.view.yrot)

        # Display main geometry
        self.display.end_grains()
        if self.display.view.show_feedback:
            self.display.unfabricatable()
            self.display.nondurable()
            self.display.unconnected()
            self.display.unbridged()
            self.display.checker()
            self.display.arrows()
            show_area = False  # <--replace by checkbox...
            if show_area:
                self.display.area()
        self.display.joint_geometry()

        if self.joint.mesh.select.sugg_state >= 0:
            index = self.joint.mesh.select.sugg_state
            if len(self.joint.suggestions) > index:
                self.display.difference_suggestion(index)

        # Display editing in action
        self.display.selected()
        self.display.moving_rotating()

        # Display milling paths
        self.display.milling_paths()

        # Suggestions
        if self.display.view.show_suggestions:
            for i in range(len(self.joint.suggestions)):
                # hquater = self.height / 4
                # wquater = self.width / 5
                GL.glViewport(self.width - self.wstep, self.height - self.hstep * (i + 1), self.wstep, self.hstep)

                if i == self.joint.mesh.select.sugg_state:
                    GL.glEnable(GL.GL_SCISSOR_TEST)
                    # glClear is determined by the scissor box.
                    GL.glScissor(self.width - self.wstep, self.height - self.hstep * (i + 1), self.wstep, self.hstep)
                    GL.glClearDepth(1.0)
                    GL.glClearColor(0.9, 0.9, 0.9, 1.0)  # light grey
                    GL.glClear(GL.GL_COLOR_BUFFER_BIT)
                    GL.glDisable(GL.GL_SCISSOR_TEST)
                self.display.joint_geometry(mesh=self.joint.suggestions[i], lw=2, hidden=False)

    def mousePressEvent(self, e):
        if e.button() == qtc.Qt.LeftButton:
            if time.time() - self.click_time < 0.2:
                self.display.view.open_joint = not self.display.view.open_joint
            elif self.joint.mesh.select.state==0: #face hovered
                self.joint.mesh.select.start_pull([self.parent.scaling * e.x(), self.parent.scaling * e.y()])
            elif self.joint.mesh.select.state==10: #body hovered
                self.joint.mesh.select.start_move([self.parent.scaling * e.x(), self.parent.scaling * e.y()], h=self.height)
            #SUGGESTION PICK
            elif self.joint.mesh.select.sugg_state>=0:
                index = self.joint.mesh.select.sugg_state
                if len(self.joint.suggestions)>index:
                    self.joint.mesh = Geometries(self.joint, hfs=self.joint.suggestions[index].height_fields)
                    self.joint.suggestions = []
                    self.joint.combine_and_buffer_indices()
                    self.joint.mesh.select.sugg_state=-1
            #GALLERY PICK -- not implemented currently
            #elif joint.mesh.select.gallstate>=0:
            #    index = joint.mesh.select.gallstate
            #    if index<len(joint.gals):
            #        joint.mesh = Geometries(joint,hfs=joint.gals[index].height_fields)
            #        joint.gals = []
            #        view_opt.gallery=False
            #        joint.gallary_start_index = -20
            #        joint.combine_and_buffer_indices()
            else: self.click_time = time.time()
        elif e.button() == qtc.Qt.RightButton:
            self.display.view.start_rotation_xy(self.parent.scaling*e.x(),self.parent.scaling*e.y())

    def mouseMoveEvent(self, e):
        self.x = self.parent.scaling*e.x()
        self.y = self.parent.scaling*e.y()
        if self.display.view.dragged:
            self.display.view.update_rotation_xy(self.x, self.y)

    def mouseReleaseEvent(self, e):
        if e.button() == qtc.Qt.LeftButton:
            if self.joint.mesh.select.state == 2:  # face pulled
                self.joint.mesh.select.end_pull()
            elif self.joint.mesh.select.state == 12:  # body moved
                self.joint.mesh.select.end_move()
        elif e.button() == qtc.Qt.RightButton:
            self.display.view.end_rotation()

    def minimumSizeHint(self):
        return qtc.QSize(50, 50)

    def sizeHint(self):
        # print("resize Hint!")
        return qtc.QSize(800, 800)

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

    # def initializeGL(self):
    #     self.print_system_info()
    #     self.gl_settings()
    #
    #     sax = self.parent.cmb_sliding_axis.currentIndex() # string x, y, z
    #     dim = self.parent.spb_voxel_res.value() # int [2:5]
    #     ang = self.parent.spb_angle.value() # int [-80: 80]
    #     dx = self.parent.spb_xdim.value() # float [10:150]
    #     dy = self.parent.spb_ydim.value() # float [10:150]
    #     dz = self.parent.spb_zdim.value() # float [10:150]
    #     dia = self.parent.spb_milling_diam.value() # float [1:50]
    #     tol = self.parent.spb_tolerances.value() # float [0.15, 5]
    #     spe = self.parent.spb_milling_speed.value() # int [100, 1000]
    #     spi = self.parent.spb_spindle_speed.value() # int [1000, 10000]
    #     aax = self.parent.cmb_alignment_axis.currentIndex() # str x-, y-, x+, y+
    #     inc = self.parent.chk_increm_depth.isChecked() # bool
    #     fin = self.parent.chk_arc_interp.isChecked() # bool
    #
    #     if self.parent.rdo_gcode.isChecked(): ext = "gcode"
    #     elif self.parent.rdo_nc.isChecked(): ext = "nc"
    #     elif self.parent.rdo_sbp.isChecked(): ext = "sbp"
    #     else: ext = "gcode"
    #
    #     # joint and display objects are related to OpenGL hence initialized here
    #     # instead of the __init__
    #     self.joint = Joint(self, fs=[[[2, 0]], [[2, 1]]], sax=sax, dim=dim, ang=ang, td=[dx, dy, dz], fabtol=tol, fabdia=dia, fspe=spe, fspi=spi, fabext=ext, align_ax=aax, incremental=inc, finterp=fin)
    #     self.display = Display(self, self.joint)

    def initializeGL(self):
        """Initialize OpenGL settings and create joint and display objects."""
        self.print_system_info()
        self.gl_settings()

        # Extract parameters from UI
        params = self._extract_parameters_from_ui()

        # Create joint and display objects
        self.joint = Joint(
            self,
            fs=[[[2, 0]], [[2, 1]]],
            sax=params['sliding_axis'],
            dim=params['voxel_resolution'],
            ang=params['angle'],
            td=[params['x_dim'], params['y_dim'], params['z_dim']],
            fabtol=params['tolerance'],
            fabdia=params['milling_diameter'],
            fspe=params['milling_speed'],
            fspi=params['spindle_speed'],
            fabext=params['extension'],
            align_ax=params['alignment_axis'],
            incremental=params['incremental'],
            finterp=params['interpolation']
        )
        self.display = Display(self, self.joint)

    def _extract_parameters_from_ui(self):
        """Extract parameters from UI controls."""
        # Get export file extension
        if self.parent.rdo_gcode.isChecked():
            ext = "gcode"
        elif self.parent.rdo_nc.isChecked():
            ext = "nc"
        elif self.parent.rdo_sbp.isChecked():
            ext = "sbp"
        else:
            ext = "gcode"

        return {
            'sliding_axis': self.parent.cmb_sliding_axis.currentIndex(),
            'voxel_resolution': self.parent.spb_voxel_res.value(),
            'angle': self.parent.spb_angle.value(),
            'x_dim': self.parent.spb_xdim.value(),
            'y_dim': self.parent.spb_ydim.value(),
            'z_dim': self.parent.spb_zdim.value(),
            'milling_diameter': self.parent.spb_milling_diam.value(),
            'tolerance': self.parent.spb_tolerances.value(),
            'milling_speed': self.parent.spb_milling_speed.value(),
            'spindle_speed': self.parent.spb_spindle_speed.value(),
            'alignment_axis': self.parent.cmb_alignment_axis.currentIndex(),
            'incremental': self.parent.chk_increm_depth.isChecked(),
            'interpolation': self.parent.chk_arc_interp.isChecked(),
            'extension': ext
        }

    def resizeGL(self, w, h):
        # remove the calls to matrixmode because the programmable pipeline is used
        self.width = w
        self.height = h
        self.wstep = int(0.5+w/5)
        self.hstep = int(0.5+h/4)

    def paintGL(self):
        """Main rendering function."""
        self.clear()
        GL.glLoadIdentity()
        self.display.update()

        # Set main viewport
        GL.glViewport(0, 0, self.width - self.wstep, self.height)

        # Handle color picking and editing
        self._handle_selection_and_editing()

        # Render main scene
        self._render_main_scene()

        # Render suggestions if enabled
        if self.display.view.show_suggestions:
            self._render_suggestions()

    def _handle_selection_and_editing(self):
        """Handle color picking and geometry editing."""
        if self.joint.mesh.select.state == 2:  # Edit joint geometry
            self.joint.mesh.select.edit(
                [self.x, self.y],
                self.display.view.xrot,
                self.display.view.yrot,
                w=self.width,
                h=self.height
            )
        elif self.joint.mesh.select.state == 12:  # Edit timber orientation/position
            self.joint.mesh.select.move(
                [self.x, self.y],
                self.display.view.xrot,
                self.display.view.yrot
            )
        else:  # Normal picking
            self.display.pick(self.x, self.y, self.height)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT | GL.GL_STENCIL_BUFFER_BIT)

    def _render_main_scene(self):
        """Render the main scene with all visual elements."""
        # Render end grains
        self.display.end_grains()

        # Render feedback elements if enabled
        if self.display.view.show_feedback:
            self._render_feedback_elements()

        # Render joint geometry
        self.display.joint_geometry()

        # Render suggestion differences if applicable
        self._render_suggestion_differences()

        # Render selection and movement
        self.display.selected()
        self.display.moving_rotating()

        # Render milling paths
        self.display.milling_paths()

    def _render_feedback_elements(self):
        """Render visual feedback elements."""
        self.display.unfabricatable()
        self.display.nondurable()
        self.display.unconnected()
        self.display.unbridged()
        self.display.checker()
        self.display.arrows()

        # Area display (currently disabled)
        show_area = False  # <--replace by checkbox...
        if show_area:
            self.display.area()

    def _render_suggestion_differences(self):
        """Render differences for the selected suggestion."""
        if self.joint.mesh.select.sugg_state >= 0:
            index = self.joint.mesh.select.sugg_state
            if len(self.joint.suggestions) > index:
                self.display.difference_suggestion(index)

    def _render_suggestions(self):
        """Render suggestion thumbnails."""
        for i in range(len(self.joint.suggestions)):
            # Set viewport for this suggestion
            GL.glViewport(
                self.width - self.wstep,
                self.height - self.hstep * (i + 1),
                self.wstep,
                self.hstep
            )

            # Highlight selected suggestion
            if i == self.joint.mesh.select.sugg_state:
                self._highlight_selected_suggestion(i)

            # Render suggestion geometry
            self.display.joint_geometry(
                mesh=self.joint.suggestions[i],
                lw=2,
                hidden=False
            )

    def _highlight_selected_suggestion(self, index):
        """Highlight the selected suggestion with a gray background."""
        GL.glEnable(GL.GL_SCISSOR_TEST)
        GL.glScissor(
            self.width - self.wstep,
            self.height - self.hstep * (index + 1),
            self.wstep,
            self.hstep
        )
        GL.glClearDepth(1.0)
        GL.glClearColor(0.9, 0.9, 0.9, 1.0)  # light grey
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        GL.glDisable(GL.GL_SCISSOR_TEST)

    def mousePressEvent(self, e):
        """Handle mouse press events."""
        scaled_x = self.parent.scaling * e.x()
        scaled_y = self.parent.scaling * e.y()

        if e.button() == qtc.Qt.LeftButton:
            self._handle_left_button_press(scaled_x, scaled_y)
        elif e.button() == qtc.Qt.RightButton:
            self._handle_right_button_press(scaled_x, scaled_y)

    def _handle_left_button_press(self, x, y):
        """Handle left mouse button press."""
        # Check for double click (toggle joint opening)
        if time.time() - self.click_time < 0.2:
            self.display.view.open_joint = not self.display.view.open_joint
            return

        # Handle based on selection state
        select_state = self.joint.mesh.select.state

        if select_state == 0:  # Face hovered
            self.joint.mesh.select.start_pull([x, y])
        elif select_state == 10:  # Body hovered
            self.joint.mesh.select.start_move([x, y], h=self.height)
        elif self.joint.mesh.select.sugg_state >= 0:
            self._apply_selected_suggestion()
        else:
            # Just a regular click, update click time
            self.click_time = time.time()

    def _apply_selected_suggestion(self):
        """Apply the currently selected suggestion."""
        index = self.joint.mesh.select.sugg_state
        if len(self.joint.suggestions) > index:
            self.joint.mesh = Geometries(
                self.joint,
                hfs=self.joint.suggestions[index].height_fields
            )
            self.joint.suggestions = []
            self.joint.combine_and_buffer_indices()
            self.joint.mesh.select.sugg_state = -1

    def _handle_right_button_press(self, x, y):
        """Handle right mouse button press."""
        self.display.view.start_rotation_xy(x, y)

    def mouseReleaseEvent(self, e):
        """Handle mouse release events."""
        if e.button() == qtc.Qt.LeftButton:
            self._handle_left_button_release()
        elif e.button() == qtc.Qt.RightButton:
            self._handle_right_button_release()

    def _handle_left_button_release(self):
        """Handle left mouse button release."""
        if self.joint.mesh.select.state == 2:  # Face pulled
            self.joint.mesh.select.end_pull()
        elif self.joint.mesh.select.state == 12:  # Body moved
            self.joint.mesh.select.end_move()

    def _handle_right_button_release(self):
        """Handle right mouse button release."""
        self.display.view.end_rotation()

    def mouseMoveEvent(self, e):
        """Handle mouse movement events."""
        # Update current mouse position with proper scaling
        self.x = self.parent.scaling * e.x()
        self.y = self.parent.scaling * e.y()

        # Handle rotation if dragging with right mouse button
        if self.display.view.dragged:
            self.display.view.update_rotation_xy(self.x, self.y)

    def minimumSizeHint(self):
        return qtc.QSize(50, 50)

    def sizeHint(self):
        # print("resize Hint!")
        return qtc.QSize(800, 800)

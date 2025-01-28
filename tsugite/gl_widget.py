import time
from math import tan, pi
import numpy as np
import pyrr
from ctypes import c_void_p as buffer_offset

import PyQt5.QtWidgets as qtw
import PyQt5.QtGui as qtg
import PyQt5.QtCore as qtc
import PyQt5.QtOpenGL as qgl

import OpenGL.GL as GL  # imports start with GL
import OpenGL.GL.shaders as GLSH

from joint import Joint
from geometries import Geometries
from buffer import ElementProperties
from view_settings import ViewSettings

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
        # self.display = Display(self, self.joint)

        self.view = ViewSettings()
        self.create_color_shaders()
        self.create_texture_shaders()

    def resizeGL(self, w, h):
        # remove the calls to matrixmode because the programmable pipeline is used
        self.width = w
        self.height = h
        print("width", self.width)
        print("height", self.height)
        self.wstep = int(0.5+w/5)
        self.hstep = int(0.5+h/4)

    def paintGL(self):
        self.clear()

        # technically not needed because it is part of fixed pipeline
        # https://stackoverflow.com/questions/21112570/opengl-changing-from-fixed-functions-to-programmable-pipeline
        GL.glLoadIdentity()

        self.update()

        GL.glViewport(0, 0, self.width - self.wstep, self.height)

        # Color picking / editing
        # Pick faces -1: nothing, 0: hovered, 1: adding, 2: pulling

        # Draw back buffer colors
        if not self.joint.mesh.select.state == 2 and not self.joint.mesh.select.state == 12:
            self.pick(self.x, self.y, self.height)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT | GL.GL_STENCIL_BUFFER_BIT)
        elif self.joint.mesh.select.state == 2:  # Edit joint geometry
            self.joint.mesh.select.edit([self.x, self.y], self.view.xrot, self.view.yrot, w=self.width,
                                        h=self.height)
        elif self.joint.mesh.select.state == 12:  # Edit timber orientation/position
            self.joint.mesh.select.move([self.x, self.y], self.view.xrot, self.view.yrot)

        # Display main geometry
        self.end_grains()
        if self.view.show_feedback:
            self.unfabricatable()
            self.nondurable()
            self.unconnected()
            self.unbridged()
            self.checker()
            self.arrows()
            show_area = False  # <--replace by checkbox...
            if show_area:
                self.area()
        self.joint_geometry()

        if self.joint.mesh.select.sugg_state >= 0:
            index = self.joint.mesh.select.sugg_state
            if len(self.joint.suggestions) > index:
                self.difference_suggestion(index)

        # Display editing in action
        self.selected()
        self.moving_rotating()

        # Display milling paths
        self.milling_paths()

        # Suggestions
        if self.view.show_suggestions:
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
                self.joint_geometry(mesh=self.joint.suggestions[i], lw=2, hidden=False)

    def create_color_shaders(self):
        """
        Note values explicity attrib and uniform locations are only availabl ein GL 3.3 and 4.3 respectively
        If to be use in versions lower than the above, the following are needed for vertex shaders
        """

        vertex_shader = """
        #version 150
        #extension GL_ARB_explicit_attrib_location : require
        #extension GL_ARB_explicit_uniform_location : require
        layout(location = 0) in vec3 position;
        layout(location = 1) in vec3 color;
        layout(location = 2) in vec2 inTexCoords;
        layout(location = 3) uniform mat4 translate;
        layout(location = 4) uniform mat4 transform;
        layout(location = 5) uniform vec3 myColor;
        out vec3 newColor;
        out vec2 outTexCoords;
        void main()
        {
            gl_Position = transform* translate* vec4(position, 1.0f);
            newColor = myColor;
            outTexCoords = inTexCoords;
        }
        """

        fragment_shader = """
        #version 150
        in vec3 newColor;
        in vec2 outTexCoords;
        out vec4 outColor;
        uniform sampler2D samplerTex;
        void main()
        {
            outColor = vec4(newColor, 1.0);
        }
        """
        # Compiling the shaders
        self.shader_col = GLSH.compileProgram(GLSH.compileShader(vertex_shader, GL.GL_VERTEX_SHADER),
                                                  GLSH.compileShader(fragment_shader, GL.GL_FRAGMENT_SHADER))

    def create_texture_shaders(self):
        vertex_shader = """
        #version 150
        #extension GL_ARB_explicit_attrib_location : require
        #extension GL_ARB_explicit_uniform_location : require
        layout(location = 0) in vec3 position;
        layout(location = 1) in vec3 color;
        layout(location = 2) in vec2 inTexCoords;
        layout(location = 3) uniform mat4 translate;
        layout(location = 4) uniform mat4 transform;
        out vec3 newColor;
        out vec2 outTexCoords;
        void main()
        {
            gl_Position = transform* translate* vec4(position, 1.0f);
            newColor = color;
            outTexCoords = inTexCoords;
        }
        """

        fragment_shader = """
        #version 150
        in vec3 newColor;
        in vec2 outTexCoords;
        out vec4 outColor;
        uniform sampler2D samplerTex;
        void main()
        {
            outColor = texture(samplerTex, outTexCoords);
        }
        """


        # Compiling the shaders
        self.shader_tex = GLSH.compileProgram(GLSH.compileShader(vertex_shader, GL.GL_VERTEX_SHADER),
                                                  GLSH.compileShader(fragment_shader, GL.GL_FRAGMENT_SHADER))

    def update(self):
        self.current_program = self.shader_col
        GL.glUseProgram(self.current_program)

        self.bind_view_mat_to_shader_transform_mat()
        if (self.view.open_joint and self.view.open_ratio < self.joint.noc - 1) or (not self.view.open_joint and self.view.open_ratio > 0):
            self.view.set_joint_opening_distance(self.joint.noc)

    def end_grains(self):
        self.current_program = self.shader_tex
        GL.glUseProgram(self.current_program)
        self.bind_view_mat_to_shader_transform_mat()

        G0 = self.joint.mesh.indices_fend
        G1 = self.joint.mesh.indices_not_fend
        self.draw_geometries_with_excluded_area(G0,G1)

        self.current_program = self.shader_col
        GL.glUseProgram(self.current_program)
        self.bind_view_mat_to_shader_transform_mat()

    def bind_view_mat_to_shader_transform_mat(self):
        rot_x = pyrr.Matrix44.from_x_rotation(self.view.xrot)
        rot_y = pyrr.Matrix44.from_y_rotation(self.view.yrot)

        transform_ref = GL.glGetUniformLocation(self.current_program, 'transform')
        GL.glUniformMatrix4fv(transform_ref, 1, GL.GL_FALSE, rot_x * rot_y)

    def draw_geometries(self, geos,clear_depth_buffer=True, translation_vec=np.array([0,0,0])):
        # Define translation matrices for opening
        move_vec = [0,0,0]
        move_vec[self.joint.sax] = self.view.open_ratio * self.joint.component_size
        move_vec = np.array(move_vec)
        moves = []
        for n in range(self.joint.noc):
            tot_move_vec = (2 * n + 1 - self.joint.noc) / (self.joint.noc - 1) * move_vec
            move_mat = pyrr.matrix44.create_from_translation(tot_move_vec+translation_vec)
            moves.append(move_mat)
        if clear_depth_buffer: GL.glClear(GL.GL_DEPTH_BUFFER_BIT)
        for geo in geos:
            if geo==None: continue
            if self.view.hidden[geo.n]: continue
            translate_ref = GL.glGetUniformLocation(self.current_program, 'translate')
            GL.glUniformMatrix4fv(translate_ref, 1, GL.GL_FALSE, moves[geo.n])
            GL.glDrawElements(geo.draw_type, geo.count, GL.GL_UNSIGNED_INT,  buffer_offset(4*geo.start_index))

    def draw_geometries_with_excluded_area(self, show_geos, screen_geos, translation_vec=np.array([0,0,0])):
        # Define translation matrices for opening
        move_vec = [0,0,0]
        move_vec[self.joint.sax] = self.view.open_ratio * self.joint.component_size
        move_vec = np.array(move_vec)
        moves = []
        moves_show = []
        for n in range(self.joint.noc):
            tot_move_vec = (2 * n + 1 - self.joint.noc) / (self.joint.noc - 1) * move_vec
            move_mat = pyrr.matrix44.create_from_translation(tot_move_vec)
            moves.append(move_mat)
            move_mat_show = pyrr.matrix44.create_from_translation(tot_move_vec+translation_vec)
            moves_show.append(move_mat_show)
        #
        GL.glClear(GL.GL_DEPTH_BUFFER_BIT)
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glColorMask(GL.GL_FALSE,GL.GL_FALSE,GL.GL_FALSE,GL.GL_FALSE)
        GL.glEnable(GL.GL_STENCIL_TEST)
        GL.glStencilFunc(GL.GL_ALWAYS,1,1)
        GL.glStencilOp(GL.GL_REPLACE,GL.GL_REPLACE,GL.GL_REPLACE)
        GL.glDepthRange (0.0, 0.9975)

        translate_ref = GL.glGetUniformLocation(self.current_program, 'translate')

        for geo in show_geos:
            if geo==None: continue
            if self.view.hidden[geo.n]: continue

            GL.glUniformMatrix4fv(translate_ref, 1, GL.GL_FALSE, moves_show[geo.n])
            GL.glDrawElements(geo.draw_type, geo.count, GL.GL_UNSIGNED_INT,  buffer_offset(4*geo.start_index))

        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glStencilFunc(GL.GL_EQUAL,1,1)
        GL.glStencilOp(GL.GL_KEEP,GL.GL_KEEP,GL.GL_KEEP)
        GL.glDepthRange (0.0025, 1.0)
        for geo in screen_geos:
            if geo==None: continue
            if self.view.hidden[geo.n]: continue

            GL.glUniformMatrix4fv(translate_ref, 1, GL.GL_FALSE, moves[geo.n])
            GL.glDrawElements(geo.draw_type, geo.count, GL.GL_UNSIGNED_INT,  buffer_offset(4*geo.start_index))
        GL.glDisable(GL.GL_STENCIL_TEST)
        GL.glColorMask(GL.GL_TRUE,GL.GL_TRUE,GL.GL_TRUE,GL.GL_TRUE)
        GL.glDepthRange (0.0, 0.9975)
        for geo in show_geos:
            if geo==None: continue
            if self.view.hidden[geo.n]: continue

            GL.glUniformMatrix4fv(translate_ref, 1, GL.GL_FALSE, moves_show[geo.n])
            GL.glDrawElements(geo.draw_type, geo.count, GL.GL_UNSIGNED_INT,  buffer_offset(4*geo.start_index))

    def pick(self,xpos,ypos,height):
        if not self.view.gallery:
            ######################## COLOR SHADER ###########################
            self.current_program = self.shader_col
            GL.glUseProgram(self.current_program)

            GL.glClearColor(1.0, 1.0, 1.0, 1.0) # white
            GL.glEnable(GL.GL_DEPTH_TEST)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT | GL.GL_STENCIL_BUFFER_BIT)

            self.bind_view_mat_to_shader_transform_mat()
            GL.glPolygonOffset(1.0,1.0)

            ########################## Draw colorful top faces ##########################

            # Draw colorful geometries
            col_step = 1.0/(2 + 2 * self.joint.dim * self.joint.dim)
            for n in range(self.joint.noc):
                col = np.zeros(3, dtype=np.float64)
                col[n%3] = 1.0
                if n>2: col[(n+1) % self.joint.dim] = 1.0
                GL.glUniform3f(5, col[0], col[1], col[2])
                self.draw_geometries([self.joint.mesh.indices_fpick_not_top[n]], clear_depth_buffer=False)
                if n==0 or n==self.joint.noc-1: mos = 1
                else: mos = 2
                # mos is "number of sides"
                for m in range(mos):
                    # Draw top faces
                    for i in range(self.joint.dim * self.joint.dim):
                        col -= col_step
                        GL.glUniform3f(5, col[0], col[1], col[2])
                        top = ElementProperties(GL.GL_QUADS, 4, self.joint.mesh.indices_fpick_top[n].start_index + mos * 4 * i + 4 * m, n)
                        self.draw_geometries([top],clear_depth_buffer=False)

        ############### Read pixel color at mouse position ###############
        mouse_pixel = GL.glReadPixelsub(xpos, height-ypos, 1, 1, GL.GL_RGB, outputType=GL.GL_UNSIGNED_BYTE)[0][0]
        mouse_pixel = np.array(mouse_pixel)
        pick_n = pick_d = pick_x = pick_y = None
        self.joint.mesh.select.sugg_state = -1
        self.joint.mesh.select.gallstate = -1
        if not self.view.gallery:
            if xpos>self.width-self.wstep: # suggestion side
                if ypos>0 and ypos<self.height:
                    index = int(ypos/self.hstep)
                    if self.joint.mesh.select.sugg_state!=index:
                        self.joint.mesh.select.sugg_state=index
            elif not np.all(mouse_pixel==255): # not white / background
                    non_zeros = np.where(mouse_pixel!=0)
                    if len(non_zeros)>0:
                        if len(non_zeros[0]>0):
                            pick_n = non_zeros[0][0]
                            if len(non_zeros[0])>1:
                                pick_n = pick_n+self.joint.dim
                                if mouse_pixel[0]==mouse_pixel[2]: pick_n = 5
                            val = 255-mouse_pixel[non_zeros[0][0]]
                            #i = int(0.5+val*(2+2*self.joint.dim*self.joint.dim)/255)-1
                            step_size = 128/(self.joint.dim ** 2 + 1)
                            i=round(val/step_size)-1
                            if i>=0:
                                pick_x = (int(i / self.joint.dim)) % self.joint.dim
                                pick_y = i%self.joint.dim
                            pick_d = 0
                            if pick_n==self.joint.noc-1: pick_d = 1
                            elif int(i/self.joint.dim)>=self.joint.dim: pick_d = 1
                            #print("pick",pick_n,pick_d,pick_x,pick_y)
        """
        else: #gallerymode
            if xpos>0 and xpos<2000 and ypos>0 and ypos<1600:
                i = int(xpos/400)
                j = int(ypos/400)
                index = i*4+j
                mesh.select.gallstate=index
                mesh.select.state = -1
                mesh.select.sugg_state = -1
        """
        ### Update selection
        if pick_x !=None and pick_d!=None and pick_y!=None and pick_n!=None:
            ### Initialize selection
            new_pos = False
            if pick_x!=self.joint.mesh.select.x or pick_y!=self.joint.mesh.select.y or pick_n!=self.joint.mesh.select.n or pick_d!=self.joint.mesh.select.dir or self.joint.mesh.select.refresh:
                self.joint.mesh.select.update_pick(pick_x, pick_y, pick_n, pick_d)
                self.joint.mesh.select.refresh = False
                self.joint.mesh.select.state = 0 # hovering
        elif pick_n!=None:
            self.joint.mesh.select.state = 10 # hovering component body
            self.joint.mesh.select.update_pick(pick_x, pick_y, pick_n, pick_d)
        else: self.joint.mesh.select.state = -1
        GL.glClearColor(1.0, 1.0, 1.0, 1.0)

    def selected(self):
        ################### Draw top face that is currently being hovered ##########
        # Draw base face (hovered)
        if self.joint.mesh.select.state==0:
            GL.glClear(GL.GL_DEPTH_BUFFER_BIT | GL.GL_STENCIL_BUFFER_BIT)
            GL.glUniform3f(5, 0.2, 0.2, 0.2) #dark grey
            G1 = self.joint.mesh.indices_fpick_not_top
            for face in self.joint.mesh.select.faces:
                if self.joint.mesh.select.n==0 or self.joint.mesh.select.n==self.joint.noc-1: mos = 1
                else: mos = 2
                index = int(self.joint.dim * face[0] + face[1])
                top = ElementProperties(GL.GL_QUADS, 4, self.joint.mesh.indices_fpick_top[self.joint.mesh.select.n].start_index + mos * 4 * index + (mos - 1) * 4 * self.joint.mesh.select.dir, self.joint.mesh.select.n)
                #top = ElementProperties(GL_QUADS, 4, mesh.indices_fpick_top[mesh.select.n].start_index+4*index, mesh.select.n)
                self.draw_geometries_with_excluded_area([top],G1)
        # Draw pulled face
        if self.joint.mesh.select.state==2:
            GL.glPushAttrib(GL.GL_ENABLE_BIT)
            GL.glLineWidth(3)
            GL.glEnable(GL.GL_LINE_STIPPLE)
            GL.glLineStipple(2, 0xAAAA)
            for val in range(0, abs(self.joint.mesh.select.val) + 1):
                if self.joint.mesh.select.val<0: val = -val
                pulled_vec = [0,0,0]
                pulled_vec[self.joint.sax] = val * self.joint.voxel_sizes[self.joint.sax]
                self.draw_geometries([self.joint.mesh.outline_selected_faces], translation_vec=np.array(pulled_vec))
            GL.glPopAttrib()

    def difference_suggestion(self,index):
        GL.glPushAttrib(GL.GL_ENABLE_BIT)
        # draw faces of additional part
        #glUniform3f(5, 1.0, 1.0, 1.0) # white
        #for n in range(self.joint.noc):
        #    G0 = [self.joint.suggestions[index].indices_fall[n]]
        #    G1 = self.joint.mesh.indices_fall
        #    self.draw_geometries_with_excluded_area(G0,G1)

        # draw faces of subtracted part
        #glUniform3f(5, 1.0, 0.5, 0.5) # pink/red
        #for n in range(self.joint.noc):
        #    G0 = [self.joint.mesh.indices_fall[n]]
        #    G1 = self.joint.suggestions[index].indices_fall
        #    self.draw_geometries_with_excluded_area(G0,G1)

        # draw outlines
        GL.glUniform3f(5, 0.0, 0.0, 0.0) # black
        GL.glLineWidth(3)
        GL.glEnable(GL.GL_LINE_STIPPLE)
        GL.glLineStipple(2, 0xAAAA)
        for n in range(self.joint.noc):
            G0 = [self.joint.suggestions[index].indices_lns[n]]
            G1 = self.joint.suggestions[index].indices_fall
            self.draw_geometries_with_excluded_area(G0,G1)
        GL.glPopAttrib()

    def moving_rotating(self):
        # Draw moved_rotated component before action is finalized
        if self.joint.mesh.select.state==12 and self.joint.mesh.outline_selected_component!=None:
            GL.glPushAttrib(GL.GL_ENABLE_BIT)
            GL.glLineWidth(3)
            GL.glEnable(GL.GL_LINE_STIPPLE)
            GL.glLineStipple(2, 0xAAAA)
            self.draw_geometries([self.joint.mesh.outline_selected_component])
            GL.glPopAttrib()

    def joint_geometry(self,mesh=None,lw=3,hidden=True,zoom=False):

        if mesh==None: mesh = self.joint.mesh

        ############################# Draw hidden lines #############################
        GL.glClear(GL.GL_DEPTH_BUFFER_BIT)
        GL.glUniform3f(5,0.0,0.0,0.0) # black
        GL.glPushAttrib(GL.GL_ENABLE_BIT)
        GL.glLineWidth(1)
        GL.glLineStipple(3, 0xAAAA) #dashed line
        GL.glEnable(GL.GL_LINE_STIPPLE)
        if hidden and self.view.show_hidden_lines:
            for n in range(mesh.parent.noc):
                G0 = [mesh.indices_lns[n]]
                G1 = [mesh.indices_fall[n]]
                self.draw_geometries_with_excluded_area(G0,G1)
        GL.glPopAttrib()

        ############################ Draw visible lines #############################
        for n in range(mesh.parent.noc):
            if not mesh.mainmesh or (mesh.eval.interlocks[n] and self.view.show_feedback) or not self.view.show_feedback:
                GL.glUniform3f(5,0.0,0.0,0.0) # black
                GL.glLineWidth(lw)
            else:
                GL.glUniform3f(5,1.0,0.0,0.0) # red
                GL.glLineWidth(lw+1)
            G0 = [mesh.indices_lns[n]]
            G1 = mesh.indices_fall
            self.draw_geometries_with_excluded_area(G0,G1)


        if mesh.mainmesh:
            ################ When joint is fully open, draw dahsed lines ################
            if hidden and not self.view.hidden[0] and not self.view.hidden[1] and self.view.open_ratio==1+0.5*(mesh.parent.noc-2):
                GL.glUniform3f(5,0.0,0.0,0.0) # black
                GL.glPushAttrib(GL.GL_ENABLE_BIT)
                GL.glLineWidth(2)
                GL.glLineStipple(1, 0x00FF)
                GL.glEnable(GL.GL_LINE_STIPPLE)
                G0 = mesh.indices_open_lines
                G1 = mesh.indices_fall
                self.draw_geometries_with_excluded_area(G0,G1)
                GL.glPopAttrib()

    def unfabricatable(self):
        col = [1.0, 0.8, 0.5] # orange
        GL.glUniform3f(5, col[0], col[1], col[2])
        for n in range(self.joint.noc):
            if not self.joint.mesh.eval.fab_direction_ok[n]:
                G0 = [self.joint.mesh.indices_fall[n]]
                G1 = []
                for n2 in range(self.joint.noc):
                    if n2!=n: G1.append(self.joint.mesh.indices_fall[n2])
                self.draw_geometries_with_excluded_area(G0,G1)

    def unconnected(self):
        # 1. Draw hidden geometry
        col = [1.0, 0.8, 0.7]  # light red orange
        GL.glUniform3f(5, col[0], col[1], col[2])
        for n in range(self.joint.mesh.parent.noc):
            if not self.joint.mesh.eval.connected[n]:
                self.draw_geometries([self.joint.mesh.indices_not_fcon[n]])

        # 1. Draw visible geometry
        col = [1.0, 0.2, 0.0] # red orange
        GL.glUniform3f(5, col[0], col[1], col[2])
        G0 = self.joint.mesh.indices_not_fcon
        G1 = self.joint.mesh.indices_fcon
        self.draw_geometries_with_excluded_area(G0,G1)

    def unbridged(self):
        # Draw colored faces when unbridged
        for n in range(self.joint.noc):
            if not self.joint.mesh.eval.bridged[n]:
                for m in range(2): # browse the two parts
                    # a) Unbridge part 1
                    col = self.view.unbridge_colors[n][m]
                    GL.glUniform3f(5, col[0], col[1], col[2])
                    G0 = [self.joint.mesh.indices_not_fbridge[n][m]]
                    G1 = [self.joint.mesh.indices_not_fbridge[n][1 - m],
                          self.joint.mesh.indices_fall[1 - n],
                          self.joint.mesh.indices_not_fcon[n]] # needs reformulation for 3 components
                    self.draw_geometries_with_excluded_area(G0,G1)

    def checker(self):
        # 1. Draw hidden geometry
        GL.glUniform3f(5, 1.0, 0.2, 0.0) # red orange
        GL.glLineWidth(8)
        for n in range(self.joint.mesh.parent.noc):
            if self.joint.mesh.eval.checker[n]:
                self.draw_geometries([self.joint.mesh.indices_chess_lines[n]])
        GL.glUniform3f(5, 0.0, 0.0, 0.0) # back to black

    def arrows(self):
        #glClear(GL_DEPTH_BUFFER_BIT)
        GL.glUniform3f(5, 0.0, 0.0, 0.0)
        ############################## Direction arrows ################################
        for n in range(self.joint.noc):
            if (self.joint.mesh.eval.interlocks[n]): GL.glUniform3f(5, 0.0, 0.0, 0.0) # black
            else: GL.glUniform3f(5,1.0,0.0,0.0) # red
            GL.glLineWidth(3)
            G1 = self.joint.mesh.indices_fall
            G0 = self.joint.mesh.indices_arrows[n]
            d0 = 2.55*self.joint.component_size
            d1 = 1.55*self.joint.component_size
            if len(self.joint.fixed.sides[n])==2: d0 = d1
            for side in self.joint.fixed.sides[n]:
                vec = d0 * (2*side.dir-1) * self.joint.pos_vecs[side.ax] / np.linalg.norm(self.joint.pos_vecs[side.ax])
                #draw_geometries_with_excluded_area(window,G0,G1,translation_vec=vec)
                self.draw_geometries(G0,translation_vec=vec)

    def nondurable(self):
        # 1. Draw hidden geometry
        col = [1.0, 1.0, 0.8] # super light yellow
        GL.glUniform3f(5, col[0], col[1], col[2])
        for n in range(self.joint.noc):
            self.draw_geometries_with_excluded_area([self.joint.mesh.indices_fbrk[n]], [self.joint.mesh.indices_not_fbrk[n]])

        # Draw visible geometry
        col = [1.0, 1.0, 0.4] # light yellow
        GL.glUniform3f(5, col[0], col[1], col[2])
        self.draw_geometries_with_excluded_area(self.joint.mesh.indices_fbrk, self.joint.mesh.indices_not_fbrk)

    def milling_paths(self):
        if len(self.joint.mesh.indices_milling_path)==0: self.view.show_milling_path = False
        if self.view.show_milling_path:
            cols = [[1.0,0,0],[0,1.0,0],[0,0,1.0],[1.0,1.0,0],[0.0,1.0,1.0],[1.0,0,1.0]]
            GL.glLineWidth(3)
            for n in range(self.joint.noc):
                if self.joint.mesh.eval.fab_direction_ok[n]:
                    GL.glUniform3f(5,cols[n][0],cols[n][1],cols[n][2])
                    self.draw_geometries([self.joint.mesh.indices_milling_path[n]])

    def mousePressEvent(self, e):
        if e.button() == qtc.Qt.LeftButton:
            if time.time() - self.click_time < 0.2:
                self.view.open_joint = not self.view.open_joint
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
            self.view.start_rotation_xy(self.parent.scaling*e.x(),self.parent.scaling*e.y())

    def mouseMoveEvent(self, e):
        self.x = self.parent.scaling*e.x()
        self.y = self.parent.scaling*e.y()
        if self.view.dragged:
            self.view.update_rotation_xy(self.x, self.y)

    def mouseReleaseEvent(self, e):
        if e.button() == qtc.Qt.LeftButton:
            if self.joint.mesh.select.state == 2:  # face pulled
                self.joint.mesh.select.end_pull()
            elif self.joint.mesh.select.state == 12:  # body moved
                self.joint.mesh.select.end_move()
        elif e.button() == qtc.Qt.RightButton:
            self.view.end_rotation()

    def minimumSizeHint(self):
        return qtc.QSize(50, 50)

    def sizeHint(self):
        # print("resize Hint!")
        return qtc.QSize(800, 800)

import numpy as np
import pyrr
from ctypes import c_void_p as buffer_offset

import OpenGL.GL as GL  # imports start with GL
import OpenGL.GL.shaders as GLSH
from OpenGL.GLU import gluPerspective

from buffer import ElementProperties
from view_settings import ViewSettings

class Display:
    def __init__(self, pwidget, joint):
        self.pwidget = pwidget
        self.joint = joint
        self.view = ViewSettings()
        self.create_color_shaders()
        self.create_texture_shaders()

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

        # there's only one myColor var so it is safe to make it an attribute
        self.myColor = GL.glGetUniformLocation(self.current_program, 'myColor')

    def bind_view_mat_to_shader_transform_mat(self):
        rot_x = pyrr.Matrix44.from_x_rotation(self.view.xrot)
        rot_y = pyrr.Matrix44.from_y_rotation(self.view.yrot)

        transform_ref = GL.glGetUniformLocation(self.current_program, 'transform')
        GL.glUniformMatrix4fv(transform_ref, 1, GL.GL_FALSE, rot_x * rot_y)

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

        translate_ref = GL.glGetUniformLocation(self.current_program, 'translate')

        for geo in geos:
            if geo==None: continue
            if self.view.hidden[geo.n]: continue

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
                GL.glUniform3f(self.myColor, col[0], col[1], col[2])
                self.draw_geometries([self.joint.mesh.indices_fpick_not_top[n]], clear_depth_buffer=False)
                if n==0 or n==self.joint.noc-1: mos = 1
                else: mos = 2
                # mos is "number of sides"
                for m in range(mos):
                    # Draw top faces
                    for i in range(self.joint.dim * self.joint.dim):
                        col -= col_step
                        GL.glUniform3f(self.myColor, col[0], col[1], col[2])
                        top = ElementProperties(GL.GL_QUADS, 4, self.joint.mesh.indices_fpick_top[n].start_index + mos * 4 * i + 4 * m, n)
                        self.draw_geometries([top],clear_depth_buffer=False)

        ############### Read pixel color at mouse position ###############
        mouse_pixel = GL.glReadPixelsub(xpos, height-ypos, 1, 1, GL.GL_RGB, outputType=GL.GL_UNSIGNED_BYTE)[0][0]
        mouse_pixel = np.array(mouse_pixel)
        pick_n = pick_d = pick_x = pick_y = None
        self.joint.mesh.select.sugg_state = -1
        self.joint.mesh.select.gallstate = -1
        if not self.view.gallery:
            if xpos>self.pwidget.width-self.pwidget.wstep: # suggestion side
                if ypos>0 and ypos<self.pwidget.height:
                    index = int(ypos/self.pwidget.hstep)
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
            GL.glUniform3f(self.myColor, 0.2, 0.2, 0.2) #dark grey
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
        # glUniform3f(self.myColor, 1.0, 1.0, 1.0) # white
        # for n in range(self.joint.noc):
        #    G0 = [self.joint.suggestions[index].indices_fall[n]]
        #    G1 = self.joint.mesh.indices_fall
        #    self.draw_geometries_with_excluded_area(G0,G1)

        # draw faces of subtracted part
        # glUniform3f(self.myColor, 1.0, 0.5, 0.5) # pink/red
        # for n in range(self.joint.noc):
        #    G0 = [self.joint.mesh.indices_fall[n]]
        #    G1 = self.joint.suggestions[index].indices_fall
        #    self.draw_geometries_with_excluded_area(G0,G1)

        # draw outlines
        GL.glUniform3f(self.myColor, 0.0, 0.0, 0.0) # black
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
        GL.glUniform3f(self.myColor,0.0,0.0,0.0) # black
        GL.glPushAttrib(GL.GL_ENABLE_BIT)
        GL.glLineWidth(1)
        GL.glLineStipple(3, 0xAAAA) #dashed line
        GL.glEnable(GL.GL_LINE_STIPPLE)
        if hidden and self.view.show_hidden_lines:
            for n in range(mesh.pjoint.noc):
                G0 = [mesh.indices_lns[n]]
                G1 = [mesh.indices_fall[n]]
                self.draw_geometries_with_excluded_area(G0,G1)
        GL.glPopAttrib()

        ############################ Draw visible lines #############################
        for n in range(mesh.pjoint.noc):
            if not mesh.mainmesh or (mesh.eval.interlocks[n] and self.view.show_feedback) or not self.view.show_feedback:
                GL.glUniform3f(self.myColor,0.0,0.0,0.0) # black
                GL.glLineWidth(lw)
            else:
                GL.glUniform3f(self.myColor,1.0,0.0,0.0) # red
                GL.glLineWidth(lw+1)
            G0 = [mesh.indices_lns[n]]
            G1 = mesh.indices_fall
            self.draw_geometries_with_excluded_area(G0,G1)


        if mesh.mainmesh:
            ################ When joint is fully open, draw dahsed lines ################
            if hidden and not self.view.hidden[0] and not self.view.hidden[1] and self.view.open_ratio==1+0.5*(mesh.pjoint.noc-2):
                GL.glUniform3f(self.myColor,0.0,0.0,0.0) # black
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
        GL.glUniform3f(self.myColor, col[0], col[1], col[2])
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
        GL.glUniform3f(self.myColor, col[0], col[1], col[2])
        for n in range(self.joint.mesh.pjoint.noc):
            if not self.joint.mesh.eval.connected[n]:
                self.draw_geometries([self.joint.mesh.indices_not_fcon[n]])

        # 1. Draw visible geometry
        col = [1.0, 0.2, 0.0] # red orange
        GL.glUniform3f(self.myColor, col[0], col[1], col[2])
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
                    GL.glUniform3f(self.myColor, col[0], col[1], col[2])
                    G0 = [self.joint.mesh.indices_not_fbridge[n][m]]
                    G1 = [self.joint.mesh.indices_not_fbridge[n][1 - m],
                          self.joint.mesh.indices_fall[1 - n],
                          self.joint.mesh.indices_not_fcon[n]] # needs reformulation for 3 components
                    self.draw_geometries_with_excluded_area(G0,G1)

    def checker(self):
        # 1. Draw hidden geometry
        GL.glUniform3f(self.myColor, 1.0, 0.2, 0.0) # red orange
        GL.glLineWidth(8)
        for n in range(self.joint.mesh.pjoint.noc):
            if self.joint.mesh.eval.checker[n]:
                self.draw_geometries([self.joint.mesh.indices_chess_lines[n]])
        GL.glUniform3f(self.myColor, 0.0, 0.0, 0.0) # back to black

    def arrows(self):
        #glClear(GL_DEPTH_BUFFER_BIT)
        GL.glUniform3f(self.myColor, 0.0, 0.0, 0.0)
        ############################## Direction arrows ################################
        for n in range(self.joint.noc):
            if (self.joint.mesh.eval.interlocks[n]): GL.glUniform3f(self.myColor, 0.0, 0.0, 0.0) # black
            else: GL.glUniform3f(self.myColor,1.0,0.0,0.0) # red
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
        GL.glUniform3f(self.myColor, col[0], col[1], col[2])
        for n in range(self.joint.noc):
            self.draw_geometries_with_excluded_area([self.joint.mesh.indices_fbrk[n]], [self.joint.mesh.indices_not_fbrk[n]])

        # Draw visible geometry
        col = [1.0, 1.0, 0.4] # light yellow
        GL.glUniform3f(self.myColor, col[0], col[1], col[2])
        self.draw_geometries_with_excluded_area(self.joint.mesh.indices_fbrk, self.joint.mesh.indices_not_fbrk)

    def milling_paths(self):
        if len(self.joint.mesh.indices_milling_path)==0: self.view.show_milling_path = False
        if self.view.show_milling_path:
            cols = [[1.0,0,0],[0,1.0,0],[0,0,1.0],[1.0,1.0,0],[0.0,1.0,1.0],[1.0,0,1.0]]
            GL.glLineWidth(3)
            for n in range(self.joint.noc):
                if self.joint.mesh.eval.fab_direction_ok[n]:
                    GL.glUniform3f(self.myColor,cols[n][0],cols[n][1],cols[n][2])
                    self.draw_geometries([self.joint.mesh.indices_milling_path[n]])

    def resizeEvent(self, event):
        print(' resizeEvent')
        self.resize(self.width(), self.height())

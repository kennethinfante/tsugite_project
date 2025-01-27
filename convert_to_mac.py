# gl_widget.py
class GLWidget(qgl.QGLWidget):
    def __init__(self, main_window=None, *args):
        fmt = qgl.QGLFormat()
        fmt.setVersion(2, 1)
        fmt.setProfile(qgl.QGLFormat.CoreProfile)
        fmt.setSampleBuffers(True)

        super().__init__(main_window, *args)
        
# utils.py
def initialize_shader(shader_code, shader_type):
    # Specify required OpenGL/GLSL version
    # major = GL.glGetInteger(GL.GL_MAJOR_VERSION)
    # minor = GL.glGetInteger(GL.GL_MINOR_VERSION)

    shader_code = '#version 120\n' + shader_code
    # Create empty shader object and return reference value
    shader_ref = GL.glCreateShader(shader_type)
    # Stores the source code in the shader
    GL.glShaderSource(shader_ref, shader_code)
    # Compiles source code previously stored in the shader object
    GL.glCompileShader(shader_ref)
    # Queries whether shader compile was successful
    compile_success = GL.glGetShaderiv(shader_ref, GL.GL_COMPILE_STATUS)
    if not compile_success:
        # Retrieve error message
        error_message = GL.glGetShaderInfoLog(shader_ref)
        # free memory used to store shader program
        GL.glDeleteShader(shader_ref)
        # Convert byte string to character string
        error_message = '\n' + error_message.decode('utf-8')
        # Raise exception: halt program and print error message
        raise Exception(error_message)
    # Compilation was successful; return shader reference value
    return shader_ref

def initialize_program(vertex_shader_code, fragment_shader_code):
    vertex_shader_ref = initialize_shader(vertex_shader_code, GL.GL_VERTEX_SHADER)
    fragment_shader_ref = initialize_shader(fragment_shader_code, GL.GL_FRAGMENT_SHADER)
    # Create empty program object and store reference to it
    program_ref = GL.glCreateProgram()
    # Attach previously compiled shader programs
    GL.glAttachShader(program_ref, vertex_shader_ref)
    GL.glAttachShader(program_ref, fragment_shader_ref)
    # Link vertex shader to fragment shader
    GL.glLinkProgram(program_ref)
    # queries whether program link was successful
    link_success = GL.glGetProgramiv(program_ref, GL.GL_LINK_STATUS)
    if not link_success:
        # Retrieve error message
        error_message = GL.glGetProgramInfoLog(program_ref)
        # free memory used to store program
        GL.glDeleteProgram(program_ref)
        # Convert byte string to character string
        error_message = '\n' + error_message.decode('utf-8')
        # Raise exception: halt application and print error message
        raise Exception(error_message)
    # Linking was successful; return program reference value
    return program_ref

# display.py
def create_color_shaders(self):
    """
    Note values explicity attrib and uniform locations are only availabl ein GL 3.3 and 4.3 respectively
    If to be use in versions lower than the above, the following are needed for vertex shaders
    """

    vertex_shader = """
    attribute vec3 position;
    attribute vec2 inTexCoords;
    uniform mat4 transform;
    uniform mat4 translate;
    attribute vec3 myColor;
    varying vec3 newColor;
    varying vec2 outTexCoords;
    void main()
    {
        gl_Position = transform* translate* vec4(position, 1.0f);
        newColor = myColor;
        outTexCoords = inTexCoords;
    }
    """

    fragment_shader = """
    attribute vec3 newColor;
    attribute vec2 outTexCoords;
    varying vec4 outColor;
    uniform sampler2D samplerTex;
    void main()
    {
        outColor = vec4(newColor, 1.0);
    }
    """
    # Compiling the shaders
    self.shader_col = Utils.initialize_program(vertex_shader, fragment_shader)

def create_texture_shaders(self):
    vertex_shader = """
    attribute vec3 position;
    attribute vec3 color;
    attribute vec2 inTexCoords;
    uniform mat4 transform;
    uniform mat4 translate;
    varying vec3 newColor;
    varying vec2 outTexCoords;
    void main()
    {
        gl_Position = transform* translate* vec4(position, 1.0f);
        newColor = color;
        outTexCoords = inTexCoords;
    }
    """

    fragment_shader = """
    attribute vec3 newColor;
    attribute vec2 outTexCoords;
    varying vec4 outColor;
    uniform sampler2D samplerTex;
    void main()
    {
        outColor = texture(samplerTex, outTexCoords);
    }
    """
    

    # Compiling the shaders
    self.shader_tex = Utils.initialize_program(vertex_shader, fragment_shader)
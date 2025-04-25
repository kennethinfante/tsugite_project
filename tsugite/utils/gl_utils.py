from OpenGL import GL

def setup_dashed_line_style(line_width=3, stipple_factor=2, stipple_pattern=0xAAAA):
    """
    Set up OpenGL state for drawing dashed lines.

    Args:
        line_width: Width of the lines
        stipple_factor: Stipple factor for dashed lines
        stipple_pattern: Stipple pattern for dashed lines
    """
    GL.glPushAttrib(GL.GL_ENABLE_BIT)
    GL.glLineWidth(line_width)
    GL.glEnable(GL.GL_LINE_STIPPLE)
    GL.glLineStipple(stipple_factor, stipple_pattern)

def restore_line_style():
    """
    Restore OpenGL state after drawing lines.
    """
    GL.glPopAttrib()

def set_color(location, color):
    """
    Set the current drawing color.

    Args:
        color: RGB color as a list or tuple of 3 values
    """
    GL.glUniform3f(location, color[0], color[1], color[2])

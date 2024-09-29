import glfw
import numpy as np
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import pyrr

# Shades GLSL
vertex_src = """

# version 330

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;

uniform mat4 model;
out vec3 v_color;

void main()
{
    gl_Position  = model * vec4(position, 1.0);
    v_color = color;
}
"""
fragment_src = """
# version 330

in vec3 v_color;

out vec4 out_color;

void main()
{
    out_color = vec4(v_color, 1.0);
}
"""

# window initialization
glfw.init()
window = glfw.create_window(800, 600, "Activity Window", None, None)

if not window:
    glfw.terminate()
    exit()

glfw.make_context_current(window)

# Object Creation
# Vertex Definition

cube_vertices = [-0.5, 0.5, -0.5, 1.0, 0.0, 0.0,
                 -0.5, -0.5, -0.5, 0.0, 1.0, 0.0,
                 0.5, -0.5, -0.5, 0.0, 0.0, 1.0,
                 0.5, 0.5, -0.5, 1.0, 0.0, 1.0,
                 -0.5, 0.5, 0.5, 1.0, 1.0, 0.0,
                 -0.5, -0.5, 0.5, 0.0, 0.0, 1.0,
                 0.5, -0.5, 0.5, 0.0, 1.0, 0.0,
                 0.5, 0.5, 0.5, 1.0, 0.0, 0.0]

cube_indices = [0, 1, 2, 0, 2, 3,
                3, 2, 6, 3, 6, 7,
                7, 6, 5, 7, 5, 4,
                4, 5, 1, 4, 1, 0,
                4, 0, 3, 4, 3, 7,
                1, 5, 6, 1, 6, 2]
cube_vertices = np.array(cube_vertices, dtype=np.float32)
cube_indices = np.array(cube_indices, dtype=np.uint32)


pyramid_vertices = [0.0, 0.5, 0.0, 1.0, 0.0, 0.0,
                    -0.5, -0.5, -0.5, 0.0, 1.0, 0.0,
                    0.5, -0.5, -0.5, 0.0, 0.0, 1.0,
                    0.5, -0.5, 0.5, 1.0, 1.0, 0.0,
                    -0.5, -0.5, 0.5, 0.0, 1.0, 1.0]
# triangle creation by indexing method
pyramid_indices = [0, 1, 2,
                   0, 2, 3,
                   0, 3, 4,
                   0, 4, 1]
pyramid_vertices = np.array(pyramid_vertices, dtype=np.float32)
pyramid_indices = np.array(pyramid_indices, dtype=np.uint32)
# Sending Data
# vao = generate a vao for an object
vao = glGenVertexArrays(1)
glBindVertexArray(vao)
# vbo
vbo = glGenBuffers(1)
glBindBuffer(GL_ARRAY_BUFFER, vbo)
glBufferData(GL_ARRAY_BUFFER, pyramid_vertices.nbytes, pyramid_vertices, GL_STATIC_DRAW)
# ebo
ebo = glGenBuffers(1)
glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
glBufferData(GL_ELEMENT_ARRAY_BUFFER, pyramid_indices.nbytes, pyramid_indices, GL_STATIC_DRAW)
# Position
glEnableVertexAttribArray(0)
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
# color
glEnableVertexAttribArray(1)
glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))
glBindVertexArray(0)

vao1 = glGenVertexArrays(1)
glBindVertexArray(vao1)

vbo1 = glGenBuffers(1)
glBindBuffer(GL_ARRAY_BUFFER, vbo1)
glBufferData(GL_ARRAY_BUFFER, cube_vertices.nbytes, cube_vertices, GL_STATIC_DRAW)

ebo1 = glGenBuffers(1)
glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo1)
glBufferData(GL_ELEMENT_ARRAY_BUFFER, cube_indices.nbytes, cube_indices, GL_STATIC_DRAW)

glEnableVertexAttribArray(0)
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
# color
glEnableVertexAttribArray(1)
glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))
glBindVertexArray(0)

print(vao)
print(vbo)
print(ebo)
print(vao1)
print(vbo1)
print(ebo1)
# Transformation
# SCALING SCALE(X, Y, Z)
scale = pyrr.Matrix44.from_scale(pyrr.Vector3([0.5, 0.65, 0.5]))
trans = pyrr.matrix44.create_from_translation([-0.5, 0, 0])# left
trans1 = pyrr.matrix44.create_from_translation([0.5, 0, 0])# right

# VIEW MATRIX

# PROJECTION MATRIX

# Shader Space
shader = compileProgram(compileShader(vertex_src, GL_VERTEX_SHADER), compileShader(fragment_src, GL_FRAGMENT_SHADER))
glUseProgram(shader)
model_loc = glGetUniformLocation(shader, "model")


# Rendering Space
# Set up the color for background
glClearColor(0.1, 0.1, 0.1, 1)
glEnable(GL_DEPTH_TEST) # active the z-buffer

while not glfw.window_should_close(window):
    glfw.poll_events() # get the events happening in the  window
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # Object Transformation

    # Object Assembly and Rendering
# left
    rot_y = pyrr.matrix44.create_from_y_rotation(5 * glfw.get_time())
    model = pyrr.matrix44.multiply(scale, rot_y)
    model = pyrr.matrix44.multiply(model, trans)
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
    glBindVertexArray(vao1)
    glDrawElements(GL_TRIANGLES, len(cube_indices), GL_UNSIGNED_INT, None)
    glBindVertexArray(0)

# right
    rot_y = pyrr.matrix44.create_from_y_rotation(5 * glfw.get_time())
    model = pyrr.matrix44.multiply(scale, rot_y)
    model = pyrr.matrix44.multiply(model, trans1)
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
    glBindVertexArray(vao)
    glDrawElements(GL_TRIANGLES, len(pyramid_indices), GL_UNSIGNED_INT, None)
    glBindVertexArray(0)

    glfw.swap_interval(1)
    glfw.swap_buffers(window)

# Memory Clearing
glDeleteBuffers(4, [vbo, ebo, ebo1, vbo1])
glDeleteVertexArrays(2, [vao, vao1])
glfw.terminate()
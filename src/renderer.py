import OpenGL.GL as gl
import numpy as np
import glm
from OpenGL.GL import shaders
from ldraw import get_color
import ctypes

class Renderer:
    def __init__(self):
        self.prog = self._create_program()
        self.vaos = {} # part_id -> (vao, count)
        self._init_grid()
        self._init_marker()

    def _create_program(self):
        vertex_shader = """
        #version 330
        in vec3 in_position;
        in vec3 in_normal;
        in vec4 in_color;

        out vec3 v_normal;
        out vec3 v_frag_pos; // World space position
        out vec4 v_color;

        uniform mat4 m_model;
        uniform mat4 m_view;
        uniform mat4 m_proj;

        void main() {
            vec4 pos = m_model * vec4(in_position, 1.0);
            v_frag_pos = pos.xyz;
            // Normal matrix should be inverse transpose to handle non-uniform scale
            v_normal = mat3(transpose(inverse(m_model))) * in_normal; 
            v_color = in_color;
            gl_Position = m_proj * m_view * pos;
        }
        """

        fragment_shader = """
        #version 330
        in vec3 v_normal;
        in vec3 v_frag_pos;
        in vec4 v_color;

        out vec4 f_color;

        uniform vec3 view_pos;
        uniform vec3 light_pos;
        
        // Simple Plastic Material
        const float shininess = 64.0;
        const float specular_strength = 0.5;

        void main() {
            vec3 norm = normalize(v_normal);
            vec3 view_dir = normalize(view_pos - v_frag_pos);
            
            // 1. Directional Light (Main Key Light)
            // Simulating a sun-like directional light from light_pos
            vec3 light_dir = normalize(light_pos - v_frag_pos); 
            
            // Ambient
            float ambient_strength = 0.3;
            vec3 ambient = ambient_strength * v_color.rgb;
            
            // Diffuse
            float diff = max(dot(norm, light_dir), 0.0);
            vec3 diffuse = diff * v_color.rgb;
            
            // Specular (Blinn-Phong)
            vec3 halfway_dir = normalize(light_dir + view_dir);
            float spec = pow(max(dot(norm, halfway_dir), 0.0), shininess);
            vec3 specular = specular_strength * spec * vec3(1.0, 1.0, 1.0); // White highlight
            
            // Rim Light (Fresnel-ish)
            float rim = 1.0 - max(dot(view_dir, norm), 0.0);
            rim = pow(rim, 3.0);
            vec3 rim_color = 0.2 * rim * vec3(1.0);

            vec3 result = ambient + diffuse + specular + rim_color;
            
            // Gamma Correction
            float gamma = 2.2;
            result = pow(result, vec3(1.0/gamma));
            
            f_color = vec4(result, v_color.a);
        }
        """
        
        vs = shaders.compileShader(vertex_shader, gl.GL_VERTEX_SHADER)
        fs = shaders.compileShader(fragment_shader, gl.GL_FRAGMENT_SHADER)
        return shaders.compileProgram(vs, fs)

    def _init_marker(self):
        # Small cube for markers (studs/tubes)
        size = 2.0
        # 8 corners
        v = [
            -size, -size, -size,
             size, -size, -size,
             size,  size, -size,
            -size,  size, -size,
            -size, -size,  size,
             size, -size,  size,
             size,  size,  size,
            -size,  size,  size,
        ]
        # Triangles
        indices = [
            0, 1, 2, 2, 3, 0, # Back
            4, 5, 6, 6, 7, 4, # Front
            0, 1, 5, 5, 4, 0, # Bottom
            2, 3, 7, 7, 6, 2, # Top
            0, 3, 7, 7, 4, 0, # Left
            1, 2, 6, 6, 5, 1, # Right
        ]
        
        verts = []
        norms = [] # Fake normals
        
        for i in indices:
            idx = i * 3
            verts.extend([v[idx], v[idx+1], v[idx+2]])
            norms.extend([0, 1, 0]) # Dummy
            
        verts = np.array(verts, dtype='f4')
        norms = np.array(norms, dtype='f4')
        colors = np.array([1, 1, 0, 1] * len(indices), dtype='f4') # Yellow default
        
        self.marker_vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.marker_vao)
        
        # VBOs
        vbo_pos = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo_pos)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, verts.nbytes, verts, gl.GL_STATIC_DRAW)
        pos_loc = gl.glGetAttribLocation(self.prog, 'in_position')
        gl.glEnableVertexAttribArray(pos_loc)
        gl.glVertexAttribPointer(pos_loc, 3, gl.GL_FLOAT, False, 0, None)
        
        vbo_norm = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo_norm)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, norms.nbytes, norms, gl.GL_STATIC_DRAW)
        norm_loc = gl.glGetAttribLocation(self.prog, 'in_normal')
        gl.glEnableVertexAttribArray(norm_loc)
        gl.glVertexAttribPointer(norm_loc, 3, gl.GL_FLOAT, False, 0, None)

        vbo_col = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo_col)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, colors.nbytes, colors, gl.GL_STATIC_DRAW)
        col_loc = gl.glGetAttribLocation(self.prog, 'in_color')
        gl.glEnableVertexAttribArray(col_loc)
        gl.glVertexAttribPointer(col_loc, 4, gl.GL_FLOAT, False, 0, None)
        
        gl.glBindVertexArray(0)
        self.marker_count = len(verts) // 3

    def render_markers(self, positions, color=(1, 1, 0, 1), view=None, proj=None):
        gl.glUseProgram(self.prog)
        
        # Set View/Proj
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(self.prog, 'm_view'), 1, gl.GL_FALSE, glm.value_ptr(view))
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(self.prog, 'm_proj'), 1, gl.GL_FALSE, glm.value_ptr(proj))
        gl.glUniform3f(gl.glGetUniformLocation(self.prog, 'light_pos'), 500.0, -500.0, 500.0)

        gl.glBindVertexArray(self.marker_vao)
        
        model_loc = gl.glGetUniformLocation(self.prog, 'm_model')
        
        for pos in positions:
            mat = glm.translate(glm.mat4(1.0), pos)
            gl.glUniformMatrix4fv(model_loc, 1, gl.GL_FALSE, glm.value_ptr(mat))
            gl.glDrawArrays(gl.GL_TRIANGLES, 0, self.marker_count)
            
        gl.glBindVertexArray(0)
        gl.glUseProgram(0)

    def _init_grid(self):
        size = 2000.0
        step = 100.0
        
        vertices = []
        colors = []
        
        # Grid lines (Grey)
        for i in range(int(-size), int(size)+1, int(step)):
            # X lines
            vertices.extend([-size, 0, i, size, 0, i])
            colors.extend([0.5, 0.5, 0.5, 1.0] * 2)
            # Z lines
            vertices.extend([i, 0, -size, i, 0, size])
            colors.extend([0.5, 0.5, 0.5, 1.0] * 2)
            
        # Axes (RGB)
        vertices.extend([0, 0, 0, 1000, 0, 0])
        colors.extend([1, 0, 0, 1] * 2)
        vertices.extend([0, 0, 0, 0, 1000, 0])
        colors.extend([0, 1, 0, 1] * 2)
        vertices.extend([0, 0, 0, 0, 0, 1000])
        colors.extend([0, 0, 1, 1] * 2)
        
        vertices = np.array(vertices, dtype='f4')
        colors = np.array(colors, dtype='f4')
        
        # Grid Shader
        vs_src = """
        #version 330
        in vec3 in_position;
        in vec4 in_color;
        out vec4 v_color;
        uniform mat4 m_view;
        uniform mat4 m_proj;
        void main() {
            gl_Position = m_proj * m_view * vec4(in_position, 1.0);
            v_color = in_color;
        }
        """
        fs_src = """
        #version 330
        in vec4 v_color;
        out vec4 f_color;
        void main() {
            f_color = v_color;
        }
        """
        
        vs = shaders.compileShader(vs_src, gl.GL_VERTEX_SHADER)
        fs = shaders.compileShader(fs_src, gl.GL_FRAGMENT_SHADER)
        self.grid_prog = shaders.compileProgram(vs, fs)
        
        self.grid_vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.grid_vao)
        
        vbo_pos = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo_pos)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices.nbytes, vertices, gl.GL_STATIC_DRAW)
        pos_loc = gl.glGetAttribLocation(self.grid_prog, 'in_position')
        gl.glEnableVertexAttribArray(pos_loc)
        gl.glVertexAttribPointer(pos_loc, 3, gl.GL_FLOAT, False, 0, None)
        
        vbo_col = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo_col)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, colors.nbytes, colors, gl.GL_STATIC_DRAW)
        col_loc = gl.glGetAttribLocation(self.grid_prog, 'in_color')
        gl.glEnableVertexAttribArray(col_loc)
        gl.glVertexAttribPointer(col_loc, 4, gl.GL_FLOAT, False, 0, None)
        
        gl.glBindVertexArray(0)
        self.grid_count = len(vertices) // 3

    def render(self, scene, view, proj):
        gl.glEnable(gl.GL_DEPTH_TEST)
        # LDraw models often have mixed winding orders or open geometry.
        # Disabling backface culling ensures all faces are visible.
        gl.glDisable(gl.GL_CULL_FACE)
        
        # Render Grid
        gl.glUseProgram(self.grid_prog)
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(self.grid_prog, 'm_view'), 1, gl.GL_FALSE, glm.value_ptr(view))
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(self.grid_prog, 'm_proj'), 1, gl.GL_FALSE, glm.value_ptr(proj))
        
        gl.glBindVertexArray(self.grid_vao)
        gl.glDrawArrays(gl.GL_LINES, 0, self.grid_count)
        gl.glBindVertexArray(0)
        
        # Render Parts
        gl.glUseProgram(self.prog)
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(self.prog, 'm_view'), 1, gl.GL_FALSE, glm.value_ptr(view))
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(self.prog, 'm_proj'), 1, gl.GL_FALSE, glm.value_ptr(proj))
        
        # Lighting Uniforms
        gl.glUniform3f(gl.glGetUniformLocation(self.prog, 'light_pos'), 300.0, 500.0, 500.0)
        
        # Calculate Camera Position
        inv_view = glm.inverse(view)
        cam_pos = glm.vec3(inv_view[3])
        gl.glUniform3f(gl.glGetUniformLocation(self.prog, 'view_pos'), cam_pos.x, cam_pos.y, cam_pos.z)
        
        model_loc = gl.glGetUniformLocation(self.prog, 'm_model')
        
        all_objects = scene.get_all_objects()
        for obj in all_objects:
            part = obj.part
            model_matrix = obj.world_matrix
            
            vao, count = self.upload_part(part)
            if vao:
                gl.glUniformMatrix4fv(model_loc, 1, gl.GL_FALSE, glm.value_ptr(model_matrix))
                gl.glBindVertexArray(vao)
                gl.glDrawArrays(gl.GL_TRIANGLES, 0, count)
                gl.glBindVertexArray(0)
        
        gl.glUseProgram(0)

    def render_outline(self, part, view, proj, color=(1, 0.5, 0, 1)):
        # Render part in wireframe/outline
        gl.glUseProgram(self.prog)
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(self.prog, 'm_view'), 1, gl.GL_FALSE, glm.value_ptr(view))
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(self.prog, 'm_proj'), 1, gl.GL_FALSE, glm.value_ptr(proj))
        gl.glUniform3f(gl.glGetUniformLocation(self.prog, 'light_pos'), 500.0, -500.0, 500.0)
        
        # Override color in shader? 
        # The current shader uses vertex color. We need a uniform override or a new shader.
        # For simplicity, let's use a "solid color" mode in the shader or just ignore vertex color.
        # Actually, let's just create a simple solid color shader for outline to be safe/clean.
        pass

    def _init_outline_shader(self):
        vs = """
        #version 330
        in vec3 in_position;
        uniform mat4 m_model;
        uniform mat4 m_view;
        uniform mat4 m_proj;
        uniform float offset;
        void main() {
            vec4 pos = m_model * vec4(in_position, 1.0);
            // Push vertex along normal? No normals here.
            // Just standard transform. PolygonOffset will handle depth.
            gl_Position = m_proj * m_view * pos;
        }
        """
        fs = """
        #version 330
        out vec4 f_color;
        uniform vec4 u_color;
        void main() {
            f_color = u_color;
        }
        """
        try:
            v = shaders.compileShader(vs, gl.GL_VERTEX_SHADER)
            f = shaders.compileShader(fs, gl.GL_FRAGMENT_SHADER)
            self.outline_prog = shaders.compileProgram(v, f)
        except Exception as e:
            print(f"Outline shader error: {e}")
            self.outline_prog = None

    def render_selected_outline(self, obj, view, proj):
        if not hasattr(self, 'outline_prog'):
            self._init_outline_shader()
            
        if not self.outline_prog: return

        gl.glUseProgram(self.outline_prog)
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(self.outline_prog, 'm_view'), 1, gl.GL_FALSE, glm.value_ptr(view))
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(self.outline_prog, 'm_proj'), 1, gl.GL_FALSE, glm.value_ptr(proj))
        gl.glUniform4f(gl.glGetUniformLocation(self.outline_prog, 'u_color'), 1.0, 0.5, 0.0, 1.0) # Orange
        
        model_loc = gl.glGetUniformLocation(self.outline_prog, 'm_model')
        
        # Enable Polygon Offset to draw wireframe on top of solid
        gl.glEnable(gl.GL_POLYGON_OFFSET_LINE)
        gl.glPolygonOffset(-1.0, -1.0) # Move closer to camera
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
        
        vao, count = self.upload_part(obj.part)
        if vao:
            gl.glUniformMatrix4fv(model_loc, 1, gl.GL_FALSE, glm.value_ptr(obj.world_matrix))
            gl.glBindVertexArray(vao)
            gl.glDrawArrays(gl.GL_TRIANGLES, 0, count)
            gl.glBindVertexArray(0)
            
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
        gl.glDisable(gl.GL_POLYGON_OFFSET_LINE)
        gl.glUseProgram(0)

    def upload_part(self, part):
        if part in self.vaos:
            return self.vaos[part]
        
        vertices = np.array(part.vertices, dtype='f4')
        normals = np.array(part.normals, dtype='f4')
        
        colors = []
        for c in part.colors:
            colors.extend(get_color(c))
        colors = np.array(colors, dtype='f4')
        
        if len(vertices) == 0:
            return None, 0

        vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(vao)
        
        vbo_pos = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo_pos)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices.nbytes, vertices, gl.GL_STATIC_DRAW)
        pos_loc = gl.glGetAttribLocation(self.prog, 'in_position')
        gl.glEnableVertexAttribArray(pos_loc)
        gl.glVertexAttribPointer(pos_loc, 3, gl.GL_FLOAT, False, 0, None)
        
        vbo_norm = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo_norm)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, normals.nbytes, normals, gl.GL_STATIC_DRAW)
        norm_loc = gl.glGetAttribLocation(self.prog, 'in_normal')
        gl.glEnableVertexAttribArray(norm_loc)
        gl.glVertexAttribPointer(norm_loc, 3, gl.GL_FLOAT, False, 0, None)

        vbo_col = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo_col)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, colors.nbytes, colors, gl.GL_STATIC_DRAW)
        col_loc = gl.glGetAttribLocation(self.prog, 'in_color')
        gl.glEnableVertexAttribArray(col_loc)
        gl.glVertexAttribPointer(col_loc, 4, gl.GL_FLOAT, False, 0, None)
        
        gl.glBindVertexArray(0)
        
        count = len(vertices) // 3
        self.vaos[part] = (vao, count)
        return vao, count

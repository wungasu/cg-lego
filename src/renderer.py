import OpenGL.GL as gl
import numpy as np
import glm
from OpenGL.GL import shaders
from ldraw import get_color
import ctypes

class Renderer:
    def __init__(self):
        self.prog = self._create_program()
        self.vaos = {} 
        self._init_grid()
        self._init_light_viz()
        self._init_marker()

        self.shadow_width = 2048
        self.shadow_height = 2048
        self._init_shadow_map()
        self.shadow_prog = self._create_shadow_program()

        self.light_pos = glm.vec3(300.0, 500.0, 500.0)
        self.light_color = glm.vec3(1.0, 1.0, 1.0)
        self.ambient_strength = 0.3

        self.default_shininess = 64.0
        self.default_specular = 0.5
        self.default_reflectivity = 0.0
        self.default_metallic = 0.0
        self.default_rim = 0.2 

        self.env_map_size = 512
        self._init_env_map()

        self._init_outline_shader()

    def _init_env_map(self):
        self.env_map_fbo = gl.glGenFramebuffers(1)
        self.env_map = gl.glGenTextures(1)
        
        gl.glBindTexture(gl.GL_TEXTURE_CUBE_MAP, self.env_map)
        for i in range(6):
            gl.glTexImage2D(gl.GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, gl.GL_RGB, 
                            self.env_map_size, self.env_map_size, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, None)
                            
        gl.glTexParameteri(gl.GL_TEXTURE_CUBE_MAP, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_CUBE_MAP, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_CUBE_MAP, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_CUBE_MAP, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_CUBE_MAP, gl.GL_TEXTURE_WRAP_R, gl.GL_CLAMP_TO_EDGE)

        self.env_map_depth = gl.glGenRenderbuffers(1)
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, self.env_map_depth)
        gl.glRenderbufferStorage(gl.GL_RENDERBUFFER, gl.GL_DEPTH_COMPONENT, self.env_map_size, self.env_map_size)
        
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.env_map_fbo)
        gl.glFramebufferRenderbuffer(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT, gl.GL_RENDERBUFFER, self.env_map_depth)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

    def _init_shadow_map(self):
        self.depth_map_fbo = gl.glGenFramebuffers(1)
        self.depth_map = gl.glGenTextures(1)
        
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.depth_map)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_DEPTH_COMPONENT, 
                        self.shadow_width, self.shadow_height, 0, 
                        gl.GL_DEPTH_COMPONENT, gl.GL_FLOAT, None)
        
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_BORDER)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_BORDER)
        borderColor = [1.0, 1.0, 1.0, 1.0]
        gl.glTexParameterfv(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_BORDER_COLOR, borderColor)
        
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.depth_map_fbo)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT, 
                                  gl.GL_TEXTURE_2D, self.depth_map, 0)
        gl.glDrawBuffer(gl.GL_NONE)
        gl.glReadBuffer(gl.GL_NONE)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

    def _create_shadow_program(self):
        vertex_shader = """
        #version 330 core
        layout (location = 0) in vec3 in_position;
        
        uniform mat4 lightSpaceMatrix;
        uniform mat4 m_model;
        
        void main()
        {
            gl_Position = lightSpaceMatrix * m_model * vec4(in_position, 1.0);
        }
        """
        fragment_shader = """
        #version 330 core
        void main()
        {             
            // gl_FragDepth = gl_FragCoord.z;
        }  
        """
        vs = shaders.compileShader(vertex_shader, gl.GL_VERTEX_SHADER)
        fs = shaders.compileShader(fragment_shader, gl.GL_FRAGMENT_SHADER)
        return shaders.compileProgram(vs, fs)

    def _create_program(self):
        vertex_shader = """
        #version 330
        in vec3 in_position;
        in vec3 in_normal;
        in vec4 in_color;

        out vec3 v_normal;
        out vec3 v_frag_pos;
        out vec4 v_color;
        out vec4 v_frag_pos_light_space;

        uniform mat4 m_model;
        uniform mat4 m_view;
        uniform mat4 m_proj;
        uniform mat4 lightSpaceMatrix;

        void main() {
            vec4 pos = m_model * vec4(in_position, 1.0);
            v_frag_pos = pos.xyz;
            v_normal = mat3(transpose(inverse(m_model))) * in_normal; 
            v_color = in_color;
            v_frag_pos_light_space = lightSpaceMatrix * pos;
            gl_Position = m_proj * m_view * pos;
        }
        """

        fragment_shader = """
        #version 330
        in vec3 v_normal;
        in vec3 v_frag_pos;
        in vec4 v_color;
        in vec4 v_frag_pos_light_space;

        out vec4 f_color;

        uniform vec3 view_pos;
        uniform vec3 light_pos;
        uniform sampler2D shadowMap;
        
        // Material Uniforms
        uniform float shininess;
        uniform float specular_strength;
        uniform float reflectivity;
        uniform float metallic;
        uniform float rim_strength;
        
        uniform samplerCube skybox; // For reflection if implemented
        uniform int use_reflection; // 0 = no, 1 = yes

        // ACES Tone Mapping
        vec3 aces_tone_mapping(vec3 x) {
            const float a = 2.51;
            const float b = 0.03;
            const float c = 2.43;
            const float d = 0.59;
            const float e = 0.14;
            return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
        }

        float ShadowCalculation(vec4 fragPosLightSpace)
        {
            // perform perspective divide
            vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
            // transform to [0,1] range
            projCoords = projCoords * 0.5 + 0.5;
            
            // keep the shadow at 0.0 when outside the far_plane region of the light's frustum.
            if(projCoords.z > 1.0)
                return 0.0;
            
            // get closest depth value from light's perspective (using [0,1] range fragPosLight as coords)
            float closestDepth = texture(shadowMap, projCoords.xy).r; 
            // get depth of current fragment from light's perspective
            float currentDepth = projCoords.z;
            
            vec3 normal = normalize(v_normal);
            vec3 lightDir = normalize(light_pos - v_frag_pos);
            
            // Reduced bias for better accuracy
            float bias = max(0.002 * (1.0 - dot(normal, lightDir)), 0.0002);  

            // PCF
            float shadow = 0.0;
            vec2 texelSize = 1.0 / textureSize(shadowMap, 0);
            for(int x = -1; x <= 1; ++x)
            {
                for(int y = -1; y <= 1; ++y)
                {
                    float pcfDepth = texture(shadowMap, projCoords.xy + vec2(x, y) * texelSize).r; 
                    shadow += currentDepth - bias > pcfDepth ? 1.0 : 0.0;        
                }    
            }
            shadow /= 9.0;
            
            return shadow;
        }

        void main() {
            vec3 norm = normalize(v_normal);
            vec3 view_dir = normalize(view_pos - v_frag_pos);
            vec3 light_dir = normalize(light_pos - v_frag_pos); 
            
            // Ambient (Boosted for Tone Mapping)
            float ambient_strength = 0.5;
            vec3 ambient = ambient_strength * v_color.rgb;
            
            // Diffuse
            float diff = max(dot(norm, light_dir), 0.0);
            // Metallic: Metals have no diffuse (black), Plastic has colored diffuse
            vec3 diffuse_albedo = mix(v_color.rgb, vec3(0.0), metallic);
            vec3 diffuse = diff * diffuse_albedo;
            
            // Specular
            vec3 halfway_dir = normalize(light_dir + view_dir);
            float spec = pow(max(dot(norm, halfway_dir), 0.0), shininess);
            // Metallic: Metals have colored specular (albedo), Plastic has white specular
            vec3 spec_color = mix(vec3(1.0), v_color.rgb, metallic);
            vec3 specular = specular_strength * spec * spec_color; 
            
            // Rim Lighting (Backlight)
            float rim = 1.0 - max(dot(view_dir, norm), 0.0);
            rim = smoothstep(0.4, 1.0, rim);
            // Rim is stronger when looking at light (backlight)? Or just generic rim?
            // Let's add a directional term to rim so it appears mostly on the lit side or backlit side?
            // Standard rim is view dependent. Let's just add it.
            // To make "backlit" details popping, we can boost rim based on light direction too.
            // float rim_light_align = max(dot(view_dir, -light_dir), 0.0);
            vec3 rim_emission = rim * rim_strength * vec3(1.0);

            // Shadow
            float shadow = ShadowCalculation(v_frag_pos_light_space);       
            
            // Combine
            // Ambient + (Diffuse + Specular) * Shadow + Rim
            // Rim should be affected by shadow? Usually rim represents light wrapping around.
            // If we want "backlit" details, rim should persist or be stronger in shadow?
            // Let's leave Rim outside shadow for the "glowing edge" effect.
            vec3 lighting = ambient + (1.0 - shadow) * (diffuse + specular) + rim_emission;    
            
            // Reflection
            vec3 final_color = lighting;
            
            if (use_reflection == 1) {
                vec3 reflection_vec = reflect(-view_dir, norm);
                // Sample from skybox (environment map)
                vec3 env_color = texture(skybox, reflection_vec).rgb;
                
                // Fresnel
                float fresnel = 0.04 + (1.0 - 0.04) * pow(1.0 - clamp(dot(norm, view_dir), 0.0, 1.0), 5.0);
                
                // Mix based on reflectivity and Fresnel
                float mix_factor = clamp(reflectivity + fresnel * 0.5, 0.0, 1.0);
                final_color = mix(final_color, env_color, mix_factor);
            }

            // Tone Mapping
            final_color = aces_tone_mapping(final_color);

            // Gamma
            float gamma = 2.2;
            final_color = pow(final_color, vec3(1.0/gamma));
            
            f_color = vec4(final_color, v_color.a);
        }
        """

        vs = shaders.compileShader(vertex_shader, gl.GL_VERTEX_SHADER)
        fs = shaders.compileShader(fragment_shader, gl.GL_FRAGMENT_SHADER)
        return shaders.compileProgram(vs, fs)

    def _init_marker(self):

        size = 2.0

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
        norms = [] 
        
        for i in indices:
            idx = i * 3
            verts.extend([v[idx], v[idx+1], v[idx+2]])
            norms.extend([0, 1, 0]) 
            
        verts = np.array(verts, dtype='f4')
        norms = np.array(norms, dtype='f4')
        colors = np.array([1, 1, 0, 1] * len(indices), dtype='f4') 
        
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

    def render_light_source(self, view, proj):
        gl.glUseProgram(self.grid_prog)
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(self.grid_prog, 'm_view'), 1, gl.GL_FALSE, glm.value_ptr(view))
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(self.grid_prog, 'm_proj'), 1, gl.GL_FALSE, glm.value_ptr(proj))
        
        line_verts = np.array([
            0.0, 0.0, 0.0,
            self.light_pos.x, self.light_pos.y, self.light_pos.z
        ], dtype='f4')
        
        gl.glBindVertexArray(self.light_line_vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.light_line_vbo)
        gl.glBufferSubData(gl.GL_ARRAY_BUFFER, 0, line_verts.nbytes, line_verts)
        

        ident = glm.mat4(1.0)
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(self.grid_prog, 'm_model'), 1, gl.GL_FALSE, glm.value_ptr(ident))
        
        gl.glDrawArrays(gl.GL_LINES, 0, 2)
        

        gl.glDisable(gl.GL_DEPTH_TEST) 
        

        model = glm.translate(glm.mat4(1.0), self.light_pos)
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(self.grid_prog, 'm_model'), 1, gl.GL_FALSE, glm.value_ptr(model))
        
        gl.glBindVertexArray(self.light_vao)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, self.light_cube_count)
        
        gl.glEnable(gl.GL_DEPTH_TEST)
        
        gl.glBindVertexArray(0)
        gl.glUseProgram(0)

    def _init_grid(self):
        size = 2000.0
        step = 100.0
        
        vertices = []
        colors = []

        for i in range(int(-size), int(size)+1, int(step)):

            vertices.extend([-size, 0, i, size, 0, i])
            colors.extend([0.5, 0.5, 0.5, 1.0] * 2)

            vertices.extend([i, 0, -size, i, 0, size])
            colors.extend([0.5, 0.5, 0.5, 1.0] * 2)
            

        y_off = 0.5
        vertices.extend([0, y_off, 0, 1000, y_off, 0])
        colors.extend([1, 0, 0, 1] * 2)
        vertices.extend([0, y_off, 0, 0, 1000, 0])
        colors.extend([0, 1, 0, 1] * 2)
        vertices.extend([0, y_off, 0, 0, y_off, 1000])
        colors.extend([0, 0, 1, 1] * 2)
        
        vertices = np.array(vertices, dtype='f4')
        colors = np.array(colors, dtype='f4')

        vs_src = """
        #version 330
        in vec3 in_position;
        in vec4 in_color;
        out vec4 v_color;
        uniform mat4 m_view;
        uniform mat4 m_proj;
        uniform mat4 m_model;
        void main() {
            gl_Position = m_proj * m_view * m_model * vec4(in_position, 1.0);
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

    def _init_light_viz(self):

        self.light_line_vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.light_line_vao)

        self.light_line_vbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.light_line_vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, 2 * 3 * 4, None, gl.GL_DYNAMIC_DRAW)
        
        pos_loc = gl.glGetAttribLocation(self.grid_prog, 'in_position')
        gl.glEnableVertexAttribArray(pos_loc)
        gl.glVertexAttribPointer(pos_loc, 3, gl.GL_FLOAT, False, 0, None)

        self.light_line_col_vbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.light_line_col_vbo)
        colors = np.array([1, 1, 0, 1] * 2, dtype='f4') # Yellow
        gl.glBufferData(gl.GL_ARRAY_BUFFER, colors.nbytes, colors, gl.GL_STATIC_DRAW)
        
        col_loc = gl.glGetAttribLocation(self.grid_prog, 'in_color')
        gl.glEnableVertexAttribArray(col_loc)
        gl.glVertexAttribPointer(col_loc, 4, gl.GL_FLOAT, False, 0, None)
        
        gl.glBindVertexArray(0)

        size = 10.0
        v = [
            -size, -size, -size, size, -size, -size, size,  size, -size, -size,  size, -size,
            -size, -size,  size, size, -size,  size, size,  size,  size, -size,  size,  size,
        ]
        indices = [
            0, 1, 2, 2, 3, 0, 4, 5, 6, 6, 7, 4, 0, 1, 5, 5, 4, 0,
            2, 3, 7, 7, 6, 2, 0, 3, 7, 7, 4, 0, 1, 2, 6, 6, 5, 1,
        ]
        verts = []
        for i in indices:
            idx = i * 3
            verts.extend([v[idx], v[idx+1], v[idx+2]])
            
        verts = np.array(verts, dtype='f4')
        colors = np.array([1, 1, 0, 1] * (len(verts)//3), dtype='f4')
        
        self.light_cube_count = len(verts) // 3
        
        self.light_vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.light_vao)
        
        vbo_pos = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo_pos)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, verts.nbytes, verts, gl.GL_STATIC_DRAW)
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

    def render_shadow_map(self, scene, light_space_matrix):
        gl.glViewport(0, 0, self.shadow_width, self.shadow_height)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.depth_map_fbo)
        gl.glClear(gl.GL_DEPTH_BUFFER_BIT)
        gl.glUseProgram(self.shadow_prog)
        
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(self.shadow_prog, 'lightSpaceMatrix'), 1, gl.GL_FALSE, glm.value_ptr(light_space_matrix))
        
        model_loc = gl.glGetUniformLocation(self.shadow_prog, 'm_model')

        
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
                
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        gl.glUseProgram(0)

    def render(self, scene, view, proj):

        light_projection = glm.ortho(-1000.0, 1000.0, -1000.0, 1000.0, 1.0, 2000.0) 
        light_view = glm.lookAt(self.light_pos, glm.vec3(0.0, 0.0, 0.0), glm.vec3(0.0, 1.0, 0.0))
        light_space_matrix = light_projection * light_view

        viewport = gl.glGetIntegerv(gl.GL_VIEWPORT)
        self.render_shadow_map(scene, light_space_matrix)
        

        gl.glViewport(0, 0, self.env_map_size, self.env_map_size)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.env_map_fbo)
        
        center = glm.vec3(0.0, 0.0, 0.0) 
        views = [
            glm.lookAt(center, center + glm.vec3( 1.0,  0.0,  0.0), glm.vec3(0.0, -1.0,  0.0)),
            glm.lookAt(center, center + glm.vec3(-1.0,  0.0,  0.0), glm.vec3(0.0, -1.0,  0.0)),
            glm.lookAt(center, center + glm.vec3( 0.0,  1.0,  0.0), glm.vec3(0.0,  0.0,  1.0)),
            glm.lookAt(center, center + glm.vec3( 0.0, -1.0,  0.0), glm.vec3(0.0,  0.0, -1.0)),
            glm.lookAt(center, center + glm.vec3( 0.0,  0.0,  1.0), glm.vec3(0.0, -1.0,  0.0)),
            glm.lookAt(center, center + glm.vec3( 0.0,  0.0, -1.0), glm.vec3(0.0, -1.0,  0.0)),
        ]
        env_proj = glm.perspective(glm.radians(90.0), 1.0, 1.0, 2000.0)
        
        gl.glUseProgram(self.prog)
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(self.prog, 'm_proj'), 1, gl.GL_FALSE, glm.value_ptr(env_proj))
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(self.prog, 'lightSpaceMatrix'), 1, gl.GL_FALSE, glm.value_ptr(light_space_matrix))
        gl.glUniform3f(gl.glGetUniformLocation(self.prog, 'light_pos'), self.light_pos.x, self.light_pos.y, self.light_pos.z)
        gl.glUniform1i(gl.glGetUniformLocation(self.prog, 'use_reflection'), 0)
        
        # Bind Shadow Map for reflections
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.depth_map)
        gl.glUniform1i(gl.glGetUniformLocation(self.prog, 'shadowMap'), 0)
        
        model_loc = gl.glGetUniformLocation(self.prog, 'm_model')
        view_loc = gl.glGetUniformLocation(self.prog, 'm_view')
        
        all_objects = scene.get_all_objects()
        
        for i in range(6):
            gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, 
                                      gl.GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, self.env_map, 0)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
            
            gl.glUniformMatrix4fv(view_loc, 1, gl.GL_FALSE, glm.value_ptr(views[i]))
            

            gl.glUniform3f(gl.glGetUniformLocation(self.prog, 'view_pos'), 0, 0, 0)
            
            for obj in all_objects:

                gl.glUniform1f(gl.glGetUniformLocation(self.prog, 'shininess'), self.default_shininess)
                gl.glUniform1f(gl.glGetUniformLocation(self.prog, 'specular_strength'), self.default_specular)
                gl.glUniform1f(gl.glGetUniformLocation(self.prog, 'reflectivity'), 0.0)
                gl.glUniform1f(gl.glGetUniformLocation(self.prog, 'metallic'), 0.0)
                gl.glUniform1f(gl.glGetUniformLocation(self.prog, 'rim_strength'), 0.0)
                
                vao, count = self.upload_part(obj.part)
                if vao:
                    gl.glUniformMatrix4fv(model_loc, 1, gl.GL_FALSE, glm.value_ptr(obj.world_matrix))
                    gl.glBindVertexArray(vao)
                    gl.glDrawArrays(gl.GL_TRIANGLES, 0, count)
                    gl.glBindVertexArray(0)
                    
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)


        gl.glViewport(viewport[0], viewport[1], viewport[2], viewport[3])
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDisable(gl.GL_CULL_FACE)

        gl.glUseProgram(self.grid_prog)
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(self.grid_prog, 'm_view'), 1, gl.GL_FALSE, glm.value_ptr(view))
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(self.grid_prog, 'm_proj'), 1, gl.GL_FALSE, glm.value_ptr(proj))
        ident = glm.mat4(1.0)
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(self.grid_prog, 'm_model'), 1, gl.GL_FALSE, glm.value_ptr(ident))
        gl.glBindVertexArray(self.grid_vao)
        gl.glDrawArrays(gl.GL_LINES, 0, self.grid_count)
        gl.glBindVertexArray(0)

        gl.glUseProgram(self.prog)
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(self.prog, 'm_view'), 1, gl.GL_FALSE, glm.value_ptr(view))
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(self.prog, 'm_proj'), 1, gl.GL_FALSE, glm.value_ptr(proj))
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(self.prog, 'lightSpaceMatrix'), 1, gl.GL_FALSE, glm.value_ptr(light_space_matrix))

        gl.glUniform3f(gl.glGetUniformLocation(self.prog, 'light_pos'), self.light_pos.x, self.light_pos.y, self.light_pos.z)

        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.depth_map)
        gl.glUniform1i(gl.glGetUniformLocation(self.prog, 'shadowMap'), 0)

        gl.glActiveTexture(gl.GL_TEXTURE1)
        gl.glBindTexture(gl.GL_TEXTURE_CUBE_MAP, self.env_map)
        gl.glUniform1i(gl.glGetUniformLocation(self.prog, 'skybox'), 1)

        inv_view = glm.inverse(view)
        cam_pos = glm.vec3(inv_view[3])
        gl.glUniform3f(gl.glGetUniformLocation(self.prog, 'view_pos'), cam_pos.x, cam_pos.y, cam_pos.z)
        
        model_loc = gl.glGetUniformLocation(self.prog, 'm_model')

        shininess_loc = gl.glGetUniformLocation(self.prog, 'shininess')
        spec_loc = gl.glGetUniformLocation(self.prog, 'specular_strength')
        refl_loc = gl.glGetUniformLocation(self.prog, 'reflectivity')
        metal_loc = gl.glGetUniformLocation(self.prog, 'metallic')
        rim_loc = gl.glGetUniformLocation(self.prog, 'rim_strength')
        use_refl_loc = gl.glGetUniformLocation(self.prog, 'use_reflection')
        
        for obj in all_objects:
            part = obj.part
            model_matrix = obj.world_matrix

            mat_shininess = self.default_shininess
            mat_specular = self.default_specular
            mat_reflectivity = self.default_reflectivity
            mat_metallic = self.default_metallic
            mat_rim = self.default_rim
            
            if hasattr(obj, 'material') and obj.material:
                mat = obj.material
                mat_shininess = mat.get('shininess', self.default_shininess)
                mat_specular = mat.get('specular', self.default_specular)
                mat_reflectivity = mat.get('reflectivity', self.default_reflectivity)
                mat_metallic = mat.get('metallic', self.default_metallic)
                mat_rim = mat.get('rim', self.default_rim)
            
            gl.glUniform1f(shininess_loc, mat_shininess)
            gl.glUniform1f(spec_loc, mat_specular)
            gl.glUniform1f(refl_loc, mat_reflectivity)
            gl.glUniform1f(metal_loc, mat_metallic)
            gl.glUniform1f(rim_loc, mat_rim)
            
            gl.glUniform1i(use_refl_loc, 1 if mat_reflectivity > 0 else 0)
            
            vao, count = self.upload_part(part)
            if vao:
                gl.glUniformMatrix4fv(model_loc, 1, gl.GL_FALSE, glm.value_ptr(model_matrix))
                gl.glBindVertexArray(vao)
                gl.glDrawArrays(gl.GL_TRIANGLES, 0, count)
                gl.glBindVertexArray(0)
        
        gl.glUseProgram(0)

    def render_outline(self, part, view, proj, color=(1, 0.5, 0, 1)):

        gl.glUseProgram(self.prog)
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(self.prog, 'm_view'), 1, gl.GL_FALSE, glm.value_ptr(view))
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(self.prog, 'm_proj'), 1, gl.GL_FALSE, glm.value_ptr(proj))
        gl.glUniform3f(gl.glGetUniformLocation(self.prog, 'light_pos'), 500.0, -500.0, 500.0)

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

        gl.glEnable(gl.GL_POLYGON_OFFSET_LINE)
        gl.glPolygonOffset(-1.0, -1.0) 
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

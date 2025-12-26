import glfw
import OpenGL.GL as gl
import imgui
from imgui.integrations.glfw import GlfwRenderer
import glm
import math
import os
import sys
import numpy as np

from renderer import Renderer
from camera import Camera
from ldraw import LDrawLoader
from scene import Scene, SceneObject
from utils import export_stl

SAMPLES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'samples'))
PARTS_DIR = os.path.join(SAMPLES_DIR, 'parts')
P_DIR = os.path.join(SAMPLES_DIR, 'p')

class App:
    def __init__(self):
        if not glfw.init():
            sys.exit(1)
            
        # Core Profile 3.3
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, gl.GL_TRUE)
        
        self.window = glfw.create_window(1280, 720, "Lego Editor", None, None)
        if not self.window:
            glfw.terminate()
            sys.exit(1)
            
        glfw.make_context_current(self.window)
        
        # Enable Depth Test
        gl.glEnable(gl.GL_DEPTH_TEST)
        # Disable Cull Face for LDraw compatibility
        gl.glDisable(gl.GL_CULL_FACE)
        
        # ImGui Setup
        imgui.create_context()
        self.impl = GlfwRenderer(self.window)
        
        # Callbacks
        glfw.set_mouse_button_callback(self.window, self.mouse_button_callback)
        glfw.set_scroll_callback(self.window, self.scroll_callback)
        glfw.set_window_size_callback(self.window, self.resize_callback)
        # Note: We don't set cursor_pos_callback to avoid overriding ImGui's. We poll instead.
        
        # Systems
        self.renderer = Renderer()
        self.camera = Camera(1280, 720)
        self.loader = LDrawLoader([PARTS_DIR, P_DIR])
        self.scene = Scene()
        
        # State
        self.running = True
        self.selected_part = None
        self.drag_start_pos = None
        self.drag_start_part_pos = None
        self.is_dragging_part = False
        self.is_rotating_camera = False
        self.is_panning_camera = False
        
        self.show_file_browser = False
        self.file_browser_path = PARTS_DIR
        self.part_search_text = ""
        self.files_cache = []
        self._refresh_files()
        
        # Mouse state
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        
        # Message for UI
        self.status_message = ""
        self.status_timer = 0.0

    def _refresh_files(self):
        if os.path.exists(self.file_browser_path):
            self.files_cache = [f for f in os.listdir(self.file_browser_path) if f.lower().endswith('.dat')]
            self.files_cache.sort()

    def run(self):
        while not glfw.window_should_close(self.window):
            glfw.poll_events()
            self.impl.process_inputs()
            
            # Mouse polling for movement
            x, y = glfw.get_cursor_pos(self.window)
            dx = x - self.last_mouse_x
            dy = y - self.last_mouse_y
            self.last_mouse_x = x
            self.last_mouse_y = y
            
            if not imgui.get_io().want_capture_mouse:
                if self.is_rotating_camera:
                    self.camera.rotate(dx, dy)
                elif self.is_panning_camera:
                    self.camera.pan(dx, dy)
                elif self.is_dragging_part and self.selected_part:
                     # Move part logic
                    sensitivity = self.camera.distance * 0.001
                    view = self.camera.get_view_matrix()
                    inv_view = glm.inverse(view)
                    cam_right = glm.vec3(inv_view[0])
                    cam_up = glm.vec3(inv_view[1]) 
                    move_vec = cam_right * dx * sensitivity - cam_up * dy * sensitivity
                    # We implement a simple screen-plane move
                    # Ideally we want XZ plane movement
                    # Let's just update X/Z based on camera orientation for now?
                    # Or just apply the vector to position (free movement)
                    self.selected_part.matrix[3] += glm.vec4(move_vec, 0.0)

            self.update()
            self.render()
            
            glfw.swap_buffers(self.window)
        
        self.impl.shutdown()
        glfw.terminate()

    def mouse_button_callback(self, window, button, action, mods):
        if imgui.get_io().want_capture_mouse:
            return
            
        x, y = glfw.get_cursor_pos(window)
        
        if action == glfw.PRESS:
            if button == glfw.MOUSE_BUTTON_LEFT:
                hit_part = self.raycast((x, y))
                if hit_part:
                    if self.selected_part == hit_part:
                        # Toggle selection off
                        self.selected_part = None
                        self.is_dragging_part = False
                    else:
                        # Select new part
                        self.selected_part = hit_part
                        self.is_dragging_part = True
                else:
                    self.is_rotating_camera = True
                    self.selected_part = None
                        
            elif button == glfw.MOUSE_BUTTON_RIGHT:
                self.is_panning_camera = True
                
        elif action == glfw.RELEASE:
            if button == glfw.MOUSE_BUTTON_LEFT:
                self.is_dragging_part = False
                self.is_rotating_camera = False
            elif button == glfw.MOUSE_BUTTON_RIGHT:
                self.is_panning_camera = False

    def scroll_callback(self, window, x_offset, y_offset):
        if imgui.get_io().want_capture_mouse:
            return
            
        self.camera.zoom(y_offset)

    def resize_callback(self, window, width, height):
        gl.glViewport(0, 0, width, height)
        self.camera.update_aspect(width, height)

    def raycast(self, mouse_pos):
        x, y = mouse_pos
        w, h = glfw.get_window_size(self.window)
        if w == 0 or h == 0: return None
        
        # NDC
        ndc_x = (2.0 * x) / w - 1.0
        ndc_y = 1.0 - (2.0 * y) / h
        ndc = glm.vec4(ndc_x, ndc_y, -1.0, 1.0)
        
        # World Ray
        proj = self.camera.get_proj_matrix()
        view = self.camera.get_view_matrix()
        inv_proj_view = glm.inverse(proj * view)
        
        world_pos = inv_proj_view * ndc
        world_pos /= world_pos.w
        
        ray_origin = glm.vec3(glm.inverse(view)[3])
        ray_dir = glm.normalize(glm.vec3(world_pos) - ray_origin)
        
        # Test against all parts AABB
        closest_t = float('inf')
        closest_part = None
        
        all_objects = self.scene.get_all_objects()
        for obj in all_objects:
            world_mat = obj.world_matrix
            inv_world = glm.inverse(world_mat)
            
            local_origin = glm.vec3(inv_world * glm.vec4(ray_origin, 1.0))
            local_dir = glm.vec3(inv_world * glm.vec4(ray_dir, 0.0))
            
            hit, t = self.intersect_aabb(local_origin, local_dir, obj.part.aabb_min, obj.part.aabb_max)
            if hit and t < closest_t:
                closest_t = t
                closest_part = obj
                
        return closest_part

    def intersect_aabb(self, ro, rd, box_min, box_max):
        # Safeguard: If AABB is invalid (min > max), return False
        if box_min.x > box_max.x or box_min.y > box_max.y or box_min.z > box_max.z:
            return False, 0.0
            
        inv_d = glm.vec3(0)
        inv_d.x = 1.0 / rd.x if rd.x != 0 else 1e20
        inv_d.y = 1.0 / rd.y if rd.y != 0 else 1e20
        inv_d.z = 1.0 / rd.z if rd.z != 0 else 1e20
        
        t0s = (box_min - ro) * inv_d
        t1s = (box_max - ro) * inv_d
        
        tsmaller = glm.min(t0s, t1s)
        tbigger = glm.max(t0s, t1s)
        
        tmin = max(tsmaller.x, max(tsmaller.y, tsmaller.z))
        tmax = min(tbigger.x, min(tbigger.y, tbigger.z))
        
        return (tmin < tmax and tmax > 0), tmin

    def update(self):
        # Update Status Timer
        if self.status_timer > 0:
            self.status_timer -= 1.0 / 60.0
            if self.status_timer <= 0:
                self.status_message = ""

        # Physics (Simple Gravity)
        dt = 1.0 / 60.0
        gravity = 200.0
        
        for obj in self.scene.objects:
            if not self.is_dragging_part or obj != self.selected_part:
                pos = obj.matrix[3]
                if pos.y < 0: 
                    obj.matrix[3].y += gravity * dt
                    if obj.matrix[3].y > 0:
                        obj.matrix[3].y = 0

    def check_collision(self, moving_part, new_matrix, ignore_part=None):
        # Simple AABB collision check
        # Get World AABB for moving part at new_matrix
        min_a, max_a = moving_part.get_world_aabb(override_matrix=new_matrix)
        
        # Shrink AABB slightly to allow touching
        epsilon = 0.5
        min_a += epsilon
        max_a -= epsilon
        
        for obj in self.scene.get_all_objects():
            if obj == moving_part or obj == ignore_part:
                continue
                
            # Check if obj is a child/parent of moving_part?
            # Ideally we check collision against independent objects.
            # For now, just check all others.
            
            min_b, max_b = obj.get_world_aabb()
            min_b += epsilon
            max_b -= epsilon
            
            # Check overlap
            overlap = (min_a.x < max_b.x and max_a.x > min_b.x and
                       min_a.y < max_b.y and max_a.y > min_b.y and
                       min_a.z < max_b.z and max_a.z > min_b.z)
                       
            if overlap:
                return True
                
        return False

    def render(self):
        gl.glClearColor(0.2, 0.2, 0.2, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        
        view = self.camera.get_view_matrix()
        proj = self.camera.get_proj_matrix()
        
        # Render Scene
        self.renderer.render(self.scene, view, proj)
        
        # Render Selected Part Outline
        if self.selected_part:
             self.renderer.render_selected_outline(self.selected_part, view, proj)
        
        # Render Light Gizmo
        self.renderer.render_light_source(view, proj)

        # Render UI
        imgui.new_frame()
        
        # Draw Light Label
        viewport = gl.glGetIntegerv(gl.GL_VIEWPORT) # [x, y, w, h]
        # Project light pos to screen
        screen_pos = glm.project(self.renderer.light_pos, view, proj, glm.vec4(viewport))
        # screen_pos.y is from bottom, imgui uses from top
        window_h = glfw.get_window_size(self.window)[1]
        
        # Check if light is in front of camera
        # Simple check: distance to plane? Or just check z in clip space?
        # glm.project returns window coordinates. Z is depth (0-1).
        if 0.0 <= screen_pos.z <= 1.0:
            imgui.set_next_window_position(screen_pos.x, window_h - screen_pos.y)
            imgui.begin("LightLabel", flags=imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_ALWAYS_AUTO_RESIZE | imgui.WINDOW_NO_BACKGROUND | imgui.WINDOW_NO_INPUTS)
            imgui.text_colored("Light Source", 1.0, 1.0, 0.0, 1.0) # Yellow text
            imgui.end()

        self.draw_ui()
        
        # Overlay Status Message
        if self.status_message:
            w, h = glfw.get_window_size(self.window)
            imgui.set_next_window_position(w/2, 50, pivot_x=0.5)
            imgui.begin("Status", flags=imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_ALWAYS_AUTO_RESIZE | imgui.WINDOW_NO_MOVE)
            imgui.text(self.status_message)
            imgui.end()
            
        imgui.render()
        self.impl.render(imgui.get_draw_data())

    def draw_ui(self):
        if imgui.begin_main_menu_bar():
            if imgui.begin_menu("File", True):
                if imgui.menu_item("Open Part Browser", "", False, True)[0]:
                    self.show_file_browser = True
                if imgui.menu_item("Export STL", "", False, True)[0]:
                    export_stl(self.scene, "export.stl")
                imgui.end_menu()
            imgui.end_main_menu_bar()
            
        # Connection Panel - REMOVED
        
        # File Browser
        if self.show_file_browser:
            expanded, opened = imgui.begin("Part Browser", True)
            if not opened:
                self.show_file_browser = False
            
            # Search Bar
            changed, self.part_search_text = imgui.input_text("Search", self.part_search_text, 256)
            
            imgui.separator()
            
            # List files
            # Ensure we are listing from self.file_browser_path which is PARTS_DIR
            for fname in self.files_cache:
                # Filter based on search text
                if self.part_search_text.lower() in fname.lower():
                    if imgui.button(fname):
                        self.load_part(fname)
                    if imgui.is_item_hovered():
                        imgui.set_tooltip(f"Load {fname}")
            imgui.end()
            
        # Properties Panel
        imgui.begin("Properties")
        imgui.text("Application Average: %.3f ms/frame (%.1f FPS)" % (1000.0/imgui.get_io().framerate, imgui.get_io().framerate))
        
        if imgui.collapsing_header("Lighting", None, imgui.TREE_NODE_DEFAULT_OPEN)[0]:
            l_pos = self.renderer.light_pos
            changed, new_l_pos = imgui.drag_float3("Light Position", l_pos.x, l_pos.y, l_pos.z, change_speed=10.0)
            if changed:
                self.renderer.light_pos = glm.vec3(new_l_pos[0], new_l_pos[1], new_l_pos[2])
        
        if self.selected_part:
            imgui.separator()
            imgui.text(f"Selected: {self.selected_part.name}")
            if imgui.button("Delete"):
                self.scene.remove_object(self.selected_part)
                self.selected_part = None
            
            if self.selected_part: 
                if self.selected_part.parent:
                    if imgui.button("Detach from Parent"):
                        self.selected_part.set_parent(None)
                        
                imgui.separator()
                imgui.text("Transform")
                pos = self.selected_part.matrix[3].xyz
                changed, new_pos = imgui.drag_float3("Position", *pos, change_speed=1.0)
                if changed:
                    self.selected_part.matrix[3] = glm.vec4(new_pos[0], new_pos[1], new_pos[2], 1.0)
                    
                if imgui.button("Rotate Y +90"):
                    rot = glm.rotate(glm.mat4(1.0), glm.radians(90), glm.vec3(0, 1, 0))
                    self.selected_part.matrix = self.selected_part.matrix * rot

                if imgui.button("Rotate X +90"):
                    rot = glm.rotate(glm.mat4(1.0), glm.radians(90), glm.vec3(1, 0, 0))
                    self.selected_part.matrix = self.selected_part.matrix * rot

                if imgui.button("Rotate Z +90"):
                    rot = glm.rotate(glm.mat4(1.0), glm.radians(90), glm.vec3(0, 0, 1))
                    self.selected_part.matrix = self.selected_part.matrix * rot

                imgui.separator()
                imgui.text("Material")
                
                # Ensure material dict exists
                if not hasattr(self.selected_part, 'material') or self.selected_part.material is None:
                    self.selected_part.material = {
                        'shininess': self.renderer.default_shininess,
                        'specular': self.renderer.default_specular,
                        'reflectivity': self.renderer.default_reflectivity,
                        'metallic': self.renderer.default_metallic,
                        'rim': self.renderer.default_rim
                    }
                
                mat = self.selected_part.material
                
                # Use Roughness slider instead of Shininess for better UX
                # Roughness 0 -> Shininess 256, Roughness 1 -> Shininess 2
                current_roughness = 1.0 - math.sqrt((mat['shininess'] - 2.0) / 254.0)
                current_roughness = max(0.0, min(1.0, current_roughness))
                
                changed_ro, ro = imgui.slider_float("Roughness", current_roughness, 0.0, 1.0)
                if changed_ro:
                    # Convert Roughness back to Shininess
                    # S = 254 * (1 - R)^2 + 2
                    mat['shininess'] = 254.0 * ((1.0 - ro) ** 2) + 2.0
                
                changed_sp, sp = imgui.slider_float("Specular Strength", mat['specular'], 0.0, 5.0)
                if changed_sp: mat['specular'] = sp
                
                changed_m, m = imgui.slider_float("Metallic", mat['metallic'], 0.0, 1.0)
                if changed_m: mat['metallic'] = m
                
                changed_r, r = imgui.slider_float("Reflectivity (Mirror)", mat['reflectivity'], 0.0, 1.0)
                if changed_r: mat['reflectivity'] = r
                
                changed_rim, rim = imgui.slider_float("Rim Light (Backlight)", mat['rim'], 0.0, 5.0)
                if changed_rim: mat['rim'] = rim
                
                if imgui.button("Reset Material"):
                    self.selected_part.material = None # Reset to defaults

        imgui.end()

    def load_part(self, filename):
        part = self.loader.load(filename)
        if part:
            obj = SceneObject(part, filename)
            self.scene.add_object(obj)

if __name__ == "__main__":
    app = App()
    app.run()

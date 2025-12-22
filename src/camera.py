import glm
import math

class Camera:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        
        self.target = glm.vec3(0, 0, 0)
        self.distance = 300.0
        self.yaw = 45.0
        self.pitch = 30.0 # Positive pitch to look down
        
        self.min_dist = 10.0
        self.max_dist = 2000.0
        
    def update_aspect(self, width, height):
        self.width = width
        self.height = height

    @property
    def aspect(self):
        return self.width / max(1.0, self.height)

    def get_view_matrix(self):
        rad_yaw = math.radians(self.yaw)
        rad_pitch = math.radians(self.pitch)
        
        # Calculate eye position relative to target
        # Assuming Y is Down (LDraw style)
        # We want to orbit around Y axis.
        
        # x = r * cos(p) * sin(y)
        # y = r * sin(p)  <-- height
        # z = r * cos(p) * cos(y)
        
        # If Y is down, positive Y is "below" target.
        # To look from "above", we need negative Y (or just handle pitch correctly).
        # Let's stick to standard math:
        # y is vertical.
        
        y = self.distance * math.sin(rad_pitch)
        r = self.distance * math.cos(rad_pitch)
        x = r * math.sin(rad_yaw)
        z = r * math.cos(rad_yaw)
        
        eye = self.target + glm.vec3(x, y, z)
        
        # Up vector is (0, 1, 0)
        return glm.lookAt(eye, self.target, glm.vec3(0, 1, 0))

    def get_proj_matrix(self):
        return glm.perspective(glm.radians(45.0), self.aspect, 1.0, 5000.0)

    def rotate(self, dx, dy):
        sensitivity = 0.5
        self.yaw += dx * sensitivity
        self.pitch += dy * sensitivity
        self.pitch = max(-89.0, min(89.0, self.pitch))

    def pan(self, dx, dy):
        sensitivity = self.distance * 0.001
        
        # Calculate Right and Up vectors relative to camera
        rad_yaw = math.radians(self.yaw)
        
        # Right vector (on XZ plane)
        right = glm.vec3(math.cos(rad_yaw), 0, -math.sin(rad_yaw))
        
        # Camera Up (approximate for panning on plane)
        # If we want to pan parallel to view plane:
        # We need view matrix basis.
        # But usually panning moves target on XZ plane or screen plane.
        # Let's move on screen plane logic.
        
        view = self.get_view_matrix()
        # Extract Right and Up from view matrix (inverse)
        inv_view = glm.inverse(view)
        cam_right = glm.vec3(inv_view[0])
        cam_up = glm.vec3(inv_view[1])
        
        self.target -= cam_right * dx * sensitivity
        self.target += cam_up * dy * sensitivity

    def zoom(self, yoffset):
        sensitivity = 20.0
        self.distance -= yoffset * sensitivity
        self.distance = max(self.min_dist, min(self.max_dist, self.distance))

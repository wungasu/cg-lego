import glm
import math

class Camera:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        
        self.target = glm.vec3(0, 0, 0)
        self.distance = 300.0
        self.yaw = 45.0
        self.pitch = 30.0 
        
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
        
        y = self.distance * math.sin(rad_pitch)
        r = self.distance * math.cos(rad_pitch)
        x = r * math.sin(rad_yaw)
        z = r * math.cos(rad_yaw)
        
        eye = self.target + glm.vec3(x, y, z)
        
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
        rad_yaw = math.radians(self.yaw)
        right = glm.vec3(math.cos(rad_yaw), 0, -math.sin(rad_yaw))
        view = self.get_view_matrix()
        inv_view = glm.inverse(view)
        cam_right = glm.vec3(inv_view[0])
        cam_up = glm.vec3(inv_view[1])
        
        self.target -= cam_right * dx * sensitivity
        self.target += cam_up * dy * sensitivity

    def zoom(self, yoffset):
        sensitivity = 20.0
        self.distance -= yoffset * sensitivity
        self.distance = max(self.min_dist, min(self.max_dist, self.distance))

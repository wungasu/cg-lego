import glm

class SceneObject:
    def __init__(self, part, name="Part"):
        self.part = part
        self.name = name
        self.matrix = glm.mat4(1.0) 
        self.parent = None
        self.children = []
        self.selected = False
        
    def set_parent(self, new_parent):

        current_world = self.world_matrix
        

        if self.parent:
            self.parent.children.remove(self)
        
        self.parent = new_parent
        
        if new_parent:
            new_parent.children.append(self)

            inv_parent = glm.inverse(new_parent.world_matrix)
            self.matrix = inv_parent * current_world
        else:
           
            self.matrix = current_world

    @property
    def world_matrix(self):
        if self.parent:
            return self.parent.world_matrix * self.matrix
        return self.matrix

    @world_matrix.setter
    def world_matrix(self, mat):
        if self.parent:
            self.matrix = glm.inverse(self.parent.world_matrix) * mat
        else:
            self.matrix = mat

    def get_world_aabb(self, override_matrix=None):
        mat = override_matrix if override_matrix is not None else self.world_matrix

        min_p = self.part.aabb_min
        max_p = self.part.aabb_max
        
        corners = [
            glm.vec3(min_p.x, min_p.y, min_p.z),
            glm.vec3(max_p.x, min_p.y, min_p.z),
            glm.vec3(min_p.x, max_p.y, min_p.z),
            glm.vec3(max_p.x, max_p.y, min_p.z),
            glm.vec3(min_p.x, min_p.y, max_p.z),
            glm.vec3(max_p.x, min_p.y, max_p.z),
            glm.vec3(min_p.x, max_p.y, max_p.z),
            glm.vec3(max_p.x, max_p.y, max_p.z),
        ]
        
        world_min = glm.vec3(float('inf'))
        world_max = glm.vec3(float('-inf'))
        
        for p in corners:
            wp = glm.vec3(mat * glm.vec4(p, 1.0))
            world_min = glm.min(world_min, wp)
            world_max = glm.max(world_max, wp)
            
        return world_min, world_max

class Scene:
    def __init__(self):
        self.objects = [] 
        self.selection = [] 
        
    def add_object(self, obj):
        self.objects.append(obj)
        
    def remove_object(self, obj):
        if obj in self.objects:
            self.objects.remove(obj)

        
    def get_all_objects(self):

        all_objs = []
        def recurse(obj):
            all_objs.append(obj)
            for child in obj.children:
                recurse(child)
        for root in self.objects:
            recurse(root)
        return all_objs

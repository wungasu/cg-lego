import glm
import os

def export_stl(scene, filename):
    with open(filename, 'w') as f:
        f.write("solid lego_export\n")
        
        all_objects = scene.get_all_objects()
        for obj in all_objects:
            world_mat = obj.world_matrix
            # Inverse transpose for normals
            norm_mat = glm.transpose(glm.inverse(glm.mat3(world_mat)))
            
            part = obj.part
            # Iterate triangles
            # vertices list is flat [x,y,z, x,y,z, ...]
            count = len(part.vertices) // 9 # 3 vertices * 3 coords
            
            for i in range(count):
                idx = i * 9
                v1 = glm.vec3(part.vertices[idx], part.vertices[idx+1], part.vertices[idx+2])
                v2 = glm.vec3(part.vertices[idx+3], part.vertices[idx+4], part.vertices[idx+5])
                v3 = glm.vec3(part.vertices[idx+6], part.vertices[idx+7], part.vertices[idx+8])
                
                # Transform
                v1 = glm.vec3(world_mat * glm.vec4(v1, 1.0))
                v2 = glm.vec3(world_mat * glm.vec4(v2, 1.0))
                v3 = glm.vec3(world_mat * glm.vec4(v3, 1.0))
                
                # Normal
                n = glm.normalize(glm.cross(v2 - v1, v3 - v1))
                
                f.write(f"facet normal {n.x} {n.y} {n.z}\n")
                f.write("outer loop\n")
                f.write(f"vertex {v1.x} {v1.y} {v1.z}\n")
                f.write(f"vertex {v2.x} {v2.y} {v2.z}\n")
                f.write(f"vertex {v3.x} {v3.y} {v3.z}\n")
                f.write("endloop\n")
                f.write("endfacet\n")
                
        f.write("endsolid lego_export\n")

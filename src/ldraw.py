import os
import glm
import numpy as np

# Basic LDraw Color Table (Simplified)
COLORS = {
    0: (0.1, 0.1, 0.1, 1.0),  # Black
    1: (0.0, 0.0, 1.0, 1.0),  # Blue
    2: (0.0, 1.0, 0.0, 1.0),  # Green
    3: (0.0, 1.0, 1.0, 1.0),  # Dark Turquoise
    4: (1.0, 0.0, 0.0, 1.0),  # Red
    5: (1.0, 0.0, 1.0, 1.0),  # Dark Pink
    6: (0.5, 0.25, 0.0, 1.0),  # Brown
    7: (0.8, 0.8, 0.8, 1.0),  # Light Gray
    8: (0.4, 0.4, 0.4, 1.0),  # Dark Gray
    14: (1.0, 1.0, 0.0, 1.0),  # Yellow
    15: (1.0, 1.0, 1.0, 1.0),  # White
    71: (0.7, 0.7, 0.7, 1.0),  # Light Bluish Gray
    72: (0.3, 0.3, 0.3, 1.0),  # Dark Bluish Gray
    47: (1.0, 1.0, 1.0, 0.5),  # Trans-Clear (Alpha 0.5)
}


def get_color(code, current_color_code=None):
    if code == 16:
        if current_color_code is not None:
            return get_color(current_color_code)
        return COLORS.get(7, (0.8, 0.8, 0.8, 1.0))  # Default
    if code == 24:
        return (0.2, 0.2, 0.2, 1.0)  # Edge color, simplified
    return COLORS.get(code, (0.8, 0.8, 0.8, 1.0))


class LDrawLoader:
    def __init__(self, search_paths):
        self.search_paths = search_paths
        self.cache = {}  # filename -> LDrawPart data
        self.mpd_cache = {}

    def load(self, filename, color_code=16):
        # MPD MODIFY: 处理MPD虚拟路径（如 "model.mpd/part.dat"）
        if '/' in filename:
            mpd_name, sub_filename = filename.split('/', 1)
            # 先尝试解析MPD文件
            mpd_part = self._load_mpd_main(mpd_name)
            if mpd_part:
                # 检查MPD缓存中是否有该子文件
                mpd_full_path = mpd_part.mpd_full_path if hasattr(mpd_part, 'mpd_full_path') else None
                if mpd_full_path in self.mpd_cache and sub_filename in self.mpd_cache[mpd_full_path]:
                    # 构建MPD子文件的虚拟Part
                    if filename in self.cache:
                        return self.cache[filename]
                    part = LDrawPart(filename, self)
                    part.is_mpd_subfile = True
                    part.mpd_full_path = mpd_full_path
                    # 直接使用MPD缓存中的子文件内容解析
                    part.parse_mpd_subfile(self.mpd_cache[mpd_full_path][sub_filename])
                    self.cache[filename] = part
                    return part

        if filename in self.cache:
            return self.cache[filename]

        part = LDrawPart(filename, self)
        part.parse()
        self.cache[filename] = part
        return part

    def _load_mpd_main(self, mpd_filename):
        if mpd_filename in self.cache:
            return self.cache[mpd_filename]
        # 先解析MPD主文件
        part = LDrawPart(mpd_filename, self)
        part.parse()
        self.cache[mpd_filename] = part
        return part


class LDrawPart:
    def __init__(self, filename, loader):
        self.filename = filename
        self.loader = loader
        self.commands = []

        # Geometry data (local space)
        self.vertices = []
        self.normals = []
        self.colors = []

        # Logical components
        self.studs = []  # List of (transform_matrix, filename)
        self.tubes = []  # List of (transform_matrix, filename)

        self.aabb_min = glm.vec3(float('inf'))
        self.aabb_max = glm.vec3(float('-inf'))

        self.is_mpd_file = False  # 是否是MPD主文件
        self.is_mpd_subfile = False  # 是否是MPD子文件
        self.mpd_full_path = None  # MPD文件的完整路径
        self.mpd_subfiles = {}  # MPD主文件包含的子文件 {sub_filename: lines_list}

    def resolve_path(self, filename):
        # Normalize slashes
        filename = filename.replace('\\', '/')

        # Candidate names
        candidates = [filename, filename.lower()]

        # 1. Check strict search paths
        for p in self.loader.search_paths:
            for cand in candidates:
                full = os.path.join(p, cand)
                if os.path.exists(full):
                    return full

        # 2. Heuristic: Check common LDraw subdirectories if not present in filename
        # e.g. if looking for "stud.dat" and it's not in roots, check "p/stud.dat" or "parts/stud.dat"
        # Only if filename doesn't already have path components
        if '/' not in filename:
            subdirs = ['p', 'parts', 'P', 'PARTS']
            for p in self.loader.search_paths:
                for sd in subdirs:
                    for cand in candidates:
                        full = os.path.join(p, sd, cand)
                        if os.path.exists(full):
                            return full

        return None

    def parse_mpd_subfile(self, lines):
        for line in lines:
            line = line.strip()
            if not line or line.startswith('0'):
                continue

            parts = line.split()
            if not parts:
                continue

            cmd = parts[0]
            self._parse_ldraw_command(cmd, parts)

    def parse(self):
        if self.is_mpd_subfile:
            return

        full_path = self.resolve_path(self.filename)
        if not full_path:
            print(f"Warning: Could not find {self.filename}")
            return

        self.mpd_full_path = full_path
        lines = []
        with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()

        current_subfile = None
        mpd_subfiles = {}
        mpd_main_lines = []

        for line in lines:
            stripped_line = line.strip()
            # 识别MPD的0 FILE指令
            if stripped_line.startswith('0 FILE '):
                self.is_mpd_file = True
                # 提取子文件名（如 "0 FILE part.dat" -> "part.dat"）
                current_subfile = stripped_line.split('0 FILE ')[1].strip()
                mpd_subfiles[current_subfile] = []
                continue

            # 根据当前状态分配行到主文件或子文件
            if self.is_mpd_file and current_subfile is not None:
                mpd_subfiles[current_subfile].append(line)
            else:
                mpd_main_lines.append(line)

        if self.is_mpd_file:
            self.mpd_subfiles = mpd_subfiles
            self.loader.mpd_cache[full_path] = mpd_subfiles
            # 解析MPD主文件的内容（0 FILE之外的部分）
            parse_lines = mpd_main_lines
        else:
            parse_lines = lines

        # 解析LDraw指令（主文件或普通文件）
        for line in parse_lines:
            line = line.strip()
            if not line or line.startswith('0'):
                continue

            parts = line.split()
            if not parts:
                continue

            cmd = parts[0]
            self._parse_ldraw_command(cmd, parts)

    def _parse_ldraw_command(self, cmd, parts):
        if cmd == '1':  # Sub-file
            # 1 <colour> x y z a b c d e f g h i <file>
            if len(parts) < 15: return
            color = int(parts[1])
            x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
            a, b, c = float(parts[5]), float(parts[6]), float(parts[7])
            d, e, f = float(parts[8]), float(parts[9]), float(parts[10])
            g, h, i = float(parts[11]), float(parts[12]), float(parts[13])
            sub_file = ' '.join(parts[14:])

            mpd_subfile_path = None
            # 检查当前Part是否关联MPD文件
            if hasattr(self, 'mpd_full_path') and self.mpd_full_path in self.loader.mpd_cache:
                # 检查子文件是否在MPD缓存中
                if sub_file in self.loader.mpd_cache[self.mpd_full_path]:
                    mpd_subfile_path = f"{self.filename}/{sub_file}"

            # 使用拼接后的路径
            if mpd_subfile_path:
                sub_file = mpd_subfile_path

            transform = glm.mat4(
                a, d, g, 0,
                b, e, h, 0,
                c, f, i, 0,
                x, y, z, 1
            )

            lower_sub = sub_file.lower()
            if 'stud' in lower_sub and '.dat' in lower_sub:
                self.studs.append((transform, sub_file))

            # 加载子零件
            sub_part = self.loader.load(sub_file)
            if sub_part:
                self._bake_subpart(sub_part, transform, color)

        elif cmd == '3':  # Triangle
            # 3 <colour> x1 y1 z1 x2 y2 z2 x3 y3 z3
            if len(parts) < 11: return
            color = int(parts[1])
            v1 = glm.vec3(float(parts[2]), float(parts[3]), float(parts[4]))
            v2 = glm.vec3(float(parts[5]), float(parts[6]), float(parts[7]))
            v3 = glm.vec3(float(parts[8]), float(parts[9]), float(parts[10]))
            self._add_triangle(v1, v2, v3, color)

        elif cmd == '4':  # Quad
            # 4 <colour> x1 y1 z1 x2 y2 z2 x3 y3 z3 x4 y4 z4
            if len(parts) < 14: return
            color = int(parts[1])
            v1 = glm.vec3(float(parts[2]), float(parts[3]), float(parts[4]))
            v2 = glm.vec3(float(parts[5]), float(parts[6]), float(parts[7]))
            v3 = glm.vec3(float(parts[8]), float(parts[9]), float(parts[10]))
            v4 = glm.vec3(float(parts[11]), float(parts[12]), float(parts[13]))
            self._add_triangle(v1, v2, v3, color)
            self._add_triangle(v1, v3, v4, color)  # Quad split into 2 tris

    def _add_triangle(self, v1, v2, v3, color_code):
        # Update AABB
        self.aabb_min = glm.min(self.aabb_min, v1)
        self.aabb_min = glm.min(self.aabb_min, v2)
        self.aabb_min = glm.min(self.aabb_min, v3)

        self.aabb_max = glm.max(self.aabb_max, v1)
        self.aabb_max = glm.max(self.aabb_max, v2)
        self.aabb_max = glm.max(self.aabb_max, v3)

        # Calculate normal
        edge1 = v2 - v1
        edge2 = v3 - v1
        normal = glm.cross(edge1, edge2)
        if glm.length(normal) > 0:
            normal = glm.normalize(normal)

        # Add to lists
        self.vertices.extend([v1.x, v1.y, v1.z])
        self.normals.extend([normal.x, normal.y, normal.z])
        self.colors.append(color_code)

        self.vertices.extend([v2.x, v2.y, v2.z])
        self.normals.extend([normal.x, normal.y, normal.z])
        self.colors.append(color_code)

        self.vertices.extend([v3.x, v3.y, v3.z])
        self.normals.extend([normal.x, normal.y, normal.z])
        self.colors.append(color_code)

    def _bake_subpart(self, sub_part, transform, color_code):
        # Transform sub-part geometry and add to self
        # Apply transform to vertices and normals

        # Optimization: Use numpy for bulk transform if possible, but here we iterate
        # Since we parse once, it's okay.

        # Transform matrix3 for normals (inverse transpose, but if orthogonal, just rotation)
        # LDraw transforms can include scaling (-1 for inversion).
        # So we should use inverse transpose of the upper-left 3x3.
        mat3 = glm.mat3(transform)
        norm_mat = glm.transpose(glm.inverse(mat3))

        # Extract data from sub_part
        # sub_part.vertices is flat list [x,y,z, x,y,z...]
        # sub_part.normals is flat list

        # We need to process vertices in groups of 3
        count = len(sub_part.vertices) // 3

        for i in range(count):
            idx = i * 3
            v = glm.vec3(sub_part.vertices[idx], sub_part.vertices[idx + 1], sub_part.vertices[idx + 2])
            n = glm.vec3(sub_part.normals[idx], sub_part.normals[idx + 1], sub_part.normals[idx + 2])
            c = sub_part.colors[i]  # One color per vertex? No, code structure above adds color per vertex.

            # Apply color inheritance
            final_color = c
            if c == 16:
                final_color = color_code

            # Transform
            v_world = transform * glm.vec4(v, 1.0)
            n_world = norm_mat * n
            if glm.length(n_world) > 0:
                n_world = glm.normalize(n_world)

            # Update AABB
            v_vec3 = glm.vec3(v_world)
            self.aabb_min = glm.min(self.aabb_min, v_vec3)
            self.aabb_max = glm.max(self.aabb_max, v_vec3)

            self.vertices.extend([v_world.x, v_world.y, v_world.z])
            self.normals.extend([n_world.x, n_world.y, n_world.z])
            self.colors.append(final_color)

        # Bake Studs and Tubes with transform
        for stud_trans, stud_file in sub_part.studs:
            new_trans = transform * stud_trans
            self.studs.append((new_trans, stud_file))

        for tube_trans, tube_file in sub_part.tubes:
            new_trans = transform * tube_trans
            self.tubes.append((new_trans, tube_file))
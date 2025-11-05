#!/usr/bin/env python3
import json
import random
import time
from typing import List, Dict, Any, Tuple
from collections import Counter
class DNALibrary:
    @staticmethod
    def identity(g: Grid) -> Grid:
        return [row[:] for row in g]
    @staticmethod
    def reflect_horizontal(g: Grid) -> Grid:
        return [row[::-1] for row in g]
    @staticmethod
    def reflect_vertical(g: Grid) -> Grid:
        return g[::-1]
    @staticmethod
    def rotate_90(g: Grid) -> Grid:
        if not g or not g[0]:
            return g
        h, w = len(g), len(g[0])
        return [[g[h-1-y][x] for y in range(h)] for x in range(w)]
    @staticmethod
    def rotate_180(g: Grid) -> Grid:
        return DNALibrary.reflect_vertical(DNALibrary.reflect_horizontal(g))
    @staticmethod
    def rotate_270(g: Grid) -> Grid:
        return DNALibrary.rotate_90(DNALibrary.rotate_90(DNALibrary.rotate_90(g)))
    @staticmethod
    def transpose(g: Grid) -> Grid:
        if not g or not g[0]:
            return g
        h, w = len(g), len(g[0])
        return [[g[y][x] for y in range(h)] for x in range(w)]
    @staticmethod
    def transpose_anti(g: Grid) -> Grid:
        return DNALibrary.rotate_180(DNALibrary.transpose(g))
    @staticmethod
    def scale_up_2x(g: Grid) -> Grid:
        if not g or not g[0]:
            return g
        h, w = len(g), len(g[0])
        result = [[0]*(w*2) for _ in range(h*2)]
        for y in range(h):
            for x in range(w):
                result[y*2][x*2] = g[y][x]
                result[y*2][x*2+1] = g[y][x]
                result[y*2+1][x*2] = g[y][x]
                result[y*2+1][x*2+1] = g[y][x]
        return result
    @staticmethod
    def scale_up_3x(g: Grid) -> Grid:
        if not g or not g[0]:
            return g
        h, w = len(g), len(g[0])
        result = [[0]*(w*3) for _ in range(h*3)]
        for y in range(h):
            for x in range(w):
                for dy in range(3):
                    for dx in range(3):
                        result[y*3+dy][x*3+dx] = g[y][x]
        return result
    @staticmethod
    def scale_down_2x(g: Grid) -> Grid:
        if not g or not g[0]:
            return g
        h, w = len(g), len(g[0])
        return [[g[y][x] for x in range(0, w, 2)] for y in range(0, h, 2)]
    @staticmethod
    def tile_2x2(g: Grid) -> Grid:
        if not g or not g[0]:
            return g
        h, w = len(g), len(g[0])
        result = [[0]*(w*2) for _ in range(h*2)]
        for ty in range(2):
            for tx in range(2):
                for y in range(h):
                    for x in range(w):
                        result[ty*h+y][tx*w+x] = g[y][x]
        return result
    @staticmethod
    def tile_3x3(g: Grid) -> Grid:
        if not g or not g[0]:
            return g
        h, w = len(g), len(g[0])
        result = [[0]*(w*3) for _ in range(h*3)]
        for ty in range(3):
            for tx in range(3):
                for y in range(h):
                    for x in range(w):
                        result[ty*h+y][tx*w+x] = g[y][x]
        return result
    @staticmethod
    def extract_top_half(g: Grid) -> Grid:
        if not g:
            return g
        h = len(g)
        return g[:h//2]
    @staticmethod
    def extract_bottom_half(g: Grid) -> Grid:
        if not g:
            return g
        h = len(g)
        return g[h//2:]
    @staticmethod
    def extract_left_half(g: Grid) -> Grid:
        if not g or not g[0]:
            return g
        w = len(g[0])
        return [row[:w//2] for row in g]
    @staticmethod
    def invert_colors(g: Grid) -> Grid:
        return [[10-cell if cell > 0 else 0 for cell in row] for row in g]
    @staticmethod
    def increment_colors(g: Grid) -> Grid:
        return [[(cell+1) % 10 if cell > 0 else 0 for cell in row] for row in g]
    @staticmethod
    def decrement_colors(g: Grid) -> Grid:
        return [[max(0, cell-1) if cell > 0 else 0 for cell in row] for row in g]
    @staticmethod
    def map_to_grayscale(g: Grid) -> Grid:
        def to_gray(c):
            if c == 0:
                return 0
            elif c <= 3:
                return 1
            elif c <= 6:
                return 2
            else:
                return 3
        return [[to_gray(cell) for cell in row] for row in g]
    @staticmethod
    def extract_color(g: Grid, color: int = 1) -> Grid:
        return [[cell if cell == color else 0 for cell in row] for row in g]
    @staticmethod
    def remove_color(g: Grid, color: int = 0) -> Grid:
        return [[0 if cell == color else cell for cell in row] for row in g]
    @staticmethod
    def replace_color(g: Grid, old: int = 1, new: int = 2) -> Grid:
        return [[new if cell == old else cell for cell in row] for row in g]
    @staticmethod
    def most_common_color(g: Grid) -> Grid:
        from collections import Counter
        colors = []
        for row in g:
            for cell in row:
                if cell > 0:
                    colors.append(cell)
        if not colors:
            return g
        most_common = Counter(colors).most_common(1)[0][0]
        return [[most_common if cell > 0 else 0 for cell in row] for row in g]
    @staticmethod
    def background_to_foreground(g: Grid) -> Grid:
        from collections import Counter
        colors = []
        for row in g:
            for cell in row:
                if cell > 0:
                    colors.append(cell)
        if not colors:
            return g
        fg = Counter(colors).most_common(1)[0][0]
        return [[fg if cell == 0 else 0 for cell in row] for row in g]
    @staticmethod
    def binarize(g: Grid) -> Grid:
        return [[1 if cell > 0 else 0 for cell in row] for row in g]
    @staticmethod
    def extract_largest_object(g: Grid) -> Grid:
        if not g or not g[0]:
            return g
        h, w = len(g), len(g[0])
        visited = [[False]*w for _ in range(h)]
        def flood_fill(sy, sx):
            stack = [(sy, sx)]
            cells = []
            color = g[sy][sx]
            if color == 0:
                return cells
            while stack:
                y, x = stack.pop()
                if y < 0 or y >= h or x < 0 or x >= w:
                    continue
                if visited[y][x] or g[y][x] != color:
                    continue
                visited[y][x] = True
                cells.append((y, x))
                stack.extend([(y+1,x), (y-1,x), (y,x+1), (y,x-1)])
            return cells
        objects = []
        for y in range(h):
            for x in range(w):
                if not visited[y][x] and g[y][x] > 0:
                    obj = flood_fill(y, x)
                    if obj:
                        objects.append(obj)
        if not objects:
            return [[0]*w for _ in range(h)]
        largest = max(objects, key=len)
        result = [[0]*w for _ in range(h)]
        for y, x in largest:
            result[y][x] = g[y][x]
        return result
    @staticmethod
    def count_objects(g: Grid) -> Grid:
        if not g or not g[0]:
            return [[0]]
        h, w = len(g), len(g[0])
        visited = [[False]*w for _ in range(h)]
        count = 0
        def flood_fill(sy, sx):
            stack = [(sy, sx)]
            color = g[sy][sx]
            if color == 0:
                return False
            while stack:
                y, x = stack.pop()
                if y < 0 or y >= h or x < 0 or x >= w:
                    continue
                if visited[y][x] or g[y][x] != color:
                    continue
                visited[y][x] = True
                stack.extend([(y+1,x), (y-1,x), (y,x+1), (y,x-1)])
            return True
        for y in range(h):
            for x in range(w):
                if not visited[y][x] and g[y][x] > 0:
                    if flood_fill(y, x):
                        count += 1
        return [[count]]
    @staticmethod
    def extract_borders(g: Grid) -> Grid:
        if not g or not g[0]:
            return g
        h, w = len(g), len(g[0])
        result = [[0]*w for _ in range(h)]
        for y in range(h):
            for x in range(w):
                if g[y][x] > 0:
                    is_border = False
                    for dy, dx in [(0,1), (0,-1), (1,0), (-1,0)]:
                        ny, nx = y + dy, x + dx
                        if ny < 0 or ny >= h or nx < 0 or nx >= w:
                            is_border = True
                        elif g[ny][nx] == 0:
                            is_border = True
                    if is_border:
                        result[y][x] = g[y][x]
        return result
    @staticmethod
    def fill_holes(g: Grid) -> Grid:
        if not g or not g[0]:
            return g
        h, w = len(g), len(g[0])
        result = [row[:] for row in g]
        for y in range(1, h-1):
            for x in range(1, w-1):
                if g[y][x] == 0:
                    neighbors = [
                        g[y-1][x], g[y+1][x], g[y][x-1], g[y][x+1]
                    ]
                    if all(n > 0 for n in neighbors):
                        result[y][x] = max(neighbors)
        return result
    @staticmethod
    def extract_skeleton(g: Grid) -> Grid:
        return DNALibrary.extract_borders(g)
    @staticmethod
    def bounding_box(g: Grid) -> Grid:
        if not g or not g[0]:
            return g
        h, w = len(g), len(g[0])
        min_y, max_y = h, -1
        min_x, max_x = w, -1
        for y in range(h):
            for x in range(w):
                if g[y][x] > 0:
                    min_y = min(min_y, y)
                    max_y = max(max_y, y)
                    min_x = min(min_x, x)
                    max_x = max(max_x, x)
        if max_y == -1:
            return g
        return [row[min_x:max_x+1] for row in g[min_y:max_y+1]]
    @staticmethod
    def center_objects(g: Grid) -> Grid:
        bb = DNALibrary.bounding_box(g)
        if not bb or not bb[0]:
            return g
        h, w = len(g), len(g[0])
        bb_h, bb_w = len(bb), len(bb[0])
        offset_y = (h - bb_h) // 2
        offset_x = (w - bb_w) // 2
        result = [[0]*w for _ in range(h)]
        for y in range(bb_h):
            for x in range(bb_w):
                ny, nx = y + offset_y, x + offset_x
                if 0 <= ny < h and 0 <= nx < w:
                    result[ny][nx] = bb[y][x]
        return result
    @staticmethod
    def object_to_corners(g: Grid) -> Grid:
        bb = DNALibrary.bounding_box(g)
        if not bb or not bb[0]:
            return g
        h, w = len(g), len(g[0])
        bb_h, bb_w = len(bb), len(bb[0])
        result = [[0]*w for _ in range(h)]
        for y in range(min(bb_h, h)):
            for x in range(min(bb_w, w)):
                result[y][x] = bb[y][x]
        return result
    @staticmethod
    def duplicate_objects(g: Grid) -> Grid:
        if not g or not g[0]:
            return g
        return DNALibrary.tile_2x2(g)
    @staticmethod
    def sort_by_size(g: Grid) -> Grid:
        return g
    @staticmethod
    def detect_symmetry(g: Grid) -> Grid:
        return g
    @staticmethod
    def create_symmetric(g: Grid) -> Grid:
        if not g or not g[0]:
            return g
        h, w = len(g), len(g[0])
        result = [row[:] for row in g]
        mid = w // 2
        for y in range(h):
            for x in range(mid):
                result[y][w-1-x] = result[y][x]
        return result
    @staticmethod
    def repeat_pattern(g: Grid) -> Grid:
        return DNALibrary.tile_2x2(g)
    @staticmethod
    def extend_lines(g: Grid) -> Grid:
        if not g or not g[0]:
            return g
        h, w = len(g), len(g[0])
        result = [row[:] for row in g]
        for y in range(h):
            for x in range(w):
                if g[y][x] > 0:
                    for xx in range(x):
                        if result[y][xx] == 0:
                            result[y][xx] = g[y][x]
                    for xx in range(x+1, w):
                        if result[y][xx] == 0:
                            result[y][xx] = g[y][x]
        return result
    @staticmethod
    def connect_nearest(g: Grid) -> Grid:
        return g
    @staticmethod
    def fill_pattern(g: Grid) -> Grid:
        return g
    @staticmethod
    def gravity_down(g: Grid) -> Grid:
        if not g or not g[0]:
            return g
        h, w = len(g), len(g[0])
        result = [[0]*w for _ in range(h)]
        for x in range(w):
            column = [g[y][x] for y in range(h)]
            non_zero = [c for c in column if c > 0]
            for i, val in enumerate(reversed(non_zero)):
                result[h-1-i][x] = val
        return result
    @staticmethod
    def gravity_up(g: Grid) -> Grid:
        if not g or not g[0]:
            return g
        h, w = len(g), len(g[0])
        result = [[0]*w for _ in range(h)]
        for x in range(w):
            column = [g[y][x] for y in range(h)]
            non_zero = [c for c in column if c > 0]
            for i, val in enumerate(non_zero):
                result[i][x] = val
        return result
    @staticmethod
    def compress_horizontal(g: Grid) -> Grid:
        if not g or not g[0]:
            return g
        h, w = len(g), len(g[0])
        non_empty_cols = []
        for x in range(w):
            if any(g[y][x] > 0 for y in range(h)):
                non_empty_cols.append(x)
        if not non_empty_cols:
            return g
        return [[g[y][x] for x in non_empty_cols] for y in range(h)]
    @staticmethod
    def compress_vertical(g: Grid) -> Grid:
        if not g:
            return g
        non_empty_rows = []
        for y, row in enumerate(g):
            if any(cell > 0 for cell in row):
                non_empty_rows.append(y)
        if not non_empty_rows:
            return g
        return [g[y] for y in non_empty_rows]
    @staticmethod
    def crop_to_content(g: Grid) -> Grid:
        return DNALibrary.bounding_box(g)
    @staticmethod
    def pad_uniform(g: Grid, padding: int = 1) -> Grid:
        if not g or not g[0]:
            return g
        h, w = len(g), len(g[0])
        result = [[0]*(w + 2*padding) for _ in range(h + 2*padding)]
        for y in range(h):
            for x in range(w):
                result[y+padding][x+padding] = g[y][x]
        return result
    @staticmethod
    def add_border(g: Grid, color: int = 1) -> Grid:
        if not g or not g[0]:
            return g
        h, w = len(g), len(g[0])
        result = [[color]*(w+2) for _ in range(h+2)]
        for y in range(h):
            for x in range(w):
                result[y+1][x+1] = g[y][x]
        return result
    @staticmethod
    def remove_border(g: Grid) -> Grid:
        if not g or not g[0] or len(g) < 3 or len(g[0]) < 3:
            return g
        h, w = len(g), len(g[0])
        return [row[1:-1] for row in g[1:-1]]
    @staticmethod
    def grid_overlay(g: Grid) -> Grid:
        if not g or not g[0]:
            return g
        h, w = len(g), len(g[0])
        result = [row[:] for row in g]
        for y in range(0, h, 3):
            for x in range(w):
                result[y][x] = 5
        for x in range(0, w, 3):
            for y in range(h):
                result[y][x] = 5
        return result
    @staticmethod
    def checkerboard_mask(g: Grid) -> Grid:
        if not g or not g[0]:
            return g
        h, w = len(g), len(g[0])
        return [[g[y][x] if (y+x) % 2 == 0 else 0 for x in range(w)] for y in range(h)]
    @staticmethod
    def diagonal_mask(g: Grid) -> Grid:
        if not g or not g[0]:
            return g
        h, w = len(g), len(g[0])
        return [[g[y][x] if y == x else 0 for x in range(w)] for y in range(h)]
    @staticmethod
    def outer_ring(g: Grid) -> Grid:
        if not g or not g[0]:
            return g
        h, w = len(g), len(g[0])
        result = [[0]*w for _ in range(h)]
        for x in range(w):
            result[0][x] = g[0][x]
            result[h-1][x] = g[h-1][x]
        for y in range(h):
            result[y][0] = g[y][0]
            result[y][w-1] = g[y][w-1]
        return result
    @staticmethod
    def logical_and(g: Grid) -> Grid:
        return g
    @staticmethod
    def logical_or(g: Grid) -> Grid:
        return g
    @staticmethod
    def logical_xor(g: Grid) -> Grid:
        return g
    @staticmethod
    def logical_not(g: Grid) -> Grid:
        if not g or not g[0]:
            return g
        return [[1 if cell == 0 else 0 for cell in row] for row in g]
    @staticmethod
    def difference(g: Grid) -> Grid:
        return g
    @staticmethod
    def intersection(g: Grid) -> Grid:
        return g
    @staticmethod
    def noise_reduction(g: Grid) -> Grid:
        if not g or not g[0]:
            return g
        h, w = len(g), len(g[0])
        result = [row[:] for row in g]
        for y in range(h):
            for x in range(w):
                if g[y][x] > 0:
                    neighbors = 0
                    for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and g[ny][nx] > 0:
                            neighbors += 1
                    if neighbors == 0:
                        result[y][x] = 0
        return result
    @staticmethod
    def majority_filter(g: Grid) -> Grid:
        if not g or not g[0]:
            return g
        h, w = len(g), len(g[0])
        result = [row[:] for row in g]
        for y in range(1, h-1):
            for x in range(1, w-1):
                neighbors = [
                    g[y-1][x], g[y+1][x], g[y][x-1], g[y][x+1],
                    g[y-1][x-1], g[y-1][x+1], g[y+1][x-1], g[y+1][x+1]
                ]
                from collections import Counter
                if neighbors:
                    most_common = Counter(neighbors).most_common(1)[0][0]
                    result[y][x] = most_common
        return result
    @staticmethod
    def sobel_edges(g: Grid) -> Grid:
        if not g or not g[0]:
            return g
        h, w = len(g), len(g[0])
        result = [[0]*w for _ in range(h)]
        for y in range(1, h-1):
            for x in range(1, w-1):
                gx = (g[y-1][x+1] + 2*g[y][x+1] + g[y+1][x+1]) - \
                     (g[y-1][x-1] + 2*g[y][x-1] + g[y+1][x-1])
                gy = (g[y+1][x-1] + 2*g[y+1][x] + g[y+1][x+1]) - \
                     (g[y-1][x-1] + 2*g[y-1][x] + g[y-1][x+1])
                magnitude = abs(gx) + abs(gy)
                result[y][x] = 1 if magnitude > 0 else 0
        return result
    @staticmethod
    def fractal_expand(g: Grid) -> Grid:
        return DNALibrary.tile_2x2(g)
    @staticmethod
    def voronoi_regions(g: Grid) -> Grid:
        return g
def get_all_operations() -> Dict[str, callable]:
    ops = {}
    for name in dir(DNALibrary):
        if not name.startswith('_'):
            attr = getattr(DNALibrary, name)
            if callable(attr):
                ops[name] = attr
    return ops
class OptimizedEvolutionarySolver:
    def __init__(self, ops: Dict[str, callable]):
        self.ops = ops
        self.gene_pool = list(ops.keys())
        self.population_size = 50
        self.max_dna_length = 7
        self.cw5_coffee = 0
        self.cw5_cigarettes = 0
        self.cw5_interventions = 0
        self.operation_usage = Counter()
    def fingerprint_task(self, task: Dict) -> Dict:
        if 'train' not in task or not task['train']:
            return {}
        example = task['train'][0]
        input_grid = example['input']
        output_grid = example['output']
        h_in = len(input_grid)
        w_in = len(input_grid[0]) if input_grid else 0
        h_out = len(output_grid)
        w_out = len(output_grid[0]) if output_grid else 0
        fingerprint = {
            'input_size': (h_in, w_in),
            'output_size': (h_out, w_out),
            'size_change': (h_out / h_in if h_in > 0 else 1,
                           w_out / w_in if w_in > 0 else 1)
        }
        suggested_ops = []
        if fingerprint['size_change'][0] > 1.5 or fingerprint['size_change'][1] > 1.5:
            suggested_ops.extend(['scale_up_2x', 'scale_up_3x', 'tile_2x2', 'tile_3x3'])
        elif fingerprint['size_change'][0] < 0.7 or fingerprint['size_change'][1] < 0.7:
            suggested_ops.extend(['scale_down_2x', 'bounding_box', 'crop_to_content',
                                'compress_horizontal', 'compress_vertical'])
        if 0.9 < fingerprint['size_change'][0] < 1.1 and 0.9 < fingerprint['size_change'][1] < 1.1:
            suggested_ops.extend(['reflect_horizontal', 'reflect_vertical', 'rotate_90',
                                'rotate_180', 'transpose', 'gravity_down', 'gravity_up'])
        if h_out <= 3 and w_out <= 3:
            suggested_ops.extend(['count_objects', 'extract_largest_object',
                                'bounding_box', 'center_objects'])
        fingerprint['suggested_ops'] = suggested_ops
        return fingerprint
    def calculate_fitness_advanced(self, result_grid: List[List[int]],
                                   expected_grid: List[List[int]]) -> float:
        if result_grid == expected_grid:
            return 1.0
        fit = 0.0
        h_res = len(result_grid)
        w_res = len(result_grid[0]) if result_grid else 0
        h_exp = len(expected_grid)
        w_exp = len(expected_grid[0]) if expected_grid else 0
        if h_res == h_exp and w_res == w_exp:
            fit += 0.2
            correct_pixels = 0
            total_pixels = h_exp * w_exp
            for y in range(h_exp):
                for x in range(w_exp):
                    if result_grid[y][x] == expected_grid[y][x]:
                        correct_pixels += 1
            fit += 0.6 * (correct_pixels / total_pixels if total_pixels > 0 else 0)
        colors_res = Counter()
        colors_exp = Counter()
        for row in result_grid:
            colors_res.update(row)
        for row in expected_grid:
            colors_exp.update(row)
        all_colors = set(colors_res.keys()) | set(colors_exp.keys())
        if all_colors:
            color_similarity = 0
            for color in all_colors:
                res_count = colors_res.get(color, 0)
                exp_count = colors_exp.get(color, 0)
                total = max(res_count, exp_count)
                if total > 0:
                    color_similarity += min(res_count, exp_count) / total
            fit += 0.2 * (color_similarity / len(all_colors))
        return min(1.0, fit)
    def solve_task(self, task: Dict, max_generations: int = 50,
                   timeout_seconds: int = 60) -> Dict:
        start_time = time.time()
        fingerprint = self.fingerprint_task(task)
        suggested_ops = fingerprint.get('suggested_ops', [])
        pop = []
        for i in range(self.population_size):
            dna_length = random.randint(2, self.max_dna_length)
            if i < self.population_size // 2 and suggested_ops:
                dna = [random.choice(suggested_ops + self.gene_pool)
                       for _ in range(dna_length)]
            else:
                underused = [op for op in self.gene_pool
                           if self.operation_usage[op] < 5]
                if underused and len(underused) > dna_length:
                    dna = random.sample(underused, dna_length)
                else:
                    dna = [random.choice(self.gene_pool) for _ in range(dna_length)]
            pop.append({'dna': dna, 'fit': 0.0, 'age': 0})
        best_ever = {'dna': [], 'fit': 0.0}
        generations_stuck = 0
        mutation_rate = 1.0
        for gen in range(max_generations):
            if time.time() - start_time > timeout_seconds:
                break
            for ind in pop:
                fit = self._evaluate_fitness_advanced(ind['dna'], task)
                ind['fit'] = fit
                if fit > best_ever['fit']:
                    best_ever = {'dna': ind['dna'][:], 'fit': fit}
                    generations_stuck = 0
                else:
                    generations_stuck += 1
            if best_ever['fit'] >= 0.99:
                print(f"      ✓ SOLVED in gen {gen+1}!")
                break
            if generations_stuck > 5:
                mutation_rate = min(3.0, mutation_rate * 1.2)
            else:
                mutation_rate = max(1.0, mutation_rate * 0.95)
            if generations_stuck > 10 and gen % 5 == 0:
                self._cw5_intervene(pop, task)
            pop = self._breed_generation_optimized(
                pop, mutation_rate, suggested_ops
            )
        time_elapsed = time.time() - start_time
        return {
            'solved': best_ever['fit'] >= 0.99,
            'best_fitness': best_ever['fit'],
            'best_dna': best_ever['dna'],
            'generations': gen + 1,
            'time_elapsed': time_elapsed,
            'cw5_interventions': self.cw5_interventions
        }
    def _evaluate_fitness_advanced(self, dna: List[str], task: Dict) -> float:
        if 'train' not in task or not task['train']:
            return 0.0
        total_fitness = 0.0
        total = len(task['train'])
        for example in task['train']:
            try:
                input_grid = example['input']
                expected_output = example['output']
                result = input_grid
                for gene in dna:
                    if gene in self.ops:
                        result = self.ops[gene](result)
                        self.operation_usage[gene] += 1
                fit = self.calculate_fitness_advanced(result, expected_output)
                total_fitness += fit
            except:
                pass
        return total_fitness / total if total > 0 else 0.0
    def _breed_generation_optimized(self, pop: List[Dict],
                                    mutation_rate: float,
                                    suggested_ops: List[str]) -> List[Dict]:
        sorted_pop = sorted(pop, key=lambda x: x['fit'], reverse=True)
        elite_count = max(1, len(pop) * 15 // 100)
        new_population = []
        for ind in sorted_pop[:elite_count]:
            new_ind = {'dna': ind['dna'][:], 'fit': 0.0, 'age': ind['age'] + 1}
            new_population.append(new_ind)
        while len(new_population) < self.population_size:
            tournament_size = 5
            tournament = random.sample(sorted_pop[:len(sorted_pop)//2], tournament_size)
            parent = max(tournament, key=lambda x: x['fit'])
            child_dna = parent['dna'][:]
            num_mutations = max(1, int(mutation_rate))
            for _ in range(num_mutations):
                child_dna = self._mutate_diverse(child_dna, suggested_ops)
            new_population.append({'dna': child_dna, 'fit': 0.0, 'age': 0})
        return new_population
    def _mutate_diverse(self, dna: List[str], suggested_ops: List[str]) -> List[str]:
        if not dna:
            return [random.choice(self.gene_pool)]
        new_dna = dna[:]
        mutation_type = random.choice(['insert', 'delete', 'modify', 'swap',
                                     'replace_with_suggested', 'replace_with_rare'])
        if mutation_type == 'insert' and len(new_dna) < self.max_dna_length:
            pos = random.randint(0, len(new_dna))
            underused = [op for op in self.gene_pool if self.operation_usage[op] < 10]
            gene = random.choice(underused) if underused else random.choice(self.gene_pool)
            new_dna.insert(pos, gene)
        elif mutation_type == 'delete' and len(new_dna) > 1:
            pos = random.randint(0, len(new_dna) - 1)
            new_dna.pop(pos)
        elif mutation_type == 'modify' and new_dna:
            pos = random.randint(0, len(new_dna) - 1)
            underused = [op for op in self.gene_pool if self.operation_usage[op] < 10]
            new_dna[pos] = random.choice(underused) if underused else random.choice(self.gene_pool)
        elif mutation_type == 'swap' and len(new_dna) >= 2:
            i, j = random.sample(range(len(new_dna)), 2)
            new_dna[i], new_dna[j] = new_dna[j], new_dna[i]
        elif mutation_type == 'replace_with_suggested' and suggested_ops and new_dna:
            pos = random.randint(0, len(new_dna) - 1)
            new_dna[pos] = random.choice(suggested_ops)
        elif mutation_type == 'replace_with_rare' and new_dna:
            rarest = min(self.gene_pool, key=lambda op: self.operation_usage[op])
            pos = random.randint(0, len(new_dna) - 1)
            new_dna[pos] = rarest
        return new_dna
    def _cw5_intervene(self, pop: List[Dict], task: Dict):
        self.cw5_coffee += 1
        self.cw5_cigarettes += 1
        self.cw5_interventions += 1
        # CW5's black magic: inject radical diversity
        pop.sort(key=lambda x: x['fit'], reverse=True)
        worst_count = len(pop) // 5
        rarest_ops = sorted(self.gene_pool, key=lambda op: self.operation_usage[op])[:20]
        for i in range(-worst_count, 0):
            dna_length = random.randint(3, self.max_dna_length)
            new_dna = [random.choice(rarest_ops) for _ in range(dna_length)]
            pop[i]['dna'] = new_dna
            pop[i]['fit'] = 0.0
            pop[i]['age'] = 0
def solve_arc_task(task: Dict) -> List[List[List[int]]]:
    ops = get_all_operations()
    solver = OptimizedEvolutionarySolver(ops)
    result = solver.solve_task(task, max_generations=50, timeout_seconds=60)
    if 'test' in task:
        predictions = []
        for test_input in task['test']:
            pred = test_input['input']
            for gene in result['best_dna']:
                if gene in ops:
                    pred = ops[gene](pred)
            predictions.append(pred)
        return predictions
    return []
def main():
    import sys
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            tasks = json.load(f)
        results = {}
        for task_id, task in tasks.items():
            predictions = solve_arc_task(task)
            results[task_id] = predictions
        print(json.dumps(results))
if __name__ == "__main__":
    main()
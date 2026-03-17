# OpenSKIZZE
# Copyright (C) 2025 [Alexander Hagg]
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Fitness function."""
from typing import Dict, List, Tuple
import numpy as np
import yaml
from scipy.ndimage import label, center_of_mass
from scipy import ndimage
from numba import njit

try:
    from .gp_utils import eval_gp, train_gp, acquire_ucb
except ImportError:
    try:
        from gp_utils import eval_gp, train_gp, acquire_ucb
    except ImportError:
        print("Warning: gp_utils not found. Surrogate modeling will be disabled.")
        eval_gp, train_gp, acquire_ucb = None, None, None 

def generate_base_environment(xy_cells=10, max_floors=3):
    """Creates an open 3D environment based on cell count."""
    return np.zeros((xy_cells, xy_cells, max_floors), dtype=np.int32)
    
def embed_3d_design_in_environment(env, design):
    exy, _, ez = env.shape
    dxy, _, dz = design.shape
    offset_xy = (exy - dxy) // 2
    z_layers_to_copy = min(ez, dz)
    env[offset_xy:offset_xy+dxy, offset_xy:offset_xy+dxy, 0:z_layers_to_copy] = design[:, :, 0:z_layers_to_copy]
    return env

@njit
def flood_fill_3d_njit(env, start_positions, max_length, design_xy_size):
    r_size, c_size, z_size = env.shape
    offset_xy = (r_size - design_xy_size) // 2
    visited = np.zeros(env.shape, dtype=np.bool_)
    distance_map = np.full(env.shape, np.inf, dtype=np.float64)
    total_voxels = env.shape[0] * env.shape[1] * env.shape[2]
    queue = np.empty((total_voxels, 3), dtype=np.int64)
    head = 0
    tail = 0
    for i in range(start_positions.shape[0]):
        r0, c0, z0 = start_positions[i, 0], start_positions[i, 1], start_positions[i, 2]
        if env[r0, c0, z0] == 0:
            visited[r0, c0, z0] = True
            distance_map[r0, c0, z0] = 0.0
            queue[tail, 0], queue[tail, 1], queue[tail, 2] = r0, c0, z0
            tail += 1
    neighbors = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, -1]], dtype=np.int64)
    costs = np.array([2, 2, 1, 2, 2], dtype=np.int64)
    while head < tail:
        cur_r, cur_c, cur_z = queue[head, 0], queue[head, 1], queue[head, 2]
        head += 1
        cur_dist = distance_map[cur_r, cur_c, cur_z]
        if cur_dist >= max_length: continue
        for i in range(5):
            dr, dc, dz, new_cost = neighbors[i, 0], neighbors[i, 1], neighbors[i, 2], costs[i]
            nr, nc, nz = cur_r + dr, cur_c + dc, cur_z + dz
            if not (0 <= nz < z_size and offset_xy <= nr < offset_xy + design_xy_size and offset_xy <= nc < offset_xy + design_xy_size): continue
            if env[nr, nc, nz] == 0 and not visited[nr, nc, nz]:
                visited[nr, nc, nz] = True
                distance_map[nr, nc, nz] = cur_dist + new_cost
                queue[tail, 0], queue[tail, 1], queue[tail, 2] = nr, nc, nz
                tail += 1
    return visited, distance_map

def init_environment(config_environment: Dict) -> Dict:
    with open("encodings/parametric/cfg.yml") as f:
        config_solution = yaml.safe_load(f)
    
    # Calculate grid size from physical size and scale
    xy_scale = config_solution.get('xy_scale', 1.0)
    env_cells = int(config_environment['environment_xy_size'] / xy_scale)

    env = generate_base_environment(
        xy_cells=env_cells,
        max_floors=config_solution['max_num_floors']
    )
    config_environment['env'] = env
    config_environment['env_cells'] = env_cells
    config_environment['length_design'] = config_solution['length_design'] # in cells
    return config_environment


def compute_fitness_3d(config_environment, env_with_solution):
    env_cells = config_environment['env_cells']
    design_cells = config_environment['length_design']
    offset_cells = (env_cells - design_cells) // 2
    max_flow_val = config_environment.get('max_flow_length')
    max_length = design_cells if max_flow_val is None or str(max_flow_val).lower() == 'none' else max_flow_val
    start_positions = []
    for r in range(env_cells):
        for c in range(offset_cells):
            if env_with_solution[r, c, 0] == 0:
                start_positions.append((r, c, 0))
    start_positions = np.array(start_positions, dtype=np.int64)
    visited, _ = flood_fill_3d_njit(env_with_solution, start_positions, max_length, design_cells)
    column_weights = np.linspace(1, 2, design_cells)**10
    column_weights /= column_weights.sum()
    visited_slice = visited[offset_cells:offset_cells+design_cells, offset_cells:offset_cells+design_cells, 0]
    reached_cells = visited_slice.sum(axis=0)
    weighted_reached_sum = np.dot(reached_cells, column_weights)
    return weighted_reached_sum / design_cells, visited


def calculate_compactness(heightmap: np.ndarray, pixel_size: float) -> float:
    """
    Calculate Surface-to-Volume ratio (A/V).
    Lower values indicate more compact forms (better for heat retention).
    """
    volume = np.sum(heightmap) * (pixel_size ** 2)
    if volume == 0:
        return 0.0
    
    # Roof area
    roof_area = np.sum(heightmap > 0) * (pixel_size ** 2)
    
    # Wall area calculation (vectorized)
    # Pad heightmap to handle boundaries (assume 0 height outside)
    padded = np.pad(heightmap, 1, mode='constant', constant_values=0)
    
    # Calculate positive height differences with 4 neighbors
    # (height - neighbor) > 0 contributes to wall area
    diff_n = np.maximum(0, padded[1:-1, 1:-1] - padded[0:-2, 1:-1])
    diff_s = np.maximum(0, padded[1:-1, 1:-1] - padded[2:, 1:-1])
    diff_w = np.maximum(0, padded[1:-1, 1:-1] - padded[1:-1, 0:-2])
    diff_e = np.maximum(0, padded[1:-1, 1:-1] - padded[1:-1, 2:])
    
    total_height_diff = np.sum(diff_n + diff_s + diff_w + diff_e)
    wall_area = total_height_diff * pixel_size
    
    surface_area = roof_area + wall_area
    
    return surface_area / volume

def calculate_park_factor(heightmap: np.ndarray, pixel_size: float) -> float:
    """
    Calculate 'Park Factor' (Green Space Radius).
    Average distance from any open space pixel to the nearest building.
    Higher values indicate larger contiguous open spaces.
    """
    open_space = heightmap == 0
    if not np.any(open_space):
        return 0.0
        
    # Distance transform: calculates distance to nearest background (0) pixel.
    # We want distance to nearest BUILDING. So Buildings should be 0, Open Space 1.
    # heightmap > 0 gives True (1) for buildings.
    # We invert it: heightmap == 0 gives True (1) for open space.
    # So input to edt is 1 for open space, 0 for buildings.
    # edt returns distance to nearest 0 (building).
    dist_map_pixels = ndimage.distance_transform_edt(heightmap == 0)
    
    dist_map_meters = dist_map_pixels * pixel_size
    
    # Average over open pixels only
    return np.mean(dist_map_meters[open_space])

def calculate_planning_features(heightmap: np.ndarray, config_encoding: Dict) -> np.ndarray:
    """
    Calculate the 8 CONSOLIDATED features matching OpenSKIZZE GUI (MVP).
    
    Args:
        heightmap: Building heights in meters (grid)
        config_encoding: Encoding configuration with xy_scale, z_scale, etc.
    
    Returns:
        Array of planning-focused features:
        [0] GRZ (Site Coverage Ratio) - ratio 0-1
        [1] GFZ (Floor Area Ratio) - ratio
        [2] Average Building Height (m)
        [3] Height Variability (m) - Standard Deviation
        [4] Average Building Distance (m) - Porosity
        [5] Number of Buildings (count) - Grain
        [6] Compactness (A/V Ratio) - 1/m - Energy
        [7] Park Factor (Green Space Radius) - m - Social
    """
    grid_res_y, grid_res_x = heightmap.shape
    occupied = heightmap > 0
    pixel_size = config_encoding.get('xy_scale', 1.0)
    pixel_area = pixel_size ** 2
    
    # Calculate buildable area (entire grid for now)
    buildable_area_in_sq_meters = grid_res_y * grid_res_x * pixel_area
    
    building_heights = heightmap[occupied]
    if not building_heights.any():
        return np.zeros(8)  # Return zeros for all 8 features
    
    # [0] GRZ (Grundflächenzahl) - Site Coverage Ratio
    # GRZ = Built Area / Total Site Area
    occupied_pixels = np.sum(occupied)
    built_area_m2 = occupied_pixels * pixel_area
    grz = built_area_m2 / buildable_area_in_sq_meters if buildable_area_in_sq_meters > 0 else 0.0
    grz = np.clip(grz, 0.0, 1.0)
    
    # [1] GFZ (Geschossflächenzahl) - Floor Area Ratio
    # GFZ = Total Floor Area / Total Site Area
    # Each meter of height contributes to floor area
    total_floor_area_m2 = np.sum(heightmap) * pixel_area
    gfz = total_floor_area_m2 / buildable_area_in_sq_meters if buildable_area_in_sq_meters > 0 else 0.0
    
    # [2] Average Height - heightmap is already in METERS
    avg_height_meters = np.mean(building_heights)
    
    # [3] Height Variability (m) - Standard Deviation
    height_std = np.std(building_heights)
    
    # [4] Average Building Distance (m) - Porosity
    labeled_array, num_buildings = label(occupied)
    if num_buildings > 1:
        centroids = np.array(center_of_mass(occupied, labeled_array, range(1, num_buildings + 1)))
        diff = centroids[:, None, :] - centroids[None, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        avg_spacing_pixels = np.mean(dists[np.triu_indices(num_buildings, k=1)])
        avg_spacing_meters = avg_spacing_pixels * pixel_size
    else:
        avg_spacing_meters = 0.0
        
    # [5] Number of Buildings (count) - Grain
    # num_buildings already calculated
    
    # [6] Compactness (A/V Ratio) - Energy
    compactness = calculate_compactness(heightmap, pixel_size)
    
    # [7] Park Factor (Green Space Radius) - Social
    park_factor = calculate_park_factor(heightmap, pixel_size)
    
    return np.array([
        grz, gfz, avg_height_meters, height_std,
        avg_spacing_meters, num_buildings, compactness, park_factor
    ])


def convert_from_numpy(solution, solution_template):
    solution_template.set_genome(solution)
    return solution_template

def eval(solution, config_environment: Dict, config_encoding: Dict, solution_template, use_surrogate=False, debug=False) -> Tuple:
    solution_obj = convert_from_numpy(solution, solution_template)
    phenotype_voxels = solution_obj.express(as_height_map=False)
    phenotype_floors = solution_obj.express(as_height_map=True)
    
    # Scale factors for physical dimensions
    xy_scale = config_encoding.get('xy_scale', 1.0)
    z_scale = config_encoding.get('z_scale', 3.0)
    phenotype_heightmap = phenotype_floors * z_scale

    # --- Calculate Planning Features (GUI implementation) ---
    all_features = calculate_planning_features(phenotype_heightmap, config_encoding)
    
    # --- Run flood fill simulation if needed ---
    debug_data = (None, None)
    if not use_surrogate:
        env_with_solution = embed_3d_design_in_environment(config_environment['env'].copy(), phenotype_voxels)
        fitness, visited_map = compute_fitness_3d(config_environment, env_with_solution)
        if debug:
            debug_data = (visited_map, None)
    else:
        fitness = 0.0

    # Select features based on config
    features = all_features[config_environment['features']]
    result_array = np.concatenate(([fitness], features, phenotype_floors.flatten()))
    return result_array, debug_data

def eval_multiple(solutions, config_environment: Dict, config_encoding: Dict, solution_template, surrogate_model=None, lambda_ucb=0.1, pool=None, debug=False) -> Tuple:
    """
    Evaluates multiple solutions.
    Returns standard numpy results AND a list of debug data if debug=True.
    """
    use_surrogate = surrogate_model is not None
    
    eval_results = []
    if pool is None:
        eval_results = [eval(sol, config_environment, config_encoding, solution_template, use_surrogate, debug=debug) for sol in solutions]
    else:
        async_results = [pool.apply_async(eval, args=(sol, config_environment, config_encoding, solution_template, use_surrogate, debug)) for sol in solutions]
        eval_results = [ar.get() for ar in async_results]
        
    # Unpack the list of tuples into separate lists
    results_arrays = np.array([res[0] for res in eval_results])
    debug_data_list = [res[1] for res in eval_results] if debug else None

    # Surrogate logic
    if surrogate_model is not None:
        heightmap_start = len(config_environment['features']) + 1
        heightmaps = results_arrays[:, heightmap_start:]
        ucb = acquire_ucb(surrogate_model, heightmaps, lambda_ucb=lambda_ucb)
        results_arrays[:, 0] = ucb
        
    return np.atleast_2d(results_arrays), debug_data_list
#!/usr/bin/env python3
"""
Compare KLAM_21 configurations to analyze landuse boundary effects.

This script tests different landuse configurations with a street canyon design
to investigate the harsh transition observed at landuse boundaries.

Configurations:
1. Current: Landuse 7 (upwind) → 2 (parcel+downwind), 1° slope upwind
2. Uniform vegetation: Landuse 7 everywhere, 1° slope upwind
3. Design-only built: Landuse 7 everywhere except building footprints = 2, 1° slope upwind
4. Gradual transition: Landuse 7 → 4 → 2 (smooth boundary), 1° slope upwind
5. Flat terrain: Landuse 7 (upwind) → 2 (parcel+downwind), no slope (0°)

Author: OpenSKIZZE Team
Date: December 2025
"""

import os
import sys
import argparse
import logging
import tempfile
import shutil
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import yaml

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from domain_description.evaluation_klam import (
    write_terrain_asc_file,
    write_buildings_asc_file,
    write_landuse_asc_file,
    generate_klam_in,
    read_asc_file,
    collect_all_timestamps
)
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StreetCanyonDesign:
    """Generate a street canyon building layout."""
    
    def __init__(self, parcel_size_m: int = 51, xy_scale: float = 3.0, z_scale: float = 3.0):
        """
        Initialize street canyon design.
        
        Args:
            parcel_size_m: Parcel size in meters
            xy_scale: Meters per grid cell
            z_scale: Meters per floor
        """
        self.parcel_size_m = parcel_size_m
        self.xy_scale = xy_scale
        self.z_scale = z_scale
        self.parcel_size_cells = int(parcel_size_m / xy_scale)
        
    def create_heightmap(self) -> np.ndarray:
        """
        Create street canyon heightmap: two parallel buildings on north/south edges.
        
        Returns:
            Heightmap array (parcel_size_cells, parcel_size_cells) in meters
        """
        heightmap = np.zeros((self.parcel_size_cells, self.parcel_size_cells), dtype=np.float32)
        
        # Building parameters
        building_height_m = 15.0  # 5 floors × 3m
        building_depth_cells = 4  # ~12m deep buildings
        
        # Building 1: North edge (top rows)
        heightmap[:building_depth_cells, :] = building_height_m
        
        # Building 2: South edge (bottom rows)
        heightmap[-building_depth_cells:, :] = building_height_m
        
        # Street canyon in the middle remains at 0
        
        logger.info(f"Street canyon: 2 buildings × {building_depth_cells * self.xy_scale}m deep, "
                   f"{building_height_m}m tall, canyon width ~{(self.parcel_size_cells - 2*building_depth_cells) * self.xy_scale}m")
        
        return heightmap
    
    def get_building_mask(self) -> np.ndarray:
        """
        Get binary mask of building footprints.
        
        Returns:
            Boolean array (parcel_size_cells, parcel_size_cells), True = building
        """
        heightmap = self.create_heightmap()
        return heightmap > 0


class KLAMConfiguration:
    """Configure and run KLAM_21 simulations with different setups."""
    
    def __init__(
        self,
        parcel_size_m: int = 51,
        xy_scale: float = 3.0,
        wind_speed: float = 0.0,
        wind_direction: float = 270.0,
        sim_duration: int = 14400,
        klam_binary: str = "domain_description/klam_21_v2012"
    ):
        """
        Initialize KLAM configuration manager.
        
        Args:
            parcel_size_m: Parcel size in meters
            xy_scale: Meters per grid cell
            wind_speed: Wind speed in m/s
            wind_direction: Wind direction in degrees
            sim_duration: Simulation duration in seconds
            klam_binary: Path to KLAM binary
        """
        self.parcel_size_m = parcel_size_m
        self.xy_scale = xy_scale
        self.parcel_size_cells = int(parcel_size_m / xy_scale)
        self.wind_speed = wind_speed
        self.wind_direction = wind_direction
        self.sim_duration = sim_duration
        self.klam_binary = Path(project_root) / klam_binary
        
        # Domain setup (100% upwind extension)
        self.environment_size = 200  # Base environment size
        self.extension_factor = 1.0  # 100% extension upwind
        
    def compute_domain_geometry(self, heightmap: np.ndarray) -> Dict:
        """
        Compute extended domain geometry.
        
        Args:
            heightmap: Design heightmap (parcel_size_cells, parcel_size_cells)
            
        Returns:
            Dictionary with domain parameters
        """
        extension = int(self.environment_size * self.extension_factor)
        total_width = self.environment_size + extension
        total_height = self.environment_size
        
        nx = int(total_width / self.xy_scale)
        ny = int(total_height / self.xy_scale)
        
        # Center parcel vertically, offset horizontally by extension
        parcel_offset_x = int(extension / self.xy_scale)
        parcel_offset_y = int((total_height - self.parcel_size_m) / (2 * self.xy_scale))
        
        return {
            'nx': nx,
            'ny': ny,
            'dx': self.xy_scale,
            'total_width': total_width,
            'total_height': total_height,
            'extension': extension,
            'parcel_offset_x': parcel_offset_x,
            'parcel_offset_y': parcel_offset_y
        }
    
    def create_terrain_current(self, geom: Dict) -> np.ndarray:
        """
        Create terrain with 1° upslope (current setup).
        
        Args:
            geom: Domain geometry dict
            
        Returns:
            Terrain grid (ny, nx) in meters
        """
        terrain = np.zeros((geom['ny'], geom['nx']), dtype=np.float32)
        
        # 2° slope upwind (left of parcel)
        slope_angle_deg = 2.0
        slope_gradient = np.tan(np.radians(slope_angle_deg))
        
        parcel_left_edge = geom['parcel_offset_x']
        
        for i in range(geom['nx']):
            if i < parcel_left_edge:
                # Distance from parcel edge
                distance = (parcel_left_edge - i) * self.xy_scale
                terrain[:, i] = distance * slope_gradient
        
        return terrain
    
    def create_terrain_flat(self, geom: Dict) -> np.ndarray:
        """
        Create flat terrain (no slope).
        
        Args:
            geom: Domain geometry dict
            
        Returns:
            Terrain grid (ny, nx) in meters
        """
        return np.zeros((geom['ny'], geom['nx']), dtype=np.float32)
    
    def create_terrain_continuous_slope(self, geom: Dict) -> np.ndarray:
        """
        Create terrain with 1° slope continuing through entire domain (no discontinuity).
        
        Args:
            geom: Domain geometry dict
            
        Returns:
            Terrain grid (ny, nx) in meters
        """
        terrain = np.zeros((geom['ny'], geom['nx']), dtype=np.float32)
        
        # 2° slope across entire domain (continuous from left to right)
        slope_angle_deg = 2.0
        slope_gradient = np.tan(np.radians(slope_angle_deg))
        
        # Apply slope to ALL columns
        for i in range(geom['nx']):
            # Distance from right edge (slope increases toward left)
            distance_from_right = (geom['nx'] - i) * self.xy_scale
            terrain[:, i] = distance_from_right * slope_gradient
        
        return terrain
    
    def create_landuse_current(self, geom: Dict, building_mask: np.ndarray) -> np.ndarray:
        """
        Create landuse: 7 (upwind) → 2 (parcel+downwind).
        
        Args:
            geom: Domain geometry dict
            building_mask: Building footprint mask
            
        Returns:
            Landuse grid (ny, nx)
        """
        landuse = np.full((geom['ny'], geom['nx']), 2, dtype=np.int32)
        
        # Landuse 7 upwind (left of parcel)
        parcel_left_edge = geom['parcel_offset_x']
        landuse[:, :parcel_left_edge] = 7
        
        return landuse
    
    def create_landuse_uniform_vegetation(self, geom: Dict, building_mask: np.ndarray) -> np.ndarray:
        """
        Create landuse: 7 everywhere.
        
        Args:
            geom: Domain geometry dict
            building_mask: Building footprint mask
            
        Returns:
            Landuse grid (ny, nx)
        """
        return np.full((geom['ny'], geom['nx']), 7, dtype=np.int32)
    
    def create_landuse_design_only(self, geom: Dict, building_mask: np.ndarray) -> np.ndarray:
        """
        Create landuse: 7 everywhere except building footprints (2).
        
        Args:
            geom: Domain geometry dict
            building_mask: Building footprint mask (parcel coordinates)
            
        Returns:
            Landuse grid (ny, nx)
        """
        landuse = np.full((geom['ny'], geom['nx']), 7, dtype=np.int32)
        
        # Set landuse=2 only under buildings
        y_start = geom['parcel_offset_y']
        y_end = y_start + building_mask.shape[0]
        x_start = geom['parcel_offset_x']
        x_end = x_start + building_mask.shape[1]
        
        landuse[y_start:y_end, x_start:x_end][building_mask] = 2
        
        return landuse
    
    def create_landuse_gradual_transition(self, geom: Dict, building_mask: np.ndarray) -> np.ndarray:
        """
        Create landuse: 7 → 4 → 2 (gradual transition).
        
        Args:
            geom: Domain geometry dict
            building_mask: Building footprint mask
            
        Returns:
            Landuse grid (ny, nx)
        """
        landuse = np.full((geom['ny'], geom['nx']), 2, dtype=np.int32)
        
        parcel_left_edge = geom['parcel_offset_x']
        transition_width = int(30 / self.xy_scale)  # 30m transition zone
        
        # Zone 7: Far upwind
        landuse[:, :parcel_left_edge - transition_width] = 7
        
        # Zone 4: Transition zone
        landuse[:, parcel_left_edge - transition_width:parcel_left_edge] = 4
        
        # Zone 2: Parcel + downwind (already set)
        
        return landuse
    
    def create_landuse_boundary_after_parcel(self, geom: Dict, building_mask: np.ndarray) -> np.ndarray:
        """
        Create landuse: 7 upwind and through parcel, 2 only to the RIGHT of parcel.
        
        This isolates the boundary effect to AFTER the design, not before it.
        
        Args:
            geom: Domain geometry dict
            building_mask: Building footprint mask
            
        Returns:
            Landuse grid (ny, nx)
        """
        landuse = np.full((geom['ny'], geom['nx']), 7, dtype=np.int32)
        
        # Landuse 2 starts to the RIGHT of the parcel (downwind)
        parcel_right_edge = geom['parcel_offset_x'] + building_mask.shape[1]
        landuse[:, parcel_right_edge:] = 2
        
        return landuse
    
    def create_landuse_boundary_far_right(self, geom: Dict, building_mask: np.ndarray) -> np.ndarray:
        """
        Create landuse: 7 upwind through parcel and halfway beyond, 2 only far right.
        
        Landuse 2 starts halfway between parcel right edge and domain right edge.
        
        Args:
            geom: Domain geometry dict
            building_mask: Building footprint mask
            
        Returns:
            Landuse grid (ny, nx)
        """
        landuse = np.full((geom['ny'], geom['nx']), 7, dtype=np.int32)
        
        # Landuse 2 starts halfway between parcel right edge and domain edge
        parcel_right_edge = geom['parcel_offset_x'] + building_mask.shape[1]
        domain_right_edge = geom['nx']
        boundary_position = parcel_right_edge + (domain_right_edge - parcel_right_edge) // 2
        landuse[:, boundary_position:] = 2
        
        return landuse
    
    def create_landuse_boundary_50pct_right(self, geom: Dict, building_mask: np.ndarray) -> np.ndarray:
        """
        Create landuse: 7 upwind and through parcel, 2 starts 50% parcel width to RIGHT of parcel.
        
        This places the boundary at a distance equal to 50% of the parcel width,
        measured from the right edge of the parcel.
        
        Args:
            geom: Domain geometry dict
            building_mask: Building footprint mask
            
        Returns:
            Landuse grid (ny, nx)
        """
        landuse = np.full((geom['ny'], geom['nx']), 7, dtype=np.int32)
        
        # Landuse 2 starts 50% of parcel width to the right of parcel right edge
        parcel_width_cells = building_mask.shape[1]
        parcel_right_edge = geom['parcel_offset_x'] + parcel_width_cells
        boundary_offset = int(0.5 * parcel_width_cells)
        boundary_position = parcel_right_edge + boundary_offset
        
        # Ensure boundary doesn't exceed domain
        boundary_position = min(boundary_position, geom['nx'])
        landuse[:, boundary_position:] = 2
        
        return landuse
    
    def create_landuse_boundary_after_with_parcel_built(self, geom: Dict, building_mask: np.ndarray) -> np.ndarray:
        """
        Create landuse: 7 upwind, 2 on parcel itself, 2 downwind (starting at parcel RIGHT edge).
        
        The parcel area is landuse 2, upwind remains landuse 7, downwind is landuse 2.
        This isolates the building effect while having the parcel as a built-up area.
        
        Args:
            geom: Domain geometry dict
            building_mask: Building footprint mask (parcel coordinates)
            
        Returns:
            Landuse grid (ny, nx)
        """
        landuse = np.full((geom['ny'], geom['nx']), 7, dtype=np.int32)
        
        # Landuse 2 on the parcel itself (not starting at left edge)
        parcel_left_edge = geom['parcel_offset_x']
        parcel_right_edge = parcel_left_edge + building_mask.shape[1]
        parcel_top = geom['parcel_offset_y']
        parcel_bottom = parcel_top + building_mask.shape[0]
        
        # Set parcel area to landuse 2
        landuse[parcel_top:parcel_bottom, parcel_left_edge:parcel_right_edge] = 2
        
        # Landuse 2 also starts at parcel right edge and continues downwind
        landuse[:, parcel_right_edge:] = 2
        
        return landuse
    
    def embed_buildings(self, geom: Dict, heightmap: np.ndarray) -> np.ndarray:
        """
        Embed building heightmap into full domain.
        
        Args:
            geom: Domain geometry dict
            heightmap: Building heightmap (parcel_size_cells, parcel_size_cells)
            
        Returns:
            Building grid (ny, nx) in meters
        """
        buildings = np.zeros((geom['ny'], geom['nx']), dtype=np.float32)
        
        y_start = geom['parcel_offset_y']
        y_end = y_start + heightmap.shape[0]
        x_start = geom['parcel_offset_x']
        x_end = x_start + heightmap.shape[1]
        
        buildings[y_start:y_end, x_start:x_end] = heightmap
        
        return buildings
    
    def run_klam_simulation(
        self,
        terrain: np.ndarray,
        buildings: np.ndarray,
        landuse: np.ndarray,
        geom: Dict,
        output_dir: Path
    ) -> Dict[str, np.ndarray]:
        """
        Run KLAM_21 simulation and return output fields.
        
        Args:
            terrain: Terrain grid (ny, nx)
            buildings: Building grid (ny, nx)
            landuse: Landuse grid (ny, nx)
            geom: Domain geometry dict
            output_dir: Output directory for KLAM files
            
        Returns:
            Dictionary of output fields (uq, vq, uz, vz, Hx, Ex) at final timestep
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        examples_dir = output_dir / "examples"
        examples_dir.mkdir(exist_ok=True)
        results_dir = output_dir / "results"
        results_dir.mkdir(exist_ok=True)
        
        # Write input files
        write_terrain_asc_file(str(examples_dir), "terrain.asc", terrain)
        write_buildings_asc_file(str(examples_dir), "buildings.asc", buildings)
        write_landuse_asc_file(str(examples_dir), "landuse.asc", landuse)
        
        # Generate KLAM control file
        klam_config = {
            'nx': geom['nx'],
            'ny': geom['ny'],
            'dx': geom['dx'],
            'wind_speed': self.wind_speed,
            'wind_direction': self.wind_direction,
            'sim_duration': self.sim_duration
        }
        generate_klam_in(str(output_dir), klam_config)
        
        # Run KLAM binary
        logger.info(f"Running KLAM_21 simulation (nx={geom['nx']}, ny={geom['ny']})...")
        
        try:
            result = subprocess.run(
                [str(self.klam_binary)],
                cwd=str(output_dir),
                check=True,
                timeout=300,
                capture_output=True,
                text=True
            )
            logger.info(f"KLAM_21 completed successfully")
            if result.stdout:
                logger.debug(f"STDOUT: {result.stdout[:500]}")
        except subprocess.CalledProcessError as e:
            logger.error(f"KLAM_21 failed with return code {e.returncode}")
            logger.error(f"STDERR: {e.stderr}")
            logger.error(f"STDOUT: {e.stdout}")
            raise
        except subprocess.TimeoutExpired:
            logger.error("KLAM_21 timeout (300s)")
            raise
        
        # Read output fields at final timestep
        results_dir = output_dir / "results"
        
        # Find available timestamps from uq files
        import re
        all_files = sorted(os.listdir(str(results_dir))) if results_dir.exists() else []
        uq_files = [f for f in all_files if f.startswith('uq') and f.endswith('.asc')]
        
        if not uq_files:
            raise RuntimeError("No KLAM_21 output files found")
        
        # Parse timestamps
        timestamps = []
        for f in uq_files:
            match = re.match(r'uq(\d+)\.asc', f)
            if match:
                timestamps.append(int(match.group(1)))
        
        timestamps.sort()
        final_time = max(timestamps)
        logger.info(f"Reading KLAM_21 outputs at t={final_time}s")
        
        outputs = {}
        for field in ['uq', 'vq', 'uz', 'vz', 'Hx', 'Ex']:
            # Try different filename formats
            patterns = []
            if field in ['Hx', 'Ex']:
                # Hx/Ex use 6-digit padding
                patterns.append(f"{field}{final_time:06d}.asc")
            else:
                # Velocity fields may use 6-digit padding or no padding
                patterns.append(f"{field}{final_time:06d}.asc")
                patterns.append(f"{field}{final_time}.asc")
            
            found = False
            for pattern in patterns:
                file_path = results_dir / pattern
                if file_path.exists():
                    data, metadata = read_asc_file(str(file_path))
                    outputs[field] = data
                    logger.info(f"  {field}: {data.shape}, range=[{data.min():.3f}, {data.max():.3f}]")
                    found = True
                    break
            
            if not found:
                logger.warning(f"  {field}: file not found (tried {patterns})")
        
        return outputs


class ConfigurationComparator:
    """Compare multiple KLAM configurations."""
    
    def __init__(self, base_output_dir: Path):
        """
        Initialize comparator.
        
        Args:
            base_output_dir: Base directory for all outputs
        """
        self.base_output_dir = base_output_dir
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
    
    def _run_single_config(self, args):
        """
        Run a single configuration (for parallel execution).
        
        Args:
            args: Tuple of (config_id, config_data, klam, geom, buildings, output_dir)
            
        Returns:
            Tuple of (config_id, result_dict or None, error_message or None)
        """
        config_id, config_data, klam_params, geom, buildings, output_dir = args
        
        # Recreate KLAM instance in this process
        klam = KLAMConfiguration(**klam_params)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Running {config_id}: {config_data['name']}")
        logger.info(f"{'='*60}")
        
        config_output_dir = output_dir / config_id
        
        try:
            outputs = klam.run_klam_simulation(
                terrain=config_data['terrain'],
                buildings=buildings,
                landuse=config_data['landuse'],
                geom=geom,
                output_dir=config_output_dir
            )
            
            result = {
                'name': config_data['name'],
                'outputs': outputs,
                'terrain': config_data['terrain'],
                'landuse': config_data['landuse'],
                'buildings': buildings,
                'geom': geom
            }
            
            logger.info(f"✓ {config_id} complete")
            return (config_id, result, None)
            
        except Exception as e:
            logger.error(f"✗ {config_id} failed: {e}")
            return (config_id, None, str(e))
    
    def run_all_configurations(
        self,
        parcel_size_m: int = 51,
        xy_scale: float = 3.0,
        n_parallel: int = 8
    ) -> Dict[str, Dict]:
        """
        Run all 8 configurations and collect results.
        
        Args:
            parcel_size_m: Parcel size in meters
            xy_scale: Meters per grid cell
            n_parallel: Number of parallel KLAM simulations
            
        Returns:
            Dictionary mapping config name to results dict
        """
        # Create street canyon design
        design = StreetCanyonDesign(parcel_size_m, xy_scale)
        heightmap = design.create_heightmap()
        building_mask = design.get_building_mask()
        
        # Initialize KLAM manager
        klam = KLAMConfiguration(parcel_size_m, xy_scale)
        geom = klam.compute_domain_geometry(heightmap)
        buildings = klam.embed_buildings(geom, heightmap)
        
        # KLAM parameters for reconstruction in parallel processes
        klam_params = {
            'parcel_size_m': parcel_size_m,
            'xy_scale': xy_scale,
            'wind_speed': klam.wind_speed,
            'wind_direction': klam.wind_direction,
            'sim_duration': klam.sim_duration,
            'klam_binary': str(klam.klam_binary)
        }
        
        # Configuration definitions
        # NOTE: Removed uniform landuse configs (2, 3, 9) - KLAM_21 requires landuse transitions for flow
        configs = {
            'config1_current': {
                'name': 'Current (7→2 at parcel, 1° slope upwind only)',
                'terrain': klam.create_terrain_current(geom),
                'landuse': klam.create_landuse_current(geom, building_mask)
            },
            'config4_gradual': {
                'name': 'Gradual transition (7→4→2, 1° slope upwind only)',
                'terrain': klam.create_terrain_current(geom),
                'landuse': klam.create_landuse_gradual_transition(geom, building_mask)
            },
            'config5_flat': {
                'name': 'Flat terrain (7→2 at parcel, 0° slope)',
                'terrain': klam.create_terrain_flat(geom),
                'landuse': klam.create_landuse_current(geom, building_mask)
            },
            'config6_boundary_after': {
                'name': 'Boundary after parcel (7→2 after parcel, 1° slope upwind only)',
                'terrain': klam.create_terrain_current(geom),
                'landuse': klam.create_landuse_boundary_after_parcel(geom, building_mask)
            },
            'config7_continuous_slope': {
                'name': 'Continuous slope (7→2 at parcel, 1° slope everywhere)',
                'terrain': klam.create_terrain_continuous_slope(geom),
                'landuse': klam.create_landuse_current(geom, building_mask)
            },
            'config8_boundary_after_continuous_slope': {
                'name': 'Boundary after + continuous slope (7→2 after parcel, 1° everywhere)',
                'terrain': klam.create_terrain_continuous_slope(geom),
                'landuse': klam.create_landuse_boundary_after_parcel(geom, building_mask)
            },
            'config10_parcel_built_boundary_after_continuous': {
                'name': 'Parcel built + boundary after + continuous slope (7|parcel=2|2, 1° everywhere)',
                'terrain': klam.create_terrain_continuous_slope(geom),
                'landuse': klam.create_landuse_boundary_after_with_parcel_built(geom, building_mask)
            },
            'config11_boundary_50pct_right_continuous': {
                'name': 'Boundary 50% right + continuous slope (7→2 at +50% parcel width, 1° everywhere)',
                'terrain': klam.create_terrain_continuous_slope(geom),
                'landuse': klam.create_landuse_boundary_50pct_right(geom, building_mask)
            }
        }
        
        # Prepare arguments for parallel execution
        job_args = [
            (config_id, config_data, klam_params, geom, buildings, self.base_output_dir)
            for config_id, config_data in configs.items()
        ]
        
        results = {}
        
        # Run configurations in parallel
        logger.info(f"Running {len(configs)} configurations with {n_parallel} parallel workers...")
        
        with ProcessPoolExecutor(max_workers=n_parallel) as executor:
            futures = {executor.submit(self._run_single_config, args): args[0] 
                      for args in job_args}
            
            for future in as_completed(futures):
                config_id = futures[future]
                try:
                    config_id, result, error = future.result()
                    if result is not None:
                        results[config_id] = result
                    else:
                        results[config_id] = None
                        logger.error(f"Configuration {config_id} failed: {error}")
                except Exception as e:
                    logger.error(f"Exception in {config_id}: {e}")
                    results[config_id] = None
        
        return results
    
    def compute_metrics(self, results: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Compute quantitative comparison metrics.
        
        Args:
            results: Results from run_all_configurations
            
        Returns:
            Dictionary of metrics per configuration
        """
        metrics = {}
        
        for config_id, result in results.items():
            if result is None:
                continue
            
            outputs = result['outputs']
            geom = result['geom']
            
            # Convert units
            uq = outputs['uq'] / 100.0  # cm/s → m/s
            vq = outputs['vq'] / 100.0  # cm/s → m/s
            uz = outputs.get('uz', np.zeros_like(uq)) / 100.0  # cm/s → m/s
            vz = outputs.get('vz', np.zeros_like(vq)) / 100.0  # cm/s → m/s
            Hx = outputs.get('Hx', np.zeros_like(uq)) / 10.0  # 1/10 m → m
            Ex = outputs['Ex']  # 100 J/m²
            
            # Wind speed at 2m
            u_2m = np.sqrt(uq**2 + vq**2)
            
            # Cold air flux
            flux = Ex * u_2m  # 100 W/m²
            
            # ROI: parcel + buffer
            parcel_left = geom['parcel_offset_x']
            parcel_right = parcel_left + int(result['geom']['ny'] * 0.17)  # Approximate parcel width
            roi_left = max(0, parcel_left - int(0.5 * parcel_right - parcel_left))
            roi_right = min(geom['nx'], geom['nx'])  # To right edge
            
            # Compute metrics over ROI
            roi_slice = (slice(None), slice(roi_left, roi_right))
            
            metrics[config_id] = {
                'name': result['name'],
                'mean_velocity_2m': float(np.mean(u_2m[roi_slice])),
                'max_velocity_2m': float(np.max(u_2m[roi_slice])),
                'mean_Ex': float(np.mean(Ex[roi_slice])),
                'mean_Hx': float(np.mean(Hx[roi_slice])),
                'mean_flux': float(np.mean(flux[roi_slice])),
                'total_flux': float(np.sum(flux[roi_slice])),
                'velocity_std': float(np.std(u_2m[roi_slice])),
                'Ex_std': float(np.std(Ex[roi_slice]))
            }
            
            logger.info(f"\n{config_id} metrics:")
            for key, value in metrics[config_id].items():
                if key != 'name':
                    logger.info(f"  {key}: {value:.4f}")
        
        return metrics
    
    def plot_comparisons(self, results: Dict[str, Dict], metrics: Dict[str, Dict]):
        """
        Generate comparison plots for all configurations.
        
        Args:
            results: Results from run_all_configurations
            metrics: Metrics from compute_metrics
        """
        plots_dir = self.base_output_dir / "comparison_plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Filter out failed configs
        valid_results = {k: v for k, v in results.items() if v is not None}
        
        if not valid_results:
            logger.error("No valid results to plot")
            return
        
        # Plot 1: Terrain configurations
        self._plot_terrain_comparison(valid_results, plots_dir)
        
        # Plot 2: Velocity fields (uq at 2m)
        self._plot_velocity_fields(valid_results, plots_dir)
        
        # Plot 3: Cold air content (Ex)
        self._plot_cold_air_content(valid_results, plots_dir)
        
        # Plot 4: Cold air height (Hx)
        self._plot_cold_air_height(valid_results, plots_dir)
        
        # Plot 5: Cold air flux
        self._plot_cold_air_flux(valid_results, plots_dir)
        
        # Plot 6: Landuse configurations
        self._plot_landuse_comparison(valid_results, plots_dir)
        
        # Plot 7: Velocity profiles (cross-section)
        self._plot_velocity_profiles(valid_results, plots_dir)
        
        # Plot 8: Terrain profiles
        self._plot_terrain_profiles(valid_results, plots_dir)
        
        # Plot 9: Metrics bar charts
        self._plot_metrics_comparison(metrics, plots_dir)
        
        logger.info(f"\nPlots saved to: {plots_dir}")
    
    def _plot_terrain_comparison(self, results: Dict[str, Dict], output_dir: Path):
        """Plot terrain configurations for all configurations."""
        n_configs = len(results)
        fig, axes = plt.subplots(n_configs, 1, figsize=(16, 3*n_configs))
        
        if n_configs == 1:
            axes = [axes]
        
        # Compute global min/max for consistent color range
        all_displays = []
        for config_id, result in results.items():
            terrain = result['terrain']
            buildings = result['buildings']
            display = terrain.copy()
            display[buildings > 0] = terrain[buildings > 0] + buildings[buildings > 0]
            all_displays.append(display)
        vmin = min(d.min() for d in all_displays)
        vmax = max(d.max() for d in all_displays)
        
        for idx, (config_id, result) in enumerate(results.items()):
            terrain = result['terrain']
            buildings = result['buildings']
            
            # Create composite: terrain + buildings overlay
            display = terrain.copy()
            display[buildings > 0] = terrain[buildings > 0] + buildings[buildings > 0]
            
            im = axes[idx].imshow(display, cmap='terrain', origin='lower', aspect='auto', vmin=vmin, vmax=vmax)
            axes[idx].set_title(f"{result['name']} - Terrain + Buildings", fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('X (cells)')
            axes[idx].set_ylabel('Y (cells)')
            
            # Mark parcel boundaries
            geom = result['geom']
            parcel_left = geom['parcel_offset_x']
            parcel_right = parcel_left + int(51 / geom['dx'])
            axes[idx].axvline(parcel_left, color='white', linestyle='--', linewidth=2, alpha=0.7, label='Parcel left')
            axes[idx].axvline(parcel_right, color='yellow', linestyle='--', linewidth=2, alpha=0.7, label='Parcel right')
            axes[idx].legend(loc='upper right')
            
            plt.colorbar(im, ax=axes[idx], label='Elevation (m)')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'terrain_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("  ✓ terrain_comparison.png")
    
    def _plot_velocity_fields(self, results: Dict[str, Dict], output_dir: Path):
        """Plot velocity fields for all configurations."""
        n_configs = len(results)
        fig, axes = plt.subplots(n_configs, 1, figsize=(16, 4*n_configs))
        
        if n_configs == 1:
            axes = [axes]
        
        # Compute global min/max for consistent color range
        all_velocities = []
        for config_id, result in results.items():
            uq = result['outputs']['uq'] / 100.0  # cm/s → m/s
            vq = result['outputs']['vq'] / 100.0
            u_2m = np.sqrt(uq**2 + vq**2)
            all_velocities.append(u_2m)
        vmin = min(v.min() for v in all_velocities)
        vmax = max(v.max() for v in all_velocities)
        
        for idx, (config_id, result) in enumerate(results.items()):
            uq = result['outputs']['uq'] / 100.0  # cm/s → m/s
            vq = result['outputs']['vq'] / 100.0
            u_2m = np.sqrt(uq**2 + vq**2)
            
            im = axes[idx].imshow(u_2m, cmap='jet', origin='lower', aspect='auto', vmin=vmin, vmax=vmax)
            axes[idx].set_title(f"{result['name']} - Wind Speed at 2m", fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('X (cells)')
            axes[idx].set_ylabel('Y (cells)')
            
            # Mark parcel boundaries
            geom = result['geom']
            parcel_left = geom['parcel_offset_x']
            parcel_right = parcel_left + int(51 / geom['dx'])
            axes[idx].axvline(parcel_left, color='white', linestyle='--', linewidth=2, alpha=0.7, label='Parcel left')
            axes[idx].axvline(parcel_right, color='yellow', linestyle='--', linewidth=2, alpha=0.7, label='Parcel right')
            axes[idx].legend(loc='upper right')
            
            plt.colorbar(im, ax=axes[idx], label='Wind speed (m/s)')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'velocity_fields_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("  ✓ velocity_fields_comparison.png")
    
    def _plot_cold_air_content(self, results: Dict[str, Dict], output_dir: Path):
        """Plot cold air content (Ex) for all configurations."""
        n_configs = len(results)
        fig, axes = plt.subplots(n_configs, 1, figsize=(16, 4*n_configs))
        
        if n_configs == 1:
            axes = [axes]
        
        # Compute global min/max for consistent color range
        all_Ex = [result['outputs']['Ex'] for result in results.values()]
        vmin = min(ex.min() for ex in all_Ex)
        vmax = max(ex.max() for ex in all_Ex)
        
        for idx, (config_id, result) in enumerate(results.items()):
            Ex = result['outputs']['Ex']
            
            im = axes[idx].imshow(Ex, cmap='viridis', origin='lower', aspect='auto', vmin=vmin, vmax=vmax)
            axes[idx].set_title(f"{result['name']} - Cold Air Content (Ex)", fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('X (cells)')
            axes[idx].set_ylabel('Y (cells)')
            
            # Mark parcel boundaries
            geom = result['geom']
            parcel_left = geom['parcel_offset_x']
            parcel_right = parcel_left + int(51 / geom['dx'])
            axes[idx].axvline(parcel_left, color='white', linestyle='--', linewidth=2, alpha=0.7)
            axes[idx].axvline(parcel_right, color='red', linestyle='--', linewidth=2, alpha=0.7)
            
            plt.colorbar(im, ax=axes[idx], label='Ex (100 J/m²)')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'cold_air_content_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("  ✓ cold_air_content_comparison.png")
    
    def _plot_cold_air_height(self, results: Dict[str, Dict], output_dir: Path):
        """Plot cold air layer height (Hx) for all configurations."""
        n_configs = len(results)
        fig, axes = plt.subplots(n_configs, 1, figsize=(16, 4*n_configs))
        
        if n_configs == 1:
            axes = [axes]
        
        # Compute global min/max for consistent color range
        all_Hx = [result['outputs']['Hx'] / 10.0 for result in results.values()]  # 1/10 m → m
        vmin = min(hx.min() for hx in all_Hx)
        vmax = max(hx.max() for hx in all_Hx)
        
        for idx, (config_id, result) in enumerate(results.items()):
            Hx = result['outputs']['Hx'] / 10.0  # 1/10 m → m
            
            im = axes[idx].imshow(Hx, cmap='coolwarm', origin='lower', aspect='auto', vmin=vmin, vmax=vmax)
            axes[idx].set_title(f"{result['name']} - Cold Air Layer Height (Hx)", fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('X (cells)')
            axes[idx].set_ylabel('Y (cells)')
            
            # Mark parcel boundaries
            geom = result['geom']
            parcel_left = geom['parcel_offset_x']
            parcel_right = parcel_left + int(51 / geom['dx'])
            axes[idx].axvline(parcel_left, color='white', linestyle='--', linewidth=2, alpha=0.7)
            axes[idx].axvline(parcel_right, color='yellow', linestyle='--', linewidth=2, alpha=0.7)
            
            plt.colorbar(im, ax=axes[idx], label='Cold air height (m)')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'cold_air_height_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("  ✓ cold_air_height_comparison.png")
    
    def _plot_cold_air_flux(self, results: Dict[str, Dict], output_dir: Path):
        """Plot cold air flux for all configurations."""
        n_configs = len(results)
        fig, axes = plt.subplots(n_configs, 1, figsize=(16, 4*n_configs))
        
        if n_configs == 1:
            axes = [axes]
        
        # Compute global min/max for consistent color range
        all_flux = []
        for config_id, result in results.items():
            uq = result['outputs']['uq'] / 100.0
            vq = result['outputs']['vq'] / 100.0
            Ex = result['outputs']['Ex']
            u_2m = np.sqrt(uq**2 + vq**2)
            flux = Ex * u_2m
            all_flux.append(flux)
        vmin = min(f.min() for f in all_flux)
        vmax = max(f.max() for f in all_flux)
        
        for idx, (config_id, result) in enumerate(results.items()):
            uq = result['outputs']['uq'] / 100.0
            vq = result['outputs']['vq'] / 100.0
            Ex = result['outputs']['Ex']
            u_2m = np.sqrt(uq**2 + vq**2)
            flux = Ex * u_2m
            
            im = axes[idx].imshow(flux, cmap='plasma', origin='lower', aspect='auto', vmin=vmin, vmax=vmax)
            axes[idx].set_title(f"{result['name']} - Cold Air Flux", fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('X (cells)')
            axes[idx].set_ylabel('Y (cells)')
            
            # Mark parcel boundaries
            geom = result['geom']
            parcel_left = geom['parcel_offset_x']
            parcel_right = parcel_left + int(51 / geom['dx'])
            axes[idx].axvline(parcel_left, color='white', linestyle='--', linewidth=2, alpha=0.7)
            axes[idx].axvline(parcel_right, color='cyan', linestyle='--', linewidth=2, alpha=0.7)
            
            plt.colorbar(im, ax=axes[idx], label='Flux (100 W/m²)')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'cold_air_flux_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("  ✓ cold_air_flux_comparison.png")
    
    def _plot_landuse_comparison(self, results: Dict[str, Dict], output_dir: Path):
        """Plot landuse configurations."""
        n_configs = len(results)
        fig, axes = plt.subplots(n_configs, 1, figsize=(16, 4*n_configs))
        
        if n_configs == 1:
            axes = [axes]
        
        # Custom colormap for landuse (9 colors for categories 2, 4, 7, and buildings=8)
        # Index mapping: 0-1 unused, 2=low-density, 3 unused, 4=transition, 5-6 unused, 7=vegetation, 8=buildings, 9 unused
        cmap = mcolors.ListedColormap(['#CCCCCC', '#CCCCCC', '#90EE90', '#CCCCCC', '#FFD700', 
                                       '#CCCCCC', '#CCCCCC', '#006400', '#696969', '#CCCCCC'])
        bounds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)
        
        for idx, (config_id, result) in enumerate(results.items()):
            landuse = result['landuse']
            buildings = result['buildings']
            
            # Create composite view: landuse + buildings overlay
            display = landuse.astype(float)
            display[buildings > 0] = 8  # Buildings as separate category
            
            im = axes[idx].imshow(display, cmap=cmap, norm=norm, origin='lower', aspect='auto')
            axes[idx].set_title(f"{result['name']} - Landuse Configuration", fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('X (cells)')
            axes[idx].set_ylabel('Y (cells)')
            
            # Mark parcel boundaries
            geom = result['geom']
            parcel_left = geom['parcel_offset_x']
            parcel_right = parcel_left + int(51 / geom['dx'])
            axes[idx].axvline(parcel_left, color='red', linestyle='--', linewidth=2, alpha=0.7)
            axes[idx].axvline(parcel_right, color='red', linestyle='--', linewidth=2, alpha=0.7)
            
            # Custom legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#006400', label='Landuse 7 (vegetation)'),
                Patch(facecolor='#FFD700', label='Landuse 4 (transition)'),
                Patch(facecolor='#90EE90', label='Landuse 2 (low-density)'),
                Patch(facecolor='#696969', label='Buildings')
            ]
            axes[idx].legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'landuse_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("  ✓ landuse_comparison.png")
    
    def _plot_velocity_profiles(self, results: Dict[str, Dict], output_dir: Path):
        """Plot velocity cross-sections along flow direction."""
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Get centerline row
        first_result = list(results.values())[0]
        ny = first_result['geom']['ny']
        centerline_y = ny // 2
        
        for config_id, result in results.items():
            uq = result['outputs']['uq'] / 100.0
            vq = result['outputs']['vq'] / 100.0
            Ex = result['outputs']['Ex']
            
            u_2m = np.sqrt(uq**2 + vq**2)
            
            x_coords = np.arange(result['geom']['nx']) * result['geom']['dx']
            
            # Velocity profile
            axes[0].plot(x_coords, u_2m[centerline_y, :], label=result['name'], linewidth=2)
            
            # Cold air content profile
            axes[1].plot(x_coords, Ex[centerline_y, :], label=result['name'], linewidth=2)
        
        # Mark parcel boundaries
        parcel_left = first_result['geom']['parcel_offset_x'] * first_result['geom']['dx']
        parcel_right = parcel_left + 51
        
        for ax in axes:
            ax.axvline(parcel_left, color='gray', linestyle='--', linewidth=1.5, alpha=0.5, label='Parcel edges')
            ax.axvline(parcel_right, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('X position (m)')
        
        axes[0].set_ylabel('Wind speed at 2m (m/s)')
        axes[0].set_title('Velocity Profile Along Centerline', fontweight='bold')
        
        axes[1].set_ylabel('Cold air content (100 J/m²)')
        axes[1].set_title('Cold Air Content Along Centerline', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'velocity_profiles.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("  ✓ velocity_profiles.png")
    
    def _plot_terrain_profiles(self, results: Dict[str, Dict], output_dir: Path):
        """Plot terrain elevation profiles along flow direction."""
        fig, ax = plt.subplots(1, 1, figsize=(14, 6))
        
        # Get centerline row
        first_result = list(results.values())[0]
        ny = first_result['geom']['ny']
        centerline_y = ny // 2
        
        for config_id, result in results.items():
            terrain = result['terrain']
            buildings = result['buildings']
            
            x_coords = np.arange(result['geom']['nx']) * result['geom']['dx']
            
            # Terrain elevation profile
            ax.plot(x_coords, terrain[centerline_y, :], label=f"{result['name']} (terrain)", linewidth=2)
            
            # Terrain + buildings profile
            combined = terrain + buildings
            if buildings.max() > 0:
                ax.plot(x_coords, combined[centerline_y, :], label=f"{result['name']} (terrain+buildings)", 
                       linewidth=1.5, linestyle='--', alpha=0.7)
        
        # Mark parcel boundaries
        parcel_left = first_result['geom']['parcel_offset_x'] * first_result['geom']['dx']
        parcel_right = parcel_left + 51
        
        ax.axvline(parcel_left, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Parcel edges')
        ax.axvline(parcel_right, color='red', linestyle='--', linewidth=2, alpha=0.5)
        ax.legend(fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X position (m)')
        ax.set_ylabel('Elevation (m)')
        ax.set_title('Terrain Elevation Profiles Along Centerline', fontweight='bold', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'terrain_profiles.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("  ✓ terrain_profiles.png")
    
    def _plot_metrics_comparison(self, metrics: Dict[str, Dict], output_dir: Path):
        """Plot bar charts comparing metrics."""
        if not metrics:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        config_names = [m['name'] for m in metrics.values()]
        
        metric_specs = [
            ('mean_velocity_2m', 'Mean Velocity at 2m (m/s)'),
            ('max_velocity_2m', 'Max Velocity at 2m (m/s)'),
            ('mean_Ex', 'Mean Cold Air Content (100 J/m²)'),
            ('mean_Hx', 'Mean Cold Air Height (m)'),
            ('mean_flux', 'Mean Cold Air Flux (100 W/m²)'),
            ('total_flux', 'Total Cold Air Flux (100 W/m²)')
        ]
        
        for idx, (metric_key, title) in enumerate(metric_specs):
            values = [m[metric_key] for m in metrics.values()]
            
            bars = axes[idx].bar(range(len(values)), values, color='steelblue', alpha=0.7)
            axes[idx].set_xticks(range(len(values)))
            axes[idx].set_xticklabels([f"C{i+1}" for i in range(len(values))], rotation=0)
            axes[idx].set_ylabel(title.split('(')[0].strip())
            axes[idx].set_title(title, fontsize=10, fontweight='bold')
            axes[idx].grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar, val in zip(bars, values):
                height = bar.get_height()
                axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                             f'{val:.2f}', ha='center', va='bottom', fontsize=8)
        
        # Add legend
        legend_text = '\n'.join([f"C{i+1}: {name}" for i, name in enumerate(config_names)])
        fig.text(0.5, 0.02, legend_text, ha='center', fontsize=9, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout(rect=[0, 0.08, 1, 1])
        plt.savefig(output_dir / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("  ✓ metrics_comparison.png")
    
    def save_metrics_table(self, metrics: Dict[str, Dict]):
        """Save metrics as CSV table."""
        import csv
        
        output_file = self.base_output_dir / "metrics_summary.csv"
        
        if not metrics:
            logger.warning("No metrics to save")
            return
        
        # Get all metric keys (excluding 'name')
        metric_keys = [k for k in list(metrics.values())[0].keys() if k != 'name']
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow(['Configuration', 'Name'] + metric_keys)
            
            # Data rows
            for config_id, config_metrics in metrics.items():
                row = [config_id, config_metrics['name']]
                row.extend([config_metrics[k] for k in metric_keys])
                writer.writerow(row)
        
        logger.info(f"\n✓ Metrics saved to: {output_file}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Compare KLAM_21 configurations to analyze landuse boundary effects"
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/exp9_klam_boundary_analysis',
        help='Output directory for results'
    )
    parser.add_argument(
        '--parcel-size',
        type=int,
        default=51,
        help='Parcel size in meters (default: 51)'
    )
    parser.add_argument(
        '--xy-scale',
        type=float,
        default=3.0,
        help='Meters per grid cell (default: 3.0)'
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    logger.info(f"\n{'='*60}")
    logger.info("KLAM_21 Configuration Comparison")
    logger.info(f"{'='*60}")
    logger.info(f"Parcel size: {args.parcel_size}m")
    logger.info(f"Grid resolution: {args.xy_scale}m/cell")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"{'='*60}\n")
    
    # Initialize comparator
    comparator = ConfigurationComparator(output_dir)
    
    # Run all configurations
    logger.info("Running all configurations...")
    results = comparator.run_all_configurations(
        parcel_size_m=args.parcel_size,
        xy_scale=args.xy_scale
    )
    
    # Compute metrics
    logger.info("\nComputing metrics...")
    metrics = comparator.compute_metrics(results)
    
    # Save metrics table
    comparator.save_metrics_table(metrics)
    
    # Generate plots
    logger.info("\nGenerating comparison plots...")
    comparator.plot_comparisons(results, metrics)
    
    logger.info(f"\n{'='*60}")
    logger.info("Analysis complete!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"{'='*60}\n")


if __name__ == '__main__':
    main()

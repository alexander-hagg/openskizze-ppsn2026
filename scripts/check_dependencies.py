#!/usr/bin/env python3
# OpenSKIZZE
# Copyright (C) 2025 [Alexander Hagg]
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
Dependency Checker for OpenSKIZZE KLAM21 Optimization

This script checks whether all necessary packages are installed and reports
any missing dependencies to the user.

Usage:
    python check_dependencies.py
"""

import sys
import importlib
import subprocess
from typing import List, Tuple, Optional


# Define required packages with their import names and optional minimum versions
# Format: (import_name, package_display_name, min_version_or_None)
REQUIRED_PACKAGES = [
    # Core scientific computing
    ("numpy", "numpy", "1.20.0"),
    ("scipy", "scipy", "1.7.0"),
    ("pandas", "pandas", "1.3.0"),
    ("sklearn", "scikit-learn", "1.0.0"),
    
    # Deep learning / GP
    ("torch", "PyTorch", "2.0.0"),
    ("gpytorch", "GPyTorch", "1.10.0"),
    
    # Visualization
    ("matplotlib", "matplotlib", "3.5.0"),
    
    # Optimization / QD
    ("ribs", "pyribs", "0.6.0"),
    
    # Geospatial
    ("geopandas", "geopandas", None),
    ("rasterio", "rasterio", None),
    
    # Utilities
    ("yaml", "PyYAML", None),
    ("tqdm", "tqdm", None),
    ("numba", "numba", None),
    
    # Configuration
    ("psutil", "psutil", None),
    ("joblib", "joblib", None),
]

# Optional packages (nice to have but not required)
OPTIONAL_PACKAGES = [
    ("PIL", "Pillow", None),
    ("cv2", "opencv-python", None),
]


def get_version(module) -> Optional[str]:
    """Try to get the version of an imported module."""
    for attr in ["__version__", "VERSION", "version"]:
        if hasattr(module, attr):
            ver = getattr(module, attr)
            if callable(ver):
                ver = ver()
            return str(ver)
    return None


def parse_version(version_str: str) -> Tuple[int, ...]:
    """Parse a version string into a tuple of integers for comparison."""
    # Handle versions like "2.5.1+cu124" or "1.14.2.post1"
    clean_version = version_str.split("+")[0].split(".post")[0].split("a")[0].split("b")[0].split("rc")[0]
    parts = []
    for part in clean_version.split("."):
        try:
            parts.append(int(part))
        except ValueError:
            break
    return tuple(parts) if parts else (0,)


def check_version(current: str, minimum: str) -> bool:
    """Check if current version meets minimum requirement."""
    try:
        return parse_version(current) >= parse_version(minimum)
    except Exception:
        return True  # If we can't parse, assume it's fine


def check_package(import_name: str, display_name: str, min_version: Optional[str]) -> Tuple[bool, str, Optional[str]]:
    """
    Check if a package is installed and meets version requirements.
    
    Returns:
        (is_ok, status_message, installed_version)
    """
    try:
        module = importlib.import_module(import_name)
        version = get_version(module)
        
        if min_version and version:
            if check_version(version, min_version):
                return (True, f"✓ {display_name} {version} (>= {min_version})", version)
            else:
                return (False, f"✗ {display_name} {version} (requires >= {min_version})", version)
        elif version:
            return (True, f"✓ {display_name} {version}", version)
        else:
            return (True, f"✓ {display_name} (version unknown)", None)
            
    except ImportError as e:
        return (False, f"✗ {display_name} - NOT INSTALLED", None)
    except Exception as e:
        return (False, f"? {display_name} - Error: {e}", None)


def check_cuda() -> Tuple[bool, str]:
    """Check CUDA availability for PyTorch."""
    try:
        import torch
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda or "unknown"
            device_count = torch.cuda.device_count()
            device_names = [torch.cuda.get_device_name(i) for i in range(device_count)]
            return (True, f"✓ CUDA {cuda_version} available ({device_count} GPU(s): {', '.join(device_names)})")
        else:
            return (False, "⚠ CUDA not available (will use CPU)")
    except ImportError:
        return (False, "⚠ PyTorch not installed, cannot check CUDA")
    except Exception as e:
        return (False, f"? CUDA check failed: {e}")


def check_klam21() -> Tuple[bool, str]:
    """Check if KLAM_21 simulation files are present."""
    from pathlib import Path
    
    # The KLAM_21 executable is required
    klam_executable = Path("domain_description/klam_21_v2012")
    # The example directory is optional (just for reference/testing)
    klam_example = Path("domain_description/klam_21_v2012_example")
    
    if klam_executable.exists():
        if klam_example.exists():
            return (True, f"✓ KLAM_21 executable found (with example files)")
        else:
            return (True, f"✓ KLAM_21 executable found")
    else:
        return (False, f"✗ KLAM_21 executable not found: {klam_executable}")


def check_local_modules() -> List[Tuple[bool, str]]:
    """Check if local project modules can be imported."""
    results = []
    
    local_modules = [
        ("domain_description.evaluation", "Domain evaluation module"),
        ("domain_description.evaluation_klam", "KLAM evaluation module"),
        ("encodings.parametric.parametric", "Parametric encoding"),
        ("optimization.sail_optimizer", "SAIL optimizer"),
    ]
    
    for module_name, display_name in local_modules:
        try:
            importlib.import_module(module_name)
            results.append((True, f"✓ {display_name}"))
        except ImportError as e:
            results.append((False, f"✗ {display_name} - {e}"))
        except Exception as e:
            results.append((False, f"? {display_name} - Error: {e}"))
    
    return results


def main():
    print("=" * 70)
    print("OpenSKIZZE KLAM21 Optimization - Dependency Checker")
    print("=" * 70)
    print()
    
    # Track results
    all_ok = True
    required_missing = []
    optional_missing = []
    
    # Check required packages
    print("REQUIRED PACKAGES:")
    print("-" * 50)
    for import_name, display_name, min_version in REQUIRED_PACKAGES:
        ok, message, version = check_package(import_name, display_name, min_version)
        print(f"  {message}")
        if not ok:
            all_ok = False
            required_missing.append(display_name)
    print()
    
    # Check optional packages
    print("OPTIONAL PACKAGES:")
    print("-" * 50)
    for import_name, display_name, min_version in OPTIONAL_PACKAGES:
        ok, message, version = check_package(import_name, display_name, min_version)
        print(f"  {message}")
        if not ok:
            optional_missing.append(display_name)
    print()
    
    # Check CUDA
    print("GPU/CUDA:")
    print("-" * 50)
    cuda_ok, cuda_message = check_cuda()
    print(f"  {cuda_message}")
    print()
    
    # Check KLAM_21
    print("KLAM_21 SIMULATION:")
    print("-" * 50)
    klam_ok, klam_message = check_klam21()
    print(f"  {klam_message}")
    if not klam_ok:
        all_ok = False
    print()
    
    # Check local modules
    print("LOCAL PROJECT MODULES:")
    print("-" * 50)
    for ok, message in check_local_modules():
        print(f"  {message}")
        if not ok:
            all_ok = False
    print()
    
    # Summary
    print("=" * 70)
    print("SUMMARY:")
    print("=" * 70)
    
    if all_ok and not required_missing:
        print("✓ All required dependencies are installed!")
        if optional_missing:
            print(f"  (Optional missing: {', '.join(optional_missing)})")
        print()
        print("You can run the experiments with:")
        print("  python experiments/test_experiment_pipeline.py --skip-klam  # Fast test")
        print("  python experiments/test_experiment_pipeline.py              # Full test")
        return 0
    else:
        print("✗ Some dependencies are missing or have issues!")
        print()
        
        if required_missing:
            print("Missing required packages:")
            for pkg in required_missing:
                print(f"  - {pkg}")
            print()
            print("Install missing packages with:")
            print("  pip install " + " ".join(required_missing))
            print()
            print("Or if using conda:")
            print("  conda install " + " ".join(required_missing))
        
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())

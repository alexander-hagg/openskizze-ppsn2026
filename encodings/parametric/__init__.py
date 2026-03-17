# OpenSKIZZE
# Copyright (C) 2025 [Alexander Hagg]
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This file makes 'parametric' a sub-package of 'encodings'.

# NumbaFastEncoding is the optimized implementation (16× faster)
# Exported as both ParametricEncoding (for backwards compatibility) and FastEncoding (new name)
from .fast_encoding import NumbaFastEncoding, compute_features_batch_numba, numba_calculate_features

# Alias for backwards compatibility - all existing code using ParametricEncoding will now use the fast version
ParametricEncoding = NumbaFastEncoding
FastEncoding = NumbaFastEncoding

# Legacy import (deprecated, use NumbaFastEncoding or ParametricEncoding instead)
# from .parametric import ParametricEncoding as LegacyParametricEncoding

__all__ = [
    "ParametricEncoding",       # Backwards compatible alias (now uses NumbaFastEncoding)
    "FastEncoding",             # Explicit fast encoding name
    "NumbaFastEncoding",        # Original name for clarity
    "compute_features_batch_numba",
    "numba_calculate_features",
]
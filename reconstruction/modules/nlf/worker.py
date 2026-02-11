import os
import multiprocessing
import sys
import inspect

# Force spawn start method for multiprocessing to avoid CUDA hangs in Celery workers
try:
    if multiprocessing.get_start_method(allow_none=True) != 'spawn':
        multiprocessing.set_start_method('spawn', force=True)
        sys.stderr.write("Successfully set multiprocessing start method to 'spawn'\n")
        sys.stderr.flush()
except RuntimeError:
    sys.stderr.write(f"Could not set multiprocessing start method to 'spawn', already set to {multiprocessing.get_start_method()}\n")
    sys.stderr.flush()

if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec

import numpy as np
# Patch numpy for chumpy compatibility
if not hasattr(np, 'bool'): np.bool = bool
if not hasattr(np, 'int'): np.int = int
if not hasattr(np, 'float'): np.float = float
if not hasattr(np, 'complex'): np.complex = complex
if not hasattr(np, 'object'): np.object = object
if not hasattr(np, 'unicode'): np.unicode = str
if not hasattr(np, 'str'): np.str = str

from modules.nlf.tasks import celery_app



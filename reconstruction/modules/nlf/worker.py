import inspect
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

if __name__ == "__main__":
    celery_app.start()



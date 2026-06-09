from .. import constraints as _constraints
from mt.base import logger

for _x, _y in _constraints.__dict__.items():
    if _x.startswith("_"):
        continue
    globals()[_x] = _y
__doc__ = _constraints.__doc__

try:
    from .center_around import *
except Exception as e:
    logger.warn(f"Unable to import mt.tkeras_src.constraints_src.center_around: {e}")


__api__ = [
    "CenterAround",
]

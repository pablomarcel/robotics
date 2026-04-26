# Lightweight decorator for call tracing (optional).
from functools import wraps
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

def track(label: str):
    def deco(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            logging.getLogger(__name__).debug("→ %s", label)
            return fn(*args, **kwargs)
        return wrapper
    return deco

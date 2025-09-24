# time/tools/trace_uml.py
def track(_src: str, _dst: str):
    """No-op decorator placeholder for runtime class-diagram tracing."""
    def deco(fn):
        def inner(*a, **k): return fn(*a, **k)
        return inner
    return deco

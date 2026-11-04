"""
RetainNet Dependency Availability Checker.
Provides clean flags for optional packages without injecting mock modules.
"""
import importlib.util

def check_module(name: str) -> bool:
    try:
        spec = importlib.util.find_spec(name)
        return spec is not None
    except Exception:
        return False

HAS_TORCH = check_module("torch")
HAS_SKLEARN = check_module("sklearn")
HAS_LIFELINES = check_module("lifelines")
HAS_SKSURV = check_module("sksurv")
HAS_SCIPY = check_module("scipy")

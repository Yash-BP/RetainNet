# --- RetainNet tests/conftest.py ---
import sys, os
_this_dir = os.path.dirname(os.path.abspath(__file__))
_root_dir = os.path.abspath(os.path.join(_this_dir, ".."))
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)

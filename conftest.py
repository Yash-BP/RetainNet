# --- RetainNet sys.path bootstrap (strictly binds to current project root) ---
import os
import sys

_root_dir = os.path.dirname(os.path.abspath(__file__))
if _root_dir in sys.path:
    sys.path.remove(_root_dir)
sys.path.insert(0, _root_dir)
# -----------------------------------------------------------------------------

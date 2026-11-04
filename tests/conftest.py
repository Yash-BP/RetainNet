# --- RetainNet sys.path bootstrap (strictly binds to parent project root) ---
import os
import sys

_this_dir = os.path.dirname(os.path.abspath(__file__))
_root_dir = os.path.abspath(os.path.join(_this_dir, ".."))
if _root_dir in sys.path:
    sys.path.remove(_root_dir)
sys.path.insert(0, _root_dir)

# Prevent Windows loky subprocess physical core detection warning
if "LOKY_MAX_CPU_COUNT" not in os.environ:
    os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 4)
# ----------------------------------------------------------------------------

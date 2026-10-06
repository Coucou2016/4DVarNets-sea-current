import os

# Prevent broken optional pytest plugins (zarr/numcodecs) on some Windows conda envs
os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")

# polarix_sh/__init__.py

# Export the environment and task classes. Alias the env class to the
# historical name `PolarixRedvsBlue` for backward compatibility.
from .env import PolarixRedvsBlueEnv as PolarixRedvsBlue

# Importing `polarix.task` pulls an external dependency; avoid importing
# `RedvsBluePolarixTask` at package-import time so tests that import
# `polarix_sh` submodules don't require `polarix` to be installed.
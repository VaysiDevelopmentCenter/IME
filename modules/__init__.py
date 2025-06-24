# This file makes the 'modules' directory a Python package.

# To avoid circular import issues when modules are complexly interrelated,
# keep this __init__.py minimal. Users should import directly from the submodules,
# e.g., from modules.engine import SmartMutationEngine
# or from modules.graph_schema import NetworkGraph.

# We can expose very fundamental, stable classes if desired, but with caution.
# For now, let's only expose MutableEntity as an example, if even that.
# from .engine import MutableEntity

# By keeping this minimal, we let each module handle its own dependencies
# without __init__.py forcing a specific load order that might conflict
# when a script within the package is run as __main__.
pass

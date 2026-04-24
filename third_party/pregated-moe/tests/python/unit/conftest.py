"""Unit-test conftest: stub optional/GPU dependencies before collection.

Prevents ImportError when pregated_moe submodules are imported in
environments where the optional 'nvtx' profiling library or compiled
CUDA extensions (_store, _engine) are not installed.
"""

import sys
from unittest.mock import MagicMock


def _stub_if_missing(name: str) -> None:
    if name in sys.modules:
        return
    try:
        __import__(name)
    except ImportError:
        sys.modules[name] = MagicMock()


# nvtx: optional NVIDIA profiling library used in pregated_moe.models.*
_stub_if_missing("nvtx")

# Compiled CUDA extensions: may be absent in CPU-only test environments
_stub_if_missing("pregated_moe._store")
_stub_if_missing("pregated_moe._engine")

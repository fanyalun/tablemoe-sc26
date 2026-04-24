__version__ = "0.0.1"
__all__ = ["MoE", "OffloadEngine", "__version__"]


def __getattr__(name):
    if name == "MoE":
        from moe_infinity.entrypoints import MoE

        return MoE
    if name == "OffloadEngine":
        from moe_infinity.runtime import OffloadEngine

        return OffloadEngine
    raise AttributeError(f"module 'moe_infinity' has no attribute {name!r}")

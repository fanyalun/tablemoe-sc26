__version__ = "0.0.1"
__all__ = ["MoE", "OffloadEngine", "__version__"]


def __getattr__(name):
    if name == "MoE":
        from pregated_moe.entrypoints import MoE

        return MoE
    if name == "OffloadEngine":
        from pregated_moe.runtime import OffloadEngine

        return OffloadEngine
    raise AttributeError(f"module 'pregated_moe' has no attribute {name!r}")

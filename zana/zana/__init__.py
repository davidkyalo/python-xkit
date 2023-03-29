import importlib.metadata

__version__ = tuple(
    int(v) if v.isnumeric() else v
    for v in importlib.metadata.version("zana").partition("-")[0].split(".")
)
__version__ = __version__ + (0, 0, 0)[len(__version__) :]


__all__ = []

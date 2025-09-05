from typing import TYPE_CHECKING, Any
import importlib

if TYPE_CHECKING:
    from .adapter.api import AdapterAPI
    from .utilities.logger import logger


__lazy_imports__ = {
    "AdapterAPI": (".adapter", "AdapterAPI"),
    "logger": (".utilities.logger", "logger"),
    # "QuantizationAPI": (".quantization", "QuantizationAPI"),
}


def __getattr__(name: str) -> Any:
    if name in __lazy_imports__:
        module_name, attr = __lazy_imports__[name]
        return getattr(importlib.import_module(module_name, package=__name__), attr)
    raise AttributeError(f"module {__name__} has no attribute {name}")


__all__ = list(__lazy_imports__.keys())

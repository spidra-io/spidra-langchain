"""LangChain integration for Spidra."""

from langchain_spidra.document_loaders import (
    SpidraLoader,
    SpidraMapLoader,
    SpidraParseLoader,
    SpidraSearchLoader,
)
from langchain_spidra.retrievers import SpidraRetriever
from langchain_spidra.tools import (
    SpidraBatchScrape,
    SpidraBulkScrape,
    SpidraCrawl,
    SpidraMap,
    SpidraMonitor,
    SpidraParse,
    SpidraRunPreset,
    SpidraRunTemplate,
    SpidraScrape,
    SpidraSearch,
)

try:
    import importlib.metadata as _metadata

    try:
        __version__ = _metadata.version(__package__ or "langchain-spidra")
    except _metadata.PackageNotFoundError:
        __version__ = ""
    finally:
        del _metadata
except Exception:
    __version__ = ""


__all__ = [
    "SpidraBatchScrape",
    "SpidraBulkScrape",
    "SpidraCrawl",
    "SpidraLoader",
    "SpidraMap",
    "SpidraMapLoader",
    "SpidraMonitor",
    "SpidraParse",
    "SpidraParseLoader",
    "SpidraRetriever",
    "SpidraRunPreset",
    "SpidraRunTemplate",
    "SpidraScrape",
    "SpidraSearch",
    "SpidraSearchLoader",
    "__version__",
]

"""LangChain integration for Spidra."""

from importlib import metadata

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
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

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

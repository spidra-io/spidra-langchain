"""Unit tests for imports in ``langchain_spidra``."""

from langchain_spidra import __all__

EXPECTED_ALL = [
    "SpidraBatchScrape",
    "SpidraBulkScrape",
    "SpidraCrawl",
    "SpidraLoader",
    "SpidraMap",
    "SpidraMapLoader",
    "SpidraParse",
    "SpidraParseLoader",
    "SpidraRetriever",
    "SpidraScrape",
    "SpidraSearch",
    "SpidraSearchLoader",
    "__version__",
]


def test_all_imports() -> None:
    """Test that all expected names are exported in ``__all__``."""
    assert sorted(EXPECTED_ALL) == sorted(__all__)


def test_loader_imports() -> None:
    """Loader classes are importable from the top-level package."""
    from langchain_spidra import (  # noqa: F401
        SpidraLoader,
        SpidraMapLoader,
        SpidraParseLoader,
        SpidraSearchLoader,
    )

    assert SpidraLoader is not None
    assert SpidraSearchLoader is not None
    assert SpidraParseLoader is not None
    assert SpidraMapLoader is not None


def test_tools_importable() -> None:
    """All tool classes are importable from the top-level package."""
    from langchain_spidra import (  # noqa: F401
        SpidraBatchScrape,
        SpidraBulkScrape,
        SpidraCrawl,
        SpidraMap,
        SpidraParse,
        SpidraScrape,
        SpidraSearch,
    )

    assert SpidraScrape is not None
    assert SpidraCrawl is not None
    assert SpidraBatchScrape is not None
    assert SpidraSearch is not None
    assert SpidraMap is not None
    assert SpidraParse is not None
    assert SpidraBulkScrape is not None


def test_retriever_importable() -> None:
    """SpidraRetriever is importable from top-level package."""
    from langchain_spidra import SpidraRetriever  # noqa: F401

    assert SpidraRetriever is not None

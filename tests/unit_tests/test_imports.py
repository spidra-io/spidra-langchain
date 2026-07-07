"""Unit tests for imports in ``langchain_spidra``."""

from langchain_spidra import __all__

EXPECTED_ALL = [
    "SpidraBatchScrape",
    "SpidraCrawl",
    "SpidraScrape",
    "SpidraLoader",
    "__version__",
]


def test_all_imports() -> None:
    """Test that all expected names are exported in ``__all__``."""
    assert sorted(EXPECTED_ALL) == sorted(__all__)


def test_loader_import() -> None:
    """SpidraLoader is importable from the top-level package."""
    from langchain_spidra import SpidraLoader  # noqa: F401


def test_tools_importable() -> None:
    """All tool classes are importable from the top-level package."""
    from langchain_spidra import SpidraBatchScrape, SpidraCrawl, SpidraScrape  # noqa: F401

    assert SpidraScrape is not None
    assert SpidraCrawl is not None
    assert SpidraBatchScrape is not None

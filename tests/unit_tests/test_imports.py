from langchain_spidra import SpidraLoader
from langchain_spidra.document_loaders import SpidraLoader as _SpidraLoaderDirect


def test_import_spidra_loader() -> None:
    assert SpidraLoader is _SpidraLoaderDirect


def test_all_public_exports() -> None:
    import langchain_spidra

    assert hasattr(langchain_spidra, "SpidraLoader")

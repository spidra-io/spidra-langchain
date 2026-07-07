"""Standard LangChain unit tests for the Spidra tools.

These are the ``langchain-tests`` ``ToolsUnitTests`` that LangChain uses to bless
an integration package: init, init-from-env, name, input schema, and that the
example invoke params validate against the declared schema.
"""

from typing import Any

from langchain_tests.unit_tests import ToolsUnitTests

from langchain_spidra import SpidraBatchScrape, SpidraCrawl, SpidraScrape

_ENV = {"SPIDRA_API_KEY": "spd-test"}
_EXPECTED = {"api_key": "spd-test"}
_PARAMS = {"api_key": "spd-test"}


class TestSpidraScrapeUnit(ToolsUnitTests):
    """Standard unit tests for ``SpidraScrape``."""

    @property
    def tool_constructor(self) -> type[SpidraScrape]:
        """The tool class under test."""
        return SpidraScrape

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        """Params passed to the constructor."""
        return _PARAMS

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        """Example args matching the tool's input schema."""
        return {"url": "https://example.com"}

    @property
    def init_from_env_params(self) -> tuple[dict, dict, dict]:
        """Env vars, init args, and expected attrs for env init."""
        return _ENV, {}, _EXPECTED


class TestSpidraCrawlUnit(ToolsUnitTests):
    """Standard unit tests for ``SpidraCrawl``."""

    @property
    def tool_constructor(self) -> type[SpidraCrawl]:
        """The tool class under test."""
        return SpidraCrawl

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        """Params passed to the constructor."""
        return _PARAMS

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        """Example args matching the tool's input schema."""
        return {"url": "https://example.com"}

    @property
    def init_from_env_params(self) -> tuple[dict, dict, dict]:
        """Env vars, init args, and expected attrs for env init."""
        return _ENV, {}, _EXPECTED


class TestSpidraBatchScrapeUnit(ToolsUnitTests):
    """Standard unit tests for ``SpidraBatchScrape``."""

    @property
    def tool_constructor(self) -> type[SpidraBatchScrape]:
        """The tool class under test."""
        return SpidraBatchScrape

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        """Params passed to the constructor."""
        return _PARAMS

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        """Example args matching the tool's input schema."""
        return {"urls": ["https://example.com"]}

    @property
    def init_from_env_params(self) -> tuple[dict, dict, dict]:
        """Env vars, init args, and expected attrs for env init."""
        return _ENV, {}, _EXPECTED

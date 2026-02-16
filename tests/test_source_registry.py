"""Tests for core.sources.base and core.sources.registry.

Covers SourceAdapter ABC behavior, SourceRegistry registration,
lookup, listing, and edge cases.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from core.models import Paper
from core.sources.base import SourceAdapter
from core.sources.registry import SourceRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class DummyAdapter(SourceAdapter):
    """Minimal concrete adapter for testing."""

    @property
    def source_type(self) -> str:
        return "dummy"

    def collect(
        self,
        agent1_output: dict[str, Any],
        categories: list[str],
        window_start: datetime,
        window_end: datetime,
        config: dict[str, Any],
    ) -> list[Paper]:
        return []


class AnotherAdapter(SourceAdapter):
    """Second concrete adapter for multi-registration tests."""

    @property
    def source_type(self) -> str:
        return "another"

    def collect(
        self,
        agent1_output: dict[str, Any],
        categories: list[str],
        window_start: datetime,
        window_end: datetime,
        config: dict[str, Any],
    ) -> list[Paper]:
        return []


class ReturningAdapter(SourceAdapter):
    """Adapter that returns a fixed list of Papers."""

    @property
    def source_type(self) -> str:
        return "returning"

    def collect(
        self,
        agent1_output: dict[str, Any],
        categories: list[str],
        window_start: datetime,
        window_end: datetime,
        config: dict[str, Any],
    ) -> list[Paper]:
        return [
            Paper(
                source="returning",
                native_id="001",
                paper_key="returning:001",
                url="https://example.com/001",
                title="Test Paper",
                abstract="Abstract text.",
                authors=["Author A"],
                categories=["cs.AI"],
                published_at_utc=datetime(2025, 1, 1, tzinfo=timezone.utc),
            ),
        ]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_registry():
    """Ensure a clean registry for every test."""
    SourceRegistry.clear()
    yield
    SourceRegistry.clear()


# ---------------------------------------------------------------------------
# Tests: SourceAdapter ABC
# ---------------------------------------------------------------------------


class TestSourceAdapterABC:
    """Tests for the SourceAdapter abstract base class."""

    def test_cannot_instantiate_directly(self):
        """SourceAdapter cannot be instantiated because it is abstract."""
        with pytest.raises(TypeError):
            SourceAdapter()  # type: ignore[abstract]

    def test_collect_is_abstract(self):
        """A subclass missing collect() cannot be instantiated."""

        class MissingCollect(SourceAdapter):
            @property
            def source_type(self) -> str:
                return "missing_collect"

        with pytest.raises(TypeError):
            MissingCollect()  # type: ignore[abstract]

    def test_source_type_is_abstract(self):
        """A subclass missing source_type cannot be instantiated."""

        class MissingSourceType(SourceAdapter):
            def collect(
                self,
                agent1_output: dict[str, Any],
                categories: list[str],
                window_start: datetime,
                window_end: datetime,
                config: dict[str, Any],
            ) -> list[Paper]:
                return []

        with pytest.raises(TypeError):
            MissingSourceType()  # type: ignore[abstract]

    def test_concrete_subclass_instantiates(self):
        """A fully concrete subclass can be instantiated normally."""
        adapter = DummyAdapter()
        assert adapter.source_type == "dummy"


# ---------------------------------------------------------------------------
# Tests: SourceRegistry
# ---------------------------------------------------------------------------


class TestSourceRegistry:
    """Tests for the SourceRegistry class."""

    def test_register_via_classmethod(self):
        """Register an adapter using register() classmethod."""
        SourceRegistry.register(DummyAdapter)
        assert "dummy" in SourceRegistry.list_types()

    def test_register_as_decorator(self):
        """Register an adapter using register() as a decorator."""

        @SourceRegistry.register
        class DecoratedAdapter(SourceAdapter):
            @property
            def source_type(self) -> str:
                return "decorated"

            def collect(
                self,
                agent1_output: dict[str, Any],
                categories: list[str],
                window_start: datetime,
                window_end: datetime,
                config: dict[str, Any],
            ) -> list[Paper]:
                return []

        assert "decorated" in SourceRegistry.list_types()
        assert SourceRegistry.get("decorated") is DecoratedAdapter

    def test_get_registered_adapter(self):
        """get() returns the correct adapter class."""
        SourceRegistry.register(DummyAdapter)
        assert SourceRegistry.get("dummy") is DummyAdapter

    def test_get_unregistered_raises_key_error(self):
        """get() raises KeyError for an unregistered source type."""
        with pytest.raises(KeyError, match="nonexistent"):
            SourceRegistry.get("nonexistent")

    def test_list_types_returns_all(self):
        """list_types() returns all registered type strings."""
        SourceRegistry.register(DummyAdapter)
        SourceRegistry.register(AnotherAdapter)
        types = SourceRegistry.list_types()
        assert set(types) == {"dummy", "another"}

    def test_list_types_empty_after_clear(self):
        """list_types() returns empty list after clear()."""
        SourceRegistry.register(DummyAdapter)
        SourceRegistry.clear()
        assert SourceRegistry.list_types() == []

    def test_clear_removes_all_registrations(self):
        """clear() removes all previously registered adapters."""
        SourceRegistry.register(DummyAdapter)
        SourceRegistry.register(AnotherAdapter)
        SourceRegistry.clear()
        assert SourceRegistry.list_types() == []
        with pytest.raises(KeyError):
            SourceRegistry.get("dummy")

    def test_register_duplicate_overwrites(self):
        """Registering the same source_type twice overwrites silently."""
        SourceRegistry.register(DummyAdapter)

        class DummyAdapterV2(SourceAdapter):
            @property
            def source_type(self) -> str:
                return "dummy"

            def collect(
                self,
                agent1_output: dict[str, Any],
                categories: list[str],
                window_start: datetime,
                window_end: datetime,
                config: dict[str, Any],
            ) -> list[Paper]:
                return []

        SourceRegistry.register(DummyAdapterV2)
        assert SourceRegistry.get("dummy") is DummyAdapterV2

    def test_multiple_adapters_registered(self):
        """Multiple distinct adapters can coexist in the registry."""
        SourceRegistry.register(DummyAdapter)
        SourceRegistry.register(AnotherAdapter)
        assert SourceRegistry.get("dummy") is DummyAdapter
        assert SourceRegistry.get("another") is AnotherAdapter

    def test_registered_adapter_can_be_instantiated(self):
        """An adapter retrieved from the registry can be instantiated."""
        SourceRegistry.register(DummyAdapter)
        adapter_cls = SourceRegistry.get("dummy")
        instance = adapter_cls()
        assert instance.source_type == "dummy"

    def test_collect_returns_list_of_papers(self):
        """A mock adapter's collect() returns a list of Paper objects."""
        SourceRegistry.register(ReturningAdapter)
        adapter_cls = SourceRegistry.get("returning")
        instance = adapter_cls()

        now = datetime.now(tz=timezone.utc)
        papers = instance.collect(
            agent1_output={
                "concepts": ["deep learning"],
                "cross_domain_keywords": [],
                "exclude_keywords": [],
            },
            categories=["cs.AI"],
            window_start=now,
            window_end=now,
            config={"max_results": 10},
        )

        assert isinstance(papers, list)
        assert len(papers) == 1
        assert isinstance(papers[0], Paper)
        assert papers[0].paper_key == "returning:001"

    def test_register_returns_adapter_class(self):
        """register() returns the adapter class for decorator chaining."""
        result = SourceRegistry.register(DummyAdapter)
        assert result is DummyAdapter

    def test_clear_is_idempotent(self):
        """Calling clear() on an already-empty registry does not error."""
        SourceRegistry.clear()
        SourceRegistry.clear()
        assert SourceRegistry.list_types() == []

    def test_list_types_returns_list_type(self):
        """list_types() returns a list (not dict_keys or other view)."""
        SourceRegistry.register(DummyAdapter)
        result = SourceRegistry.list_types()
        assert type(result) is list

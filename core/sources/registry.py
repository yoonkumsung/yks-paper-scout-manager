"""Registry for source adapters.

Maps source type strings to adapter classes.  Used by the pipeline to look
up the correct adapter for each configured source.
"""

from __future__ import annotations

from core.sources.base import SourceAdapter


class SourceRegistry:
    """Registry for source adapters.

    Maps source type strings to adapter classes.
    Used by the pipeline to look up the correct adapter for each
    configured source.
    """

    _adapters: dict[str, type[SourceAdapter]] = {}

    @classmethod
    def register(cls, adapter_class: type[SourceAdapter]) -> type[SourceAdapter]:
        """Register an adapter class.  Can be used as a decorator.

        Usage::

            @SourceRegistry.register
            class ArxivAdapter(SourceAdapter):
                ...

        Args:
            adapter_class: A concrete SourceAdapter subclass.

        Returns:
            The adapter class (unchanged), enabling decorator usage.
        """
        # Instantiate temporarily to read source_type property.
        # We avoid this by using the class-level attribute approach:
        # instead, we create a temporary instance or read from the class.
        # Since source_type is a property on instances, we need an instance.
        # However, we can also just register and resolve at get() time.
        # For simplicity, we instantiate to read the property.
        instance = adapter_class.__new__(adapter_class)
        source_key = instance.source_type
        cls._adapters[source_key] = adapter_class
        return adapter_class

    @classmethod
    def get(cls, source_type: str) -> type[SourceAdapter]:
        """Get adapter class by source type.

        Args:
            source_type: The source identifier (e.g., ``'arxiv'``).

        Returns:
            The registered adapter class.

        Raises:
            KeyError: If *source_type* is not registered.
        """
        if source_type not in cls._adapters:
            raise KeyError(
                f"No adapter registered for source type: {source_type!r}"
            )
        return cls._adapters[source_type]

    @classmethod
    def list_types(cls) -> list[str]:
        """List all registered source types."""
        return list(cls._adapters.keys())

    @classmethod
    def clear(cls) -> None:
        """Clear all registrations.  Useful for testing."""
        cls._adapters = {}

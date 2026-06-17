"""Core observation system for Rolland - replaces traitlets observe functionality.

This module provides:
- @observe: Decorator to mark methods that observe attribute changes
- @observable: Class decorator to enable observation on any class
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np

if TYPE_CHECKING:
    from typing import TypedDict

    class ChangeDict(TypedDict, total=False):
        """Dictionary describing an attribute change event."""

        name: str
        old: Any
        new: Any
        owner: Any
        type: str

T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])


# =============================================================================
# DECORATORS
# =============================================================================

def observe(*names: str) -> Callable[[F], F]:
    """Mark methods that observe specific attributes for changes.

    Parameters
    ----------
    *names : str
        Names of attributes to observe.

    Examples
    --------
    >>> class MyClass:
    ...     @observe('value')
    ...     def on_value_change(self, change):
    ...         print(f"{change['name']} changed from {change['old']} to {change['new']}")
    """
    def decorator(func: F) -> F:
        func.__observes__ = names  # type: ignore[attr-defined]
        return func
    return decorator


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _values_differ(old: Any, new: Any) -> bool:
    """Safely compare values, including numpy arrays.

    Parameters
    ----------
    old : Any
        Previous value.
    new : Any
        New value.

    Returns
    -------
    bool
        True if values differ, False otherwise.
    """
    try:
        result = old != new
        # Handle numpy arrays and other iterables
        if hasattr(result, '__iter__') and hasattr(result, '__len__'):
            return bool(np.any(result))
        return bool(result)
    except (TypeError, ValueError, AttributeError):
        # Fallback for uncomparable types
        return True


def _register_observers(self: Any) -> None:
    """Register all observer methods for an instance.

    Scans all methods of the instance for the `__observes__` attribute
    and registers them in the `_observers` registry.
    """
    self._observers = {}  # type: ignore # type: dict[str, list[Callable[[dict[str, Any]], None]]]
    for method_name in dir(self):
        try:
            method = getattr(self, method_name)
        except AttributeError:
            continue
        if callable(method) and hasattr(method, '__observes__'):
            for attr_name in method.__observes__:  # type: ignore[attr-defined]
                self._observers.setdefault(attr_name, []).append(method)


def _setattr(self: Any, name: str, value: Any) -> None:
    """Trigger observers on attribute changes via custom __setattr__.

    Parameters
    ----------
    self : Any
        The instance.
    name : str
        Attribute name.
    value : Any
        New value.
    """
    # Skip observer setup for internal attributes
    if name.startswith('_') and not name.startswith('__'):
        object.__setattr__(self, name, value)
        return

    # Initialize _observers if not present
    if not hasattr(self, '_observers'):
        object.__setattr__(self, name, value)
        return

    old_value = getattr(self, name, None)
    object.__setattr__(self, name, value)

    # Trigger observers if value changed
    if name in self._observers and _values_differ(old_value, value):
        change: dict[str, Any] = {
            'name': name,
            'old': old_value,
            'new': value,
            'owner': self,
            'type': 'change',
        }
        for callback in self._observers[name]:
            callback(change)


# =============================================================================
# CLASS DECORATOR
# =============================================================================

def observable(cls: type[T]) -> type[T]:  # noqa: UP047
    """Class decorator that enables observation on any class without inheritance.

    Adds to the class:
    - __init__ wrapper for observer registration
    - __setattr__ for change detection
    - observe() and unobserve() methods for manual observer management

    Parameters
    ----------
    cls : Type[T]
        The class to modify.

    Returns
    -------
    Type[T]
        The modified class.

    Examples
    --------
    >>> @observable
    ... class MyTrack:
    ...     def __init__(self):
    ...         self.value = 0
    ...
    ...     @observe('value')
    ...     def on_value_change(self, change):
    ...         print(f"Value changed: {change['new']}")
    """
    # Save original __init__ if it exists
    original_init = cls.__init__

    @functools.wraps(original_init) # dunder functions are not wrapped by default, need to preserve metadata manually
    def __init__(self: Any, *args: Any, **kwargs: Any) -> None:  # noqa: N807
        """Initialize instance and register observers."""
        # Call original __init__
        original_init(self, *args, **kwargs)
        # Register observers after initialization
        _register_observers(self)

    # Override class methods
    cls.__init__ = __init__  # type: ignore[method-assign]
    cls.__setattr__ = _setattr  # type: ignore[method-assign]

    # Add helper methods for manual observer management
    def observe_handler(
        self: Any,
        handler: Callable[[dict[str, Any]], None],
        names: list[str],
    ) -> None:
        """Manually register an observer for specific attributes."""
        # Ensure _observers exists
        if not hasattr(self, '_observers'):
            _register_observers(self)
        for n in names:
            self._observers.setdefault(n, []).append(handler)

    def unobserve_handler(
        self: Any,
        handler: Callable[[dict[str, Any]], None],
        names: list[str],
    ) -> None:
        """Manually unregister an observer for specific attributes."""
        if hasattr(self, '_observers'):
            for n in names:
                if n in self._observers:
                    self._observers[n] = [
                        cb for cb in self._observers[n] if cb != handler
                    ]

    cls.observe = observe_handler  # type: ignore[attr-defined]
    cls.unobserve = unobserve_handler  # type: ignore[attr-defined]

    return cls

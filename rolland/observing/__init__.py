"""Observation system for Rolland.

This package provides reactive programming capabilities to replace traitlets.
It enables classes to observe and react to attribute changes.

Main Components:
- observe: Method decorator to mark observer methods
- observable: Class decorator to enable observation on any class
- _values_differ: Helper for safe value comparison (including numpy arrays)
"""

from .core import (
    _register_observers,
    _setattr,
    _values_differ,
    observable,
    observe,
)

__all__ = [
    'observe',
    'observable',
    '_values_differ',
    '_register_observers',
    '_setattr',
]

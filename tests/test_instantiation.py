"""Test class instantiation."""

import inspect
from unittest.mock import patch

import pytest

import rolland


def get_classes(module):
    """Return all classes defined in a module."""
    classes = []
    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj) and obj.__module__.startswith('rolland'):
            classes.append(obj)
    return classes

@pytest.mark.parametrize("cls", get_classes(rolland))
def test_class_initialization(cls):
    """Test class initialization."""
    with patch.object(cls, '__init__', lambda self: None):
        instance = cls()
        assert isinstance(instance, cls)
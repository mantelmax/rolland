"""Tests for the custom observation system.

This module tests the observe decorator and observable class decorator
that replace traitlets functionality.
"""

import numpy as np

from rolland.observing import observable, observe
from rolland.observing.core import _values_differ


class TestObserveDecorator:
    """Tests for the @observe decorator."""

    def test_observe_marks_method(self):
        """Test that @observe decorator adds __observes__ attribute."""

        @observable
        class TestClass:
            @observe('value')
            def handler(self, change):
                pass

        obj = TestClass()
        assert hasattr(obj.handler, '__observes__')
        assert obj.handler.__observes__ == ('value',)

    def test_observe_multiple_attributes(self):
        """Test that @observe can mark multiple attributes."""

        @observable
        class TestClass:
            @observe('x', 'y', 'z')
            def handler(self, change):
                pass

        obj = TestClass()
        assert obj.handler.__observes__ == ('x', 'y', 'z')


class TestObservableDecorator:
    """Tests for the @observable class decorator."""

    def test_observable_triggers_on_change(self):
        """Test that changing an observed attribute triggers the callback."""

        @observable
        class TestClass:
            def __init__(self):
                self.value = 0
                self.last_change = None

            @observe('value')
            def on_change(self, change):
                self.last_change = change

        obj = TestClass()
        obj.value = 10

        assert obj.last_change is not None
        assert obj.last_change['name'] == 'value'
        assert obj.last_change['old'] == 0
        assert obj.last_change['new'] == 10
        assert obj.last_change['type'] == 'change'

    def test_observable_multiple_observers(self):
        """Test that multiple @observe methods are triggered."""

        @observable
        class TestClass:
            def __init__(self):
                self.x = 0
                self.y = 0
                self.changes_x = []
                self.changes_y = []

            @observe('x')
            def on_x_change(self, change):
                self.changes_x.append(change)

            @observe('y')
            def on_y_change(self, change):
                self.changes_y.append(change)

        obj = TestClass()
        obj.x = 1
        obj.y = 2

        assert len(obj.changes_x) == 1
        assert len(obj.changes_y) == 1
        assert obj.changes_x[0]['name'] == 'x'
        assert obj.changes_y[0]['name'] == 'y'

    def test_observable_multiple_attributes_single_handler(self):
        """Test that a single handler can observe multiple attributes."""

        @observable
        class TestClass:
            def __init__(self):
                self.x = 0
                self.y = 0
                self.changes = []

            @observe('x', 'y')
            def on_change(self, change):
                self.changes.append(change)

        obj = TestClass()
        obj.x = 1
        obj.y = 2

        assert len(obj.changes) == 2
        assert obj.changes[0]['name'] == 'x'
        assert obj.changes[1]['name'] == 'y'

    def test_observable_no_trigger_on_same_value(self):
        """Test that observers are not triggered when value doesn't change."""

        @observable
        class TestClass:
            def __init__(self):
                self.value = 0
                self.change_count = 0

            @observe('value')
            def on_change(self, change):
                self.change_count += 1

        obj = TestClass()
        obj.value = 0  # Same value
        obj.value = 0  # Same value again

        assert obj.change_count == 0

    def test_observable_internal_attributes_ignored(self):
        """Test that internal attributes (starting with _) don't trigger observers."""

        @observable
        class TestClass:
            def __init__(self):
                self.value = 0
                self._internal = 0
                self.change_count = 0

            @observe('value')
            def on_change(self, change):
                self.change_count += 1

        obj = TestClass()
        obj._internal = 10  # Should not trigger observer  # noqa: SLF001

        assert obj.change_count == 0

    def test_observable_init_registration(self):
        """Test that observers are registered during __init__."""

        @observable
        class TestClass:
            def __init__(self):
                self.value = 0
                self.registrations_called = False

            @observe('value')
            def on_change(self, change):
                self.registrations_called = True

        obj = TestClass()
        # After __init__, _observers should exist
        assert hasattr(obj, '_observers')
        assert 'value' in obj._observers  # noqa: SLF001
        assert len(obj._observers['value']) == 1  # noqa: SLF001

    def test_observable_preserves_original_init(self):
        """Test that original __init__ is preserved and called."""

        @observable
        class TestClass:
            def __init__(self):
                self.value = 0
                self.init_called = True

        obj = TestClass()
        assert obj.init_called is True
        assert obj.value == 0

    def test_observable_manual_observer_registration(self):
        """Test manual observer registration via observe() method."""

        @observable
        class TestClass:
            def __init__(self):
                self.value = 0
                self.changes = []

            def custom_handler(self, change):
                self.changes.append(change)

        obj = TestClass()
        obj.observe(obj.custom_handler, ['value'])
        obj.value = 10

        assert len(obj.changes) == 1
        assert obj.changes[0]['name'] == 'value'

    def test_observable_manual_unobserve(self):
        """Test manual observer unregistration via unobserve() method."""

        @observable
        class TestClass:
            def __init__(self):
                self.value = 0
                self.changes = []

            def custom_handler(self, change):
                self.changes.append(change)

        obj = TestClass()
        obj.observe(obj.custom_handler, ['value'])
        obj.value = 10
        assert len(obj.changes) == 1

        obj.unobserve(obj.custom_handler, ['value'])
        obj.value = 20
        assert len(obj.changes) == 1  # No new changes

    def test_observable_with_init_args(self):
        """Test that __init__ arguments are passed correctly."""

        @observable
        class TestClass:
            def __init__(self, initial_value: int):
                self.value = initial_value

        obj = TestClass(42)
        assert obj.value == 42
        assert hasattr(obj, '_observers')

    def test_observable_with_kwargs(self):
        """Test that __init__ kwargs are passed correctly."""

        @observable
        class TestClass:
            def __init__(self, **kwargs):
                self.data = kwargs

        obj = TestClass(a=1, b=2)
        assert obj.data == {'a': 1, 'b': 2}
        assert hasattr(obj, '_observers')


class TestValuesDiffer:
    """Tests for the _values_differ helper function."""

    def test_values_differ_integers(self):
        """Test comparison of integers."""
        assert _values_differ(1, 2) is True
        assert _values_differ(1, 1) is False

    def test_values_differ_floats(self):
        """Test comparison of floats."""
        assert _values_differ(1.0, 2.0) is True
        assert _values_differ(1.0, 1.0) is False

    def test_values_differ_strings(self):
        """Test comparison of strings."""
        assert _values_differ('a', 'b') is True
        assert _values_differ('a', 'a') is False

    def test_values_differ_numpy_arrays(self):
        """Test comparison of numpy arrays."""
        arr1 = np.array([1, 2, 3])
        arr2 = np.array([1, 2, 4])
        arr3 = np.array([1, 2, 3])

        assert _values_differ(arr1, arr2) is True
        assert _values_differ(arr1, arr3) is False

    def test_values_differ_numpy_2d_arrays(self):
        """Test comparison of 2D numpy arrays."""
        arr1 = np.array([[1, 2], [3, 4]])
        arr2 = np.array([[1, 2], [3, 5]])
        arr3 = np.array([[1, 2], [3, 4]])

        assert _values_differ(arr1, arr2) is True
        assert _values_differ(arr1, arr3) is False

    def test_values_differ_none_values(self):
        """Test comparison with None values."""
        assert _values_differ(None, 1) is True
        assert _values_differ(None, None) is False

    def test_values_differ_different_types(self):
        """Test comparison of different types."""
        assert _values_differ(1, '1') is True
        assert _values_differ([1, 2], (1, 2)) is True

    def test_values_differ_lists(self):
        """Test comparison of lists."""
        assert _values_differ([1, 2], [1, 3]) is True
        assert _values_differ([1, 2], [1, 2]) is False

    def test_values_differ_dicts(self):
        """Test comparison of dictionaries."""
        assert _values_differ({'a': 1}, {'a': 2}) is True
        assert _values_differ({'a': 1}, {'a': 1}) is False

    def test_values_differ_uncomparable_types(self):
        """Test that uncomparable types return True (fallback)."""
        # Custom class without __eq__
        class Uncomparable:
            pass

        assert _values_differ(Uncomparable(), Uncomparable()) is True


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_observe_no_attributes(self):
        """Test @observe with no attributes (should still work)."""

        @observable
        class TestClass:
            def __init__(self):
                self.value = 0

            @observe()
            def handler(self, change):
                pass

        obj = TestClass()
        assert hasattr(obj.handler, '__observes__')
        assert obj.handler.__observes__ == ()

    def test_observable_empty_class(self):
        """Test @observable on a class with no attributes or methods."""

        @observable
        class EmptyClass:
            pass

        obj = EmptyClass()
        # Should not raise any errors
        assert hasattr(obj, '_observers')

    def test_change_dict_structure(self):
        """Test that change dictionary has all expected keys."""

        @observable
        class TestClass:
            def __init__(self):
                self.value = 0
                self.last_change = None

            @observe('value')
            def on_change(self, change):
                self.last_change = change

        obj = TestClass()
        obj.value = 10

        change = obj.last_change
        assert change is not None
        assert 'name' in change
        assert 'old' in change
        assert 'new' in change
        assert 'owner' in change
        assert 'type' in change
        assert change['owner'] is obj

    def test_multiple_observers_same_attribute(self):
        """Test that multiple observers on the same attribute all get called."""

        @observable
        class TestClass:
            def __init__(self):
                self.value = 0
                self.changes1 = []
                self.changes2 = []

            @observe('value')
            def observer1(self, change):
                self.changes1.append(change)

            @observe('value')
            def observer2(self, change):
                self.changes2.append(change)

        obj = TestClass()
        obj.value = 10

        assert len(obj.changes1) == 1
        assert len(obj.changes2) == 1
        assert obj.changes1[0]['new'] == 10
        assert obj.changes2[0]['new'] == 10

    def test_observable_inheritance(self):
        """Test that @observable works with class inheritance."""

        @observable
        class BaseClass:
            def __init__(self):
                self.base_value = 0
                self.base_changes = []

            @observe('base_value')
            def on_base_change(self, change):
                self.base_changes.append(change)

        @observable
        class DerivedClass(BaseClass):
            def __init__(self):
                super().__init__()
                # Initialize observer handler attributes BEFORE setting observed attributes
                self.derived_changes = []
                self.derived_value = 0

            @observe('derived_value')
            def on_derived_change(self, change):
                self.derived_changes.append(change)

        obj = DerivedClass()
        # Note: derived_changes already has 1 entry from self.derived_value = 0 in __init__
        obj.base_value = 1
        obj.derived_value = 2

        assert len(obj.base_changes) == 1
        # 1 from __init__ (derived_value=0) + 1 from test (derived_value=2)
        assert len(obj.derived_changes) == 2

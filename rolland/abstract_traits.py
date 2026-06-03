"""Module for creating abstract classes with traitlets.

.. autosummary::
    :toctree: abstract traits

    ABCMetaHasTraits
    ABCHasTraits
    ABCHasStrictTraits
"""

import abc

from traitlets import Any, HasTraits, MetaHasTraits

# ABC classes with traitlets

class ABCMetaHasTraits(abc.ABCMeta, MetaHasTraits):
    """A MetaHasTraits subclass which also inherits from abc.ABCMeta."""


class ABCHasTraits(HasTraits, metaclass=ABCMetaHasTraits):
    """A HasTraits subclass which enables the features of Abstract Base Classes (ABC).

    See the 'abc' module in the standard library for more information.
    """


class ABCHasStrictTraits(ABCHasTraits):
    """A HasTraits subclass which behaves like HasStrictTraits.

    See the 'abc' module in the standard library for more information.
    """

    # Use Any() to enforce strict trait behavior
    _ = Any()

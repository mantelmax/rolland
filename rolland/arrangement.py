"""Arrangement classes for the definition of non-uniform mounting properties.

.. autosummary::
    :toctree: arrangement

    Arrangement
    PeriodicArrangement
    RandomArrangement
"""

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from scipy.stats import truncnorm


@dataclass(kw_only=True)
class Arrangement(ABC):
    r"""Abstract base class for the definition of non-uniform mounting properties.

    Attributes
    ----------
    item : any
        Characteristic object or objects to repeat.
    """

    item: Any

    @abstractmethod
    def generate(self, num_mount):
        """Generate `num_mount` repetitions of objects.

        Parameters
        ----------
        num_mount : int
            The number of repetitions to generate.

        Yields
        ------
        Any
            The repeated item(s).
        """


class PeriodicArrangement(Arrangement):
    r"""Periodic arrangement of given objects.

    Given sequence of objects is repeated periodically when building track using
    :class:`~rolland.track.ArrangedSlabSingleRailTrack` or
    :class:`~rolland.track.ArrangedBallastedSingleRailTrack` class. Mounting positions start at
    :math:`x=0`.

    Attributes
    ----------
    item : any
        Characteristic object or objects to repeat.

    Example
    --------
    >>> from rolland.database.rail.db_rail import rail_60E1
    >>> from rolland.components import Ballast, DiscrPad, Sleeper
    >>> from rolland.arrangement import PeriodicArrangement
    >>> from rolland.track import ArrangedBallastedSingleRailTrack

    >>> thepadA = DiscrPad(
    ...     sp_z=120e6, sp_y=40e6, sp_x=40e6,
    ...     etap_z=0.2, etap_y=0.2, etap_x=0.2, etap_r=0.2,
    ...     wdthp=0.15)
    >>> thepadB = DiscrPad(
    ...     sp_z=60e6, sp_y=20e6, sp_x=20e6,
    ...     etap_z=0.2, etap_y=0.2, etap_x=0.2, etap_r=0.2,
    ...     wdthp=0.15)
    >>> thesleeper = Sleeper(
    ...     ms=300.05, rhos=2648, Is_x=0.0593, Is_y=0.00089, Is_z=0.0596,
    ...     lengs=2.5, wdths=0.245, heights=0.185, z_st=-0.0925, z_sb=0.0925)
    >>> theballast = Ballast(
    ...     sb_z=120e6, sb_y=120e6, sb_x=120e6,
    ...     etab_z=1.0, etab_y=2.0, etab_x=2.0, etab_r=2.0)
    >>> pad = PeriodicArrangement(item=[thepadA, thepadB])
    >>> distance = PeriodicArrangement(item=[0.65, 0.5])
    >>> sleeper = PeriodicArrangement(item=[thesleeper])
    >>> ballast = PeriodicArrangement(item=[theballast])
    >>> tr = ArrangedBallastedSingleRailTrack(
    ...     rail=rail_60E1,
    ...     pad=pad,
    ...     sleeper=sleeper,
    ...     ballast=ballast,
    ...     distance=distance,
    ...     num_mount=100,
    ...     z_f=0.081,
    ...     y_f=0.0)
    """

    def generate(self, num_mount):
        """Generate `num_mount` repetitions of objects.

        Parameters
        ----------
        num_mount : int
            The number of repetitions to generate.

        Yields
        ------
        Any
            The repeated item(s).
        """
        c = 0
        while c < num_mount:
            if isinstance(self.item, list):
                for item_ in self.item:
                    yield item_
                    c += 1
            else:
                yield self.item
                c += 1


class RandomArrangement(Arrangement):
    r"""Random arrangement of given objects.

    Given sequence of objects is repeated randomly when building track using
    :class:`~rolland.track.ArrangedSlabSingleRailTrack` or
    :class:`~rolland.track.ArrangedBallastedSingleRailTrack` class. Mounting positions start at
    :math:`x=0`.

    Attributes
    ----------
    item : any
        Characteristic object or objects to repeat.

    Example
    --------
    >>> from rolland.database.rail.db_rail import rail_60E1
    >>> from rolland.components import Ballast, DiscrPad, Sleeper
    >>> from rolland.arrangement import RandomArrangement
    >>> from rolland.track import ArrangedBallastedSingleRailTrack

    >>> thepadA = DiscrPad(
    ...     sp_z=120e6, sp_y=40e6, sp_x=40e6,
    ...     etap_z=0.2, etap_y=0.2, etap_x=0.2, etap_r=0.2,
    ...     wdthp=0.15)
    >>> thepadB = DiscrPad(
    ...     sp_z=60e6, sp_y=20e6, sp_x=20e6,
    ...     etap_z=0.2, etap_y=0.2, etap_x=0.2, etap_r=0.2,
    ...     wdthp=0.15)
    >>> thesleeper = Sleeper(
    ...     ms=300.05, rhos=2648, Is_x=0.0593, Is_y=0.00089, Is_z=0.0596,
    ...     lengs=2.5, wdths=0.245, heights=0.185, z_st=-0.0925, z_sb=0.0925)
    >>> theballast = Ballast(
    ...     sb_z=120e6, sb_y=120e6, sb_x=120e6,
    ...     etab_z=1.0, etab_y=2.0, etab_x=2.0, etab_r=2.0)
    >>> pad = RandomArrangement(item=[thepadA, thepadB])
    >>> distance = RandomArrangement(item=[0.65, 0.5])
    >>> sleeper = RandomArrangement(item=[thesleeper])
    >>> ballast = RandomArrangement(item=[theballast])
    >>> tr = ArrangedBallastedSingleRailTrack(
    ...     rail=rail_60E1,
    ...     pad=pad,
    ...     sleeper=sleeper,
    ...     ballast=ballast,
    ...     distance=distance,
    ...     num_mount=100,
    ...     z_f=0.081,
    ...     y_f=0.0)
    """

    def generate(self, num_mount):
        """Generate `num_mount` repetitions of objects.

        Parameters
        ----------
        num_mount : int
            The number of repetitions to generate.

        Yields
        ------
        Any
            The repeated item(s).
        """
        for _ in range(num_mount):
            if isinstance(self.item, list):
                yield random.choice(self.item)
            else:
                yield self.item

    @staticmethod
    def trunc_norm(mean, sd, minv, max_v) -> float:
        """Calculate truncated normal distribution.

        Parameters
        ----------
        mean : float
            Mean value.
        sd : float
            Standard deviation.
        minv : float
            Minimum value.
        max_v : float
            Maximum value.

        Returns
        -------
        float
            A random value from the truncated normal distribution.
        """
        sd = sd + 1e-6  # Low value added to avoid error.
        minv = minv - 1e-6
        max_v = max_v + 1e-6
        return float(truncnorm((minv - mean) / sd, (max_v - mean) / sd, loc=mean, scale=sd).rvs(1).item())

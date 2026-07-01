"""Defines deflection classes for FDM simulation.

.. autosummary::
    :toctree: deflection

    Deflection
    DeflectionEBBVertic
"""

import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from numpy import empty, linspace, zeros
from scipy.sparse.linalg import splu

from .discretization import Discretization
from .excitation import Excitation
from .track import Track


@dataclass(kw_only=True)
class Deflection(ABC):
    r"""Abstract base class for deflection classes.

    Attributes
    ----------
    excit : Excitation
        Excitation instance.
    discr : Discretization
        Discretization instance.
    """

    discr: Discretization
    excit: Excitation
    track: Track = field(init=False)

    def __post_init__(self, *args, **kwargs):
        """post_init method to set track attribute after initialization."""
        self.track = self.discr.track

    @abstractmethod
    def validate_deflection(self):
        """Validate deflection."""


class DeflectionEBBVertic(Deflection):
    r"""Calculate deflection according to :cite:t:`stampka2022a`.

    Attributes
    ----------
    track : Track
        Track instance.
    excit : Excitation
        Excitation instance.
    discr : Discretization
        Discretization instance.
    deflection : numpy.ndarray
        Deflection array :math:`[m]`.
    ind_excit : int
        Index of excitation point :math:`[-]`.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the DeflectionFDMStampka class.

        Parameters
        ----------
        *args : tuple
            Variable length argument list.
        **kwargs : dict
            Arbitrary keyword arguments.

        Attributes
        ----------
        deflection : numpy.ndarray
            Array of calculated deflections with shape (2 * nx, nt + 1).
        """
        super().__init__(*args, **kwargs)
        # Initialize starting values
        self.calc_force()
        defl = self.initialize_start_values()
        # Calculate deflection
        self.deflection = self.calc_deflection(defl)

    __init__.__signature__ = inspect.signature(Deflection.__init__)

    def validate_deflection(self):
        """Validate deflection."""

    def initialize_start_values(self):
        """Set starting values of deflections to zero.

        Returns
        -------
        defl : numpy.ndarray
            Array of deflections initialized to zero with shape (2 * nx, nt + 1).
        """
        defl = empty((2 * self.discr.nx, self.discr.nt + 1))

        # Set starting values to zero for two time steps
        defl[:, 0:2] = zeros((2 * self.discr.nx, 2))
        return defl

    def calc_force(self):
        """Calculate force array."""
        t = linspace(0, self.discr.sim_t, self.discr.nt)
        self.force = self.excit.force(t)


    def calc_rightside_crank_nicolson(self, u1, u0, ind_excit, t):
        """Calculate the right-hand side of the equation according to :cite:t:`stampka2022a`.

        Parameters
        ----------
        u1 : numpy.ndarray
            Deflection array at the current time step.
        u0 : numpy.ndarray
            Deflection array at the previous time step.
        ind_excit : int
            Index of the excitation point.
        t : int
            Current time step.

        Returns
        -------
        numpy.ndarray
            Right-hand side of the equation.
        """
        # Write excitation force for time step t into force array
        f = zeros(2 * self.discr.nx)

        if isinstance(ind_excit, list):
            for idx in ind_excit:
                f[idx] = self.force[t]
        else:
            f[ind_excit] = self.force[t]

        return (self.discr.B.dot(u1) + self.discr.C.dot(u0) + self.discr.dt ** 2 /
                (self.track.rail.mr * self.discr.dx) * f)


    def calc_deflection(self, defl):
        """
        Calculate deflection.

        Parameters
        ----------
        defl : numpy.ndarray
            Array of deflections initialized to zero with shape (2 * nx, nt + 1).

        Returns
        -------
        defl : numpy.ndarray
            Array of calculated deflections with shape (2 * nx, nt + 1).
        """
        # Index of excitation point/points
        if isinstance(self.excit.x_excit, list):
            self.ind_excit = [int(x / self.discr.dx) for x in self.excit.x_excit]
        else:
            self.ind_excit = int(self.excit.x_excit / self.discr.dx)

        # Factorization of matrix A (LU decomposition)
        factoriz = splu(self.discr.A)

        for t in range(1, self.discr.nt):
            # Calculate right hand side of equation
            b = self.calc_rightside_crank_nicolson(u1=defl[:, t], u0=defl[:, t - 1], ind_excit=self.ind_excit, t=t)

            # Calculate deflection for time step t
            u = factoriz.solve(b)

            defl[:, t + 1] = u[0:2 * self.discr.nx]
        return defl





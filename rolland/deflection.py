"""Defines deflection classes for FDM simulation.

.. autosummary::
    :toctree: deflection

    Deflection
"""

from dataclasses import dataclass

from devito import (
    ConditionalDimension,
    Eq,
    Operator,
    SparseTimeFunction,
    TimeFunction,
)

from .domainsetup import DomSetup
from .excitation import Excitation


@dataclass(kw_only=True)
class Deflection:
    r"""Simulates and records track deflection.

    This class manages the finite-difference simulation of a track under excitation.
    It handles operator building, conditional sub-sampling to manage memory, and
    extracting the resulting wavefields either at specific points or across the entire grid.

    Attributes
    ----------
    discr : DomSetup
        The discretization object containing track operator.
    excit : Excitation
        The excitation source injected into the simulation.
    store : str
        Storage mode for results:
        - 'observe' for specific positions
        - 'full' for the entire grid
        - 'excit' to track deflection at the excitation position
    skip : int
        The subsampling factor; saves every `skip` time step to reduce output array size.
    obs_pos : float | list | None
        Coordinates of the response positions to observe. Must be explicitly provided
        if store='observe'.
    track : Track
        The track model properties (automatically set from `discr`).
    u_z_obs : devito.TimeFunction or devito.SparseTimeFunction
        The observed vertical deflection at defined positions.
    u_y_obs : devito.TimeFunction or devito.SparseTimeFunction
        The observed lateral deflection at defined positions.
    phi_x_obs : devito.TimeFunction or devito.SparseTimeFunction
        The observed torsional rotation at defined positions.
    """

    discr: DomSetup
    excit: Excitation
    store: str
    skip: int = 1
    obs_pos: float | list | None = None

    def __post_init__(self):
        """Initialize derived attributes and run the simulation."""
        if self.store not in ['full', 'observe', 'excit']:
            msg = "store must be one of 'full', 'observe', or 'excit'."
            raise ValueError(msg)

        # Enforce that a float or list is provided when observing
        if self.store == 'observe' and self.obs_pos is None:
            msg_0 = "When store='observe', a float or list must be explicitly provided for obs_pos."
            raise ValueError(msg_0)

        self.track = self.discr.track

        # Only set up observation positions if we are not relying on the excitation position
        if self.store in ['full', 'observe']:
            self.obs_pos = self._setup_obs_pos(self.obs_pos)

        self.run()

    def run(self):
        """Build and execute the operator."""
        exc_terms = self.excit.inject_in_track(discr=self.discr)

        time_dim = self.discr.grid.time_dim
        time_sub = ConditionalDimension('time_sub', parent=time_dim, factor=self.skip)

        nt_sub = (self.discr.nt - 1) // self.skip + 1

        obs_funcs = {}
        obs_terms = []

        if self.store == 'full':
            self.u_z_obs = TimeFunction(name='u_z_obs', grid=self.discr.grid, save=nt_sub, time_dim=time_sub)
            self.u_y_obs = TimeFunction(name='u_y_obs', grid=self.discr.grid, save=nt_sub, time_dim=time_sub)
            self.phi_x_obs = TimeFunction(name='phi_x_obs', grid=self.discr.grid, save=nt_sub, time_dim=time_sub)

            obs_terms.append(Eq(self.u_z_obs, self.discr.u_z))
            obs_terms.append(Eq(self.u_y_obs, self.discr.u_y))
            obs_terms.append(Eq(self.phi_x_obs, self.discr.phi_x))

            obs_funcs = {'u_z_obs': self.u_z_obs, 'u_y_obs': self.u_y_obs, 'phi_x_obs': self.phi_x_obs}

        elif self.store == 'observe':
            npoint = len(self.obs_pos)
            fields_map = {
                'u_z_obs': self.discr.u_z,
                'u_y_obs': self.discr.u_y,
                'phi_x_obs': self.discr.phi_x,
            }

            for name, field_expr in fields_map.items():
                obs = SparseTimeFunction(
                    name=name,
                    grid=self.discr.grid,
                    npoint=npoint,
                    nt=self.discr.nt,
                    coordinates=self.obs_pos,
                )
                obs_funcs[name] = obs
                obs_terms.append(obs.interpolate(expr=field_expr))

        elif self.store == 'excit':
            # Call the excitation's observe method and add the terms to the operator
            obs_funcs_excit, obs_terms_excit = self.excit.observe_excitation(discr=self.discr)
            obs_funcs.update(obs_funcs_excit)
            obs_terms.extend(obs_terms_excit)

        # Build and apply the DEVITO operator
        op_exc = Operator(self.discr.op_track + list(exc_terms) + obs_terms)
        op_exc.apply(dt=self.discr.dt)

        # Extract data onto the class attributes
        for name, obs in obs_funcs.items():
            if self.store == 'full':
                setattr(self, name, obs.data)
            else:
                # Store subsampled data
                setattr(self, name, obs.data[:: self.skip])

    def _setup_obs_pos(self, obs_pos):
        """Format observation positions based on the selected storage mode."""
        if self.store == 'full':
            nx, dx, origin = (self.discr.grid.shape[0], self.discr.grid.spacing[0], self.discr.grid.origin[0])
            return [[origin + i * dx] for i in range(nx)]

        # Handle a single float or integer position
        if isinstance(obs_pos, (float, int)):
            return [[float(obs_pos)]]

        # Handle list of positions
        return [[pos] if not isinstance(pos, (list, tuple)) else pos for pos in obs_pos]

    def _abstract(self) -> None:
        pass

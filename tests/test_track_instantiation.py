"""Tests for track class instantiation behavior.

This module tests that abstract track classes cannot be instantiated
and concrete track classes can be instantiated.
"""

import pytest

from rolland.components import Ballast, ContPad, DiscrPad, Slab, Sleeper
from rolland.database.rail.db_rail import UIC60
from rolland.track import (
    ArrangedBallastedSingleRailTrack,
    ArrangedSlabSingleRailTrack,
    BallastedSingleRailTrack,
    ContBallastedSingleRailTrack,
    ContSlabSingleRailTrack,
    DiscrBallastedSingleRailTrack,
    DiscrSlabSingleRailTrack,
    SimplePeriodicBallastedSingleRailTrack,
    SimplePeriodicSlabSingleRailTrack,
    SingleRailTrack,
    SlabSingleRailTrack,
    Track,
)


# Fixtures for reusable component instances
@pytest.fixture
def rail():
    """Provide a pre-defined rail instance."""
    return UIC60


@pytest.fixture
def cont_pad():
    """Provide a pre-defined continuous pad instance."""
    return ContPad(sp_z=300e6, sp_y=0, sp_x=0, dp_z=30000, dp_y=0, dp_x=0, dp_xr=0, wdthp=0.5)


@pytest.fixture
def discr_pad():
    """Provide a pre-defined discrete pad instance."""
    return DiscrPad(sp_z=300e6, sp_y=0, sp_x=0, etap_z=0.25, etap_y=0.25, etap_x=0.25, etap_r=0.25, wdthp=0.5)


@pytest.fixture
def slab():
    """Provide a pre-defined slab instance."""
    return Slab(
        ms=250,
        Is_z=1.0,
        Is_y=1.0,
        Is_x=1.0,
        rhos=2500,
        lengs=1.0,
        equ_wdths=1.0,
        heights=0.5,
        z_st=0.25,
        z_sb=0.25,
    )


@pytest.fixture
def sleeper():
    """Provide a pre-defined sleeper instance."""
    return Sleeper(
        ms=162,
        Is_x=1.0,
        Is_y=1.0,
        Is_z=1.0,
        rhos=2500,
        lengs=2.5,
        wdths=0.26,
        heights=0.2,
        z_st=0.1,
        z_sb=0.1,
    )


@pytest.fixture
def ballast():
    """Provide a pre-defined ballast instance."""
    return Ballast(sb_z=50e6, sb_y=0, sb_x=0, etab_z=1.0, etab_y=1.0, etab_x=1.0, etab_r=1.0)


class TestAbstractTrackClasses:
    """Tests for abstract track base classes."""

    def test_track_cannot_be_instantiated(self):
        """Test that Track (base ABC) cannot be instantiated."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class Track"):
            Track()

    def test_single_rail_track_cannot_be_instantiated(self, rail):
        """Test that SingleRailTrack cannot be instantiated."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class SingleRailTrack"):
            SingleRailTrack(rail=rail)

    def test_slab_single_rail_track_cannot_be_instantiated(self, rail):
        """Test that SlabSingleRailTrack cannot be instantiated."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class SlabSingleRailTrack"):
            SlabSingleRailTrack(rail=rail)

    def test_discr_slab_single_rail_track_cannot_be_instantiated(self, rail, discr_pad):
        """Test that DiscrSlabSingleRailTrack cannot be instantiated."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class DiscrSlabSingleRailTrack"):
            DiscrSlabSingleRailTrack(rail=rail, pad=discr_pad)

    def test_ballasted_single_rail_track_cannot_be_instantiated(self, rail, ballast):
        """Test that BallastedSingleRailTrack cannot be instantiated."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class BallastedSingleRailTrack"):
            BallastedSingleRailTrack(rail=rail, ballast=ballast)

    def test_discr_ballasted_single_rail_track_cannot_be_instantiated(self, rail, ballast):
        """Test that DiscrBallastedSingleRailTrack cannot be instantiated."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class DiscrBallastedSingleRailTrack"):
            DiscrBallastedSingleRailTrack(rail=rail, ballast=ballast)


class TestConcreteSlabTrackClasses:
    """Tests for concrete slab track classes."""

    def test_cont_slab_single_rail_track_can_be_instantiated(self, rail, cont_pad, slab):
        """Test that ContSlabSingleRailTrack can be instantiated."""
        track = ContSlabSingleRailTrack(
            rail=rail,
            pad=cont_pad,
            slab=slab,
            l_track=100.0,
            z_f=0.0,
            y_f=0.0,
        )
        assert isinstance(track, ContSlabSingleRailTrack)

    def test_simple_periodic_slab_single_rail_track_can_be_instantiated(self, rail, discr_pad, slab):
        """Test that SimplePeriodicSlabSingleRailTrack can be instantiated."""
        track = SimplePeriodicSlabSingleRailTrack(
            rail=rail,
            pad=discr_pad,
            slab=slab,
            distance=0.6,
            num_mount=10,
            z_f=0.0,
            y_f=0.0,
        )
        assert isinstance(track, SimplePeriodicSlabSingleRailTrack)

    def test_arranged_slab_single_rail_track_can_be_instantiated(self, rail, discr_pad, slab):
        """Test that ArrangedSlabSingleRailTrack can be instantiated."""
        from rolland.arrangement import PeriodicArrangement

        pad = PeriodicArrangement(item=[discr_pad])
        distance = PeriodicArrangement(item=[0.6])

        track = ArrangedSlabSingleRailTrack(
            rail=rail,
            pad=pad,
            slab=slab,
            distance=distance,
            num_mount=10,
            z_f=0.0,
            y_f=0.0,
        )
        assert isinstance(track, ArrangedSlabSingleRailTrack)


class TestConcreteBallastedTrackClasses:
    """Tests for concrete ballasted track classes."""

    def test_cont_ballasted_single_rail_track_can_be_instantiated(self, rail, cont_pad, slab, ballast):
        """Test that ContBallastedSingleRailTrack can be instantiated."""
        track = ContBallastedSingleRailTrack(
            rail=rail,
            pad=cont_pad,
            slab=slab,
            ballast=ballast,
            l_track=100.0,
            z_f=0.0,
            y_f=0.0,
        )
        assert isinstance(track, ContBallastedSingleRailTrack)

    def test_simple_periodic_ballasted_single_rail_track_can_be_instantiated(self, rail, discr_pad, sleeper, ballast):
        """Test that SimplePeriodicBallastedSingleRailTrack can be instantiated."""
        track = SimplePeriodicBallastedSingleRailTrack(
            rail=rail,
            pad=discr_pad,
            sleeper=sleeper,
            ballast=ballast,
            distance=0.6,
            num_mount=10,
            z_f=0.0,
            y_f=0.0,
        )
        assert isinstance(track, SimplePeriodicBallastedSingleRailTrack)

    def test_arranged_ballasted_single_rail_track_can_be_instantiated(self, rail, discr_pad, sleeper, ballast):
        """Test that ArrangedBallastedSingleRailTrack can be instantiated."""
        from rolland.arrangement import PeriodicArrangement

        pad = PeriodicArrangement(item=[discr_pad])
        sleeper_arr = PeriodicArrangement(item=[sleeper])
        ballast_arr = PeriodicArrangement(item=[ballast])
        distance = PeriodicArrangement(item=[0.6])

        track = ArrangedBallastedSingleRailTrack(
            rail=rail,
            pad=pad,
            sleeper=sleeper_arr,
            ballast=ballast_arr,
            distance=distance,
            num_mount=10,
            z_f=0.0,
            y_f=0.0,
        )
        assert isinstance(track, ArrangedBallastedSingleRailTrack)

"""Tests for visualization module."""

import gc

import matplotlib as mpl
import numpy as np
import pytest
from matplotlib.backend_bases import MouseEvent
from matplotlib.collections import PatchCollection
from matplotlib.offsetbox import AnnotationBbox

import rolland
from rolland.arrangement import PeriodicArrangement
from rolland.boundary import CFSPML
from rolland.components import Ballast, ContPad, DiscrPad, Slab, Sleeper
from rolland.database.rail.db_rail import rail_60E1
from rolland.excitation import GaussianImpulse, RandomForce
from rolland.track import (
    ArrangedBallastedSingleRailTrack,
    ArrangedSlabSingleRailTrack,
    ContBallastedSingleRailTrack,
    ContSlabSingleRailTrack,
    SimplePeriodicBallastedSingleRailTrack,
)
from rolland.visualization import TrackSchematic

mpl.use('Agg')

L_BOUND = 2.0  # Boundary domain length [m]
SPACINGS = (0.65, 0.5)  # Alternating mounting distances of the arranged tracks [m]
PAD_STIFFNESSES = (120e6, 700e6)  # Alternating vertical pad stiffnesses [N/m]
SLEEPER_MASSES = (150.0, 600.0)  # Alternating sleeper masses [kg]


# --- Fixtures ---------------------------------------------------------------------


def make_pad(sp_z):
    """Discrete pad with the given vertical stiffness."""
    return DiscrPad(sp_z=sp_z, sp_y=40e6, sp_x=40e6, etap_z=0.2, etap_y=0.2, etap_x=0.2, etap_r=0.2, wdthp=0.15)


def make_sleeper(ms):
    """Sleeper with the given mass."""
    return Sleeper(ms=ms, Is_x=0.059, Is_y=0.00089, Is_z=0.059, rhos=2648, lengs=2.5, wdths=0.245, heights=0.185,
                   z_st=-0.0925, z_sb=0.0925)


BALLAST = Ballast(sb_z=120e6, sb_y=120e6, sb_x=120e6, etab_z=1.0, etab_y=2.0, etab_x=2.0, etab_r=2.0)
CONT_PAD = ContPad(sp_z=200e6, sp_y=66.7e6, sp_x=66.7e6, etap_z=0.2, etap_y=0.2, etap_x=0.2, etap_r=0.2, wdthp=0.15)
SLAB = Slab(ms=500.0, Is_x=0.1, Is_y=0.0015, Is_z=0.1, rhos=2648, lengs=2.5, equ_wdths=0.245, heights=0.185,
            z_st=-0.0925, z_sb=0.0925)


@pytest.fixture(scope='module')
def varied_ballasted():
    """Ballasted track with alternating pads, sleepers and spacings (60 mounting positions)."""
    return ArrangedBallastedSingleRailTrack(
        rail=rail_60E1,
        pad=PeriodicArrangement(item=[make_pad(s) for s in PAD_STIFFNESSES]),
        sleeper=PeriodicArrangement(item=[make_sleeper(m) for m in SLEEPER_MASSES]),
        ballast=PeriodicArrangement(item=BALLAST),
        distance=PeriodicArrangement(item=list(SPACINGS)),
        num_mount=60,
        z_f=0.081,
        y_f=0.0,
    )


@pytest.fixture(scope='module')
def uniform_ballasted():
    """Ballasted track without variation."""
    return SimplePeriodicBallastedSingleRailTrack(
        rail=rail_60E1, pad=make_pad(120e6), sleeper=make_sleeper(300.0), ballast=BALLAST, distance=0.6,
        num_mount=20, z_f=0.081, y_f=0.0,
    )


@pytest.fixture(scope='module')
def varied_slab():
    """Slab track with alternating pads and mounting distances."""
    return ArrangedSlabSingleRailTrack(
        rail=rail_60E1,
        pad=PeriodicArrangement(item=[make_pad(s) for s in PAD_STIFFNESSES]),
        distance=PeriodicArrangement(item=list(SPACINGS)),
        num_mount=20,
        z_f=0.081,
        y_f=0.0,
    )


@pytest.fixture(scope='module')
def cont_ballasted():
    """Continuously supported ballasted track."""
    return ContBallastedSingleRailTrack(rail=rail_60E1, pad=CONT_PAD, slab=SLAB, ballast=BALLAST, l_track=20.0,
                                        z_f=0.081, y_f=0.0)


@pytest.fixture(scope='module')
def cont_slab():
    """Continuously supported slab track."""
    return ContSlabSingleRailTrack(rail=rail_60E1, pad=CONT_PAD, l_track=20.0, z_f=0.081, y_f=0.0)


@pytest.fixture(autouse=True)
def close_figures():
    """Close all figures after each test."""
    yield
    import matplotlib.pyplot as plt

    plt.close('all')


def labels(ax):
    """Return the boxed labels right of the track."""
    return [t.get_text() for t in ax.texts if t.get_bbox_patch() is not None]


def names(ax):
    """Return the component names (first line of the boxed labels)."""
    return {label.split('\n')[0] for label in labels(ax)}


def texts(ax):
    """Return all texts of the axes."""
    return [t.get_text() for t in ax.texts]


def colorbar_axes(ax):
    """Return the colorbar axes of the figure."""
    return [a for a in ax.figure.axes if a.get_ylabel()]


def scroll(ax, button, times=1):
    """Simulate mouse wheel steps over the center of the axes."""
    x, y = ax.transAxes.transform((0.5, 0.5))
    for _ in range(times):
        ax.figure.canvas.callbacks.process('scroll_event', MouseEvent('scroll_event', ax.figure.canvas, x, y,
                                                                      button=button))


# --- Package import ---------------------------------------------------------------


def test_import_from_package():
    """TrackSchematic is available from the package namespace."""
    assert rolland.TrackSchematic is TrackSchematic


# --- Input validation -------------------------------------------------------------


def test_moving_excitation_requires_bound(uniform_ballasted):
    """A moving excitation cannot be shown without boundary (unknown start position)."""
    with pytest.raises(ValueError, match='bound'):
        TrackSchematic(track=uniform_ballasted, excit=RandomForce(v=30.0, F_stat_z=5e4, F_stat_y=0.0))


def test_unsupported_excitation_type(uniform_ballasted):
    """Unsupported excitation types raise an error."""
    with pytest.raises(TypeError, match='not supported'):
        TrackSchematic(track=uniform_ballasted, excit=object())


def test_invalid_property(uniform_ballasted):
    """Unknown component properties raise an error."""
    with pytest.raises(ValueError, match='sp_q'):
        TrackSchematic(track=uniform_ballasted, pad_prop='sp_q').plot()


# --- Track layers -----------------------------------------------------------------


@pytest.mark.parametrize(('track_fixture', 'expected'), [
    ('uniform_ballasted', {'Rail', 'Pad', 'Sleeper', 'Ballast'}),
    ('varied_slab', {'Rail', 'Pad', 'Slab (rigid)', 'Mounting distance'}),
    ('cont_ballasted', {'Rail', 'Pad', 'Slab', 'Ballast'}),
    ('cont_slab', {'Rail', 'Pad', 'Slab (rigid)'}),
])
def test_component_names(request, track_fixture, expected):
    """Each track type shows its components."""
    ax = TrackSchematic(track=request.getfixturevalue(track_fixture)).plot()
    assert names(ax) == expected


def test_equal_blocks(varied_ballasted):
    """All support blocks have the same width and do not touch at the smallest spacing."""
    ax = TrackSchematic(track=varied_ballasted).plot()
    widths = [path.get_extents().width for coll in ax.collections if isinstance(coll, PatchCollection)
              for path in coll.get_paths()]
    assert len(widths) == 3 * len(varied_ballasted.mount_prop)
    np.testing.assert_allclose(widths, widths[0])
    assert widths[0] < min(SPACINGS)


def test_higher_value_is_darker(varied_ballasted):
    """Blocks with a higher value are drawn darker."""
    ax = TrackSchematic(track=varied_ballasted).plot()
    pads = next(coll for coll in ax.collections if isinstance(coll, PatchCollection))  # Top layer first
    first, second = pads.get_facecolors()[:2, :3].sum(axis=1)
    assert PAD_STIFFNESSES[0] < PAD_STIFFNESSES[1]
    assert second < first


def test_constant_values_with_units(uniform_ballasted, cont_ballasted):
    """Constant values are given next to the layer, with units of discrete or continuous support."""
    discrete = labels(TrackSchematic(track=uniform_ballasted).plot())
    assert 'Pad\nsp_z = 120 MN/m' in discrete
    assert 'Sleeper\nms = 300 kg' in discrete
    assert 'Ballast\nsb_z = 120 MN/m' in discrete

    continuous = labels(TrackSchematic(track=cont_ballasted, pad_prop='sp_y').plot())
    assert 'Pad\nsp_y = 66.7 MN/m²' in continuous
    assert 'Slab\nms = 500 kg/m' in continuous


# --- Variation --------------------------------------------------------------------


def test_colorbars_for_varying_layers(varied_ballasted):
    """Varying layers get a colorbar (in layer order) with ticks at the occurring values."""
    ax = TrackSchematic(track=varied_ballasted).plot()
    ax.figure.canvas.draw()
    cbars = {cax.get_ylabel(): cax for cax in colorbar_axes(ax)}

    assert set(cbars) == {'Pad: sp_z [MN/m]', 'Sleeper: ms [kg]'}  # Ballast is constant
    pad, sleeper = cbars['Pad: sp_z [MN/m]'], cbars['Sleeper: ms [kg]']
    np.testing.assert_allclose(pad.get_yticks(), PAD_STIFFNESSES)
    assert [t.get_text() for t in pad.get_yticklabels()] == ['120', '700']
    np.testing.assert_allclose(sleeper.get_yticks(), SLEEPER_MASSES)
    assert pad.get_position().x0 < sleeper.get_position().x0


def test_varying_layers_show_name_only(varied_ballasted):
    """Varying layers are labeled with their name only, constant layers with their value."""
    ax_labels = labels(TrackSchematic(track=varied_ballasted).plot())
    assert 'Pad' in ax_labels
    assert 'Sleeper' in ax_labels
    assert 'Ballast\nsb_z = 120 MN/m' in ax_labels


def test_no_variation_marks_without_variation(uniform_ballasted):
    """Without variation there is neither a brace nor a colorbar."""
    ax = TrackSchematic(track=uniform_ballasted).plot()
    assert 'Variation' not in texts(ax)
    assert colorbar_axes(ax) == []


def test_variation_brace(varied_ballasted):
    """Varying layers are marked with a brace."""
    assert 'Variation' in texts(TrackSchematic(track=varied_ballasted).plot())


def test_spacing_dimensions_in_section(varied_ballasted):
    """Spacings are dimensioned in a section and summarized in the label in any case."""
    label = f'Sleeper spacing\n{min(SPACINGS):g} … {max(SPACINGS):g} m'
    dims = {f'{s:g} m' for s in SPACINGS}

    section = TrackSchematic(track=varied_ballasted).plot(x_range=(0.0, 5.0))
    assert label in labels(section)
    assert dims <= set(texts(section))

    overview = TrackSchematic(track=varied_ballasted).plot()  # More than 40 mounting positions
    assert label in labels(overview)
    assert not dims & set(texts(overview))


# --- Boundary and excitation ------------------------------------------------------


def test_boundary_labels_in_visible_domains(uniform_ballasted):
    """Each visible boundary domain is labeled once."""
    bound = CFSPML(l_bound=L_BOUND)
    whole = TrackSchematic(track=uniform_ballasted, bound=bound).plot()
    assert texts(whole).count(f'PML ({L_BOUND:g} m)') == 2

    left = TrackSchematic(track=uniform_ballasted, bound=bound).plot(x_range=(1.0, 5.0))
    assert texts(left).count(f'PML ({L_BOUND:g} m)') == 1


@pytest.mark.parametrize(('force_dir', 'expected'), [('vertical', '$F_z$'), ('lateral', '$F_y$')])
def test_stationary_excitation(uniform_ballasted, force_dir, expected):
    """A stationary impulse is drawn at its position on the rail."""
    excit = GaussianImpulse(x_excit=5.0, force_dir=force_dir)
    ax = TrackSchematic(track=uniform_ballasted, excit=excit).plot()
    force = next(t for t in ax.texts if t.get_text() == expected)
    assert force.xy[0] == excit.x_excit


def test_moving_excitation(uniform_ballasted):
    """A moving excitation is drawn as wheel at the end of the left boundary domain."""
    excit = RandomForce(v=30.0, F_stat_z=5e4, F_stat_y=0.0)
    ax = TrackSchematic(track=uniform_ballasted, excit=excit, bound=CFSPML(l_bound=L_BOUND)).plot()
    wheels = [a for a in ax.artists if isinstance(a, AnnotationBbox)]
    assert len(wheels) == 1
    assert wheels[0].xy[0] == L_BOUND
    assert 'v = 30 m/s' in texts(ax)


# --- Scrolling --------------------------------------------------------------------


def test_scroll_moves_section(varied_ballasted):
    """The mouse wheel moves the section by 10 % of its width and stops at the track ends."""
    ax = TrackSchematic(track=varied_ballasted).plot(x_range=(0.0, 10.0), scroll=True)
    gc.collect()  # The slider must stay alive without a reference held by the user

    scroll(ax, 'down')
    assert ax.get_xlim() == pytest.approx((0.0, 10.0))
    scroll(ax, 'up', times=3)
    assert ax.get_xlim() == pytest.approx((3.0, 13.0))

    l_track = varied_ballasted.l_track
    ax = TrackSchematic(track=varied_ballasted).plot(x_range=(l_track - 11.5, l_track - 1.5), scroll=True)
    scroll(ax, 'up', times=3)
    assert ax.get_xlim() == pytest.approx((l_track - 10.0, l_track))


def test_scroll_updates_section_elements(varied_ballasted):
    """Section-dependent elements are redrawn instead of accumulating."""
    ax = TrackSchematic(track=varied_ballasted, bound=CFSPML(l_bound=L_BOUND)).plot(x_range=(0.0, 5.0),
                                                                                    scroll=True)
    n_children = len(ax.get_children())
    assert texts(ax).count(f'PML ({L_BOUND:g} m)') == 1

    scroll(ax, 'up', times=6)  # Left boundary domain out of view
    assert len(ax.get_children()) == n_children - 1
    assert f'PML ({L_BOUND:g} m)' not in texts(ax)


def test_scroll_default_section(varied_ballasted):
    """Without x_range, scrolling starts with a 10 m section."""
    ax = TrackSchematic(track=varied_ballasted).plot(scroll=True)
    assert ax.get_xlim() == pytest.approx((0.0, 10.0))
    assert len(ax.child_axes) == 1  # Slider


def test_no_slider_for_whole_track(varied_ballasted):
    """No slider is added if the section covers the whole track."""
    ax = TrackSchematic(track=varied_ballasted).plot(x_range=(0.0, varied_ballasted.l_track), scroll=True)
    assert ax.child_axes == []

"""Visualization of the track setup.

.. autosummary::
    :toctree: visualization

    TrackSchematic
"""

from dataclasses import dataclass
from itertools import groupby

import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb
from matplotlib.offsetbox import AnnotationBbox, DrawingArea
from matplotlib.patches import Circle, PathPatch, Rectangle
from matplotlib.path import Path
from matplotlib.ticker import EngFormatter, FuncFormatter

from .boundary import CFSPML
from .excitation import Excitation, GaussianImpulse, RandomForce
from .track import (
    DiscrBallastedSingleRailTrack,
    DiscrSlabSingleRailTrack,
    SlabSingleRailTrack,
    Track,
)

# Vertical layout of the schematic (arbitrary units, not to scale).
_Z_TOP = 3.1
_Z_TOP_EXCIT = 4.3  # upper limit if an excitation is shown
_Z_LABEL_ROW = 0.35  # additional space at the top for the boundary labels
_Z_BOTTOM = -1.1  # lower limit if spacings are labeled, otherwise just below the ground
_Z_GROUND = -0.25
_Z_SPACING = -0.7
_Z_BALLAST = (0.25, 0.75)
_Z_SLEEPER = (1.0, 1.5)
_Z_PAD = (1.75, 2.25)
_Z_RAIL = (2.5, 2.55, 2.8, 2.95)  # bottom of foot, foot/web, web/head, top of head

# Rail parts as (bottom, top, colors from bottom to top).
_RAIL_PARTS = (
    (_Z_RAIL[0], _Z_RAIL[1], ('#1a1a1a', '#dfbf9f')),
    (_Z_RAIL[1], _Z_RAIL[2], ('#404040', '#dfbf9f', '#1a1a1a')),
    (_Z_RAIL[2], _Z_RAIL[3], ('#dfbf9f', '#dfbf9f', '#1a1a1a')),
)

_MAX_DETAIL_MOUNTS = 40  # mounting lines and spacing dimensions only up to this number of visible positions
_SCROLL_WINDOW = 10.0  # default width of the shown section when scrolling [m]
_SCROLL_STEP = 0.1  # shift per mouse wheel step relative to the section width

_BLOCK_WIDTH = 0.6  # width of the support blocks relative to the smallest mounting distance
_MAX_DISTINCT_TICKS = 4  # colorbar ticks at the occurring values if there are at most this many
_PREFIX_MIN = 1e3  # smallest value written with SI prefix

_COLORS = {'pad': 'blue', 'sleeper': '#fb3199', 'ballast': 'gray'}
_ALPHA_RANGE = (0.2, 1.0)  # color intensity for the lowest and highest value of a varying property
_ALPHA_CONST = 0.5  # color intensity of a property without variation

# Units of common properties as (discrete support, continuous support).
_UNITS = {
    **dict.fromkeys(('sp_x', 'sp_y', 'sp_z', 'sb_x', 'sb_y', 'sb_z'), ('N/m', 'N/m²')),
    **dict.fromkeys(('dp_x', 'dp_y', 'dp_z', 'db_x', 'db_y', 'db_z'), ('Ns/m', 'Ns/m²')),
    'ms': ('kg', 'kg/m'),
    'wdthp': ('m', 'm'),
    'wdths': ('m', 'm'),
}

# Unit curly brace as (depth, position) control points; the tip points away from the track.
_BRACE_VERTS = (
    (0, 0), (0.4, 0), (0.5, 0.05), (0.5, 0.12), (0.5, 0.38),
    (0.5, 0.45), (0.6, 0.5), (1, 0.5),
    (0.6, 0.5), (0.5, 0.55), (0.5, 0.62), (0.5, 0.88),
    (0.5, 0.95), (0.4, 1), (0, 1),
)
_BRACE_CODES = (
    Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.LINETO,
    Path.CURVE4, Path.CURVE4, Path.CURVE4,
    Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.LINETO,
    Path.CURVE4, Path.CURVE4, Path.CURVE4,
)

# Wheel as (relative radius, color) from outside to inside: flange, tread, inner part, hub.
_WHEEL_PARTS = ((1.1, '0.6'), (0.975, '0.4'), (0.8, '0.55'), (0.28, '0.7'))
_WHEEL_SCALE = 20  # points per relative radius; wheel and force symbols keep their size when zooming
_FORCE_LENGTH = 40  # length of the force arrow in points


@dataclass(kw_only=True)
class _Layer:
    """Support layer of the schematic (pads, sleepers, slab or ballast)."""

    label: str
    color: str
    z: tuple[float, float]
    prop: str
    unit: str
    x: np.ndarray
    width: np.ndarray
    values: np.ndarray

    @property
    def varies(self) -> bool:
        """Whether the shown property differs between the mounting positions."""
        return not np.isclose(self.values.min(), self.values.max())


@dataclass(kw_only=True)
class TrackSchematic:
    r"""Schematic side view of a track.

    Shows the rail and the support layers of the track (pads, sleepers or slab, ballast).
    Discrete support components are drawn as blocks of equal size at the mounting positions;
    their actual dimensions are not shown. The color intensity of each support layer
    represents the value of a selectable property, e.g. the vertical pad stiffness. Darker
    colors correspond to higher values. If a property varies along the track, a colorbar
    shows its values and a brace marks the varying layers; otherwise, the value is given
    next to the layer. Varying sleeper spacings are dimensioned below the ground.

    Optionally, the excitation and the boundary domains are shown. A stationary impulse is
    drawn as force arrow (vertical) or force symbol (lateral) at its position. A moving
    excitation is drawn as wheel at its start position, the end of the left boundary domain,
    together with its velocity. The boundary domains (PML) at both track ends are shaded.

    The track position (x-axis) is to scale, the vertical layout is schematic.

    Attributes
    ----------
    track : Track
        Track instance to be shown.
    excit : Excitation, optional
        Excitation instance to be shown (:class:`~rolland.excitation.GaussianImpulse` or
        :class:`~rolland.excitation.RandomForce`).
    bound : CFSPML, optional
        Boundary instance whose boundary domains are shown. Required for a moving excitation.
    pad_prop : str, default='sp_z'
        Pad property represented by the color intensity of the pads.
    sleeper_prop : str, default='ms'
        Sleeper (or slab) property represented by the color intensity of the sleepers.
    ballast_prop : str, default='sb_z'
        Ballast property represented by the color intensity of the ballast.

    Example
    --------
    >>> from rolland.visualization import TrackSchematic

    >>> schematic = TrackSchematic(track=track)  # track: any track instance
    >>> ax = schematic.plot(x_range=(0, 10))

    Section that can be moved along the track (interactive backends only):

    >>> ax = schematic.plot(x_range=(0, 10), scroll=True)

    With excitation and boundary domains:

    >>> from rolland.boundary import CFSPML
    >>> from rolland.excitation import GaussianImpulse

    >>> schematic = TrackSchematic(track=track, excit=GaussianImpulse(x_excit=50.0), bound=CFSPML())
    >>> ax = schematic.plot()
    """

    track: Track
    excit: Excitation | None = None
    bound: CFSPML | None = None
    pad_prop: str = 'sp_z'
    sleeper_prop: str = 'ms'
    ballast_prop: str = 'sb_z'

    def __post_init__(self):
        """Check that the excitation can be shown."""
        if self.excit is None:
            return
        if not isinstance(self.excit, (GaussianImpulse, RandomForce)):
            msg = f'Showing an excitation of type {type(self.excit).__name__} is not supported.'
            raise TypeError(msg)
        if isinstance(self.excit, RandomForce) and self.bound is None:
            msg = 'A boundary (bound) is required to show the start position of a moving excitation.'
            raise ValueError(msg)

    def plot(self, ax=None, x_range=None, scroll=False):
        """Plot the schematic of the track.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Existing axes to plot on. If None, a new figure is created.
        x_range : tuple[float, float], optional
            Section of the track to be shown :math:`[m]`. If None, the whole track is shown,
            or a 10 m section if ``scroll`` is True.
        scroll : bool, default=False
            If True, the shown section can be moved along the track with a slider below the
            plot or with the mouse wheel. Requires an interactive matplotlib backend (e.g. a
            separate plot window or ``%matplotlib widget`` in Jupyter). Static outputs only
            show the initial section.

        Returns
        -------
        matplotlib.axes.Axes
            The axes containing the schematic.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            msg = 'matplotlib is required for plotting.'
            raise ImportError(msg) from e

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 4.5 if scroll else 4), layout='constrained')
            # Extra padding: the layout does not account for the frames of the labels.
            fig.get_layout_engine().set(w_pad=0.1)

        l_track = self.track.l_track
        if x_range is None:
            x_range = (0.0, min(_SCROLL_WINDOW, l_track)) if scroll else (0.0, l_track)

        content_below = self._draw_static(ax)
        window_artists = self._draw_window(ax, x_range)
        if scroll and x_range[1] - x_range[0] < l_track:
            self._add_scroll(ax, x_range, window_artists)

        z_top = _Z_TOP if self.excit is None else _Z_TOP_EXCIT
        if self.bound is not None:
            z_top += _Z_LABEL_ROW
        ax.set_xlim(x_range)
        ax.set_ylim(_Z_BOTTOM if content_below else _Z_GROUND - 0.15, z_top)
        ax.set_yticks([])
        ax.spines[['left', 'right', 'top']].set_visible(False)
        ax.set_xlabel('Track position [m]')
        return ax

    def _draw_static(self, ax) -> bool:
        """Draw everything that does not depend on the shown section.

        Returns whether content is drawn below the ground.
        """
        l_track = self.track.l_track
        is_slab = isinstance(self.track, SlabSingleRailTrack)
        layers = self._collect_layers()
        content_below = False

        if self.bound is not None:
            _draw_boundary(ax, l_track, self.bound.l_bound)
        _draw_ground(ax, l_track)
        for layer in layers:
            _draw_layer(ax, layer)
        if is_slab:
            _draw_rigid_slab(ax, l_track)
        positions = self._positions
        if positions is not None:
            label = 'Mounting distance' if is_slab else 'Sleeper spacing'
            content_below |= _draw_spacing_label(ax, positions, label)
        _draw_rail(ax, l_track)
        _draw_variation(ax, layers)
        if isinstance(self.excit, RandomForce):
            _draw_wheel(ax, self.bound.l_bound, self.excit.v)
        elif isinstance(self.excit, GaussianImpulse):
            _draw_force(ax, self.excit.x_excit, self.excit.force_dir)
        return content_below

    def _draw_window(self, ax, x_range) -> list:
        """Draw the elements that depend on the shown section and return them."""
        artists = []
        positions = self._positions
        if positions is not None:
            in_range = (positions >= x_range[0]) & (positions <= x_range[1])
            if np.count_nonzero(in_range) <= _MAX_DETAIL_MOUNTS:
                artists.append(ax.vlines(positions[in_range], 0, _Z_RAIL[0], colors='k', linestyles='dashed',
                                         linewidth=0.5, alpha=0.4))
                artists += _draw_spacing_dims(ax, positions, x_range)
        if self.bound is not None:
            artists += _label_boundary(ax, self.track.l_track, self.bound.l_bound, x_range)
        return artists

    def _add_scroll(self, ax, x_range, window_artists):
        """Add a slider and mouse wheel control that move the shown section along the track."""
        from matplotlib.widgets import Slider

        width = x_range[1] - x_range[0]
        slider = Slider(ax.inset_axes((0.0, -0.25, 1.0, 0.05)), 'Section', 0.0, self.track.l_track - width,
                        valinit=x_range[0], initcolor='none')

        def show(start):
            for artist in window_artists:
                artist.remove()
            window_artists[:] = self._draw_window(ax, (start, start + width))
            ax.set_xlim(start, start + width)
            slider.valtext.set_text(f'{start:.1f} … {start + width:.1f} m')
            ax.figure.canvas.draw_idle()

        def on_wheel(event):
            if event.inaxes is not ax:
                return
            step = _SCROLL_STEP * width if event.button == 'up' else -_SCROLL_STEP * width
            start = np.clip(slider.val + step, slider.valmin, slider.valmax)
            if start != slider.val:  # no redraw at the track ends
                slider.set_val(start)

        slider.valtext.set_text(f'{x_range[0]:.1f} … {x_range[1]:.1f} m')
        slider.on_changed(show)
        # The canvas keeps this callback, and with it the slider, alive.
        ax.figure.canvas.mpl_connect('scroll_event', on_wheel)

    @property
    def _positions(self):
        """Sorted mounting positions for discrete support, None for continuous support."""
        if isinstance(self.track, (DiscrSlabSingleRailTrack, DiscrBallastedSingleRailTrack)):
            return np.array(sorted(self.track.mount_prop))
        return None

    def _collect_layers(self) -> list[_Layer]:
        """Collect the support layers of the track from top to bottom."""
        track = self.track
        is_slab = isinstance(track, SlabSingleRailTrack)
        x = self._positions
        discrete = x is not None

        if discrete:
            pads, sleepers, ballasts = zip(*(track.mount_prop[pos] for pos in x), strict=True)
            # Equal blocks that do not touch, even at the smallest mounting distance.
            min_spacing = np.diff(x).min() if x.size > 1 else 1.0
            width = np.full(x.shape, _BLOCK_WIDTH * min_spacing)
            sleeper_label = 'Sleeper'
        else:
            # Continuous support: one block over the whole track length.
            x = np.array([track.l_track / 2])
            pads, sleepers, ballasts = [track.pad], [track.slab], [getattr(track, 'ballast', None)]
            width = np.array([track.l_track])
            sleeper_label = 'Slab'

        layers = [
            _Layer(label='Pad', color=_COLORS['pad'], z=_Z_PAD, prop=self.pad_prop,
                   unit=_unit(self.pad_prop, discrete), x=x, width=width, values=_values(pads, self.pad_prop)),
        ]
        if not is_slab:
            layers += [
                _Layer(label=sleeper_label, color=_COLORS['sleeper'], z=_Z_SLEEPER, prop=self.sleeper_prop,
                       unit=_unit(self.sleeper_prop, discrete), x=x, width=width,
                       values=_values(sleepers, self.sleeper_prop)),
                _Layer(label='Ballast', color=_COLORS['ballast'], z=_Z_BALLAST, prop=self.ballast_prop,
                       unit=_unit(self.ballast_prop, discrete), x=x, width=width,
                       values=_values(ballasts, self.ballast_prop)),
            ]
        return layers


def _values(components, prop) -> np.ndarray:
    """Return the values of a property for all components."""
    try:
        return np.array([getattr(component, prop) for component in components], dtype=float)
    except (AttributeError, TypeError) as e:
        msg = f"'{prop}' is not a numeric attribute of {type(components[0]).__name__}."
        raise ValueError(msg) from e


def _unit(prop, discrete) -> str:
    """Return the unit of a property for discrete or continuous support ('' if unknown)."""
    return _UNITS.get(prop, ('', ''))[0 if discrete else 1]


def _alphas(values) -> np.ndarray:
    """Map values linearly to color intensities (lowest value -> lightest)."""
    vmin, vmax = values.min(), values.max()
    if np.isclose(vmin, vmax):
        return np.full(values.shape, _ALPHA_CONST)
    return _ALPHA_RANGE[0] + (_ALPHA_RANGE[1] - _ALPHA_RANGE[0]) * (values - vmin) / (vmax - vmin)


def _tint(color, alpha) -> np.ndarray:
    """Return the opaque mix of a color with white (alpha=1: pure color)."""
    return np.array(to_rgb(color)) * alpha + (1 - alpha)


def _si_exponent(value, unit) -> int:
    """Return the exponent of the SI prefix for a value (no prefix below 1000 and for masses)."""
    if abs(value) < _PREFIX_MIN or unit.startswith('kg'):
        return 0
    return min(3 * int(np.log10(abs(value)) // 3), 30)


def _fmt(value, unit='') -> str:
    """Format a value with SI prefix and unit, e.g. '120 MN/m'."""
    exponent = _si_exponent(value, unit)
    return f'{value / 10**exponent:.4g} {EngFormatter.ENG_PREFIXES[exponent]}{unit}'.strip()


def _label(ax, z, text):
    """Add a boxed label right of the track at height z."""
    ax.text(1.01, z, text, transform=ax.get_yaxis_transform(), ha='left', va='center', fontsize='small',
            bbox={'boxstyle': 'square,pad=0.3', 'fc': 'white', 'ec': 'black', 'lw': 0.8}, clip_on=False)


def _boundary_domains(l_track, l_bound):
    """Return the boundary domains at both track ends as (start, end)."""
    return ((0.0, l_bound), (l_track - l_bound, l_track))


def _draw_boundary(ax, l_track, l_bound):
    """Shade the boundary domains at both track ends."""
    for x0, x1 in _boundary_domains(l_track, l_bound):
        ax.axvspan(x0, x1, facecolor='gray', alpha=0.15, edgecolor='none', zorder=0)
    for inner in (l_bound, l_track - l_bound):
        ax.axvline(inner, color='gray', linestyle='-.', linewidth=0.8, zorder=0)


def _label_boundary(ax, l_track, l_bound, x_range) -> list:
    """Label the visible parts of the boundary domains and return the labels."""
    labels = []
    for x0, x1 in _boundary_domains(l_track, l_bound):
        lo, hi = max(x0, x_range[0]), min(x1, x_range[1])
        if lo < hi:
            # Inside the axes (top edge), so that an axes title does not overlap.
            labels.append(ax.text((lo + hi) / 2, 0.98, f'PML ({l_bound:g} m)', transform=ax.get_xaxis_transform(),
                                  ha='center', va='top', fontsize='small'))
    return labels


def _draw_ground(ax, l_track):
    """Draw the hatched ground below the track."""
    ax.add_patch(Rectangle((0, _Z_GROUND), l_track, -_Z_GROUND, facecolor='none', edgecolor='k', hatch='////',
                           linewidth=0))
    ax.hlines(0, 0, l_track, colors='k', linewidth=1.5)


def _draw_layer(ax, layer):
    """Draw a support layer and its label."""
    height = layer.z[1] - layer.z[0]
    rects = [Rectangle((x - w / 2, layer.z[0]), w, height) for x, w in zip(layer.x, layer.width, strict=True)]
    colors = [_tint(layer.color, alpha) for alpha in _alphas(layer.values)]
    ax.add_collection(PatchCollection(rects, facecolors=colors, edgecolors='none'))

    # Varying values are shown by a colorbar, a constant value next to the name.
    text = layer.label if layer.varies else f'{layer.label}\n{layer.prop} = {_fmt(layer.values[0], layer.unit)}'
    _label(ax, sum(layer.z) / 2, text)


def _draw_rigid_slab(ax, l_track):
    """Draw the rigid slab of a slab track."""
    ax.add_patch(Rectangle((0, 0), l_track, _Z_SLEEPER[1], facecolor='lightgray', edgecolor='k', hatch='..',
                           linewidth=0.8))
    _label(ax, _Z_SLEEPER[1] / 2, 'Slab (rigid)')


def _draw_rail(ax, l_track):
    """Draw the rail (foot, web and head) over the whole track length."""
    gradient = np.linspace(0, 1, 64)[:, np.newaxis]
    for z_bottom, z_top, colors in _RAIL_PARTS:
        cmap = LinearSegmentedColormap.from_list('rail', colors)
        ax.imshow(gradient, cmap=cmap, alpha=0.7, origin='lower', extent=(0, l_track, z_bottom, z_top),
                  aspect='auto', interpolation='bilinear')
    ax.hlines(_Z_RAIL, 0, l_track, colors='k', linewidth=0.6)
    _label(ax, (_Z_RAIL[0] + _Z_RAIL[-1]) / 2, 'Rail')


def _draw_wheel(ax, x, v):
    """Draw the wheel resting on the rail at position x and its velocity."""
    r_flange = _WHEEL_PARTS[0][0] * _WHEEL_SCALE
    r_tread = _WHEEL_PARTS[1][0] * _WHEEL_SCALE
    area = DrawingArea(2 * r_flange, 2 * r_flange, clip=False)
    for radius, color in _WHEEL_PARTS:
        area.add_artist(Circle((r_flange, r_flange), radius * _WHEEL_SCALE, facecolor=color, edgecolor='k',
                               linewidth=0.5))
    # The tread (not the flange) rests on the rail head.
    alignment = (0.5, (r_flange - r_tread) / (2 * r_flange))
    wheel = AnnotationBbox(area, (x, _Z_RAIL[-1]), box_alignment=alignment, frameon=False, pad=0)
    # The wheel lies within the axes; outside the shown section it must not affect the layout.
    wheel.set_in_layout(False)
    ax.add_artist(wheel)
    ax.annotate(f'v = {v:g} m/s', xy=(x, _Z_RAIL[-1]), xytext=(r_flange + 6, r_tread), textcoords='offset points',
                ha='left', va='center', fontsize='small',
                bbox={'boxstyle': 'rarrow,pad=0.4', 'fc': '0.85', 'ec': 'none'})


def _draw_force(ax, x, direction):
    """Draw a vertical force arrow or a lateral force symbol on the rail head at position x."""
    if direction == 'lateral':
        # Lateral force acts perpendicular to the drawing plane.
        ax.annotate(r'$\odot$', xy=(x, _Z_RAIL[-1]), xytext=(0, 3), textcoords='offset points', ha='center',
                    va='bottom', fontsize='x-large')
        ax.annotate('$F_y$', xy=(x, _Z_RAIL[-1]), xytext=(10, 6), textcoords='offset points', ha='left',
                    va='bottom')
    else:
        ax.annotate('$F_z$', xy=(x, _Z_RAIL[-1]), xytext=(0, _FORCE_LENGTH), textcoords='offset points',
                    ha='center', va='bottom',
                    arrowprops={'arrowstyle': '-|>', 'color': 'k', 'lw': 1.5, 'shrinkA': 0, 'shrinkB': 0})


def _spacings_vary(spacings) -> bool:
    """Whether the spacings between the mounting positions differ."""
    return spacings.size > 0 and not np.isclose(spacings.min(), spacings.max())


def _draw_spacing_label(ax, positions, label) -> bool:
    """Label the spacing range if the spacings vary; return whether the label was drawn."""
    spacings = np.diff(positions)
    if not _spacings_vary(spacings):
        return False
    _label(ax, _Z_SPACING, f'{label}\n{spacings.min():.3g} … {spacings.max():.3g} m')
    return True


def _draw_spacing_dims(ax, positions, x_range) -> list:
    """Dimension the shortest and the longest visible spacing if they vary; return the artists."""
    spacings = np.diff(positions)
    artists = []
    if not _spacings_vary(spacings):
        return artists

    visible = np.flatnonzero((positions[:-1] >= x_range[0]) & (positions[1:] <= x_range[1]))
    for target in (spacings.min(), spacings.max()):
        hits = visible[np.isclose(spacings[visible], target)]
        if hits.size == 0:
            continue
        x0, x1 = positions[hits[0]], positions[hits[0] + 1]
        artists += [
            ax.vlines([x0, x1], _Z_SPACING - 0.15, _Z_GROUND, colors='k', linestyles='dotted', linewidth=0.8),
            ax.annotate('', xy=(x1, _Z_SPACING), xytext=(x0, _Z_SPACING),
                        arrowprops={'arrowstyle': '<->', 'shrinkA': 0, 'shrinkB': 0, 'lw': 0.8}),
            ax.text((x0 + x1) / 2, _Z_SPACING - 0.05, f'{target:.3g} m', ha='center', va='top', fontsize='small'),
        ]
    return artists


def _draw_variation(ax, layers):
    """Mark varying layers with a brace and add a colorbar for each of them."""
    for varies, group in groupby(layers, key=lambda layer: layer.varies):
        if varies:
            members = list(group)
            _draw_brace(ax, min(m.z[0] for m in members), max(m.z[1] for m in members))

    # Each new colorbar is placed between the plot and the previous ones; reversed order
    # puts the colorbars in the order of the layers (top layer next to the plot).
    for layer in reversed(layers):
        if layer.varies:
            _draw_colorbar(ax, layer)


def _draw_brace(ax, z_bottom, z_top):
    """Draw a curly brace with the label 'Variation' left of the track."""
    x_open, depth = -0.01, 0.02  # in axes coordinates
    verts = [(x_open - d * depth, z_bottom + s * (z_top - z_bottom)) for d, s in _BRACE_VERTS]
    ax.add_patch(PathPatch(Path(verts, _BRACE_CODES), transform=ax.get_yaxis_transform(), fill=False, lw=1,
                           clip_on=False))
    ax.text(x_open - 1.2 * depth, (z_bottom + z_top) / 2, 'Variation', transform=ax.get_yaxis_transform(),
            ha='right', va='center')


def _draw_colorbar(ax, layer):
    """Add a vertical colorbar right of the plot showing the values of a varying layer."""
    vmin, vmax = layer.values.min(), layer.values.max()
    # Same colors as the blocks of the layer.
    cmap = LinearSegmentedColormap.from_list('layer', [_tint(layer.color, alpha) for alpha in _ALPHA_RANGE])
    cbar = ax.figure.colorbar(ScalarMappable(norm=Normalize(vmin, vmax), cmap=cmap), ax=ax, location='right',
                              shrink=0.8, aspect=15, pad=0.02)

    distinct = np.unique(layer.values)
    if distinct.size <= _MAX_DISTINCT_TICKS:
        cbar.set_ticks(distinct)
    exponent = _si_exponent(max(abs(vmin), abs(vmax)), layer.unit)
    cbar.formatter = FuncFormatter(lambda value, _: f'{value / 10**exponent:.4g}')
    unit = EngFormatter.ENG_PREFIXES[exponent] + layer.unit
    cbar.set_label(f'{layer.label}: {layer.prop}' + (f' [{unit}]' if unit else ''), fontsize='small')
    cbar.ax.tick_params(labelsize='small')

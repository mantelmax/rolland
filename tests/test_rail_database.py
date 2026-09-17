"""Tests for the rail profile database.

Every profile shipped in ``rolland/database/rail/profiles`` is loaded and checked
automatically, so adding a profile needs no change to this file. The rail_60E1 test
pins its values so that unintended changes to the profile file are noticed.
"""

import shutil

import pytest

from rolland.components import Rail
from rolland.database.rail import db_rail
from rolland.database.rail.db_rail import PROFILE_DIR, available_rails, load_rail

# Reference values of rail_60E1 (formerly UIC60).
RAIL_60E1_REFERENCE = {
    'E': 210e9,
    'G': 80.769e9,
    'nu': 0.3,
    'kapz': 0.39318,
    'kapy': 0.55044,
    'mr': 60.2,
    'rho': 7860,
    'etar': 0.02,
    'dr': 1000,
    'shearc': [0.0, 34.6e-3],
    'centr': [0.0, 0],
    'Iyr': 3.03813e-05,
    'Izr': 5.11963e-06,
    'Iyz': 0.0,
    'Ipr': 3.55009e-05,
    'Ar': 76.71e-4,
    'Asr': 0.688,
    'Vr': 7670.00e-6,
    'kapp_s': 1,
    'Iw': 2.20949e-08,
    'Iwz': 1.77310e-07,
    'Iwy': 0.0,
    'k_w': 0.5302,
    'J': 2.21072e-06,
    'chi': 0.0,
}


def _polygon_area(points):
    """Return the signed area of the closed outline (positive when counter-clockwise)."""
    n = len(points)
    return 0.5 * sum(
        points[i][0] * points[(i + 1) % n][1] - points[(i + 1) % n][0] * points[i][1] for i in range(n)
    )


def _polygon_centroid(points):
    """Return the area centroid (y, z) of the closed outline."""
    n = len(points)
    area = _polygon_area(points)
    centre_y = centre_z = 0.0
    for i in range(n):
        y0, z0 = points[i]
        y1, z1 = points[(i + 1) % n]
        cross = y0 * z1 - y1 * z0
        centre_y += (y0 + y1) * cross
        centre_z += (z0 + z1) * cross
    return centre_y / (6.0 * area), centre_z / (6.0 * area)


def _width_near(points, level, tolerance):
    """Return the lateral extent of the outline within `tolerance` of the given z."""
    lateral = [y for y, z in points if abs(z - level) <= tolerance]
    return max(lateral) - min(lateral)


def _width_at_min_z(points):
    """Return the lateral extent at the minimum-z end of the outline."""
    levels = [z for _, z in points]
    return _width_near(points, min(levels), 0.02 * (max(levels) - min(levels)))


def _width_at_max_z(points):
    """Return the lateral extent at the maximum-z end of the outline."""
    levels = [z for _, z in points]
    return _width_near(points, max(levels), 0.02 * (max(levels) - min(levels)))


@pytest.fixture
def profile_copy(tmp_path):
    """Return a factory for a user-owned copy of the rail_60E1 profile."""
    def _make(name='MY60', extra=''):
        shutil.copy(PROFILE_DIR / 'rail_60E1.csv', tmp_path / 'rail_60E1.csv')
        text = (PROFILE_DIR / 'rail_60E1.toml').read_text()
        path = tmp_path / f'{name}.toml'
        path.write_text(text.replace('name = "rail_60E1"', f'name = "{name}"') + extra)
        return path
    return _make


def test_profiles_are_discovered():
    """The bundled profile directory is found and is not empty."""
    assert 'rail_60E1' in available_rails()


@pytest.mark.parametrize('name', available_rails())
def test_profile_loads(name):
    """
    Every bundled profile parses into a physically plausible Rail instance.

    This is the guard for new profiles: a typo, a missing parameter or a broken
    outline file fails here without anyone writing a dedicated test.
    """
    rail = load_rail(name)

    assert isinstance(rail, Rail)
    assert len(rail.rl_geo) > 2
    assert all(len(point) == 2 for point in rail.rl_geo)  # noqa: PLR2004
    assert rail.mr > 0
    assert rail.Ar > 0
    assert rail.E > 0


def test_rail_60e1_matches_hard_coded_values():
    """rail_60E1 holds exactly the reference values."""
    rail = load_rail('rail_60E1')

    for attribute, expected in RAIL_60E1_REFERENCE.items():
        assert getattr(rail, attribute) == expected, attribute


def test_rail_60e1_outline():
    """The outline is read from the CSV file with the header skipped."""
    rail = load_rail('rail_60E1')

    assert len(rail.rl_geo) == 1000  # noqa: PLR2004
    assert rail.rl_geo[0] == (0.07299975666, 0.08091240057)


@pytest.mark.parametrize('name', available_rails())
def test_outline_conventions(name):
    """
    Every outline follows the database conventions.

    The outline is a closed polygon in metres with a positive signed area and the
    area centroid at the origin, so that ``centr = [0.0, 0.0]`` in the profile file
    is the truth. Its enclosed area must agree with the tabulated cross-section.

    The z-axis points downwards, matching the sign convention of the publication,
    so the wide rail foot sits at maximum z and the narrow railhead at minimum z.
    """
    rail = load_rail(name)
    outline = rail.rl_geo

    assert outline[0] != outline[-1], 'the closing point must not be repeated'

    signed_area = _polygon_area(outline)
    assert signed_area > 0, 'outline must have a positive signed area'
    assert signed_area == pytest.approx(rail.Ar, rel=0.01), 'outline area disagrees with Ar'

    centre_y, centre_z = _polygon_centroid(outline)
    assert centre_y == pytest.approx(0.0, abs=1e-9), 'outline is not centred on its centroid'
    assert centre_z == pytest.approx(0.0, abs=1e-9), 'outline is not centred on its centroid'

    assert _width_at_min_z(outline) < _width_at_max_z(outline), (
        'z must point downwards: the rail foot is the wider end and belongs at maximum z'
    )


def test_derived_attributes_are_calculated():
    """Rail.__post_init__ runs on loaded profiles, not just on manual instances."""
    rail = load_rail('rail_60E1')

    assert rail.ez == pytest.approx(rail.shearc[1] - rail.centr[1])
    assert rail.ey == pytest.approx(rail.shearc[0] - rail.centr[0])
    assert rail.J_rs == pytest.approx(rail.kapp_s * (rail.Ipr - rail.J))


def test_module_attribute_access():
    """``from ... import rail_60E1`` works and returns a cached instance."""
    assert isinstance(db_rail.rail_60E1, Rail)
    assert db_rail.rail_60E1 is db_rail.rail_60E1
    assert 'rail_60E1' in dir(db_rail)


def test_unknown_attribute_raises():
    """A misspelled profile attribute raises AttributeError, not something odd."""
    with pytest.raises(AttributeError, match='rail_61E1'):
        _ = db_rail.rail_61E1


def test_unknown_name_lists_alternatives():
    """An unknown profile name points the user at what is available."""
    with pytest.raises(ValueError, match='Available profiles.*rail_60E1'):
        load_rail('rail_61E1')


def test_load_from_user_path(profile_copy):
    """A profile file outside the package loads by path, giving users their own profiles."""
    rail = load_rail(profile_copy())

    assert isinstance(rail, Rail)
    assert rail.mr == RAIL_60E1_REFERENCE['mr']


def test_load_from_user_path_as_string(profile_copy):
    """A path given as a string is recognised by its .toml suffix."""
    assert isinstance(load_rail(str(profile_copy())), Rail)


def test_missing_file_raises(tmp_path):
    """A path that does not exist fails with a clear FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match='profile file not found'):
        load_rail(tmp_path / 'nope.toml')


def test_unknown_parameter_is_rejected(profile_copy):
    """A parameter that Rail does not accept is reported by name."""
    path = profile_copy(extra='\n[extra]\nbogus = 1.0\n')

    with pytest.raises(ValueError, match='unknown parameter'):
        load_rail(path)


def test_duplicate_parameter_is_rejected(profile_copy):
    """The same parameter in two tables is an error rather than a silent overwrite."""
    path = profile_copy(extra='\n[extra]\nmr = 54.0\n')

    with pytest.raises(ValueError, match='defined more than once'):
        load_rail(path)


def test_name_mismatch_is_rejected(profile_copy, tmp_path):
    """A [meta] name that disagrees with the file name is caught early."""
    path = profile_copy(name='MY60').rename(tmp_path / 'OTHER.toml')

    with pytest.raises(ValueError, match='must match the file name'):
        load_rail(path)


def test_missing_outline_is_rejected(tmp_path):
    """A profile without an outline reference is rejected."""
    path = tmp_path / 'MINIMAL.toml'
    path.write_text('[meta]\nname = "MINIMAL"\n\n[geometry]\nmr = 60.0\n')

    with pytest.raises(ValueError, match='outline'):
        load_rail(path)


def test_missing_required_parameter_is_rejected(profile_copy):
    """Leaving out a mandatory Rail parameter is reported by name."""
    path = profile_copy()
    path.write_text(path.read_text().replace('mr = 60.2', ''))

    with pytest.raises(ValueError, match='missing required parameter.*mr'):
        load_rail(path)

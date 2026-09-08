"""Tests for the rail profile database.

Every profile shipped in ``rolland/database/rail/profiles`` is loaded and checked
automatically, so adding a profile needs no change to this file. The UIC60 test
pins the values against the hard-coded instance that preceded the TOML files.
"""

import shutil

import pytest

from rolland.components import Rail
from rolland.database.rail import db_rail
from rolland.database.rail.db_rail import PROFILE_DIR, available_rails, load_rail

# UIC60 values as they were hard-coded in db_rail.py before the migration.
UIC60_REFERENCE = {
    'E': 210e9,
    'G': 80.769e9,
    'nu': 0.3,
    'kapz': 0.393,
    'kapy': 0.538,
    'mr': 60.2,
    'rho': 7860,
    'etar': 0.02,
    'dr': 1000,
    'shearc': [0.0, 33e-3],
    'centr': [0.0, 0],
    'Iyr': 3.037e-05,
    'Izr': 5.127e-06,
    'Iyz': 0.0,
    'Ipr': 3.55e-05,
    'Ar': 76.70e-4,
    'Asr': 0.688,
    'Vr': 7670.00e-6,
    'kapp_s': 1,
    'Iw': 2.161e-8,
    'Iwz': 1.6971e-7,
    'Iwy': 0.0,
    'k_w': -0.6016,
    'J': 2.212e-6,
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


@pytest.fixture
def profile_copy(tmp_path):
    """Return a factory for a user-owned copy of the UIC60 profile."""
    def _make(name='MY60', extra=''):
        shutil.copy(PROFILE_DIR / 'UIC60.csv', tmp_path / 'UIC60.csv')
        text = (PROFILE_DIR / 'UIC60.toml').read_text()
        path = tmp_path / f'{name}.toml'
        path.write_text(text.replace('name = "UIC60"', f'name = "{name}"') + extra)
        return path
    return _make


def test_profiles_are_discovered():
    """The bundled profile directory is found and is not empty."""
    assert 'UIC60' in available_rails()


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


def test_uic60_matches_hard_coded_values():
    """UIC60 still holds exactly the values it had before the TOML migration."""
    rail = load_rail('UIC60')

    for attribute, expected in UIC60_REFERENCE.items():
        assert getattr(rail, attribute) == expected, attribute


def test_uic60_outline():
    """The outline is read from the CSV file with the header skipped."""
    rail = load_rail('UIC60')

    assert len(rail.rl_geo) == 1000  # noqa: PLR2004
    assert rail.rl_geo[0] == (0.07335824865, -0.08088000914)


@pytest.mark.parametrize('name', available_rails())
def test_outline_conventions(name):
    """
    Every outline follows the database conventions.

    The outline is a closed polygon in metres, running counter-clockwise, with the
    area centroid at the origin so that ``centr = [0.0, 0.0]`` in the profile file
    is the truth. Its enclosed area must agree with the tabulated cross-section.
    """
    rail = load_rail(name)
    outline = rail.rl_geo

    assert outline[0] != outline[-1], 'the closing point must not be repeated'

    signed_area = _polygon_area(outline)
    assert signed_area > 0, 'outline must run counter-clockwise'
    assert signed_area == pytest.approx(rail.Ar, rel=0.01), 'outline area disagrees with Ar'

    centre_y, centre_z = _polygon_centroid(outline)
    assert centre_y == pytest.approx(0.0, abs=1e-9), 'outline is not centred on its centroid'
    assert centre_z == pytest.approx(0.0, abs=1e-9), 'outline is not centred on its centroid'


def test_derived_attributes_are_calculated():
    """Rail.__post_init__ runs on loaded profiles, not just on manual instances."""
    rail = load_rail('UIC60')

    assert rail.ez == pytest.approx(rail.shearc[1] - rail.centr[1])
    assert rail.ey == pytest.approx(rail.shearc[0] - rail.centr[0])
    assert rail.J_rs == pytest.approx(rail.kapp_s * (rail.Ipr - rail.J))


def test_module_attribute_access():
    """``from ... import UIC60`` keeps working and returns a cached instance."""
    assert isinstance(db_rail.UIC60, Rail)
    assert db_rail.UIC60 is db_rail.UIC60
    assert 'UIC60' in dir(db_rail)


def test_unknown_attribute_raises():
    """A misspelled profile attribute raises AttributeError, not something odd."""
    with pytest.raises(AttributeError, match='UIC61'):
        _ = db_rail.UIC61


def test_unknown_name_lists_alternatives():
    """An unknown profile name points the user at what is available."""
    with pytest.raises(ValueError, match='Available profiles.*UIC60'):
        load_rail('UIC61')


def test_load_from_user_path(profile_copy):
    """A profile file outside the package loads by path, giving users their own profiles."""
    rail = load_rail(profile_copy())

    assert isinstance(rail, Rail)
    assert rail.mr == UIC60_REFERENCE['mr']


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

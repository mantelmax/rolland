"""Rail profile database.

Rail profiles are stored as data: every profile consists of a TOML
file holding the scalar parameters and a CSV file holding the outline
coordinates, both located in the ``profiles`` directory next to this module.
This module turns them into :class:`~rolland.components.Rail` instances.

Bundled profiles are exposed as module attributes and are loaded on first
access, so the usual import keeps working::

    >>> from rolland.database.rail.db_rail import UIC60

Equivalently, and without hard-coding the name::

    >>> from rolland.database.rail.db_rail import available_rails, load_rail
    >>> load_rail('UIC60')  # doctest: +ELLIPSIS
    Rail(...)

Own profiles are loaded from anywhere by passing a path instead of a name::

    >>> load_rail('~/my_profiles/60E2.toml')  # doctest: +SKIP

File format
-----------
Each ``<name>.toml`` file groups its values into tables. Every table except
``[meta]`` contributes its keys directly as arguments to
:class:`~rolland.components.Rail`, so the grouping is purely documentation.
The special key ``outline`` names a CSV file (columns ``Y,Z``, in metres) next
to the TOML file, which is read into ``Rail.rl_geo``.

The ``[meta]`` table describes the profile itself and is not passed on. Its
``name`` entry must match the file name.

See ``profiles/UIC60.toml`` for a documented example.
"""

import csv
import tomllib
from dataclasses import MISSING, fields
from pathlib import Path

from rolland.components import Rail

PROFILE_DIR = Path(__file__).parent / 'profiles'

_SUFFIX = '.toml'
_META_TABLE = 'meta'
_OUTLINE_KEY = 'outline'


def available_rails() -> list[str]:
    """Return the names of all rail profiles bundled with rolland.

    Returns
    -------
    list[str]
        Profile names in alphabetical order, suitable as an argument to
        :func:`load_rail`.
    """
    return sorted(path.name.removesuffix(_SUFFIX) for path in PROFILE_DIR.glob(f'*{_SUFFIX}'))


def load_rail_geo(file_path: str | Path) -> list[tuple[float, float]]:
    """Load rail outline coordinates from a CSV file.

    Parameters
    ----------
    file_path : str or pathlib.Path
        CSV file with a header line followed by two columns ``Y,Z`` in metres.

    Returns
    -------
    list[tuple[float, float]]
        Outline coordinates :math:`(y, z)` in metres.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If a line does not hold two numbers.
    """
    path = Path(file_path).expanduser()
    if not path.is_file():
        msg = f'Rail outline file not found: {path}'
        raise FileNotFoundError(msg)

    outline = []
    with path.open(newline='') as stream:
        reader = csv.reader(stream)
        next(reader, None)  # skip header
        for number, row in enumerate(reader, start=2):
            try:
                outline.append((float(row[0]), float(row[1])))
            except (IndexError, ValueError) as exc:
                msg = f'{path.name}, line {number}: expected two numbers, got {row!r}.'
                raise ValueError(msg) from exc

    if not outline:
        msg = f'{path.name}: no outline coordinates found.'
        raise ValueError(msg)
    return outline


def load_rail(profile: str | Path) -> Rail:
    """Build a :class:`~rolland.components.Rail` instance from a profile file.

    Parameters
    ----------
    profile : str or pathlib.Path
        Either the name of a bundled profile, as listed by
        :func:`available_rails`, or the path to a TOML profile file. Anything
        ending in ``.toml`` is treated as a path.

    Returns
    -------
    rolland.components.Rail
        The rail described by the profile.

    Raises
    ------
    FileNotFoundError
        If a profile path or its outline file does not exist.
    ValueError
        If the profile name is unknown or the profile file is malformed.

    Examples
    --------
    >>> from rolland.database.rail.db_rail import load_rail
    >>> rail = load_rail('UIC60')
    >>> rail.mr
    60.2
    """
    path = _resolve(profile)
    with path.open('rb') as stream:
        document = tomllib.load(stream)

    _check_meta(document, path)
    parameters = _collect_parameters(document, path)

    outline = parameters.pop(_OUTLINE_KEY, None)
    if outline is None:
        msg = f'{path.name}: missing {_OUTLINE_KEY!r} key naming the outline CSV file.'
        raise ValueError(msg)
    parameters['rl_geo'] = load_rail_geo(path.parent / outline)

    _check_parameters(parameters, path)
    return Rail(**parameters)


def _resolve(profile: str | Path) -> Path:
    """Resolve a profile name or path to an existing TOML file."""
    if isinstance(profile, Path) or str(profile).endswith(_SUFFIX):
        path = Path(profile).expanduser()
        if not path.is_file():
            msg = f'Rail profile file not found: {path}'
            raise FileNotFoundError(msg)
        return path

    path = PROFILE_DIR / f'{profile}{_SUFFIX}'
    if not path.is_file():
        msg = (
            f'Unknown rail profile {str(profile)!r}. Available profiles: '
            f'{", ".join(available_rails())}. To load a profile from a file, '
            f'pass a path ending in {_SUFFIX!r}.'
        )
        raise ValueError(msg)
    return path


def _check_meta(document: dict, path: Path) -> None:
    """Verify that the ``[meta]`` table exists and names the profile correctly."""
    meta = document.get(_META_TABLE)
    if not isinstance(meta, dict):
        msg = f'{path.name}: missing [{_META_TABLE}] table.'
        raise ValueError(msg)

    expected = path.name.removesuffix(_SUFFIX)
    if meta.get('name') != expected:
        msg = f'{path.name}: [{_META_TABLE}] name is {meta.get("name")!r} but must match the file name {expected!r}.'
        raise ValueError(msg)


def _collect_parameters(document: dict, path: Path) -> dict:
    """Flatten all tables except ``[meta]`` into a single argument dictionary."""
    parameters: dict = {}
    for table, entries in document.items():
        if table == _META_TABLE:
            continue
        if not isinstance(entries, dict):
            msg = f'{path.name}: {table!r} must be placed inside a table, e.g. [material].'
            raise ValueError(msg)
        for key, value in entries.items():
            if key in parameters:
                msg = f'{path.name}: parameter {key!r} is defined more than once.'
                raise ValueError(msg)
            parameters[key] = value
    return parameters


def _check_parameters(parameters: dict, path: Path) -> None:
    """Check the collected parameters against the ``Rail`` signature."""
    required = {
        item.name
        for item in fields(Rail)
        if item.init and item.default is MISSING and item.default_factory is MISSING
    }
    known = {item.name for item in fields(Rail) if item.init}

    unknown = sorted(set(parameters) - known)
    if unknown:
        msg = f'{path.name}: unknown parameter(s) {", ".join(unknown)}; Rail has no such attribute.'
        raise ValueError(msg)

    missing = sorted(required - set(parameters))
    if missing:
        msg = f'{path.name}: missing required parameter(s) {", ".join(missing)}.'
        raise ValueError(msg)


def __getattr__(name: str) -> Rail:
    """Load a bundled profile on first attribute access and cache it."""
    if not name.startswith('_') and name in available_rails():
        rail = load_rail(name)
        globals()[name] = rail  # subsequent lookups bypass __getattr__
        return rail

    msg = f'module {__name__!r} has no attribute {name!r}'
    raise AttributeError(msg)


def __dir__() -> list[str]:
    """List module contents including the lazily loaded profiles."""
    return sorted({*globals(), *available_rails()})

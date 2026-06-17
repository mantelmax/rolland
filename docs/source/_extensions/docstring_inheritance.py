"""
docstring_inheritance.py
========================
Sphinx extension that automatically merges ``Attributes`` from parent class
docstrings into child class documentation.

Only *new* attributes need to be documented in each class. Sphinx renders
child docs as if every inherited attribute were described there directly.

How it works
------------
Napoleon converts NumPy ``Attributes`` sections into ``.. attribute::``
directives *before* the ``autodoc-process-docstring`` event fires.
This extension therefore:

1. Reads the **raw** docstrings of all ancestor classes via ``inspect.getdoc``
   and converts their NumPy ``Attributes`` sections to ``.. attribute::``
   directives itself.
2. Filters out any attributes already documented by the child.
3. Inserts the remaining inherited attributes *before* the child's own
   ``.. attribute::`` blocks so the final order is: ancestors first, child last.

Supports
--------
- NumPy-style docstrings
- Multi-level inheritance (full MRO is walked)
- ``@dataclass`` classes
- Classes with no ``Attributes`` section of their own
"""
from __future__ import annotations

import inspect

_UNDERLINE_CHARS = frozenset("-=~^")


def _is_underline(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and set(stripped) <= _UNDERLINE_CHARS


def _extract_rst_attributes(docstring: str) -> list[str]:
    """Parse a raw NumPy docstring and return ``.. attribute::`` RST blocks.

    Parameters
    ----------
    docstring : str
        Raw docstring as returned by ``inspect.getdoc``.

    Returns
    -------
    list[str]
        Lines ready to splice into a Sphinx ``autodoc-process-docstring``
        ``lines`` list.  Trailing blank lines are stripped.
    """
    lines = docstring.splitlines()
    result: list[str] = []
    in_section = False
    skip_underline = False

    current_name: str | None = None
    current_type: str | None = None
    current_desc: list[str] = []

    def _flush() -> None:
        nonlocal current_name, current_type, current_desc
        if current_name is None:
            return
        result.append(f".. attribute:: {current_name}")
        result.append("")
        for dl in current_desc:
            result.append(f"   {dl}" if dl else "")
        if current_type:
            result.append("")
            result.append(f"   :type: {current_type}")
        result.append("")
        current_name = None
        current_type = None
        current_desc = []

    for i, line in enumerate(lines):
        stripped = line.strip()

        if not in_section:
            if stripped in ("Attributes", "Attributes:"):
                ni = i + 1
                if ni < len(lines) and _is_underline(lines[ni]):
                    in_section = True
                    skip_underline = True
            continue

        if skip_underline:
            skip_underline = False
            continue

        # Stop at the next RST section header
        if stripped and i + 1 < len(lines) and _is_underline(lines[i + 1]):
            _flush()
            break

        # New attribute entry: non-indented, non-empty line
        if line and not line[0].isspace() and stripped:
            _flush()
            if " : " in stripped:
                name, typ = stripped.split(" : ", 1)
                current_name = name.strip()
                current_type = typ.strip()
            else:
                current_name = stripped
                current_type = None
            current_desc = []
        elif current_name is not None:
            current_desc.append(stripped)

    _flush()

    while result and not result[-1].strip():
        result.pop()

    return result


def _get_existing_attr_names(lines: list[str]) -> set[str]:
    """Return the set of attribute names already in *lines*."""
    names: set[str] = set()
    for line in lines:
        if line.strip().startswith(".. attribute::"):
            names.add(line.strip()[len(".. attribute::"):].strip())
    return names


def _find_first_attribute_pos(lines: list[str]) -> int | None:
    """Return index of the first ``.. attribute::`` line, or ``None``."""
    for i, line in enumerate(lines):
        if line.strip().startswith(".. attribute::"):
            return i
    return None


def _merge_attributes(app, what, name, obj, options, lines) -> None:  # noqa: ANN001
    """``autodoc-process-docstring`` event handler."""
    if what != "class":
        return

    # Walk MRO, collect RST attribute blocks from every ancestor
    parent_rst: list[str] = []
    for parent in inspect.getmro(obj)[1:]:
        if parent is object:
            continue
        raw = inspect.getdoc(parent) or ""
        attrs = _extract_rst_attributes(raw)
        parent_rst = attrs + parent_rst  # distant ancestors first

    if not parent_rst:
        return

    # Drop attributes the child already documents
    existing = _get_existing_attr_names(lines)
    to_insert: list[str] = []
    skip_block = False

    for i, line in enumerate(parent_rst):
        if line.strip().startswith(".. attribute::"):
            attr_name = line.strip()[len(".. attribute::"):].strip()
            skip_block = attr_name in existing
        if not skip_block:
            to_insert.append(line)
        # Reset at block boundary (blank line before next directive)
        if (not line.strip()
                and i + 1 < len(parent_rst)
                and parent_rst[i + 1].strip().startswith(".. attribute::")):
            skip_block = False

    if not to_insert:
        return

    insert_pos = _find_first_attribute_pos(lines)

    if insert_pos is None:
        # No attributes yet → append
        lines += [""] + to_insert
    else:
        # Ensure the inserted block ends with a blank line separator
        if to_insert[-1] != "":
            to_insert.append("")
        for j, line in enumerate(to_insert):
            lines.insert(insert_pos + j, line)


def setup(app):  # noqa: ANN001
    """Register the extension with Sphinx."""
    app.connect("autodoc-process-docstring", _merge_attributes)
    return {
        "version": "0.2",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }

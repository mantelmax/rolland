"""
docstring_inheritance.py
========================
Sphinx extension that automatically merges ``Attributes`` sections from
parent classes into child class docstrings.

Only *new* attributes need to be documented in each class; Sphinx renders
child docs as if every inherited attribute were described there directly.

Supports:
- NumPy-style docstrings (``Attributes\n----------``)
- Multi-level inheritance (MRO is fully walked)
- Classes with no ``Attributes`` section of their own
"""
from __future__ import annotations

import inspect

# Characters accepted as RST section underlines
_UNDERLINE_CHARS = frozenset("-=~^")


def _is_underline(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and set(stripped) <= _UNDERLINE_CHARS


def _extract_attributes_section(docstring: str) -> list[str]:
    """Return the *content* lines of the ``Attributes`` block.

    The section header (``Attributes``) and its underline are **not** included
    in the returned list so the caller can splice the content directly after
    the target header without duplicating the underline.

    Parameters
    ----------
    docstring : str
        A normalised docstring (as returned by ``inspect.getdoc``).

    Returns
    -------
    list[str]
        Lines between the ``Attributes`` header and the next section header
        (or end-of-string), with trailing blank lines removed.
    """
    lines = docstring.splitlines()
    result: list[str] = []
    in_section = False
    skip_next = False  # flag to discard the underline right after the header

    for i, line in enumerate(lines):
        stripped = line.strip()

        if not in_section:
            # Detect "Attributes" or "Attributes:" followed by an underline
            if stripped in ("Attributes", "Attributes:"):
                next_i = i + 1
                if next_i < len(lines) and _is_underline(lines[next_i]):
                    in_section = True
                    skip_next = True
            continue

        # Discard the underline line that directly follows the header
        if skip_next:
            skip_next = False
            continue

        # Stop when reaching the next RST section (non-empty line + underline)
        if (stripped
                and i + 1 < len(lines)
                and _is_underline(lines[i + 1])):
            break

        result.append(line)

    # Remove trailing blank lines
    while result and not result[-1].strip():
        result.pop()

    return result


def _find_section_start(lines: list[str], section: str) -> int | None:
    """Return the index of *section*'s header line, or ``None``.

    Parameters
    ----------
    lines : list[str]
        The docstring split into lines (as Sphinx provides it).
    section : str
        Section name to look for (e.g. ``"Attributes"``).

    Returns
    -------
    int or None
        Zero-based index of the section header, or ``None`` when absent.
    """
    for i, line in enumerate(lines):
        if line.strip() in (section, section + ":"):
            if i + 1 < len(lines) and _is_underline(lines[i + 1]):
                return i
    return None


def _merge_attributes(app, what, name, obj, options, lines):  # noqa: ANN001
    """``autodoc-process-docstring`` event handler.

    Walks the MRO of *obj* (classes only), collects ``Attributes`` content
    from every ancestor, and injects it into the child's docstring lines
    **before** Sphinx renders them.  The closest ancestor's attributes appear
    directly above the child's own attributes (MRO order).
    """
    if what != "class":
        return

    # Collect parent attribute lines in MRO order (skip ``object``)
    parent_attr_lines: list[str] = []
    for parent in inspect.getmro(obj)[1:]:
        if parent is object:
            continue
        parent_doc = inspect.getdoc(parent) or ""
        parent_attrs = _extract_attributes_section(parent_doc)
        # Prepend so that more-distant ancestors come first
        parent_attr_lines = parent_attrs + parent_attr_lines

    if not parent_attr_lines:
        return

    attr_start = _find_section_start(lines, "Attributes")

    if attr_start is None:
        # Child has no Attributes section → append a new one
        lines += ["", "Attributes", "----------"] + parent_attr_lines
    else:
        # Insert inherited attrs right after the section header + underline
        insert_pos = attr_start + 2
        for i, line in enumerate(parent_attr_lines):
            lines.insert(insert_pos + i, line)


def setup(app):  # noqa: ANN001
    """Register the extension with Sphinx."""
    app.connect("autodoc-process-docstring", _merge_attributes)
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }

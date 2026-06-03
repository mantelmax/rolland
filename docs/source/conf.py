# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath('../..'))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Rolland'
copyright = '2025, Maximilian Mantel, Ennes Sarradj'
author = 'Maximilian Mantel, Ennes Sarradj'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',             # For automatic doc generation from docstrings
    'sphinx.ext.napoleon',            # Supports NumPy and Google style docstrings
    'sphinx.ext.viewcode',            # Links to source code
    'sphinx.ext.mathjax',             # For LaTeX math rendering
    'sphinxcontrib.bibtex',           # Citation support
    'sphinx.ext.autosummary',         # Generate autodoc summaries
    'autodoc_traits',                 # Support for traitlets
    'sphinx_design',                  # Design extension
    'myst_parser',                    # Markdown support
    'sphinx_docsearch',                # Docsearch extension
]

# sphinxcontrib-bibtex extension settings
# ---------------------------------------
bibtex_bibfiles = ["literature/literature.bib"]
bibtex_default_style = 'unsrt'

templates_path = ['_templates']
exclude_patterns = []

autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'show-inheritance': True,
}

autodoc_typehints = "description"       # Include type hints in the description
napoleon_google_docstring = False       # Use NumPy-style docstrings
napoleon_numpy_docstring = True

# Select a color scheme for light mode
pygments_style = "default"
# Select a different color scheme for dark mode
pygments_style_dark = "github-dark"

# docsearch settings
docsearch_app_id = "TFRLLVQ6L2"
docsearch_api_key = "94b06103f2d9222f8335aad9da275035"
docsearch_index_name = "rolland-rolling-noise-and-dynamics"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinxawesome_theme'
html_permalinks_icon = "<span>¶</span>"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_title = "Rolland Documentation"
html_last_updated_fmt = ""
html_logo = "images/logo_rolland_dark.svg"
html_favicon = "images/logo_rolland_dark.svg"


from dataclasses import asdict

from sphinxawesome_theme import ThemeOptions

theme_options = ThemeOptions(

    # Add your theme options. For example:
    main_nav_links={"About": "/about"},
    show_scrolltop=True,
    show_prev_next=False,
    show_breadcrumbs=True,
    awesome_headerlinks=False,
    extra_header_link_icons={
        "repository on GitHub": {
            "link": "https://github.com/mantelmax/rolland",
            "icon": (
                '<svg height="26px" style="margin-top:-2px;display:inline" '
                'viewBox="0 0 45 44" '
                'fill="currentColor" xmlns="http://www.w3.org/2000/svg">'
                '<path fill-rule="evenodd" clip-rule="evenodd" '
                'd="M22.477.927C10.485.927.76 10.65.76 22.647c0 9.596 6.223 17.736 '
                "14.853 20.608 1.087.2 1.483-.47 1.483-1.047 "
                "0-.516-.019-1.881-.03-3.693-6.04 "
                "1.312-7.315-2.912-7.315-2.912-.988-2.51-2.412-3.178-2.412-3.178-1.972-1.346.149-1.32.149-1.32 "  # noqa
                "2.18.154 3.327 2.24 3.327 2.24 1.937 3.318 5.084 2.36 6.321 "
                "1.803.197-1.403.759-2.36 "
                "1.379-2.903-4.823-.548-9.894-2.412-9.894-10.734 "
                "0-2.37.847-4.31 2.236-5.828-.224-.55-.969-2.759.214-5.748 0 0 "
                "1.822-.584 5.972 2.226 "
                "1.732-.482 3.59-.722 5.437-.732 1.845.01 3.703.25 5.437.732 "
                "4.147-2.81 5.967-2.226 "
                "5.967-2.226 1.185 2.99.44 5.198.217 5.748 1.392 1.517 2.232 3.457 "
                "2.232 5.828 0 "
                "8.344-5.078 10.18-9.916 10.717.779.67 1.474 1.996 1.474 4.021 0 "
                "2.904-.027 5.247-.027 "
                "5.96 0 .58.392 1.256 1.493 1.044C37.981 40.375 44.2 32.24 44.2 "
                '22.647c0-11.996-9.726-21.72-21.722-21.72" '
                'fill="currentColor"/></svg>'
            ),
        },
    },
)

html_theme_options = asdict(theme_options)

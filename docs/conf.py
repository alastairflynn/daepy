# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------

project = 'DAEpy'
copyright = '2019, Alastair Flynn'
author = 'Alastair Flynn'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.intersphinx']
autodoc_member_order = 'bysource'
intersphinx_mapping = {'python': ('https://docs.python.org/3', None)}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
light_blue = '#2089df'
dark_blue = '#2854b6'

html_theme = 'classic'
html_theme_options = {'linkcolor':light_blue, 'visitedlinkcolor':light_blue, 'sidebarlinkcolor':light_blue, 'relbarlinkcolor':'#ffffff',
'bgcolor':'#ffffff', 'footerbgcolor':light_blue, 'headbgcolor':'#ffffff', 'sidebarbgcolor':'#ffffff', 'relbarbgcolor':light_blue,
'textcolor':'#121212', 'footertextcolor':'#ffffff', 'headtextcolor':dark_blue, 'sidebartextcolor':dark_blue, 'relbartextcolor':'#ffffff',
'stickysidebar':True}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

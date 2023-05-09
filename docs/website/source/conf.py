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
sys.path.insert(0, os.path.abspath('../../..')) # Source code dir relative to this file
sys.path.insert(0, os.path.abspath('.'))

# sys.path.insert(0, os.path.abspath('...'))


# -- Project information -----------------------------------------------------

project = 'FurnitureBench'
copyright = '2023'
author = 'CLVR @ KAIST'


# -- General configuration ---------------------------------------------------


# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    'sphinx.ext.autosectionlabel',
    'recommonmark',
    'sphinx_lesson',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'checkbox_directive'
]


autodoc_mock_imports = [
    "numpy",
    "torch",
    "gym",
    "torchcontrol",
    "polymetis",
    "torchvision",
    "cv2",
    "isaacgym",
    "numba",
    "dt_apriltags",
    "pyrealsense2",
    "oculus_reader",
    "r3m",
    "vip"
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_book_theme'

html_theme_options = {
    "repository_url": "https://github.com/clvrai/furniture-bench",
    "use_repository_button": True,
}


autosummary_generate = True

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

source_suffix = ['.rst', '.md']

html_css_files = [
    'css/custom.css',
]

html_js_files = [
    'scripts/custom_toc_highlight.js',
    'scripts/add_br_to_sections.js'
]


html_logo = "path/to/myimage.png"
html_title = "FurnitureBench"

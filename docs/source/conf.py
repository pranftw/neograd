import os
import sys


basedir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, basedir)

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'neograd'
copyright = '2022, Pranav Sastry'
author = 'Pranav Sastry'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.napoleon', 'sphinx.ext.viewcode']
autodoc_default_options = {"private-members": True}

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
html_style = 'alablaster.css'
html_theme_options = {
  'logo':'ng.png',
  'logo_name':True,
  'description':'A deep learning framework created from scratch with Python and NumPy',
  'fixed_sidebar':True,
  'github_user':'pranftw',
  'github_repo':'neograd',
  'github_button':True,
  'sidebar_collapse':True
}

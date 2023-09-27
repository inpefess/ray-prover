# type: ignore
# pylint: disable=all
"""Sphinx config."""
import os
import sys

sys.path.insert(0, os.path.abspath(".."))
project = "ray-prover"
copyright = "2023, Boris Shminke"
author = "Boris Shminke"
release = "0.0.6"
extensions = ["sphinx.ext.autodoc", "sphinx.ext.coverage"]
html_theme = "furo"

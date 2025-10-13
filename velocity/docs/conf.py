import importlib
project = "velocity"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]
templates_path = ["_templates"]
exclude_patterns = []
html_theme = "furo" if importlib.util.find_spec("furo") else "alabaster"
html_static_path = ["_static"]
# Sphinx >= 5 uses root_doc; default is "index" but set explicitly for safety
root_doc = "index"

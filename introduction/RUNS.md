### Sphinx

python -m introduction.cli sphinx-skel introduction/docs
python -m sphinx -b html docs docs/_build/html
open docs/_build/html/index.html
sphinx-autobuild docs docs/_build/html
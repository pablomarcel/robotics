Usage
=====

Package CLI
-----------

The package-level CLI delegates to the existing introduction modules::

   python -m introduction.cli manipulator --all --rich
   python -m introduction.cli mobius --backend plotly
   python -m introduction.cli kinematics --op cross --v1 1 2 3 --v2 4 5 6

Sphinx Skeleton
---------------

Generate or refresh this documentation skeleton with::

   python -m introduction.cli sphinx-skel introduction/docs --force

The generated pages intentionally use conservative reStructuredText so they remain friendly to GitHub Pages deployments.

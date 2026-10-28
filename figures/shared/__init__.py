"""Machinery shared by the manuscript figure scripts.

Every figure imports its colors, group labels, panel lettering, quality control
and results-locating from here, so the figures cannot diverge from one another:
a group is the same color and carries the same name in every figure, and a
simulation excluded from one analysis is excluded from all of them. A rename is
one edit here rather than one per figure.

Modules
-------
``palette``    colors, group display labels, outcome-axis wording
``panels``     panel letters (A, B, C ...)
``qc``         the shared pressure-outlier table
``results``    locating ``04_results`` and failing usefully when it is missing

Figure scripts are run as files, not imported as a package, so they add this
directory's parent to ``sys.path`` and ``import shared.palette``. See
``figures/README.md``.
"""

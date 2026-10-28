"""Stand-in for the ``opensim`` Python module inside the AddBiomechanics engine.

The engine imports ``opensim`` at module level in two files:
``src/moco_pass/moco_pass.py``, which ``src/engine.py`` imports unconditionally,
and ``src/plotting.py``. The module is used only by the Moco pass and its plots.
Moco is off in every run here (``"runMoco": false`` in ``_subject.json``), and the
plots the result writer draws (inverse kinematics, inverse dynamics, ground
reaction forces, marker errors) do not touch it.

The engine's environment has no OpenSim Python bindings, so without this file the
engine stops at the import. Any attribute access raises, so a code path that did
need OpenSim would fail loudly instead of running on a stand-in. The paper's
AddBiomechanics runs used this arrangement (an empty module on ``PYTHONPATH``).
"""


def __getattr__(name):
    raise AttributeError(
        f"opensim.{name}: this is the AddBiomechanics engine's stand-in module "
        "(tools/addbiomechanics/stub/opensim.py). The OpenSim Python API is not "
        "available to the engine; only its Moco pass needs it, and Moco is off."
    )

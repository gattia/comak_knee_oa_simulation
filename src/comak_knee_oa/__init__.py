"""Subject-specific knee simulation of cartilage contact mechanics in osteoarthritis.

Reproduction artifact for the accompanying manuscript. The pipeline reconstructs
subject-specific knee anatomy from MRI segmentations using neural shape models,
builds an OpenSim knee model per subject, simulates gait with COMAK, and relates
the resulting cartilage contact pressure to longitudinal cartilage thickness
change.

Start at ``README.md``. Every figure in the paper has one entry point under
``figures/``; every data path is resolved by ``comak_knee_oa.paths``.
"""

__version__ = "0.1.0.dev0"

"""ConfGeneration package.

This package provides utilities for generating molecular conformers, currently
focused on an XTB metadynamics-based workflow.

The public CLI entry point is exposed via the `conf-generation` console script
configured in `pyproject.toml`.
"""

from .gen_Confs_Main import ConfGenerator
from .gen_Confs_Generators import XTBMetadynamicsConfGenerator

__all__ = ["ConfGenerator", "XTBMetadynamicsConfGenerator"]

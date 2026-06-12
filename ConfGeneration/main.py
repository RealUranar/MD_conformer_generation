"""Command-line interface for conformer generation.

This module wires CLI arguments to a selected `ConfGenerator` implementation.

Entry point:
     The `conf-generation` console script (see `pyproject.toml`).
"""

import argparse
import os

from ConfGeneration.gen_Confs_Generators import XTBMetadynamicsConfGenerator


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
     """Parse CLI arguments.

     Parameters
     ----------
     argv:
          Optional argv override for easier programmatic testing.

     Returns
     -------
     argparse.Namespace
          Parsed argument namespace.
     """
     parser = argparse.ArgumentParser(
          description="Conformer Generation Tool",
          formatter_class=argparse.ArgumentDefaultsHelpFormatter,
     )
     parser.add_argument(
          "-p",
          "--program",
          type=str,
          help="Program to generate the structures",
          choices=["Metadynamics", "Crest"],
          required=True,
     )
     parser.add_argument(
          "-i",
          "--input",
          type=str,
          help="xyz-File of the starting geometry",
          required=True,
     )
     parser.add_argument(
          "-n",
          "--min-valid-molecules",
          type=int,
          default=100,
          help="Minimum unique structures that are generated",
     )
     parser.add_argument(
          "-t",
          "--threshold",
          type=str,
          default="auto",
          help="Threshold at which structures are rejected based on their RMSD value, possible values: 'auto' or a float value",
     )
     parser.add_argument(
          "-r",
          "--restart",
          action="store_true",
          help="Restart the generation from the last generated structures",
     )
     parser.add_argument(
          "--heavy-rmsd",
          action="store_true",
          help="Only consider heavy atoms in the RMSD calculation",
     )
     parser.add_argument(
          "-f",
          "--work-folder",
          type=str,
          default="work",
          help="Folder where all the intermediate files are stored",
     )
     parser.add_argument(
          "-k",
          "--kpush",
          type=float,
          default=0.1,
          help="(Only for Metadynamics) Scaling factor for RMSD criteria",
     )
     parser.add_argument(
          "-a",
          "--alp",
          type=float,
          default=0.5,
          help="(Only for Metadynamics) Width of the gaussian potential used in the RMSD criteria",
     )
     parser.add_argument("-d", "--debug", action="store_true", help="Debug")
     parser.add_argument("--cpus", type=int, default=8, help="Number of CPUs to use")
     return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
     """CLI main.

     This function is referenced by the console script entry point.
     """
     args = _parse_args(argv)

     # XTB uses OpenMP; control the number of threads through this env var.
     os.environ["OMP_NUM_THREADS"] = str(args.cpus)

     if args.program == "Metadynamics":
          # XTB metadynamics-based generator.
          confs = XTBMetadynamicsConfGenerator(
               structure_name=args.input,
               work_folder=args.work_folder,
               min_valid_molecules=args.min_valid_molecules,
               threshold=args.threshold,
               only_heavy_atoms_rmsd=args.heavy_rmsd,
               kpush=args.kpush,
               alp=args.alp,
               debug=args.debug,
               restart=args.restart,
          )
          confs.run()
          return 0

     if args.program == "Crest":
          # CREST-based generator (only bareley functional)
          os.system("xtb inStructure.xyz --opt vtight --alpb water")
          os.system("crest xtbopt.xyz --v3 --gfn2 --alpb water --noreftopo")
          return 0

     raise ValueError(f"Unsupported program: {args.program}")


if __name__ == "__main__":
     raise SystemExit(main())

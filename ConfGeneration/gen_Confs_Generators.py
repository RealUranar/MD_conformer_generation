"""Concrete conformer generator implementations.

Currently this module provides an XTB-based metadynamics generator that
produces conformer candidates via MD and then relies on the `ConfGenerator`
base class to filter a unique subset.
"""

from ConfGeneration.gen_Confs_Main import ConfGenerator

import ase
from ase.io import read, write

import subprocess
import os, sys, shutil

import numpy as np

class XTBMetadynamicsConfGenerator(ConfGenerator):
    def __init__(
        self,
        structure_name: str,
        min_valid_molecules: int,
        *,
        kpush: float = 0.1,
        alp: float = 0.5,
        xtb_path: str = "xtb",
        constraints: str | None = None,
        **kwargs,
    ):
        """Generate conformers using XTB metadynamics.

        Parameters
        ----------
        structure_name:
            Input structure file (xyz).
        min_valid_molecules:
            Target number of unique conformers.
        kpush:
            Metadynamics bias strength parameter in XTB.
        alp:
            Metadynamics bias width parameter in XTB.
        xtb_path:
            Path to the `xtb` executable (must be available in PATH or an absolute path).
        kwargs:
            Forwarded to `ConfGenerator` (e.g. `threshold`, `work_folder`, `restart`, ...).
        """
        super().__init__(structure_name, min_valid_molecules, **kwargs)

        self.kpush = kpush
        self.alp = alp
        self.xtb_path = xtb_path
        self.constraints = constraints

        if self.constraints is not None and not isinstance(self.constraints, str):
            raise TypeError("xTB constraints must be provided as a string.")

        self.log(
        f"""Generation Tool: XTB_Metadynamics
        kpush: {kpush}
        alp: {alp}
        XTB Path: {xtb_path}
        Constraints: {self.constraints if self.constraints else "None"}
        """)
        # Write an empty reference structure file. XTB will refuse to start a
        # metadynamics run if this file is missing.
        with open(os.path.join(self.work_folder, "Ref_Structs.xyz"), "w") as f:
            f.write("")
            
    def write_MD_input(self, number_of_structures: int):
        """Write `metadyn.inp` for XTB MD/metadynamics.

        Notes
        -----
        - The simulation length is derived from the requested number of conformers.
        - A minimum of 100 structures is enforced to avoid very short trajectories.
        """
        if number_of_structures < 100: number_of_structures = 100 
            
        dump_interval = 0.01  # in ps
        simulation_time = number_of_structures * 2 * dump_interval  # in ps

        constraint_lines = []

        if self.constraints:
            constraint_lines = [
                self.constraints.rstrip() + "\n"
            ]
        
        with open(os.path.join(self.work_folder, "metadyn.inp"), "w") as f:
            f.writelines([
                "$md\n",
                "   temp= 400 # in K\n",
                f"   time= {simulation_time}  # in ps\n",
                f"   dump= {dump_interval * 1000}  # in fs\n",
                "   step= 0.2  # in fs\n",
                "   velo= false\n",
                "   nvt= true\n",
                "   hmass= 4\n",
                "   shake= 0\n",
                "   sccacc= 2.0\n",
                "   restart= false\n",
                "$end\n",
                "$metadyn\n",
                f"   kpush={self.kpush}\n",
                f"   alp={self.alp}\n",
                "   coord=Ref_Structs.xyz\n",
                "$end\n",
                "$wall\n",
                f"   potential=logfermi\n",
                f"   sphere: auto, all\n",
                "$end\n",
                *constraint_lines,
                "$cma\n",
                ])


    def get_metadynamics_structures(self, molecules: list[ase.Atoms], fraction_to_add: float = 0.01) -> list[ase.Atoms]:
        """
        Select a subset of generated structures as metadynamics references.

        XTB metadynamics can bias away from provided reference structures.
        This method samples a small fraction of frames from the trajectory and
        appends them to `Ref_Structs.xyz` to diversify subsequent iterations.
        """
        n_new_ref_structs = max(1, int(len(molecules) * fraction_to_add))
        return [molecules[i] for i in np.random.choice(np.arange(len(molecules)), size=n_new_ref_structs)]
        
    
    def gen_confs(self, n_confs: int, restart: bool = False, debug: bool = False) -> list[ase.Atoms]:
        """Run an XTB metadynamics step and return trajectory frames as conformers.

        Parameters
        ----------
        n_confs:
            Approximate number of conformers to aim for (used to scale MD time).
        restart:
            If true, reads `save_structures.xyz` and continues from the last frame.
        debug:
            Currently unused (kept for signature compatibility).
        """
        if restart:
            self.log("Restarting XTB Metadynamics from previous run...")
            if not os.path.exists(os.path.join(self.work_folder, "save_structures.xyz")):
                raise FileNotFoundError("No save_structures.xyz file found for restart. Please check the work folder for previous runs.")
            molecules = read(os.path.join(self.work_folder, "save_structures.xyz"), index=":", format="xyz")
            write(os.path.join(self.work_folder, "start_struct.xyz"), molecules[-1], format="xyz") #The last structure is used as the starting structure for the next iteration
            write(os.path.join(self.work_folder, "Ref_Structs.xyz"), self.get_metadynamics_structures(molecules, 0.01), format="xyz", append=True) #Structures, used for the metadynamics
            return molecules
            
        env = os.environ.copy()
        env["OMP_STACKSIZE"] = "5G"
        
        self.write_MD_input(n_confs)
        
        xtb_trj = os.path.join(self.work_folder, "xtb.trj")
        xtb_out = os.path.join(self.work_folder, "XTB.out")

        # Remove stale trajectory before running xTB.
        # Otherwise a failed xTB run could accidentally reuse an old xtb.trj.
        if os.path.exists(xtb_trj):
            os.remove(xtb_trj)

        charge_spin = ""
        if self.charge != 0:
            charge_spin = f" --chrg {self.charge}"
        if self.multiplicity != 1:
            #--uhf takes the number of unpaired electrons, which is multiplicity - 1
            charge_spin += f" --uhf {self.multiplicity - 1}"

        with open(xtb_out, "a") as f:
            result = subprocess.run(
                f"{self.xtb_path} --metadyn 1000 --md --cma{charge_spin} --norestart --alpb water --input metadyn.inp start_struct.xyz",
                shell=True,
                check=False,
                cwd=self.work_folder,
                stdout=f,
                stderr=f,
                env=env,
            )

        if result.returncode != 0:
            raise RuntimeError(
                f"xTB metadynamics failed with return code {result.returncode}. "
                f"Check this file for details: {xtb_out}"
            )

        if not os.path.exists(xtb_trj):
            raise RuntimeError(
                f"xTB finished without creating xtb.trj. "
                f"Check this file for details: {xtb_out}"
            )

        try:
            molecules = read(xtb_trj, index=":", format="xyz")
        except Exception as exc:
            raise RuntimeError(
                f"xTB created xtb.trj, but ASE could not read it. "
                f"Check this file for details: {xtb_out}"
            ) from exc

        if len(molecules) < 10:
            raise RuntimeError(
                f"xTB generated only {len(molecules)} conformers. "
                f"Check this file for details: {xtb_out}"
            )

        os.remove(xtb_trj)
        
        write(os.path.join(self.work_folder, "Ref_Structs.xyz"), self.get_metadynamics_structures(molecules, 0.01), format="xyz", append=True) #Structures, used for the metadynamics
        write(os.path.join(self.work_folder, "save_structures.xyz"), molecules, format="xyz", append=True) #All generated structures, for debugging, visualization and restarts
        write(os.path.join(self.work_folder, "start_struct.xyz"), molecules[-1], format="xyz") #The last structure is used as the starting structure for the next iteration

        return molecules


    def optimize_molecule(self) -> ase.Atoms:
        """
        Optimize the input structure using XTB and set it as MD start structure.

        If constraints are provided, apply the same constraints during the
        initial optimization that are later used for metadynamics.

        Returns
        -------
        ase.Atoms
            The optimized structure read from `xtbopt.xyz`.
        """
        xtb_opt = os.path.join(self.work_folder, "xtbopt.xyz")
        xtb_out = os.path.join(self.work_folder, "XTB.out")
        opt_input = os.path.join(self.work_folder, "opt.inp")

        if os.path.exists(xtb_opt):
            os.remove(xtb_opt)

        # Write the supplied xTB constraint input exactly as provided.
        if self.constraints:
            with open(opt_input, "w") as f:
                f.write(self.constraints.rstrip() + "\n")

        with open(xtb_out, "a") as f:
            input_path = os.path.join("..", self.structure_name)

            # Keep the original xTB optimization command unchanged
            # when no constraints are supplied.
            command = f"{self.xtb_path} {input_path} --opt --cma --alpb water"
            
            if self.charge != 0:
                command += f" --chrg {self.charge}"
            if self.multiplicity != 1:
                #--uhf takes the number of unpaired electrons, which is multiplicity - 1
                command += f" --uhf {self.multiplicity - 1}"
            
            # Add the constraint input only when constraints exist.
            if self.constraints:
                command += " --input opt.inp"

            result = subprocess.run(
                command,
                shell=True,
                check=False,
                cwd=self.work_folder,
                stdout=f,
                stderr=f,
            )

        if result.returncode != 0:
            raise RuntimeError(
                f"xTB optimization failed with return code {result.returncode}. "
                f"Check this file for details: {xtb_out}"
            )

        if not os.path.exists(xtb_opt):
            raise RuntimeError(
                f"xTB optimization finished without creating xtbopt.xyz. "
                f"Check this file for details: {xtb_out}"
            )

        optimized_molecule = read(xtb_opt, format="xyz")

        os.remove(xtb_opt)

        write(
            os.path.join(self.work_folder, "start_struct.xyz"),
            optimized_molecule,
            format="xyz",
        )

        return optimized_molecule
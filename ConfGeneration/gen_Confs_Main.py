"""Core conformer generation workflow.

This module defines the `ConfGenerator` base class which orchestrates:

- preparing/optimizing a reference structure
- generating candidate conformers
- aligning candidates to the reference
- selecting a diverse subset using RMSD-based greedy sampling

Concrete generator implementations live in `gen_Confs_Generators.py`.
"""

import matplotlib.pyplot as plt
import numpy as np

import ase
from ase.io import read, write

import os, sys, shutil, time

class ConfGenerator:
    def __init__(self, structure_name:str, min_valid_molecules:int, threshold: str = "auto", only_heavy_atoms_rmsd:bool = False, restart:bool = False, work_folder:str = "work", debug:bool=False):
        """Base class for conformer generators.

        Parameters
        ----------
        structure_name:
            Path to the input structure (typically an `.xyz`).
        min_valid_molecules:
            Target number of *unique* conformers.
        threshold:
            RMSD threshold used by the uniqueness selection.
            Use "auto" to estimate a molecule-specific threshold.
        only_heavy_atoms_rmsd:
            If true, RMSD is computed only over non-hydrogen atoms.
        restart:
            If true, concrete generators may resume from a previous run.
        work_folder:
            Directory used for temporary files and logs.
        debug:
            Enables additional prints/plots.
        """
        self.work_folder = work_folder
        self.structure_name = structure_name
        self.min_valid_molecules = min_valid_molecules
        self.threshold = -1 if threshold == "auto" else float(threshold)
        self.only_heavy_atoms_rmsd = only_heavy_atoms_rmsd
        self.debug = debug
        self.rmsd_matrix = None
        self.restart = restart

        self.log_path = os.path.join(self.work_folder, "confs.log")

        if os.path.exists(self.log_path):
            os.remove(self.log_path)
        if not os.path.exists(work_folder):
            os.makedirs(work_folder)

        self.log(
f"""Starting Conformer Generation
-----------------------------
Work Folder: {work_folder}
Structure Name: {structure_name}
Minimum Valid Molecules: {min_valid_molecules}
Only Heavy Atoms RMSD: {only_heavy_atoms_rmsd}
Automatic Threshold Calculation: {self.threshold == -1}
Restart (reading save_structures.xyz): {restart}
Debug Mode: {debug}
""")
    
    def log(self, text: str):
        """Append text to the run log in the work folder."""
        with open(self.log_path, "a") as f:
            f.write(text + "\n")  

    def run(self):
        """Run the complete conformer generation workflow.

        The overall procedure:
        1) Optimize the starting structure to obtain a stable reference.
        2) Generate candidate conformers.
        3) Align candidates to the reference for comparable RMSD values.
        4) Select unique conformers using a greedy RMSD diversity search.
        5) Repeat generation until the requested number of unique conformers is reached.
        """
        start = time.perf_counter()

        ref_molecule = self.optimize_molecule()
        new_molecules = self.gen_confs(n_confs=self.min_valid_molecules, restart=self.restart)
        new_molecules = self.align_to_reference(ref_molecule, new_molecules)
        n_generated_confs = len(new_molecules)
        
        if self.threshold == -1:
            # Determine an RMSD threshold that results in a reasonable acceptance
            # rate for this molecule (heuristic; see `calc_molecule_specific_threshold`).
            self.threshold, _ = self.calc_molecule_specific_threshold(new_molecules)
            self.log(f"Automatically determined RMSD threshold: {self.threshold:.3f}")
        else:
            self.log(f"Using user-defined RMSD threshold: {self.threshold:.3f}")
        write(os.path.join(self.work_folder, "generated_confs.xyz"), new_molecules, format="xyz")
        unique_molecules = self.search_unique_molecules(new_molecules)
        
        if self.debug: ConfGenerator.show_RMSD_matrix(self.rmsd_matrix)
        self.log(f"{'Iteration':<10}| {'Generated / Unique':<19}| {'Minimum RMSD':>10}| {'Time Elapsed':>10}")
        self.log(f"{1:<10}|{n_generated_confs:>10} / {len(unique_molecules):<7}| {self.get_min_RMSD(self.rmsd_matrix):>8.3f}| {time.perf_counter() - start:>10.2f}s")
        
        iteration = 2
        while len(unique_molecules) < self.min_valid_molecules:
            start = time.perf_counter()
            new_molecules = self.gen_confs(n_confs=self.min_valid_molecules-len(unique_molecules))
            new_molecules = self.align_to_reference(ref_molecule, new_molecules)
            n_generated_confs = len(new_molecules)

            unique_molecules = self.search_unique_molecules(new_molecules, unique_molecules)

            if self.debug: ConfGenerator.show_RMSD_matrix(self.rmsd_matrix)
            
            self.log(f"{iteration:<10}|{n_generated_confs:>10} / {len(unique_molecules):<7}| {self.get_min_RMSD(self.rmsd_matrix):>8.3f}| {time.perf_counter() - start:>10.2f}s")
            iteration += 1
        
        self.rmsd_matrix = ConfGenerator.calc_RMSD_matrix(
            unique_molecules,
            only_heavy_atoms=self.only_heavy_atoms_rmsd,
            debug=self.debug,
        )
        self.log(f"-------------------Generation Completed-------------------")
        self.log(f"\nFinal Iterations: {iteration}\nNumber of Conformers Found: {len(unique_molecules)}\nMinimum RMSD: {self.get_min_RMSD(self.rmsd_matrix):.3f}")
        
        write(os.path.join("unique_confs.xyz"), unique_molecules)

    def align_to_reference(self, ref_mol: ase.Atoms, molecules: list[ase.Atoms]) -> list[ase.Atoms]:
        """Align all molecules to ref_mol in-place and return the list.

        Subclasses can override this if they need a different alignment strategy.
        """
        from ase.build import minimize_rotation_and_translation
        from copy import deepcopy
        
        tmp_molecules = deepcopy(molecules)
        for mol in tmp_molecules:
            minimize_rotation_and_translation(ref_mol, mol)
        return tmp_molecules
        
    def calc_molecule_specific_threshold(self, molecules: list[ase.Atoms]) -> tuple[float, float]:
        """Calculate a molecule specific threshold, based on an acceptance rate of 25 molecules per 100 generated molecules.

        Subclasses can override this if they want to use a different strategy for calculating the threshold.
        """
        n_confs_to_select = 60
        n_trials = 5
        max_iterations = 40
        if len(molecules) < n_confs_to_select:
            raise ValueError(
                f"Cannot select {n_confs_to_select} molecules from only {len(molecules)} molecules.")
        
        thresholds = []
        for i in range(n_trials):
            trial_molecules = [molecules[j] for j in np.random.choice(
                                                        np.arange(len(molecules)),
                                                        size=min(100, len(molecules)),
                                                        replace=False,)]

            # Choose broad initial bounds.
            low = 0.1
            high = 10.0
            if thresholds:
                low = max(0.01, thresholds[0] / 2)
                high = thresholds[0] * 2
            
            # Expand high until it selects <= target molecules.
            while len(self.search_unique_molecules(trial_molecules, threshold=high)) > n_confs_to_select:
                high *= 2

                if high > 1e6:
                    raise RuntimeError("Could not find an upper threshold bound.")

            best_threshold = None
            best_error = float("inf")
            best_n_mols = None

            for iteration in range(max_iterations):
                mid = (low + high) / 2

                n_mols = len(self.search_unique_molecules(trial_molecules, threshold=mid))
                error = abs(n_mols - n_confs_to_select)

                if error < best_error:
                    best_error = error
                    best_threshold = mid
                    best_n_mols = n_mols

                if self.debug: 
                    print(
                    f"Trial {i + 1}, Iteration {iteration + 1}: "
                    f"Threshold: {mid:.6f}, Unique Molecules: {n_mols}"
                )

                if n_mols == n_confs_to_select:
                    best_threshold = mid
                    best_n_mols = n_mols
                    break

                # If too many molecules are selected, threshold is too low.
                if n_mols > n_confs_to_select:
                    low = mid
                else:
                    high = mid

            if self.debug:
                print(
                    f"Trial {i + 1}: selected threshold {best_threshold:.6f}, "
                    f"gave {best_n_mols} molecules"
                )

            thresholds.append(best_threshold)

        thresholds = np.array(thresholds)

        if self.debug: print(
                f"Calculated molecule specific threshold: "
                f"{thresholds.mean():.6f} +/- {thresholds.std():.6f}"
            )

        return thresholds.mean(), thresholds.std()

    def gen_confs(self, n_confs: int, do_preopt: bool = False, restart: bool = False, debug: bool = False) -> list[ase.Atoms]:
        """
        Generate new conformers.

        Concrete subclasses must implement this.

        Parameters
        ----------
        n_confs:
            Requested number of conformers to generate (not necessarily unique).
        do_preopt:
            Optional hook for generators that want a pre-optimization stage.
        restart:
            Optional hook for generators that support restarting.
        debug:
            Optional hook for verbose behavior.

        Returns
        -------
        list[ase.Atoms]
            Generated conformers.
        """
        pass
    
    def optimize_molecule(self) -> ase.Atoms:
        """
        Function that optimizes a molecule and returns the optimized molecule.
        Subclasses can override this if they want to use a different optimization strategy.
        """
        pass
    
    def search_unique_molecules(self, molecules_test: list[ase.Atoms], molecules_ref: list[ase.Atoms] = None, threshold: float = None) -> list[ase.Atoms]:
        """Select a diverse subset of conformers using a greedy RMSD criterion.

        This method constructs an RMSD matrix and iteratively adds the conformer
        which maximizes the minimum RMSD to the already-selected set.

        Parameters
        ----------
        molecules_test:
            Candidate conformers.
        molecules_ref:
            If provided, starts from these already-accepted conformers.
        threshold:
            Minimum allowed internal RMSD within the selected set.

        Returns
        -------
        list[ase.Atoms]
            Unique conformers.
        """
        if threshold is None: threshold = self.threshold
        
        if molecules_ref is None:
            rmsd_matrix_test = ConfGenerator.calc_RMSD_matrix(
                molecules_test,
                only_heavy_atoms=self.only_heavy_atoms_rmsd,
                debug=self.debug,
            )
            max_id = rmsd_matrix_test.sum(axis=0).argmax()
            unique_molecules = [molecules_test[max_id]]
            unique_ids = [max_id]
        else:
            unique_molecules = molecules_ref[:]
            unique_ids = list(range(len(unique_molecules)))
            molecules_test = unique_molecules + molecules_test
            rmsd_matrix_test = ConfGenerator.calc_RMSD_matrix(
                molecules_test,
                only_heavy_atoms=self.only_heavy_atoms_rmsd,
                debug=self.debug,
            )


        iteration = 0
        while True:
            # Find the candidate that is farthest away from the current unique set,
            # where "distance" is the RMSD to the nearest selected conformer.
            max_id = int(rmsd_matrix_test[unique_ids, :].T.min(axis=1).argmax())
            unique_ids.append(max_id)
            
            min_rmsd = self.get_min_RMSD(rmsd_matrix_test[np.ix_(unique_ids, unique_ids)])
            
            if min_rmsd < threshold or iteration > self.min_valid_molecules:
                # The latest candidate would make the set too similar, so drop it.
                unique_ids.pop()  #Remove the last added molecule, as it does not meet the threshold
                break
            
            if self.debug:
                print(f"Iteration: {iteration}, min internal RMSD: {min_rmsd:.3f}")
            unique_molecules.append(molecules_test[max_id])
            iteration += 1
        
        self.rmsd_matrix = rmsd_matrix_test[np.ix_(unique_ids, unique_ids)]
        return unique_molecules

    @staticmethod
    def get_min_RMSD(rmsd_matrix):
        """Return the minimum off-diagonal RMSD in a square RMSD matrix."""
        return rmsd_matrix[np.triu_indices(rmsd_matrix.shape[0], k=1)].min()
    
    @staticmethod
    def calc_RMSD(ref_mol: ase.Atoms, test: ase.Atoms | list[ase.Atoms]):
        """Calculate the RMSD between a reference molecule and a test molecule or a list of test molecules.

        Parameters:
        ref_mol (ase.Atoms): The reference molecule.
        test (ase.Atoms or list of ase.Atoms): The test molecule(s) to compare against the reference.
        Returns:
        float or list of float: The RMSD value(s) between the reference and the test molecule(s).
        """
        if type(test) == list:
            ref_pos = ref_mol.get_positions()
            test_pos = np.array([mol.get_positions() for mol in test])
            return np.sqrt(np.power(ref_pos - test_pos, 2).sum(axis=1).sum(axis=1) / ref_pos.shape[0])

        return np.sqrt(np.min([np.sum((ref_mol.get_positions() - test.get_positions())**2)]) / ref_mol.get_positions().shape[0])

    @staticmethod
    def calc_RMSD_matrix(molecules:list[ase.Atoms], only_heavy_atoms:bool=False, debug:bool=False):
        """Compute the all-vs-all RMSD matrix for a list of molecules."""
        if only_heavy_atoms:
            # Use atom indices of the first molecule as reference.
            include_atoms_indices = [atom.index for atom in molecules[0] if atom.number != 1]
            from ase.filters import Filter
            molecules_tmp = [Filter(mol, indices=include_atoms_indices) for mol in molecules]
        else:
            molecules_tmp = molecules
            
        n = len(molecules_tmp)
        # Calculate the upper triangle of the RMSD matrix and mirror it to the lower triangle
        rmsd_matrix = np.zeros((n, n))
        for i in range(n):
            rmsd_matrix[i, i:] = ConfGenerator.calc_RMSD(molecules_tmp[i], molecules_tmp[i:])
            rmsd_matrix[i:, i] = rmsd_matrix[i, i:]
        
        return rmsd_matrix
       
    @staticmethod
    def show_RMSD_matrix(matrix):
        """Save an RMSD matrix plot as `RMSD_matrix.png` in the CWD."""
        fig, ax = plt.subplots()
        im = ax.imshow(matrix, cmap="viridis")
        ax.set_title("RMSD Matrix")
        fig.savefig("RMSD_matrix.png")
        
    

    
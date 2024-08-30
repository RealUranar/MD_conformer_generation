import numpy as np
from numpy import ndarray
from itertools import islice
from typing import List
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem.rdDetermineBonds import DetermineBonds, DetermineConnectivity

class Conf_Generator():
    def __init__(self,min_Valid_Molecules : int, threshold : float, weighting_Scheme: str = "N100", debug = False, OnlyHeavyAtomsRMSD = False, start_Structure_Filename="inStructure.xyz"):
        """
        Possible weighting schemes:
            diameter : uses the diameter of the first molecule as weight
            N100 : uses the maximum rmsd value of the supplied molecules as weight
            _ : If nothing is supplied, weight will be set to 1
        """
        self.unique_Molecules: List[Chem.Mol] = []
        self.min_Valid_Molecules = min_Valid_Molecules
        self.threshold = threshold
        self.weighting_Scheme = weighting_Scheme
        self.OnlyHeavyAtomsRMSD = OnlyHeavyAtomsRMSD
        self.start_Structure_Filename = start_Structure_Filename

        with open("confs.log", "w") as f:
            f.write("Starting Conformer Generation\n")
            f.write(f"min_Valid_Molecules: {min_Valid_Molecules}\n")
            f.write(f"threshold: {threshold}\n")
            f.write(f"weighting_Scheme: {weighting_Scheme}\n")
            f.write(f"OnlyHeavyAtomsRMSD: {OnlyHeavyAtomsRMSD}\n")
            f.write(f"start_Structure_Filename: {start_Structure_Filename}\n")
            f.write(f"debug: {debug}\n")

        self.debug = debug

    def calc_Weight(self, keyword : str = "", molecules :List[Chem.Mol] = None):
        """
        Possible keywords:
            diameter : uses the diameter of the first molecule as weight
            N100 : uses the maximum rmsd value of the supplied molecules as weight
            _ : If nothing is supplied, weight will be set to 1
        """
        match keyword.lower():
            case "diameter":
                self.weight = AllChem.Get3DDistanceMatrix(molecules[0]).max()
            case "n100":
                self.weight = np.max(self.calc_RMSD_matrix(molecules))
            case _:
                self.weight = 1

    def build_Temp_mol(self, mol1:Chem.Mol, mol2:List[Chem.Mol]) -> Chem.Mol:
        """Builds a temporary molecule combining all conformers of mol2 behind mol1"""
        temp = Chem.Mol(mol1)
        [temp.AddConformer(mol.GetConformer(0), assignId=True) for mol in mol2]
        Chem.rdMolAlign.AlignMolConformers(temp)
        try:
            DetermineBonds(temp)
        except:
            DetermineConnectivity(temp)
        return temp

    def calc_RMSD_matrix(self, molecules: List[Chem.Mol]) -> ndarray: 
        temp = self.build_Temp_mol(molecules[0], molecules[1:])
        if self.OnlyHeavyAtomsRMSD:
            temp = Chem.rdmolops.RemoveAllHs(temp)
        rmsd_1d = Chem.rdMolAlign.GetAllConformerBestRMS(temp, numThreads = -1, symmetrizeConjugatedTerminalGroups=False)
        num_molecules = len(molecules)
        rmsd_Matrix = np.zeros((num_molecules, num_molecules))
        k = 0
        for i in range(1, num_molecules):
            for j in range(i):
                rmsd_Matrix[j, i] = rmsd_1d[k]
                rmsd_Matrix[i, j] = rmsd_1d[k]
                k += 1

        return rmsd_Matrix
    
    def find_Unique_Molecules(self, molecules: List[Chem.Mol]) -> List[Chem.Mol]:
        num_molecules = len(molecules)
    
        # Calculate RMSD between every pair of molecules
        rmsd_matrix = self.calc_RMSD_matrix(molecules)
        
        # Normalize RMSD matrix by the weight specified
        rmsd_matrix /= self.weight

        # Algorithm to find the largest amount of molecules that are above a certain threshold
        unique_molecules_indice = np.array([], dtype=int)
        possible_molecules_indice = np.arange(num_molecules)
        while len(possible_molecules_indice) > 0:
            molecule = possible_molecules_indice[np.argmin(np.sum(rmsd_matrix[possible_molecules_indice, :] < self.threshold, axis=1))]
            unique_molecules_indice = np.append(unique_molecules_indice, molecule)
            possible_molecules_indice = np.setdiff1d(possible_molecules_indice, np.where(rmsd_matrix[molecule, :] < self.threshold))

        return [molecules[int(i)] for i in np.sort(unique_molecules_indice)]
    
    def check_New_Molecules(self, new_Molecules: List[Chem.Mol]) -> None:
        """Appends new molecules from the new calculation to the unique List if the given threashold is met"""
        # new_unique = self.find_Unique_Molecules(new_Molecules)

        for new_molecule in new_Molecules:
            temp_rmsd = self.calc_RMSD_matrix(self.unique_Molecules + [new_molecule]) / self.weight
            if self.get_Min_RMSD_From_Matrix(temp_rmsd) > self.threshold:
                # print("Accepted", self.get_Min_RMSD_From_Matrix(temp_rmsd))
                # print("real Min", self.get_Min_RMSD_From_Matrix(self.calc_RMSD_matrix(self.unique_Molecules + [new_molecule]) / self.weight))
                self.unique_Molecules.append(new_molecule)

    def read_Moleculues_XYZ(self, fileName : str) -> List[Chem.Mol]:
        xyz_molecules = []
        with open(fileName, "r") as f:
            lenMolecule = int(f.readline())
            f.seek(0)
            while True:
                molecule = list(islice(f, lenMolecule+2))
                if not molecule:
                    break
                xyz_molecules.append("".join(molecule))

        mol_molecules = [Chem.MolFromXYZBlock(xyz) for xyz in xyz_molecules]
        [Chem.rdMolTransforms.CanonicalizeMol(mol) for mol in mol_molecules]

        return mol_molecules
    
    def write_XYZ(self, fileName: str, molecules: List[Chem.Mol]) -> None:
        temp = self.build_Temp_mol(molecules[0], molecules[1:])
        with open(fileName, "w") as f:
            for i in range(len(temp.GetConformers())):
                f.write(Chem.rdmolfiles.MolToXYZBlock(temp, confId=i))

    def run(self):
        with open("confs.log", "a") as logFile:
            #Generate first set of conformers
            molecules = self.gen_Confs(iteration=0, with_Opt= True, start_Structure_Filename=self.start_Structure_Filename)
            #Calculate the weigh to normalize the rmsd
            self.calc_Weight(keyword=self.weighting_Scheme, molecules=molecules)
            #write the first set of unique molecules into self.unique_Molecules
            self.unique_Molecules = self.find_Unique_Molecules(molecules=molecules)
            print(f"Iteration   {0}:  {len(self.unique_Molecules)} unique molecules", file=logFile, flush=True)
            if self.debug:
                rmsd = self.calc_RMSD_matrix(self.unique_Molecules) / self.weight
                print(f"Minimum rmsd value in matrix: {self.get_Min_RMSD_From_Matrix(rmsd)}")
                print(f"Found {len(self.unique_Molecules)} unique molecules")
                self.show_RMSD_Matrix(self.unique_Molecules)

            iteration = 1
            while len(self.unique_Molecules) < self.min_Valid_Molecules:
                molecules = self.gen_Confs(iteration=iteration)
                new_Uniques = self.find_Unique_Molecules(molecules=molecules)
                self.check_New_Molecules(new_Uniques)
                iteration += 1
                print(f"Iteration {iteration:3}:  {len(self.unique_Molecules)}/{self.min_Valid_Molecules} molecules", file=logFile, flush=True)
                if self.debug:
                    rmsd = self.calc_RMSD_matrix(self.unique_Molecules) / self.weight
                    print(f"Minimum rmsd value in matrix: {self.get_Min_RMSD_From_Matrix(rmsd)}")
                    print(f"Found {len(self.unique_Molecules)}/{self.min_Valid_Molecules} molecules")
                    self.show_RMSD_Matrix(self.unique_Molecules)

        
    @staticmethod
    def get_Min_RMSD_From_Matrix(rmsd_matrix: ndarray) -> float:
        min_rmsd = rmsd_matrix[rmsd_matrix != 0].min()
        return min_rmsd
    
    def show_RMSD_Matrix(self, molecules : List[Chem.Mol])->None:
        import matplotlib.pyplot as plt
        rmsd_Matrix = self.calc_RMSD_matrix(molecules)
        fig, ax = plt.subplots()
        ax.set_ylim(0, rmsd_Matrix.shape[0])
        ax.imshow(rmsd_Matrix)
        fig.savefig("rmsd_Matrix.png")

    @staticmethod
    def show_Molecule(mol : Chem.Mol)-> None:
        import matplotlib.pyplot as plt
        from rdkit.Chem.rdDetermineBonds import DetermineBonds, DetermineConnectivity
        from rdkit.Chem import Draw
        #Copy Molecule to not cahnge the original
        mol_copy = Chem.Mol(mol)
        try:
            DetermineBonds(mol_copy,charge=0)
        except ValueError:
            print("Wrong charge selected in Determine Bonds!")
            exit()
        except:
            DetermineConnectivity(mol_copy)
        AllChem.Compute2DCoords(mol_copy)
        im = Draw.MolToImage(mol_copy)
        fig = plt.figure(figsize=(10,5))
        ax = fig.add_axes(111)
        ax.imshow(im)
        ax.axis('off')
        fig.savefig("mol.png")

    def gen_Confs(self, iteration = 0, with_Opt= False, start_Structure_Filename="inStructure.xyz") -> List[Chem.Mol]:
        """
        Function that generates new conformers and returs them as a List
        """
        pass

if __name__ == "__main__":
    confs = Conf_Generator(100, 200)
    Conf_Generator.show_Molecule(confs)
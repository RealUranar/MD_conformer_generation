from typing import List
import os
from rdkit import Chem
from rdkit.Chem import AllChem
os.path.join("/modules")
try:
    from ConfGeneration.gen_Confs_Main import Conf_Generator
except:
    from RMSD.conformer_generation.gen_Confs_Main_Old import Conf_Generator

class Conf_Generator_XTB(Conf_Generator):
    def __init__(self, min_Valid_Molecules: int, threshold: float, weighting_Scheme: str = "N100", debug=False, OnlyHeavyAtomsRMSD=False, xtb_Path = "xtb", start_Structure_Filename : str = "inStructure.xyz"):
        super().__init__(min_Valid_Molecules, threshold, weighting_Scheme, debug, OnlyHeavyAtomsRMSD, start_Structure_Filename)
        self.xtb_Path = xtb_Path
        with open("md.inp", "w") as f:
            f.writelines([
                "$md\n",
                "   temp=400 # in K\n",
                "   time= 1.0  # in ps\n",
                "   dump= 10.0  # in fs\n",
                "   step=  0.2  # in fs\n",
                "   velo=false\n",
                "   nvt =true\n",
                "   hmass=4\n",
                "   shake=0\n",
                "   sccacc=2.0\n",
                "   restart=false\n",
                "$end"])
        with open("confs.log", "a") as f:
            f.write(f"Generation Tool: XTB_MD\n")

    
    def gen_Confs(self, iteration = 0, with_Opt = False, start_Structure_Filename : str = "nextStart.xyz") -> List:
        #If xtb trajectory already exists, read and return them
        if os.path.exists(f"xtb_{iteration}.trj"):
            molecules = self.read_Moleculues_XYZ(f"xtb_{iteration}.trj")
            Chem.rdmolfiles.MolToXYZFile(molecules[-1], "nextStart.xyz")
            return molecules
        if with_Opt:
            ret = os.system(f"{self.xtb_Path} --omd --cma --norestart --alpb water --input md.inp {start_Structure_Filename} > out.txt 2>&1")
        else:
            ret = os.system(f"{self.xtb_Path} --md --cma --norestart --alpb water --input md.inp {start_Structure_Filename} > out.txt 2>&1")
        if ret != 0:
            raise RuntimeError("XTB did not run succesfully!")
        
        #read the trajectory File
        molecules = self.read_Moleculues_XYZ(f"xtb.trj")
        #Generate next starting xyz_File
        Chem.rdmolfiles.MolToXYZFile(molecules[-1], "nextStart.xyz")
        #Rename trajectory file for save keeping because xtb overrides them
        os.rename(f"xtb.trj", f"xtb_{iteration}.trj")
        return molecules
    
class Conf_Generator_XTB_Metadyn(Conf_Generator):
    def __init__(self, min_Valid_Molecules: int, threshold: float, weighting_Scheme: str = "N100", debug=False, OnlyHeavyAtomsRMSD=False, xtb_Path = "xtb", kpush = 0.1, alp = 0.01, start_Structure_Filename : str = "inStructure.xyz"):
        super().__init__(min_Valid_Molecules, threshold, weighting_Scheme, debug, OnlyHeavyAtomsRMSD, start_Structure_Filename)
        self.xtb_Path = xtb_Path
        with open("Ref_Structs.xyz", "w") as f:
            f.write("")
        with open("metadyn.inp", "w") as f:
            f.writelines([
                "$md\n",
                "   temp=400 # in K\n",
                "   time= 1.0  # in ps\n",
                "   dump= 10.0  # in fs\n",
                "   step=  0.2  # in fs\n",
                "   velo=false\n",
                "   nvt =true\n",
                "   hmass=4\n",
                "   shake=0\n",
                "   sccacc=2.0\n",
                "   restart=false\n",
                "$end\n",
                "$metadyn\n",
                f"   kpush={kpush}\n",
                f"   alp={alp}\n",
                "    coord=Ref_Structs.xyz\n",
                "$end"
                ])
        with open("confs.log", "a") as f:
            f.write(f"Generation Tool: XTB_Metadynamics\n")
            f.write(f"kpush: {kpush}\n")
            f.write(f"alp: {alp}\n")

    
    def gen_Confs(self, iteration = 0, with_Opt = False, start_Structure_Filename : str = "nextStart.xyz") -> List:
        #If xtb trajectory already exists, read and return them
        if os.path.exists(f"xtb_{iteration}.trj"):
            molecules = self.read_Moleculues_XYZ(f"xtb_{iteration}.trj")
            Chem.rdmolfiles.MolToXYZFile(molecules[-1], "nextStart.xyz")
            return molecules
        if with_Opt:
            ret = os.system(f"{self.xtb_Path} --omd --cma --norestart --alpb water --input metadyn.inp {start_Structure_Filename} > out.txt 2>&1")
        else:
            ret = os.system(f"{self.xtb_Path} --metadyn 100 --cma --norestart --alpb water --input metadyn.inp {start_Structure_Filename} > out.txt 2>&1")
        if ret != 0:
            raise RuntimeError("XTB did not run succesfully!")

        #read the trajectory File
        molecules = self.read_Moleculues_XYZ(f"xtb.trj")
        #Generate next starting xyz_File
        Chem.rdmolfiles.MolToXYZFile(molecules[-1], "nextStart.xyz")
        #Add first Structure of last run to Bias

        with open("Ref_Structs.xyz", "a") as f:
            f.write(Chem.rdmolfiles.MolToXYZBlock((molecules[0])))

        #Rename trajectory file for save keeping because xtb overrides them
        os.rename(f"xtb.trj", f"xtb_{iteration}.trj")
        return molecules

class Conf_Generator_rdkit(Conf_Generator):
    def __init__(self, min_Valid_Molecules: int, threshold: float, weighting_Scheme: str = "N100", debug=False, OnlyHeavyAtomsRMSD=False, xtb_Path = "xtb"):
        super().__init__(min_Valid_Molecules, threshold, weighting_Scheme, debug, OnlyHeavyAtomsRMSD)
        self.xtb_Path = xtb_Path
        with open("confs.log", "a") as f:
            f.write(f"Generation Tool: RdKit\n")

    def gen_Confs(self, iteration = 0, with_Opt = False, start_Structure_Filename : str = "nextStart.xyz"):
        def ConfToMol(mol, conf_id):
            conf = mol.GetConformer(conf_id)
            new_mol = Chem.Mol(mol)
            new_mol.RemoveAllConformers()
            new_mol.AddConformer(Chem.Conformer(conf), assignId=True)
            return new_mol
        
        if with_Opt:
            os.system(f"{self.xtb_Path} {start_Structure_Filename} --opt vtight")
            start_Structure_Filename = "xtbopt.xyz"
        
        from rdkit.Chem.rdDetermineBonds import DetermineBonds
        mol = Chem.rdmolfiles.MolFromXYZFile(start_Structure_Filename)
        DetermineBonds(mol,charge=0)

        params = AllChem.ETKDGv3()
        params.verbose = False
        params.numThreads = 0
        params.clearConfs = True
        params.onlyHeavyAtomsForRMS = False
        params.maxIterations = 1000
        params.pruneRmsThresh = 0.01
        confs = Chem.rdDistGeom.EmbedMultipleConfs(mol, numConfs=10000, params=params)
        
        AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=0, maxIters=5)
        AllChem.AlignMolConformers(mol)

        molecules = [ConfToMol(mol, confID) for confID in confs]
        import random
        Chem.rdmolfiles.MolToXYZFile(molecules[random.randint(0,len(molecules)-1)], "nextStart.xyz")
        return molecules

if __name__ == "__main__":
    print(len(Conf_Generator_rdkit(300, 0.01).gen_Confs(start_Structure_Filename="start.xyz")))
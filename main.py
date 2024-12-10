import sys, os, argparse
sys.path.append("/modules")
from ConfGeneration.gen_Confs_Generators import Conf_Generator_XTB, Conf_Generator_XTB_Metadyn, Conf_Generator_rdkit


parser = argparse.ArgumentParser(description="Conformer Generation Tool",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-p", "--program",  type=str, help="Program to generate the structures", choices=["MD", "Metadynamics", "RdKit", "Crest"], required = True)
parser.add_argument("-i", "--input",  type=str, help="xyz-File of the starting geometry", required = True)
parser.add_argument("-n", "--minValidMolecules",  type=int, default=100, help="Minimum unique structures that are generated")
parser.add_argument("-t", "--threshold",  type=float, default=0.07, help="Threshold at which structures are rejected based on thier RMSD value")
parser.add_argument("-w", "--weighting",  type=str, default="N100", help="Choose what weighting scheme to use", choices=["N100", "diameter", "1"])
parser.add_argument("-HRMSD", "--heavyRMSD",  type=bool, default=False, help="Only Consider heavy atoms in the RMSD calculation")
parser.add_argument("-k", "--kpush",  type=float, default=0.1, help="(Only for Metadynamics) Scaling factor for rmsd criteria")
parser.add_argument("-a", "--alpha",  type=float, default=0.01, help="(Only for Metadynamics) Width of the gaussian potential used in the rmsd criteria")
parser.add_argument("-d", "--debug",  type=bool, default=False, help="Debug")
parser.add_argument("-cpus", "--cpus",  type=int, default=8, help="Number of CPUs to use")

args = parser.parse_args()
print(args)

os.environ["OMP_NUM_THREADS"] = str(args.cpus)
if args.program == "MD":
	#Do Conformer search using XTB
	confs = Conf_Generator_XTB(min_Valid_Molecules = args.minValidMolecules, threshold = args.threshold, debug = args.debug, weighting_Scheme=args.weighting, OnlyHeavyAtomsRMSD = args.heavyRMSD, start_Structure_Filename = args.input)
	confs.run()
	confs.write_XYZ(fileName = "unique.xyz", molecules = confs.unique_Molecules)
	pass

if args.program == "Metadynamics":
	#Do Conformer search using XTB and Metadynamik
	confs = Conf_Generator_XTB_Metadyn(min_Valid_Molecules = args.minValidMolecules, threshold = args.threshold, debug = args.debug, weighting_Scheme=args.weighting, OnlyHeavyAtomsRMSD = args.heavyRMSD, kpush=args.kpush, alp=args.alpha, start_Structure_Filename = args.input)
	confs.run()
	confs.write_XYZ(fileName = "unique.xyz", molecules = confs.unique_Molecules)
	pass

if args.program == "RdKit":
	#Do Conformer search using RdKit
	confs = Conf_Generator_rdkit(min_Valid_Molecules = args.minValidMolecules, threshold = args.threshold, debug = args.debug, weighting_Scheme=args.weighting, OnlyHeavyAtomsRMSD = args.heavyRMSD)
	confs.run()
	confs.write_XYZ(fileName = "unique.xyz", molecules = confs.unique_Molecules)
	pass

if args.program == "Crest":
	#Do Conformer search using CREST
	os.system("xtb inStructure.xyz --opt vtight --alpb water")
	os.system("crest xtbopt.xyz --v3 --gfn2 --alpb water --noreftopo")
	pass
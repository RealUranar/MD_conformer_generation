import os
import re
import subprocess
from pathlib import Path


MAIN_FOLDER = Path("/src/Sets_canonical")
OUTPUT_FILE = "unique_confs.xyz"
SKIP_FOLDER_NAMES = {"work", "__pycache__", ".git"}


def count_xyz_structures(xyz_file: Path) -> int:
    """Count structures in a multi-XYZ file."""
    if not xyz_file.exists():
        return 0

    count = 0

    with xyz_file.open("r", errors="replace") as f:
        while True:
            line = f.readline()
            if not line:
                break

            line = line.strip()
            if not line:
                continue

            try:
                n_atoms = int(line)
            except ValueError:
                break

            # Skip comment line
            f.readline()

            # Skip atom lines
            for _ in range(n_atoms):
                f.readline()

            count += 1

    return count


def get_target_conformers(main_py: Path) -> int:
    """Read min_valid_molecules from the folder's main.py."""
    text = main_py.read_text(errors="replace")

    match = re.search(r"min_valid_molecules\s*=\s*(\d+)", text)

    if not match:
        raise ValueError(f"Could not find min_valid_molecules in {main_py}")

    return int(match.group(1))


def is_finished(folder: Path) -> bool:
    """Finished means unique_confs.xyz exists and has enough conformers."""
    main_py = folder / "main.py"
    output_file = folder / OUTPUT_FILE

    target = get_target_conformers(main_py)
    found = count_xyz_structures(output_file)

    if found >= target:
        print(f"Skipping {folder.name}: finished ({found}/{target} conformers).")
        return True

    if found > 0:
        print(f"Restarting {folder.name}: incomplete ({found}/{target} conformers).")
    else:
        print(f"Running {folder.name}: no finished output found ({found}/{target}).")

    return False


def run_main_in_subfolders(main_folder: Path) -> None:
    for subfolder in sorted(main_folder.iterdir()):
        if not subfolder.is_dir():
            continue

        if subfolder.name in SKIP_FOLDER_NAMES:
            continue

        main_py = subfolder / "main.py"

        if main_py.exists():
            try:
                if is_finished(subfolder):
                    continue

                print(f"Running sampling for {subfolder.name}...")

                subprocess.run(
                    ["python", "main.py"],
                    cwd=subfolder,
                    check=True,
                )

                print(f"Completed {subfolder.name}.")

            except Exception as exc:
                print(f"Failed in {subfolder.name}: {exc}")

            continue

        run_main_in_subfolders(subfolder)


if __name__ == "__main__":
    run_main_in_subfolders(MAIN_FOLDER)
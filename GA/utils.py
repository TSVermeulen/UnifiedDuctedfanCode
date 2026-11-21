"""
utils
=====

Utility functions used throughout the Genetic Algorithm (GA) workflow(s). 
Includes function to ensure the paths are correctly set up, a function to 
compute the required number of points for the appropriate reference directions, 
and a function to load JSON data. Additionally includes a figsize formatting 
function for matplotlib figures in the postprocessing module. 

Versioning
----------
Author: T.S. Vermeulen
Email: T.S.Vermeulen@tudelft.nl
Version: 1.2

Changelog:
- V1.0: Initial implementation.
- V1.1: Implemented reference direction point counter to ensure correct number 
        of reference directions are used in the optimisation.
        Updated documentation and switched from sys.path.append to 
        sys.path.insert(0, path) to ensure local code wins over any global 
        packages.
- V1.2: Added load_json_file function to load configuration data from a JSON 
        file. Updated documentation and added type hints for better code 
        clarity.
"""

# Import standard libraries
import sys
import math
import json
from typing import Any, Dict
from pathlib import Path


def ensure_repo_paths() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    submodels = repo_root / "Submodels"
    validation = repo_root / "Validation"
    ga = repo_root / "GA"
    for p in (repo_root, submodels, validation, ga):
        path_str = str(p)
        if path_str not in sys.path:
            # Pre-pend so local code wins over any globally installed packages
            sys.path.insert(0, path_str)


def calculate_n_reference_points(cfg: object) -> int:
    """
    Calculate the number of points needed to construct
    the right number of reference directions using a
    binomial coefficient.

    Parameters
    ----------
    - cfg: object
        Configuration object.
        We cannot import config directly in this file, since config already uses
        ensure_repo_paths, which would result in a circular import error.
        Must have the n_objectives and POPULATION_SIZE attributes.

    Returns
    -------
    - p : int
        The number of points needed
    """

    # Load data from JSON files
    optimisation_data = load_json_data("optimisation.json")

    try:
        m = cfg.n_objectives
    except AttributeError as e:
        raise AttributeError(f"Configuration object is missing required attribute: {e}") from e

    if m == 1:
        # Single objective uses only 1 reference point
        return m

    max_iter = 10000  # hard-stop for safety; tweak if needed
    population_size = optimisation_data["POPULATION_SIZE"]
    best_diff = float('inf')
    best_p = 1

    for p in range(1, max_iter):
        count = math.comb(p + m - 1, m - 1)
        diff = abs(count - population_size)
        if diff < best_diff:
            best_diff = diff
            best_p = p

            if diff == 0:
                # If an exact match is found, break from the for loop
                break
    return best_p


def get_figsize(columnwidth=448.1309, wf=0.5, hf=(5.**0.5-1.0)/2.0, ):
    """
    Parameters
      - wf [float]:  width fraction in columnwidth units
      - hf [float]:  height fraction in columnwidth units.
                     Set by default to golden ratio.
      - columnwidth [float]: width of the column in pt in latex. Get this from LaTeX
                             using \\showthe\\columnwidth
    Returns:  [fig_width,fig_height]: that should be given to matplotlib
    """

    fig_width_pt = columnwidth*wf
    inches_per_pt = 1.0/72.27               # Convert pt to inch
    fig_width = fig_width_pt*inches_per_pt  # width in inches
    fig_height = fig_width*hf      # height in inches
    return [fig_width, fig_height]


def load_json_data(filename: str) -> Dict[str, Any]:
    """
    Load data from a JSON file.

    Args:
        filename (str): The name of the JSON file containing the
        data.

    Returns:
        Dict[str, Any]: A dictionary containing the data.
    """

    path = Path(__file__).parent / "config" / filename
    with open(path, 'r') as file:
        data = json.load(file)

    return data


if __name__ == "__main__":
    import config
    print(calculate_n_reference_points(config))
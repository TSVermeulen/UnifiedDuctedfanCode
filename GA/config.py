"""
config
======

Description
-----------
This module defines the configuration settings and parameters for the optimisation problem, including aerodynamic
analysis, design variables, and constraints. It integrates with the MTFLOW executable for aerodynamic analysis.

Notes
-----
Ensure that the MTFLOW executable and required input files are present in the appropriate directories. This module
provides the necessary settings for optimisation and analysis.

References
----------
For more details on the MTFLOW solver and its input/output requirements, refer to the MTFLOW user manual:
https://web.mit.edu/drela/Public/web/mtflow/mtflow.pdf

Versioning
----------
Author: T.S. Vermeulen
Email: T.S.Vermeulen@tudelft.nl
Version: 1.5

Changelog:
- V1.0: Initial implementation.
- V1.1: Removed need for context manager by using absolute paths.
- V1.2: Wrapped blading generation routine in LRU cache to avoid
        re-running the function at every GA worker import.
- V1.3: Updated single point operating condition to correspond to endurance cruise condition at approximate mid cruise weight and endurance speed of 125kts at 10000ft standard day.
- V1.4: Updated operating conditions. Improved consistency. Added additional inputs.
- V1.5: Started migration towards .json based configuration. 
        Formatted for backwards compatibility. 
"""

# Import standard libraries
from enum import IntEnum, auto

# Import 3rd party libraries
import numpy as np
from ambiance import Atmosphere

# Ensure all paths are correctly setup
from utils import ensure_repo_paths, load_json_data  # type: ignore  # Module not in typeshed or py.typed missing
ensure_repo_paths()

# ------------------------------
# Load operating conditions and process them for backwards compatibility
# ------------------------------

operating_data = load_json_data("operating_conditions.json")

multi_oper = operating_data["operating_conditions"]
P_ref_constr = []  # List of reference condition powers
T_ref_constr = []  # List of reference condition thrust levels
reference_energy = 0  # Reference energy for multi-point optimisation
for oper_dict in multi_oper:
    oper_dict["atmos"] = Atmosphere(oper_dict["Altitude"])
    oper_dict["Vinl"] = oper_dict["atmos"].speed_of_sound[0] * oper_dict["Inlet_Mach"]
    P_ref_constr.append(oper_dict["reference_power"])
    T_ref_constr.append(oper_dict["reference_thrust"])
    reference_energy += oper_dict["reference_power"] * oper_dict["flight_phase_time"]

# ------------------------------
# Define the objective IDs used to construct the objective functions
# ------------------------------

class ObjectiveID(IntEnum):
    """
    Enumeration of objective function identifiers for the optimisation problem.
    Each member represents a different optimisation objective that can be used 
    in the genetic algorithm's fitness evaluation.
    """
    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        return count  # This makes the first member 0 rather than the default 1.

    EFFICIENCY = auto()
    FRONTAL_AREA = auto()
    WETTED_AREA = auto()
    PRESSURE_RATIO = auto()
    ENERGY = auto()

# Calculate the total number of objectives
# Note: Some objectives like energy or frontal area are computed once across all operating points
objective_IDs = [ObjectiveID.EFFICIENCY]  # Must be defined in order of which they exist in the enum!
_single_point_only = {ObjectiveID.FRONTAL_AREA, ObjectiveID.WETTED_AREA, ObjectiveID.ENERGY}
n_objectives = len(objective_IDs) * len(multi_oper) \
               - sum(1 for obj in objective_IDs if obj in _single_point_only) * (len(multi_oper) - 1)

# ------------------------------
# Load reference design and geometry for backward compatibility
# ------------------------------

baseline_design = load_json_data("baseline_design.json")

CENTERBODY_VALUES = baseline_design["CENTERBODY_VALUES"]
DUCT_VALUES = baseline_design["DUCT_VALUES"]
STAGE_BLADING_PARAMETERS = baseline_design["blading_parameters"]
STAGE_DESIGN_VARIABLES = baseline_design["design_parameters"]

geometry = load_json_data("geometry.json")

REF_FRONTAL_AREA = geometry["REF_FRONTAL_AREA"]  # m^2
ROTATING = geometry["ROTATING"]  # List of booleans indicating whether each stage is rotating or not
NUM_RADIALSECTIONS = geometry["NUM_RADIALSECTIONS"]  # Note that we cannot use more than 16 radial sections due to limitations of MTFLOW. Advice from the user manual: ~5 or less is good.
BLADE_DIAMETERS = geometry["REFERENCE_STAGE_DIAMETERS"]  # m, reference diameters of the stages
tipGap = geometry["TIP_GAP"]  # m, tip gap of the ducted fan

# ------------------------------
# Load optimisation configuration settings for backward compatibility
# ------------------------------

optimisation_config = load_json_data("optimisation.json")

GLOBAL_SEED = optimisation_config["GLOBAL_SEED"]
OPTIMIZE_CENTERBODY = optimisation_config["OPTIMISE_CENTERBODY"]
OPTIMIZE_DUCT = optimisation_config["OPTIMISE_DUCT"]
OPTIMIZE_BLADETHICKNESS = optimisation_config["OPTIMISE_BLADETHICKNESS"]
OPTIMIZE_STAGE = optimisation_config["OPTIMISE_STAGE"]
POPULATION_SIZE = optimisation_config["POPULATION_SIZE"]
INITIAL_POPULATION_SIZE = optimisation_config["INITIAL_POPULATION_SIZE"]
MAX_GENERATIONS = optimisation_config["MAX_GENERATIONS"]
SLIDING_WINDOW_SIZE = optimisation_config["SLIDING_WINDOW_SIZE"]
CHECKPOINT_INTERVAL = optimisation_config["CHECKPOINT_INTERVAL"]
SPREAD_CONTINUOUS = optimisation_config["SPREAD_CONTINUOUS"]  # Relative spread (+/- %) applied to continous variables around their reference values
ZERO_NOISE = optimisation_config["ZERO_NOISE"]  # % noise added to zero values to avoid stagnation
SPREAD_DISCRETE = (optimisation_config["SPREAD_DISCRETE_MIN"], optimisation_config["SPREAD_DISCRETE_MAX"])  # Absolute range for discrete variables (reference value to reference value + 12)
PROFILE_FEASIBILITY_OFFSET = optimisation_config["PROFILE_FEASIBILITY_OFFSET"]  # Offset value to avoid bezier control points lying on x_t/x_c
MAX_ONE2ONE_ATTEMPTS = optimisation_config["MAX_ONE2ONE_ATTEMPTS"]  # Maximum number of attempts to enforce one-to-one on the profile parameterisation.
RESERVE_THREAD = optimisation_config["RESERVE_THREAD"]  # Researve a thread for the operating system and any other programs.
deviation_range = optimisation_config["CONSTRAINT_DEVIATION_RANGE"]  # +/- x% of the reference value for the constraints
MAX_FRONTAL_AREA_RATIO = optimisation_config["MAX_FRONTAL_AREA_RATIO"]  # Maximum ratio of the frontal area to the reference frontal area

# Define the constraint IDs used to construct the constraint functions
# constraint IDs are structured as a nested list, of the form:
# [[inequality constraint 1, inequality constraint 2, ...],
#  [equality constraint 1, equality constraint 2, ...]]
class InEqConstraintID(IntEnum):
    """
    Enumeration of the inequality constraint identifiers for the optimisation problem.
    Each member represents a different inequality constraint that can be used in the genetic algorithm's evaluation.
    """
    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        return count  # This makes the first member 0 rather than the default 1.

    EFFICIENCY_LEQ_THEOR_LIMIT = auto()
    THRUST_FEASIBILITY = auto()
    MAXIMUM_FRONTAL_AREA = auto()
    ENERGY_IMPROVEMENT = auto()

class EqConstraintID(IntEnum):
    """
    Enumeration of the equality constraint identifiers for the optimisation problem.
    Each member represents a different equality constraint that can be used in the genetic algorithm's evaluation.
    """
    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        return count  # This makes the first member 0 rather than the default 1.

    CONSTANT_POWER = auto()

constraint_IDs = [[InEqConstraintID.EFFICIENCY_LEQ_THEOR_LIMIT, InEqConstraintID.THRUST_FEASIBILITY],
                  []]

# Postprocessing visualisation controls
# ref_objectives = np.array([-0.74376, 1])  # ref objective values for endurance cruise condition
# ref_objectives = np.array([-0.66469])  # ref objective values for take-off condition
# ref_objectives = np.array([-0.7645])  # ref objective values for combat condition
ref_objectives = np.array([1])  # reference energy objective value for a multi-point problem

"""
init_designvector
==================

Description
-----------
This module defines the DesignVector class, which is used to construct the
design vector for the pymoo framework optimisation problem.
The design vector supports mixed-variable optimisation,
including real and integer variables.

Classes
-------
DesignVector
    Class for constructing the design vector based on configuration toggles.

Examples
--------
>>> from init_designvector import DesignVector
>>> design_vector = DesignVector.construct_vector()

Notes
-----
Ensure the optimisation.json, geometry.json, operating_conditions.json, and
baseline_design.json files are correctly configured in the GA/config directory.

Versioning
----------
Author: T.S. Vermeulen
Email: T.S.Vermeulen@tudelft.nl
Version: 2.0

Changelog:
- V1.0: Initial implementation. Extracted from the problem_definition.py file
        for better modularity and readability.
- V1.1: Implemented optional variable pitch handling.
- V1.2: Made blade thickness distributions optional in the design vector.
        Controlled in config using the OPTIMIZE_BLADETHICKNESS boolean.
- V1.3: Changed parameterisation to use blade root chord + taper distribution
        rather than blade chord distribution. This avoids needing to specify
        a chord repair operator.
- V2.0: Updated to use new .json file-based configuration system.
"""

# Import standard libraries
from typing import ClassVar

# Import 3rd party libraries
import numpy as np
from pymoo.core.variable import Real, Integer

# Import utilities
from utils import load_json_data


class DesignVector:
    """
    This class is used to construct the design vector for the
    optimisation problem.
    """

    # Define the bounds on the BP3434 parameterisation method.
    BP_3434_bounds = {"b_0": (0.05, 0.1),
                      "b_2": (0.125, 0.3),
                      "b_8": (0.05, 0.7),
                      "b_15": (0, 0.9),
                      "b_17": (0, 0.9),
                      "x_t": (0.15, 0.4),
                      "y_t": (0.02, 0.3),
                      "x_c": (0.2, 0.5),
                      "y_c": (0, 0.2),
                      "z_TE": (0, 0.01),
                      "dz_TE": (0, 0.001),
                      "r_LE": (-0.2, -0.001),
                      "trailing_wedge_angle": (0.001, 0.3),
                      "trailing_camberline_angle": (0.05, 0.1),
                      "leading_edge_direction": (0.05, 0.2)}

    # Initialise the profile vars as None
    _cached_profile_vars: ClassVar[list | None] = None
    _cached_camberonly_profile_vars: ClassVar[list | None] = None

    # Indices for the thickness parameters for the centerbody parameters
    # We force the centerbody to be symmetric, so camber is not needed
    CENTERBODY_VAR_INDICES: ClassVar[list[int]] = [2, 3, 5, 6, 10, 11, 12]


    @classmethod
    def _create_profile_vars(cls,
                             include_thickness: bool = True) -> list:
        """
        Create the profile section variables once and cache them.

        Parameters
        ----------
        - include_thickness : bool, optional
            Optional control bool to decide if the profile camber distribution
            should be part of the profile variables.
        """
        if include_thickness:
            return [Real(bounds=cls.BP_3434_bounds["b_0"]),
                    Real(bounds=cls.BP_3434_bounds["b_2"]),
                    Real(bounds=cls.BP_3434_bounds["b_8"]),  # b8 mapping variable
                    Real(bounds=cls.BP_3434_bounds["b_15"]),
                    Real(bounds=cls.BP_3434_bounds["b_17"]),
                    Real(bounds=cls.BP_3434_bounds["x_t"]),
                    Real(bounds=cls.BP_3434_bounds["y_t"]),
                    Real(bounds=cls.BP_3434_bounds["x_c"]),
                    Real(bounds=cls.BP_3434_bounds["y_c"]),
                    Real(bounds=cls.BP_3434_bounds["z_TE"]),
                    Real(bounds=cls.BP_3434_bounds["dz_TE"]),
                    Real(bounds=cls.BP_3434_bounds["r_LE"]),
                    Real(bounds=cls.BP_3434_bounds["trailing_wedge_angle"]),
                    Real(bounds=cls.BP_3434_bounds["trailing_camberline_angle"]),
                    Real(bounds=cls.BP_3434_bounds["leading_edge_direction"])]
        else:
            return [Real(bounds=cls.BP_3434_bounds["b_0"]),
                    Real(bounds=cls.BP_3434_bounds["b_2"]),
                    Real(bounds=cls.BP_3434_bounds["b_17"]),
                    Real(bounds=cls.BP_3434_bounds["x_c"]),
                    Real(bounds=cls.BP_3434_bounds["y_c"]),
                    Real(bounds=cls.BP_3434_bounds["z_TE"]),
                    Real(bounds=cls.BP_3434_bounds["trailing_camberline_angle"]),
                    Real(bounds=cls.BP_3434_bounds["leading_edge_direction"])]


    @classmethod
    def profile_section_vars(cls) -> tuple[list, list]:
        """
        Return the standard profile section definitions.
        Bounds are based on those presented in:
            Rogalsky T. Acceleration of differential evolution for aerodynamic design.
            Ph.D. Thesis, University of Manitoba; 2004.
        """

        if cls._cached_profile_vars is None:
            cls._cached_profile_vars = cls._create_profile_vars(include_thickness=True)
        if cls._cached_camberonly_profile_vars is None:
            cls._cached_camberonly_profile_vars = cls._create_profile_vars(include_thickness=False)

        # Return copies to prevent modification.
        return cls._cached_profile_vars.copy(), cls._cached_camberonly_profile_vars.copy()


    @classmethod
    def construct_vector(cls) -> dict:
        """
        Initialise the pymoo design vector based on the toggles in config.

        Parameters
        ----------
        - cls : class
            The class. used to access the class-attribute BP_3434_bounds.

        Returns
        -------
        - dict
            A dictionary containing the design vector variables and their bounds.
        """

        # Load optimisation config data from JSON files
        optimisation_config = load_json_data("optimisation.json")
        geometry_config = load_json_data("geometry.json")
        operating_conditions = load_json_data("operating_conditions.json")
        baseline_design = load_json_data("baseline_design.json")

        # Initialise variable list with variable types.
        # This is required to handle the mixed-variable nature of the
        # optimisation, where the blade count is an integer
        vector = []
        if optimisation_config["OPTIMISE_CENTERBODY"]:
            # If the centerbody is to be optimised, initialise the variables
            complete_profile = cls.profile_section_vars()[0]
            section_profile = [complete_profile[i] for i in cls.CENTERBODY_VAR_INDICES]

            vector.extend(section_profile)
            vector.append(Real(bounds=(0.25, 4)))  # Chord Length
        if optimisation_config["OPTIMISE_DUCT"]:
            # If the duct is to be optimised, intialise the variables
            duct_profile = cls.profile_section_vars()[0]
            duct_profile[6] = Real(bounds=(0.05, 0.2))  # set y_t for the duct
            vector.extend(duct_profile)
            vector.append(Real(bounds=(1.0, 1.5)))  # Chord Length
            vector.append(Real(bounds=(-0.5, 0.5)))  # Leading Edge X-Coordinate

        # Pre-calculate the stage parameters to avoid repeated calculations
        stage_configs = []
        for i in range(len(geometry_config["ROTATING"])):
            if optimisation_config["OPTIMISE_STAGE"][i]:
                num_radial_sections = geometry_config["NUM_RADIALSECTIONS"][i]
                stage_configs.append({
                    'index': i,
                    'num_radial_sections': num_radial_sections,
                    'is_rotating': geometry_config["ROTATING"][i],
                    'pitch_angles': len(baseline_design["blading_parameters"][i]["ref_blade_angles_deg"])
                })

        # For the stage(s) marked for optimisation, add the profile design
        # variables to the design vector. Control of the thickness distribution
        # is global for all blade rows, so compute it outside of the loop
        if optimisation_config["OPTIMISE_BLADETHICKNESS"]:
            section_blade_profile = cls.profile_section_vars()[0]
        else:
            section_blade_profile = cls.profile_section_vars()[1]
        for stage_config in stage_configs:
            for _ in range(stage_config['num_radial_sections']):
                vector.extend(section_blade_profile)

        # Add stage-specific variables
        for stage_config in stage_configs:
            i = stage_config['index']
            vector.append(Real(bounds=(0, 0.4)))  # root_LE_coordinate
            for _ in range(stage_config["pitch_angles"]):  # Assign each pitch angle to in case of variable pitch
                vector.append(Real(bounds=(0.1, 5 * np.pi/18)))  # ref_blade_angle from [~5.7deg to 50 deg]
            vector.append(Integer(bounds=(3, 15)))  # blade_count
            if stage_config['is_rotating']:
                for _ in range(operating_conditions["num_operating_points"]):
                    vector.append(Real(bounds=(20, 80)))  # blade RPS
            vector.append(Real(bounds=(1.0, 3.0)))  # blade diameter

            vector.append(Real(bounds=(0.1,0.75)))  # root chord
            for _ in range(stage_config['num_radial_sections'] - 1):  # Note the -1 since the root section has no taper
                vector.append(Real(bounds=(0.75, 1)))  # blade taper ratio
            for _ in range(stage_config['num_radial_sections'] - 1):  # Note the -1 since the root section is independent of sweep.
                vector.append(Real(bounds=(0, np.pi/3)))  # sweep_angle
            for _ in range(stage_config['num_radial_sections'] - 1):  # Note the -1 since the tip has a fixed angle at 0
                vector.append(Real(bounds=(0, np.pi/3)))  # blade_angle

        # For a mixed-variable problem, PyMoo expects the vector to be a
        # dictionary, so we convert vector to a dictionary. Note that all
        # variables are given a name xi.
        var_names = [f"x{i}" for i in range(len(vector))]
        vector = dict(zip(var_names, vector))

        return vector


if __name__ == "__main__":
    test = DesignVector()
    vector = test.construct_vector()
    print(vector)

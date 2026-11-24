"""
init_population
===============

Description
-----------
This module provides functionality to initialise populations for optimisation
problems by generating a biased population based on a reference design

Classes
-------
InitPopulation
    A class to initialise populations for optimisation problems.

Examples
--------
>>> import time

>>> start_time = time.time()
>>> init_pop = InitPopulation(seed=42)
>>> population = init_pop.GeneratePopulation()
>>> print(population)
>>> print(f"Population generation took: {time.time() - start_time} seconds")

Notes
-----
Ensure the optimisation.json, geometry.json, operating_conditions.json, and
baseline_design.json files are correctly configured in the GA/config directory.


References
----------
pymoo: https://pymoo.org/

Versioning
----------
Author: T.S. Vermeulen
Email: T.S.Vermeulen@tudelft.nl
Version: 2.0

Changelog:
- V1.0: Initial version with tested biased population generation and basic
        random population generation functionality.
- V1.1: Added support for configurable random seed for reproducibility and
        improved documentation. Reworked GenerateBiasedPopulation to improve
        speed using NumPy.
- V1.2: Implemented perturbation for zero-valued real design parameters.
- V1.3: Implemented optional variable pitch handling.
- V1.4: Added support for optional blade thickness distributions to reduce
        dimensionality of problem. Controlled through the
        config.OPTIMIZE_BLADETHICKNESS boolean.
- V1.5: Updated parameterisation to use root chord + taper distribution rather
        than chord distribution. Removed option to use random initialisation.
- V2.0: Updated to use .json-based configuration files.
"""

# Import standard libraries
import numbers

# Import 3rd party libraries
import numpy as np
from pymoo.core.population import Population
from pymoo.core.individual import Individual

# Import interfacing modules
from init_designvector import DesignVector  # type: ignore
from utils import load_json_data  # type: ignore


class InitPopulation():
    """
    Population generation class to handle generation of the starting point for
    the genetic algorithm optimisation.
    """


    def __init__(self,
                 **kwargs) -> None:
        """
        Initialisation for the InitPopulation class.

        Parameters
        ----------
        - **kwargs : dict[str, Any], Optional
            Additional keyword arguments

        Returns
        -------
        None
        """

        # Create a new design vector object
        self.design_vector = DesignVector().construct_vector()
        self.design_vector_keys = list(self.design_vector.keys())

        # Load optimisation config data from JSON files
        self.opt_cfg = load_json_data("optimisation.json")
        self.geom_cfg = load_json_data("geometry.json")
        self.operating_conditions = load_json_data("operating_conditions.json")
        self.baseline_design = load_json_data("baseline_design.json")

        # Set the seed for the random number generator to ensure reproducibility
        seed = kwargs.get("seed", self.opt_cfg["GLOBAL_SEED"])
        self._np_rng = np.random.default_rng(seed)
        np.random.seed(seed)


    def DeconstructDictFromReferenceDesign(self) -> dict:
        """
        Deconstruct the reference design vector dictionaries back into a pymoo
        design vector. This is used to create the initial population for the
        optimisation problem.

        Parameters
        ----------
        None

        Returns
        -------
        - vector : dict
            A dict of design variables for the optimisation problem.
        """

        # Define a helper function to map the b_8 variable
        def map_b8_value(variable_dict: dict):
            """
            Compute a normalised b_8 mapping variables on [0, 1].
            This normalisation ensures the b_8 variable always matches the constraint:
                             0 <= b_8 <= min(y_t, sqrt(-2*r_LE*x_t/3))
            """

            denominator = min(variable_dict["y_t"],
                              np.sqrt(max(0.0, -2 * variable_dict["x_t"] * variable_dict["r_LE"] / 3)))
            return float(variable_dict["b_8"] / denominator) if denominator > 0 else 0.0

        vector = []
        if self.opt_cfg["OPTIMISE_CENTERBODY"]:
            # If the centerbody is to be optimised, read in the reference design
            # dictionary and extract the design vector x
            centerbody_vals = self.baseline_design["CENTERBODY_VALUES"]
            vector.append(map_b8_value(centerbody_vals))
            vector.append(centerbody_vals["b_15"])
            vector.append(centerbody_vals["x_t"])
            vector.append(centerbody_vals["y_t"])
            vector.append(centerbody_vals["dz_TE"])
            vector.append(centerbody_vals["r_LE"])
            vector.append(centerbody_vals["trailing_wedge_angle"])
            vector.append(centerbody_vals["Chord Length"])

        if self.opt_cfg["OPTIMISE_DUCT"]:
            # If the duct is to be optimised, read the reference values into
            # the design vector
            duct_vals = self.baseline_design["DUCT_VALUES"]
            vector.append(duct_vals["b_0"])
            vector.append(duct_vals["b_2"])
            vector.append(map_b8_value(duct_vals))
            vector.append(duct_vals["b_15"])
            vector.append(duct_vals["b_17"])
            vector.append(duct_vals["x_t"])
            vector.append(duct_vals["y_t"])
            vector.append(duct_vals["x_c"])
            vector.append(duct_vals["y_c"])
            vector.append(duct_vals["z_TE"])
            vector.append(duct_vals["dz_TE"])
            vector.append(duct_vals["r_LE"])
            vector.append(duct_vals["trailing_wedge_angle"])
            vector.append(duct_vals["trailing_camberline_angle"])
            vector.append(duct_vals["leading_edge_direction"])
            vector.append(duct_vals["Chord Length"])
            vector.append(duct_vals["Leading Edge X-Coordinate"])

        for i, stage_opt in enumerate(self.opt_cfg["OPTIMISE_STAGE"]):
            # If the blade rows are to be optimised, read the reference values
            # into the design vector to create the blade design parameters
            if stage_opt:
                # Read the reference values into the design vector
                for j in range(self.geom_cfg["NUM_RADIALSECTIONS"][i]):
                    vector.append(self.baseline_design["design_parameters"][i][j]["b_0"])
                    vector.append(self.baseline_design["design_parameters"][i][j]["b_2"])
                    if self.opt_cfg["OPTIMISE_BLADETHICKNESS"]:
                        vector.append(map_b8_value(self.baseline_design["design_parameters"][i][j]))
                        vector.append(self.baseline_design["design_parameters"][i][j]["b_15"])
                    vector.append(self.baseline_design["design_parameters"][i][j]["b_17"])
                    if self.opt_cfg["OPTIMISE_BLADETHICKNESS"]:
                        vector.append(self.baseline_design["design_parameters"][i][j]["x_t"])
                        vector.append(self.baseline_design["design_parameters"][i][j]["y_t"])
                    vector.append(self.baseline_design["design_parameters"][i][j]["x_c"])
                    vector.append(self.baseline_design["design_parameters"][i][j]["y_c"])
                    vector.append(self.baseline_design["design_parameters"][i][j]["z_TE"])
                    if self.opt_cfg["OPTIMISE_BLADETHICKNESS"]:
                        vector.append(self.baseline_design["design_parameters"][i][j]["dz_TE"])
                        vector.append(self.baseline_design["design_parameters"][i][j]["r_LE"])
                        vector.append(self.baseline_design["design_parameters"][i][j]["trailing_wedge_angle"])
                    vector.append(self.baseline_design["design_parameters"][i][j]["trailing_camberline_angle"])
                    vector.append(self.baseline_design["design_parameters"][i][j]["leading_edge_direction"])

        for i, stage_opt in enumerate(self.opt_cfg["OPTIMISE_STAGE"]):
            # If the blade rows are to be optimised, read the reference values into the design vector
            if stage_opt:
                # Read the reference values into the design vector
                vector.append(self.baseline_design["blading_parameters"][i]["root_LE_coordinate"])
                for angle in self.baseline_design["blading_parameters"][i]["ref_blade_angles_deg"]:
                    vector.append(float(np.deg2rad(angle)))  # Allow for variable pitch by giving each operating point a separate blade angle
                vector.append(int(self.baseline_design["blading_parameters"][i]["blade_count"]))
                if self.geom_cfg["ROTATING"][i]:
                    for _ in range(self.operating_conditions["num_operating_points"]):
                        vector.append(self.operating_conditions["operating_conditions"][i]["RPS"]) # Required for multipoint analysis

                # The interfaces use the radial locations, but the design varable is the blade diameter!
                vector.append(self.baseline_design["blading_parameters"][i]["radial_stations"][-1] * 2)

                vector.append(self.baseline_design["blading_parameters"][i]["root_chord"])
                vector.extend(self.baseline_design["blading_parameters"][i]["taper_distribution"])
                vector.extend(np.deg2rad(self.baseline_design["blading_parameters"][i]["sweep_angle_deg"][1:]).tolist())  # Second point onwards since root has no sweep
                vector.extend(np.deg2rad(self.baseline_design["blading_parameters"][i]["blade_angle_deg"][:-1]).tolist())  # Up to second to last point since tip angle is defined separately

        # Change vector from a list to a dictionary to match the expected structure of pymoo
        keys = list(self.design_vector.keys())

        # Ensure the vector is the same length as the design vector keys
        vector = {key: value for key, value in zip(keys, vector)}

        return vector


    def GeneratePopulation(self) -> Population:
        """
        Generate the initial population for the optimisation problem based on
        an existing design vector.

        Parameters
        ----------
        None

        Returns
        -------
        - pop : Population
            The initial population for the optimisation problem as a pymoo
            Population object.
        """

        # Get the invidivual corresponding to the reference design
        reference_individual = self.DeconstructDictFromReferenceDesign()
        ref = np.array([reference_individual[k] for k in self.design_vector_keys],
                       dtype=float)

        # Extract the lower and upper bounds of the design variables
        lower_bounds = np.array([self.design_vector[k].bounds[0] for k in self.design_vector_keys],
                                dtype=float)
        upper_bounds = np.array([self.design_vector[k].bounds[1] for k in self.design_vector_keys],
                                dtype=float)

        # Verify the reference individual is within the bounds
        ref = np.clip(ref, lower_bounds, upper_bounds)

        # Generate the initial population equal to the POPULATION_SIZE
        # reference_individuals
        pop_dict = [reference_individual.copy() for _ in range(self.opt_cfg["POPULATION_SIZE"])]

        # Compute masks for the floating point and integer design variables
        real_mask = np.array([np.issubdtype(type(v),
                                            np.floating) for v in reference_individual.values()])
        int_mask  = np.array([np.issubdtype(type(v),
                                            np.integer)  for v in reference_individual.values()])
        other_mask = ~(real_mask | int_mask)
        if other_mask.any():
            raise TypeError("Detected unsupported Non-scalar design variable(s)")

        # Compute masks to check which values of the floating point
        # variables are zero
        zero_real_mask = real_mask & (ref == 0)
        nonzero_real_mask = real_mask & (ref != 0)
        span = upper_bounds - lower_bounds

        for i in range(1, len(pop_dict)):
            # Generate some noise for the floating point variables
            # Use uniform noise to ensure an equal sampling across
            # the design space.
            noise = self._np_rng.uniform(-1, 1, size=ref.shape)

            # Apply perturbations
            # For nonzero real values, we use a simple noise perturbation
            perturbed_individual = ref.copy()
            perturbed_individual[nonzero_real_mask] += noise[nonzero_real_mask] * perturbed_individual[nonzero_real_mask] * self.opt_cfg["SPREAD_CONTINUOUS"]
            # For zero real values, we use a perturbation which is equal
            # to some constant value times the design variable span
            perturbed_individual[zero_real_mask] += noise[zero_real_mask] * span[zero_real_mask] * self.opt_cfg["ZERO_NOISE"] * self.opt_cfg["SPREAD_CONTINUOUS"]
            # For integer values, we use an integer noise function
            perturbed_individual[int_mask] += self._np_rng.integers(self.opt_cfg["SPREAD_DISCRETE_MIN"],
                                                                    self.opt_cfg["SPREAD_DISCRETE_MAX"],
                                                                    size=int_mask.sum())

            # Ensure perturbed individual still falls within
            # the design variable bounds
            perturbed_individual = np.clip(perturbed_individual, lower_bounds, upper_bounds)

            # Convert to dicitonary while casting the discrete variable(s)
            # back to int
            pop_dict[i] = {key: (int(val) if isinstance(reference_individual[key], numbers.Integral) else val)
                           for key, val in zip(self.design_vector_keys, perturbed_individual)}

        # Construct the population object
        individuals = []
        for pop in pop_dict:
            individuals.append(Individual(X=pop))
        pop = Population.create(*individuals)

        return pop


if __name__ == "__main__":
    ref_pop = InitPopulation().GeneratePopulation()

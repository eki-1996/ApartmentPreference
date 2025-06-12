# import experiments.bace.user_config as user_config
import experiments.bace.pmc_inference as pmc_inference
import experiments.bace.design_optimization as design_optimization
from mango import Tuner

# Optional: Random seed for reproducibility
import numpy as np
import random
random.seed(42)
np.random.seed(42)

class BACE():
    def __init__(self, user_config):
        # Track history (outside the function, persist between calls)
        self.design_history = []
        self.observed_answers = []
        self.user_config = user_config

    def interactive_bace_steps(self, random=False):
        # Step 1: Update history
        # self.design_history.append(latest_design)
        # self.observed_answers.append(user_answer)

        # Step 2: Update posterior using PMC
        posterior_samples = pmc_inference.pmc(
            theta_params=self.user_config.theta_params,
            answer_history=self.observed_answers,
            design_history=self.design_history,
            likelihood_pdf=self.user_config.likelihood_pdf,
            N=self.user_config.size_thetas,
            J=5
        )

        # Step 3: Define tuner
        objective = design_optimization.get_objective(self.user_config.answers, self.user_config.likelihood_pdf)
        conf_dict = design_optimization.get_conf_dict(self.user_config.conf_dict)
        tuner = Tuner(self.user_config.design_params, objective, conf_dict)

        # # Step 4: Generate multiple designs
        # next_designs = []
        # for _ in range(num_next_designs):
        #     next_design = design_optimization.get_next_design(
        #         thetas=posterior_samples.copy(),  # ensure sampler isn't mutated in-place
        #         tuner=tuner
        #     )
        #     next_designs.append(next_design)

        # return next_designs

        if random:
            return self.sample_from_design_space(self.user_config.design_params), None
            # return self.get_random_design(tuner)

        return design_optimization.get_next_design(
                thetas=posterior_samples.copy(),  # ensure sampler isn't mutated in-place
                tuner=tuner
            ), posterior_samples


    def add_record(self, latest_design, user_answer):
        self.design_history.append(latest_design)
        self.observed_answers.append(user_answer)

    def get_random_design(self, tuner):
        sample = tuner.ds.get_random_sample(size=1)
        while not sample:
            sample = tuner.ds.get_random_sample(size=1)
        return sample[0]
        # next_design = design_tuner.ds.get_random_sample(size=1)
        # while len(next_design) < 1:
        #     next_design = design_tuner.ds.get_random_sample(size=1)
        # print(next_design)
        # return next_design[0]
    
    def sample_from_design_space(self, design_params):
        sample = {}
        for key, dist in design_params.items():
            if hasattr(dist, "rvs"):  # it's a scipy distribution
                value = dist.rvs()
                # Convert numpy scalar (e.g., np.int64) to native Python type for JSON compatibility
                if hasattr(value, "item"):
                    value = value.item()
                sample[key] = value
            elif isinstance(dist, list):  # categorical options
                sample[key] = random.choice(dist)
            else:
                raise ValueError(f"Unsupported design spec: {key} -> {dist}")
        return sample
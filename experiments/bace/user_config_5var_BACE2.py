# Example configuration file for BACE simulation
#import scipy.stats
import numpy as np
import scipy.stats as stats

# Metadata
author       = 'Pen Example Application'
size_thetas  = 2500                       # Number of samples drawn from the prior
max_opt_time = 5                          # Time limit (in seconds) for Bayesian optimization per round

# Configuration for Mango Bayesian Optimizer
conf_dict = dict(
    domain_size    = 1500,
    initial_random = 1,
    num_iteration  = 15
)

# Possible answer options (binary choice)
answers = [0, 1]

# Preference parameters (prior distributions) — mu has been removed
# Group B configuration
theta_params = dict(
    floor=stats.norm(loc=-2, scale=5),    
    distance_to_university=stats.norm(loc=0.6, scale=5)
)

design_params = dict(
    price_a=stats.uniform(3, 7),
    price_b=stats.uniform(3, 7),
    floor_a=stats.randint(1, 4),
    floor_b=stats.randint(1, 4),
    distance_to_university_a=stats.uniform(5, 25),
    distance_to_university_b=stats.uniform(5, 25)
)

# Likelihood function for Group B
def likelihood_pdf(answer, thetas, design, profile=None):
    base_U_a = (
        - design['price_a']
        + thetas['floor'] * design['floor_a']
        + thetas['distance_to_university'] * design['distance_to_university_a']
    )

    base_U_b = (
        - design['price_b']
        + thetas['floor'] * design['floor_b']                
        + thetas['distance_to_university'] * design['distance_to_university_b']
    )

    base_utility_diff = base_U_b - base_U_a
    likelihood = 1 / (1 + np.exp(-base_utility_diff))

    eps = 1e-10
    likelihood = np.clip(likelihood, eps, 1 - eps)

    return likelihood if str(answer) == '1' else 1 - likelihood


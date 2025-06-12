# Example configuration file for BACE simulation
#import scipy.stats
import numpy as np
import scipy.stats as stats
from scipy.stats import randint

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
theta_params = dict(
    room_size=stats.norm(loc=0.5, scale=5),
    city_center=stats.norm(loc=1, scale=5)
)

design_params = dict(
    price_a=stats.uniform(3, 10),
    price_b=stats.uniform(3, 10),
    room_size_a=stats.uniform(6, 20),
    room_size_b=stats.uniform(6, 20),
    city_center_a=['繁華街', '住宅街'],
    city_center_b=['繁華街', '住宅街']
)


# Likelihood function for Group A
def likelihood_pdf(answer, thetas, design, profile=None):
    base_U_a = (
        - design['price_a']
        + thetas['room_size'] * design['room_size_a']        
        + thetas['city_center'] * (design['city_center_a'] == '繁華街')
    )

    base_U_b = (
        - design['price_b']
        + thetas['room_size'] * design['room_size_b']        
        + thetas['city_center'] * (design['city_center_b'] == '繁華街')
    )

    base_utility_diff = base_U_b - base_U_a
    likelihood = 1 / (1 + np.exp(-base_utility_diff))

    eps = 1e-10
    likelihood = np.clip(likelihood, eps, 1 - eps)

    return likelihood if str(answer) == '1' else 1 - likelihood
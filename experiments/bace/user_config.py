import numpy as np
import scipy.stats
from collections import OrderedDict

# -----------------------------
# PARAMETERS TO BE INFERRED
# -----------------------------
# theta_params = {
#     "rent": {"dist": "norm", "mean": 70000, "sd": 10000},
#     "area": {"dist": "norm", "mean": 25, "sd": 5},
#     "station_distance": {"dist": "norm", "mean": 10, "sd": 3},
#     "campus_distance": {"dist": "norm", "mean": 5, "sd": 2},
#     "building_age": {"dist": "norm", "mean": 20, "sd": 10},
# }
theta_params = dict(
    rent = scipy.stats.norm(loc=40, scale=20),
    area = scipy.stats.norm(loc=25, scale=5),
    station_distance = scipy.stats.norm(loc=10, scale=10),
    campus_distance = scipy.stats.norm(loc=20, scale=20),
    building_age = scipy.stats.norm(loc=20, scale=10),
    
    mu       = scipy.stats.uniform(loc=1, scale=9)
)

# -----------------------------
# DESIGN SPACE
# -----------------------------
def constract_selection_set(start, end, step, unit):
    ret = []
    ret.append(f"{start}{unit}未満")
    if isinstance(step, list):
        for i, item in enumerate(step):
            if i == 0:
                ret.append(f"{start}{unit}以上{start + item}{unit}未満")
            else:
                ret.append(f"{start + sum(step[:i])}{unit}以上{start + sum(step[:i+1])}{unit}未満")
    else:
        for i, index in enumerate(range(start, end, step)):
            ret.append(f"{index}{unit}以上{index+step}{unit}未満")
    ret.append(f"{end}{unit}以上")
    return ret

prior_setting = OrderedDict(
    rent = [30, 100, 5, "千"],
    area = [6, 21, 1, "畳"],
    station_distance = [5, 30, 5, "分"],
    campus_distance = [5, 30, 5, "分"],
    building_age = [1, 50, [4, 5, 10, 10, 10, 10], "年"]
)
design_params = OrderedDict()
for k, v in prior_setting.items():
    if k == "building_age":
        design_params[k+"_a"] = constract_selection_set(start=v[0], end=v[1], step=v[2], unit=v[3])
        design_params[k+"_b"] = constract_selection_set(start=v[0], end=v[1], step=v[2], unit=v[3])
    else:
        design_params[k+"_a"] = constract_selection_set(start=v[0], end=v[1], step=v[2], unit=v[3])
        design_params[k+"_b"] = constract_selection_set(start=v[0], end=v[1], step=v[2], unit=v[3])

# design_params = {
#     "rent": [40000, 120000],
#     "area": [15, 50],
#     "station_distance": [0, 20],
#     "campus_distance": [0, 15],
#     "building_age": [0, 40],
# }

# -----------------------------
# POSSIBLE USER RESPONSES
# -----------------------------
answers = [0, 1]  # No / Yes

# -----------------------------
# LIKELIHOOD MODEL (User Response Model)
# -----------------------------
def likelihood_pdf(answer, theta, design, profile=None):
    """
    Likelihood of user's answer given design and user preference (theta).
    Uses a logistic function.
    """
    # Utility = dot product of weights (theta) and design
    utility = sum(theta[param] * design[param] for param in design)

    # Logistic function to map utility to [0,1]
    prob = 1 / (1 + np.exp(-utility))

    if answer == 1:
        return prob
    else:
        return 1 - prob

# -----------------------------
# Other settings
# -----------------------------
size_thetas = 500
conf_dict = {
    'num_iteration': 20,
    'initial_random': 5,
}

import sys

sys.path.append("/Users/leonid/Desktop/FusionAI/Quantum/Unitary/deltakit/")

from deltakit_explorer.analysis import (calculate_lep_and_lep_stddev, compute_logical_error_per_round)
from deltakit_explorer.plotting import plot_logical_error_probability_per_round

num_failed_shots=[34, 151, 356]
num_shots=[500000] * 3
num_rounds=[2, 4, 6]

res = compute_logical_error_per_round(
     num_failed_shots=num_failed_shots,
     num_shots=num_shots,
     num_rounds=num_rounds)
lep, lep_stddev = calculate_lep_and_lep_stddev(fails=num_failed_shots, \
                                               shots=num_shots)
fig, ax = plot_logical_error_probability_per_round(res,  num_rounds=num_rounds,logical_error_probability=lep,logical_error_probability_stddev=lep_stddev,)

# PREM Earth model description
# Auxiliary functions


import numpy as np


def get_applicable_x(x: np.ndarray, r_min: float, r_max: float):
	return np.all([x >= r_min, x <= r_max], axis=0)
	

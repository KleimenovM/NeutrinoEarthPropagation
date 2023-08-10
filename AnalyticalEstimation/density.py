# PREM Earth model description
# Density distribution 

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from tools import get_applicable_x


# the Earth's radius
A = 6371.0  # km


def inner_core_density(x):
	r_min = .0  	# km
	r_max = 1221.5  # km
	
	new_x = get_applicable_x(x, r_min / A, r_max / A)
	
	return (13.0885 - 8.8381 * x**2) * new_x  # g/cm^3
	
	
def outer_core_density(x):
	r_min = 1221.5  # km
	r_max = 3480.0  # km
	
	new_x = get_applicable_x(x, r_min / A, r_max / A)
	
	return (12.5815 - 1.2638 * x - 3.6426 * x**2 - 5.5281 * x**3) * new_x  # g/cm^3
	

def lower_mantle_density(x):
	r_min = 3480.0  # km
	r_max = 5701.0  # km
	
	new_x = get_applicable_x(x, r_min / A, r_max / A)
	
	return (7.9565 - 6.4761 * x + 5.5283 * x**2 - 3.0807 * x**3) * new_x  # g/cm^3
	
			
def transition_zone_one_density(x):
	r_min = 5701.0  # km
	r_max = 5771.0  # km
	
	new_x = get_applicable_x(x, r_min / A, r_max / A)
	
	return (5.3197 - 1.4836 * x) * new_x  # g/cm^3
			
			
def transition_zone_two_density(x):
	r_min = 5771.0  # km
	r_max = 5971.0  # km
	
	new_x = get_applicable_x(x, r_min / A, r_max / A)
	
	return (11.2494 - 8.0298 * x) * new_x  # g/cm^3
					

def transition_zone_three_density(x):
	r_min = 5971.0  # km
	r_max = 6151.0  # km
	
	new_x = get_applicable_x(x, r_min / A, r_max / A)
	
	return (7.1089 - 3.8045 * x) * new_x  # g/cm^3
			

def lvz_and_lid_density(x):
	r_min = 6151.0  # km
	r_max = 6346.0  # km
	
	new_x = get_applicable_x(x, r_min / A, r_max / A)
	
	return (2.6910 + 0.6924 * x) * new_x  # g/cm^3
			
			
def crust_one_density(x):
	r_min = 6346.0  # km
	r_max = 6356.0  # km
	
	new_x = get_applicable_x(x, r_min / A, r_max / A)
	
	return 2.900 * new_x  # g/cm^3
	
	
def crust_two_density(x):
	r_min = 6356.0  # km
	r_max = 6368.0  # km
	
	new_x = get_applicable_x(x, r_min / A, r_max / A)
	
	return 2.600 * new_x  # g/cm^3


def ocean_density(x):
	r_min = 6368.0  # km
	r_max = 6371.0  # km
	
	new_x = get_applicable_x(x, r_min / A, r_max / A)
	
	return 1.020 * new_x  # g/cm^3
			

def density(x: np.ndarray):
	"""
	:x: np.ndarray - relative radii array
	return: density array
	"""
	
	# x = np.linspace(0, 1, 10**sampling_quality)  # radius sampling
	
	layer_density_functions = [
		inner_core_density, outer_core_density, lower_mantle_density,
		transition_zone_one_density, transition_zone_two_density,
		transition_zone_three_density, lvz_and_lid_density,
		crust_one_density, crust_two_density, ocean_density]
		
	layer_labels = [
		"inner core", "outer core", "lower mantle",
		"transition zone 1", "transition zone 2", "transition zone 3",
		"lvz & lid", "crust 1", "crust 2", "ocean"]
	
	total_density = np.zeros(x.size)
		
	for i, f in enumerate(layer_density_functions):
		y_i = f(x)
		total_density += y_i
	
	return total_density
	
	
if __name__ == "__main__":
	# just plot density distribution
	x = np.linspace(0, 1, 10**6)
	f = density(x)
		
	plt.plot(x, f)
	plt.show()


# PREM Earth model description
# Average molar mass calculator


import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from tools import get_applicable_x


def read_elements_data():
	filename = "layer_parameters.txt"
	mass, mantle, core = np.loadtxt(filename, unpack=True,
									skiprows=1, usecols=(1, 2, 3))
	return mass, mantle, core
	
		

def molar(x: np.ndarray):
	"""
	:x: np.ndarray - relative radii array
	return: molar mass array
	"""
	
	# x = np.linspace(0, 1, 10**sampling_quality)  # radius sampling

	mass, mantle, core = read_elements_data()
	
	r_min = .0  		# km
	r_change = 3480.0  	# km
	r_max = 6371.0		# km
	
	x_change = r_change / r_max
	
	total_molar_mass = np.zeros(x.size)  # final array
	
	# core
	core_x = get_applicable_x(x, .0, x_change)
	average_core = np.sum(mass * core)
	total_molar_mass += average_core * core_x
	# plt.fill_between(x * r_max, np.zeros(x.size), core_x * average_core, alpha=.3, label='core')
	
	# mantle
	mantle_x = get_applicable_x(x, x_change, 1.0)
	average_mantle = np.sum(mass * mantle)
	total_molar_mass += average_mantle * mantle_x
	# plt.fill_between(x * r_max, np.zeros(x.size), mantle_x * average_mantle, alpha=.3, label='mantle and crust')
		
	return total_molar_mass
	
	
if __name__ == "__main__":
	# just plot molar mass distribution
	x = np.linspace(0, 1, 10**6)
	f = molar(x)
	plt.plot(x * 6371.0, f, color='black')
	plt.xlabel(r"$r,\ km$")
	plt.ylabel(r"$\mu,\ g\,/\,mol$")
	plt.legend()
	plt.xlim(0, 6371.0)
	plt.ylim(0)
	plt.grid(linestyle='dashed')
	plt.tight_layout()
	plt.show()
	# plt.savefig("atomic.pdf")
	

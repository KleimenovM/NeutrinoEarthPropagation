# PREM Earth model application for neutrino propagation
# Flux attenuation integral calculation

import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import trapezoid, simpson

from density import density
from molar import molar


def get_number_of_samples(theta: float, sampling_quality: int) -> int:
	# calculate total number of samples
	
	total_samples_number = 10**sampling_quality
	relative_trajectory_length = np.cos(theta)
	
	return int(total_samples_number * relative_trajectory_length)


def propagation_integral(theta: float, sampling_quality: int = 4):
		
	# constant variables
	st, ct = np.cos(theta), np.sin(theta)	
			
	# integration boundaries
	x_0, x_1 = .0, 2 * st
	
	# integration sample
	sample_width = 2 * 10**(-sampling_quality)
	
	x = np.arange(x_0, x_1, sample_width)  	# sampling
	r = np.sqrt(1 + x**2 - 2 * x * st)  	# distance from the center
	d = density(r)  						# density sample
	mm = molar(r)							# molar mass sample
	
	y = d / mm
	
	result = simpson(y, x)
			
	return result  #, x, r, d, mm
	
	
if __name__ == "__main__":
	# plot propagation integral distribution
	
	n = 1000
	theta = np.linspace(0, np.pi / 2, n)
	ans = np.zeros(n)
	
	for i, t in enumerate(theta):
		ans[i] = propagation_integral(t)
	
	plt.plot(theta * 180 / np.pi, ans)
	
	plt.show()
		

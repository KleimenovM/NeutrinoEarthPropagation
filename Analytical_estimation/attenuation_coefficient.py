# PREM Earth model application for neutrino propagation
# Attenuation coefficient calculation

import numpy as np
import ROOT as rt

from cross_section import nu_cross_section, anu_cross_section
from propagation import propagation_integral


N_A = 6.02 * 1e23  	# 1/mol
R = 6371.0 * 1e5  	# cm


def get_nu_attenuation_coefficient(e: np.ndarray, theta: float):
	cross_section = nu_cross_section(e)
	propagation_coefficient = propagation_integral(theta)
	return np.exp(-R * N_A * cross_section * propagation_coefficient)
	

def get_anu_attenuation_coefficient(e: np.ndarray, theta: float):
	cross_section = anu_cross_section(e)
	propagation_coefficient = propagation_integral(theta)
	return np.exp(-R * N_A * cross_section * propagation_coefficient)
	
	
if __name__ == "__main__":
	n_theta, n_e = 10, 10
	theta = np.linspace(0, np.pi/2, n_theta)
	lg_e = np.linspace(4, 15, n_e)  # lg(energy)
	
	e_theta_matrix = np.zeros([2, n_theta, n_e])
	for i, t in enumerate(theta):
		e_theta_matrix[0, i] = get_nu_attenuation_coefficient(lg_e, theta[i])
		e_theta_matrix[1, i] = get_anu_attenuation_coefficient(lg_e, theta[i])
	
	print(e_theta_matrix)	


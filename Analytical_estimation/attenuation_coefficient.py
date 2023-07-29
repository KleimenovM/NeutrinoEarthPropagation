# PREM Earth model application for neutrino propagation
# Attenuation coefficient calculation

import numpy as np
import ROOT as rt

from cross_section import nu_cross_section, anu_cross_section
from propagation import propagation_integral

N_A = 6.02 * 1e23  # 1/mol
R = 6371.0 * 1e5  # cm


def get_nu_attenuation_coefficient(e: np.ndarray, theta: float):
    cross_section = nu_cross_section(e)
    propagation_coefficient = propagation_integral(theta)
    return np.exp(-R * N_A * cross_section * propagation_coefficient)


def get_anu_attenuation_coefficient(e: np.ndarray, theta: float):
    cross_section = anu_cross_section(e)
    propagation_coefficient = propagation_integral(theta)
    return np.exp(-R * N_A * cross_section * propagation_coefficient)


if __name__ == "__main__":
    n_theta, n_e = 30, 30
    theta_sample = np.linspace(0, np.pi / 2, n_theta)
    lg_e = np.linspace(3, 9, n_e)  # lg(energy)

    e_theta_matrix = np.zeros([2, n_theta, n_e])
    for i, t in enumerate(theta_sample):
        e_theta_matrix[0, i] = get_nu_attenuation_coefficient(lg_e, theta_sample[i])
        e_theta_matrix[1, i] = get_anu_attenuation_coefficient(lg_e, theta_sample[i])

    h2 = rt.TH2F("Hist Name", "Hist Title", n_theta, min(theta_sample), max(theta_sample), n_e, min(lg_e), max(lg_e))

    for i, theta_i in enumerate(theta_sample):
        for j, e_j in enumerate(lg_e):
            h2.Fill(theta_i, e_j, e_theta_matrix[0, i, j])

    print(e_theta_matrix.shape)

    h2.Draw("surf2")
    k = input()

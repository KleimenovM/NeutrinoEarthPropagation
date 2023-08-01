# PREM Earth model application for neutrino propagation
# Attenuation coefficient calculation

import numpy as np
import ROOT as rt
import matplotlib.pyplot as plt

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
    """
    n_theta, n_e = 30, 30
    theta_sample = np.linspace(0, np.pi / 2, n_theta)
    lg_e = np.linspace(3, 9, n_e)  # lg(energy)

    e_theta_matrix = np.zeros([2, n_theta, n_e])
    for i, t in enumerate(theta_sample):
        e_theta_matrix[0, i] = get_nu_attenuation_coefficient(lg_e, theta_sample[i])
        e_theta_matrix[1, i] = get_anu_attenuation_coefficient(lg_e, theta_sample[i])

    h2 = rt.TH2F("Hist Name", "Neutrino flux attenuation coefficient", n_theta, min(theta_sample), max(theta_sample), n_e, min(lg_e), max(lg_e))

    for i, theta_i in enumerate(theta_sample):
        for j, e_j in enumerate(lg_e):
            h2.Fill(theta_i, e_j, e_theta_matrix[0, i, j])

    print(e_theta_matrix.shape)
    h2.Draw("surf2")
    h2.GetXaxis().SetTitle("#theta, rad")
    h2.GetYaxis().SetTitle("log_{10}(E / 1 GeV)")
    k = input()
    """
    
    theta1 = np.arcsin(.1)
    theta2 = np.arcsin(.9)
    
    lg_e = np.linspace(3, 10, 100)
    at1 = get_nu_attenuation_coefficient(lg_e, theta1)
    at2 = get_nu_attenuation_coefficient(lg_e, theta2)
    
    plt.figure(figsize=(9, 4))
    plt.subplot(1, 2, 1)
    plt.plot(lg_e, at1, linestyle='dashed', label=r'$\sin\theta = 0.1$')
    plt.xlim(5, 9.7)
    plt.xlabel(r'$\log_{10}(E_\nu[GeV])$')
    plt.ylabel(r'$\kappa$')
    plt.grid(linestyle='dashed')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(lg_e, at2, linestyle='dashed', label=r'$\sin\theta = 0.9$')
    plt.xlim(3, 6.5)
    plt.xlabel(r'$\log_{10}(E_\nu[GeV])$')
    plt.ylabel(r'$\kappa$')
    plt.grid(linestyle='dashed')
    plt.legend()
    
    plt.tight_layout()
    
    plt.savefig("attenuation-ref.png")
    plt.show()
    

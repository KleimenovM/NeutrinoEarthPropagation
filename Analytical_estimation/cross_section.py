# PREM Earth model application for neutrino propagation
# Neutrino total cross-section parametrization
# ref: http://dx.doi.org/10.1103/PhysRevD.83.113009

import numpy as np
import matplotlib.pyplot as plt

NC, CC, ANC, ACC = (np.loadtxt("txt_data/cross_section_parameters.txt",
                              unpack=True, skiprows=2, usecols=(1, 2, 3, 4, 5))).T
ln10 = np.log(10)


def get_cross_section(e, c):
    value = c[1] + c[2] * np.log(e - c[0]) + c[3] * np.log(e - c[0]) ** 2 + c[4] / np.log(e - c[0])
    return np.exp(ln10 * value)  # cm^2


def nu_cross_section(e):
    # neutral current
    cs_nc = get_cross_section(e, NC)
    # charged current
    cs_cc = get_cross_section(e, CC)

    return cs_nc + cs_cc


def anu_cross_section(e):
    # neutral current
    cs_nc = get_cross_section(e, ANC)
    # charged current
    cs_cc = get_cross_section(e, ACC)

    return cs_nc + cs_cc


if __name__ == "__main__":
    e = np.linspace(4, 12, 100)  # energy, GeV
    cs = nu_cross_section(e)
    plt.plot(np.exp(ln10 * e), cs)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r"$E_\nu,\ GeV$")
    plt.ylabel(r"$\sigma_{tot},\ cm^{2}$")
    plt.grid(linestyle='dashed')
    plt.tight_layout()
    # plt.savefig("cross_section.png")
    plt.show()

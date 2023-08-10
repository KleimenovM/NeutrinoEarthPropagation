# PREM Earth model application for neutrino propagation
# Neutrino total cross-section parametrization
# ref: http://dx.doi.org/10.1103/PhysRevD.83.113009

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

Ln10 = np.log(10)


def get_cross_section(e, c):
    value = c[1] + c[2] * np.log(e - c[0]) + c[3] * np.log(e - c[0]) ** 2 + c[4] / np.log(e - c[0])
    return 10**value  # cm^2


class CrossSectionEstimation:
    def __init__(self):
        nc, cc, anc, acc = (np.loadtxt("txt_data/cross_section_parameters.txt", unpack=True, skiprows=2, usecols=(1, 2, 3, 4, 5))).T
        self.nc = nc
        self.cc = cc
        self.anc = anc
        self.acc = acc

    def nu_cross_section(self, e):
        # neutral current
        cs_nc = get_cross_section(e, self.nc)
        # charged current
        cs_cc = get_cross_section(e, self.cc)

        return cs_nc + cs_cc

    def anu_cross_section(self, e):
        # neutral current
        cs_nc = get_cross_section(e, self.anc)
        # charged current
        cs_cc = get_cross_section(e, self.acc)

        return cs_nc + cs_cc


class CrossSectionTable:
    def __init__(self):
        data = pd.read_csv("txt_data/cross-section-table.csv")

        self.e = np.log(data["E_nu, GeV"]) / Ln10
        cof = 1e-24 * 1e-12  # pb to cm^2

        self.cc = data["nu (CC), pb"] * cof  # cm^2
        self.nc = data["nu (NC), pb"] * cof  # cm^2
        self.acc = data["anu (CC), pb"] * cof  # cm^2
        self.anc = data["anu (NC), pb"] * cof  # cm^2

        self.ccf = interp1d(self.e, self.cc, "cubic")
        self.ncf = interp1d(self.e, self.nc, "cubic")
        self.acf = interp1d(self.e, self.acc, "cubic")
        self.anf = interp1d(self.e, self.anc, "cubic")

    def nu_cross_section_points(self):
        return self.e, self.cc # + self.nc

    def anu_cross_section_points(self):
        return self.e, self.acc # + self.anc

    def nu_cross_section_value(self, e):
        return self.ccf(e) # + self.ncf(e)

    def anu_cross_section_value(self, e):
        return self.acf(e) # + self.anf(e)


if __name__ == "__main__":
    cs_estimation = CrossSectionEstimation()
    cs_table = CrossSectionTable()

    lg_e = np.linspace(1.7, 11.6, 100)  # energy, GeV
    # cs = cs_estimation.nu_cross_section(lg_e)
    # acs = cs_estimation.anu_cross_section(lg_e)
    tc = cs_table.nu_cross_section_value(lg_e)
    ta = cs_table.anu_cross_section_value(lg_e)
    # plt.plot(np.exp(Ln10 * lg_e), cs, label=r"$\nu,\ high-energy\ fit$", color='royalblue', alpha=.6)
    # plt.plot(np.exp(Ln10 * lg_e), acs, label=r"$\bar\nu,\ high-energy\ fit$", color='darkred', linestyle='dashed', alpha=.6)
    plt.plot(np.exp(Ln10 * lg_e), tc, label=r"$\nu$", color='royalblue')
    plt.plot(np.exp(Ln10 * lg_e), ta, label=r"$\bar\nu$", color='darkred', linestyle='dashed')

    plt.xlim(10, 1e12)
    
    plt.xscale('log')
    plt.yscale('log')
    
    plt.xlabel(r"$E_\nu,\ GeV$", fontsize=14)
    plt.ylabel(r"$\sigma_{tot},\ cm^{2}$", fontsize=14)
    plt.tick_params(labelsize=14)
    
    plt.legend(fontsize=14)
    
    plt.grid(linestyle='dashed')
    plt.tight_layout()
    plt.savefig("cross_section.png")
    plt.show()

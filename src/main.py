from __future__ import division, print_function

import numpy as np
from itertools import count
from numpy import sqrt
from scipy.integrate import quad
from matplotlib import pyplot as plt

from qutip import *

import constants as const
from operators import ModelSpace, HamiltonianSystem


# Construct dipole operator
def dipole(model_space, mu_ge, mu_ev):
    assert isinstance(model_space, ModelSpace)
    n = model_space.n
    c_e = model_space.creator_e
    a_e = model_space.annihilator_e
    c_v = model_space.creator_v
    ge_trans = mu_ge * sum((c_e(i) for i in range(n)), 0)
    ev_trans = mu_ev * sum((c_v(i)*a_e(i) for i in range(n)), 0)
    return ge_trans + ev_trans


def alpha(model_space, dipole_op, hamiltonian_op, omega_L, omega_i):
    div_op = hamiltonian_op - (omega_L+omega_i) * model_space.identity
    mat = div_op.data.toarray()
    newmat = np.linalg.inv(mat)
    mul_op = Qobj(newmat, dims=div_op.dims)
    return dipole_op * mul_op * dipole_op


def sum_R(alpha_op, ground):
    alpha_op2 = alpha_op**2
    return (
        alpha_op2.matrix_element(ground.dag(), ground) -
        alpha_op.matrix_element(ground.dag(), ground)**2
    )


if __name__ == '__main__':
    # Script parameters
    n_parts = 1  # Number of paticles

    omega_v = 1730 * 100  # Vibrational/cavity frequency [1/m]
    omega_c = omega_v
    omega_e = 5 / (const.PLANK_CONST_H * const.C0)  # Electronic frequency [1/m]
    omega_L = omega_v * sqrt(omega_e/omega_v)
    Omega_p = omega_v**2 / omega_L

    gamma_v = 13 * 100  # [1/m]
    gamma_e = 50 * 100  # [1/m]
    kappa = 13 * 100  # Cavity losses [1/m]
    s = 2
    g = 1

    # Construct Hilbert space
    ms = ModelSpace(num_molecules=n_parts)

    # Construct Hamiltonian system
    hamiltonian = HamiltonianSystem(
        model_space=ms, omega_c=omega_c, omega_v=omega_v, omega_e=omega_e,
        omega_L=omega_L, Omega_p=Omega_p, s=s, g=g
    )

    # Plot sum_R vs g/w_v
    xdat = np.linspace(0, 0.4, 50)
    ydat = np.empty_like(xdat)
    for xi, i in zip(xdat, count()):
        gi = xi * omega_v
        ham = HamiltonianSystem(
            model_space=ms, omega_c=omega_c, omega_v=omega_v, omega_e=omega_e,
            omega_L=omega_L, Omega_p=Omega_p, s=s, g=gi
        )
        h0 = ham.h0
        mu = dipole(model_space=ms, mu_ge=omega_e, mu_ev=omega_e-omega_v)
        alpha_op = alpha(
            model_space=ms, dipole_op=mu, hamiltonian_op=h0,
            omega_L=omega_L, omega_i=0
        )
        ydat[i] = abs(sum_R(alpha_op=alpha_op, ground=h0.groundstate()[1]))
        print(ydat[i])
    plt.figure(1)
    plt.plot(xdat, ydat, '-')
    plt.show()

    # Plot S(omega) vs omega_L - omega
    xdat = np.linspace(10*100, 60*100, 50)
    ydat = np.empty_like(xdat)
    omega_dat = omega_L * np.ones_like(xdat) - xdat

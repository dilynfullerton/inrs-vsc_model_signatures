from __future__ import division, print_function

import numpy as np
from itertools import count
from numpy import sqrt, log
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
    div_op = hamiltonian_op - (omega_L+omega_i) * model_space.one
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


def c_ops(
        model_space, kappa=0, gamma_e=0, gamma_v=0,
        gamma_e_phi=0, gamma_v_phi=0
):
    cops = [sqrt(kappa) * model_space.annihilator_c]
    for i in range(model_space.n):
        an_ei = model_space.annihilator_e(i)
        an_vi = model_space.annihilator_v(i)
        cops.append(sqrt(gamma_e) * an_ei)
        cops.append(sqrt(gamma_v) * an_vi)
        cops.append(sqrt(gamma_e_phi) * an_ei.dag() * an_ei)
        cops.append(sqrt(gamma_v_phi) * an_vi.dag() * an_vi)
    return cops


if __name__ == '__main__':
    # Script parameters
    n_parts = 1  # Number of paticles

    omega_v = 1730  # Vibrational/cavity frequency [1/cm]
    omega_c = omega_v
    omega_e = 5 / (const.PLANK_CONST_H * const.C0) / 100  # Electronic frequency [1/cm]
    omega_L = omega_v * sqrt(omega_e/omega_v)
    # omega_L = 6 * omega_v
    # Omega_p = omega_v**2 / omega_L
    # Omega_p = omega_v**2 / omega_e
    Omega_p = .1 * omega_v
    Omega_R = 160  # Rabi splitting [1/cm]

    gamma_v = 13  # [1/cm]
    gamma_e = 50  # [1/cm]
    kappa = 13  # Cavity losses [1/cm]

    omega_e = .4 * omega_e
    omega_L = 0.5 * (omega_e - omega_v) + omega_v

    s = 2
    # g = .05 * omega_v
    g = Omega_R/2/sqrt(n_parts)

    # Get polariton eigenvalues
    omega_plus = omega_v * sqrt(1 + 2 * g / omega_v)
    omega_minus = omega_v * sqrt(1 - 2 * g / omega_v)

    # Construct Hilbert space
    # TODO: Figure out proper number of excitations
    ms = ModelSpace(num_molecules=n_parts, num_excitations=(3, 3, 1))

    # Construct collapse operators
    c_ops_list = c_ops(
        model_space=ms,
        kappa=kappa,
        gamma_v=gamma_v,
        gamma_e=gamma_e,
        gamma_v_phi=gamma_v,
        gamma_e_phi=gamma_e,
    )

    # Construct Hamiltonian system
    hamiltonian = HamiltonianSystem(
        model_space=ms, omega_c=omega_c, omega_v=omega_v, omega_e=omega_e,
        omega_L=omega_L, Omega_p=Omega_p, s=s, g=g
    )

    # Plot sum_R vs g/w_v
    xdat = np.linspace(0, 0.4, 50)
    ydat = np.empty_like(xdat)
    ydat2 = np.empty_like(xdat)
    ydat3 = np.empty_like(xdat)
    for xi, i in zip(xdat, count()):
        gi = xi * omega_v
        mu = dipole(model_space=ms, mu_ge=omega_e, mu_ev=omega_e-omega_v)
        ham = HamiltonianSystem(
            model_space=ms, omega_c=omega_c, omega_v=omega_v, omega_e=omega_e,
            omega_L=omega_L, Omega_p=Omega_p, s=s, g=gi
        )
        h0 = ham.h0
        alpha_op = alpha(
            model_space=ms, dipole_op=mu, hamiltonian_op=h0,
            omega_L=omega_L, omega_i=0
        )
        ham2 = ham(t=0)
        alpha_op2 = alpha(
            model_space=ms, dipole_op=mu, hamiltonian_op=ham2,
            omega_L=omega_L, omega_i=0
        )
        ham3 = ham.h0c
        alpha_op3 = alpha(
            model_space=ms, dipole_op=mu, hamiltonian_op=ham3,
            omega_L=omega_L, omega_i=0
        )
        ydat[i] = abs(sum_R(alpha_op=alpha_op, ground=h0.groundstate()[1]))
        ydat2[i] = abs(sum_R(alpha_op=alpha_op2, ground=ham2.groundstate()[1]))
        ydat3[i] = abs(sum_R(alpha_op=alpha_op3, ground=ham3.groundstate()[1]))
    title = 'Sum_R vs g/omega_v'
    print('\n\nPlot: {}'.format(title))
    for xi, y1i, y2i, y3i in zip(xdat, ydat, ydat2, ydat3):
        print('  {:16.8f}    {:16.8f}    {:16.8f}    {:16.8f}'
              ''.format(xi, y1i, y2i, y3i))
    # fig, ax = plt.subplots(1, 1)
    # ax.plot(xdat, ydat, '-', label='h0')
    # ax.plot(xdat, ydat3, '-', label='h0c')
    # ax.legend()
    # ax.set_title(title)
    # ax.set_xlabel('g/omega_v')
    # ax.set_ylabel('sum_R')
    # plt.show()

    # Plot S(omega) vs omega_L - omega
    # omegadat = np.linspace(omega_L, omega_v, 500)
    # xdat = omega_L - omegadat
    xdat = np.linspace(0, 6000, 500+1)
    ydat = np.empty_like(xdat)
    ham = HamiltonianSystem(
        model_space=ms, omega_c=omega_c, omega_v=omega_v, omega_e=omega_e,
        omega_L=omega_L, Omega_p=Omega_p, s=s, g=g
    )
    title = 'S(omega)/N vs omega_L - omega'
    print('\n\nPlot: {}'.format(title))
    print('    H = \n    {}'.format(ham.h0))
    print('    c_e = \n    {}'.format(ms.total_creator_e))
    print('    a_e = \n    {}'.format(ms.total_annihilator_e))
    ydat = spectrum(
        H=ham(t=0), wlist=-xdat,
        c_ops=c_ops_list,
        a_op=ms.total_creator_e, b_op=ms.total_annihilator_e,
    ) / n_parts
    for xi, yi in zip(xdat, ydat):
        print('  {:16.8f}    {:16.8E}'.format(xi, yi))
    fig, ax = plt.subplots(1, 1)
    ax.plot(xdat, ydat, '-')
    # Plot vertical lines
    for n in count(1):
        vline = n * omega_v
        if vline < min(xdat):
            continue
        elif vline > max(xdat):
            break
        else:
            ax.axvline(vline, color='orange')
            for k in range(n+1):
                l = n - k
                ax.axvline(k * omega_plus + l * omega_minus,
                           linestyle='dashed', color='green')
    ax.set_title(title)
    ax.set_xlabel('omega_L - omega')
    ax.set_ylabel('S(omega)/N')
    ax.set_yscale('log')
    plt.show()

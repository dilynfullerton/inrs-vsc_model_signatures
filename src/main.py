from __future__ import division, print_function

import numpy as np
from itertools import count
from numpy import sqrt
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from qutip import *

from operators import ModelSpace, HamiltonianSystem

PARALLEL = False

PLANK_CONST_H = 4.135667516e-15  # h [eV * s]
C0 = 299792458  # Speed of light in vacuum [m/s]

OMEGA_E = .02 / (PLANK_CONST_H * C0)
OMEGA_V = 1730
OMEGA_L = (OMEGA_E + OMEGA_V) / 2
OMEGA_P = .1 * OMEGA_V
OMEGA_R = 160
GAMMA_V = 13  # [1/cm]
GAMMA_E = 50  # [1/cm]
KAPPA = 13  # Cavity losses [1/cm]
S = 2
EX_C = 3
EX_V = 3
EX_E = 1


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


def c_ops(model_space, kappa=0, gamma_e=0, gamma_v=0, gamma_e_phi=0,
          gamma_v_phi=0):
    cops = [sqrt(kappa) * model_space.annihilator_c]
    for i in range(model_space.n):
        an_ei = model_space.annihilator_e(i)
        an_vi = model_space.annihilator_v(i)
        cops.append(sqrt(gamma_e) * an_ei)
        cops.append(sqrt(gamma_v) * an_vi)
        cops.append(sqrt(gamma_e_phi) * an_ei.dag() * an_ei)
        cops.append(sqrt(gamma_v_phi) * an_vi.dag() * an_vi)
    return cops


def _par_spec_ham(h, **kwargs):
    return spectrum(H=h, wlist=kwargs['wlist'], c_ops=kwargs['c_ops'],
                     a_op=kwargs['a_op'], b_op=kwargs['b_op'])


def plot2ab(excitations=(3, 3, 1), n_parts=1, omega_e=OMEGA_E, omega_L=OMEGA_L,
            omega_v=OMEGA_V, omega_c=OMEGA_V, Omega_p=OMEGA_P,
            Omega_R_strong=OMEGA_R, Omega_R_weak=0, gamma_v=GAMMA_V,
            gamma_e=GAMMA_E, kappa=KAPPA, s=S, xmin=0, xmax=6000, npts=501):
    ms = ModelSpace(num_molecules=n_parts, num_excitations=excitations)
    cops_list = c_ops(model_space=ms, kappa=kappa, gamma_e=gamma_e,
                      gamma_v=gamma_v, gamma_e_phi=gamma_e, gamma_v_phi=gamma_v)
    g_vals = [omr/2/sqrt(n_parts) for omr in [Omega_R_weak, Omega_R_strong]]
    hams = []
    for gi in g_vals:
        hi = HamiltonianSystem(
            model_space=ms, omega_c=omega_c, omega_v=omega_v, omega_e=omega_e,
            omega_L=omega_L, Omega_p=Omega_p, s=s, g=gi)
        hams.append(hi(t=0))

    # Get plots
    xdat = np.linspace(xmin, xmax, npts)
    if PARALLEL:
        ydats = parallel_map(
            _par_spec_ham, hams,
            task_kwargs={'wlist': -xdat, 'c_ops': cops_list,
                         'a_op': ms.total_creator_e,
                         'b_op': ms.total_annihilator_e}
        )
    else:
        ydats = []
        for h in hams:
            ydats.append(
                _par_spec_ham(h, wlist=-xdat, c_ops=cops_list,
                              a_op=ms.total_creator_e,
                              b_op=ms.total_annihilator_e)
            )
    ydat_weak, ydat_strong = [ydat_i / n_parts for ydat_i in ydats]

    omega_plus = omega_v * sqrt(1 + 2 * g_vals[1] / omega_v)
    omega_minus = omega_v * sqrt(1 - 2 * g_vals[1] / omega_v)

    # Print crap
    title = 'S(omega)/N vs omega_L - omega'
    print('\n\nFigure: {}'.format(title))
    lab1 = 'S(omega), weak corr.'
    print('\nPlot: {}'.format(lab1))
    for xi, yi in zip(xdat, ydat_weak):
        print('  {:16.8f}    {:16.8E}'.format(xi, yi))
    lab2 = 'S(omega), strong corr.'
    print('\nPlot: {}'.format(lab2))
    for xi, yi in zip(xdat, ydat_strong):
        print('  {:16.8f}    {:16.8E}'.format(xi, yi))

    fig, ax1 = plt.subplots(1, 1)
    divider = make_axes_locatable(ax1)
    ax2 = divider.new_vertical(size='100%', pad=0.05)
    fig.add_axes(ax2)
    ax2.plot(xdat, ydat_weak, '-', label=lab1)
    ax1.plot(xdat, ydat_strong, '-', label=lab2)
    # Plot vertical lines
    for n in count(1):
        vline = n * omega_v
        if vline < min(xdat):
            continue
        elif vline > max(xdat):
            break
        else:
            ax1.axvline(vline, color='orange')
            ax2.axvline(vline, color='orange')
            for k in range(n+1):
                l = n - k
                ax1.axvline(
                    k * omega_plus + l * omega_minus,
                    linestyle='dashed', color='green'
                )
    ax1.set_title(title)
    ax1.set_xlabel('omega_L - omega')
    ax2.set_ylabel('S(omega)/N, weak')
    ax1.set_ylabel('S(omega)/N, strong')
    ax1.set_yscale('log')
    ax2.set_yscale('log')
    plt.show()

    return xdat, ydat_weak, ydat_strong


def _par_spec_ops(ops, **kwargs):
    return spectrum(H=kwargs['H'], wlist=kwargs['wlist'], c_ops=kwargs['c_ops'],
                    a_op=ops[0], b_op=ops[1])


def plot3(excitations=(2, 2, 1), omega_e=OMEGA_E, omega_L=OMEGA_L,
          omega_v=OMEGA_V, omega_c=OMEGA_V, Omega_p=OMEGA_P, Omega_R=OMEGA_R,
          gamma_v=GAMMA_V, gamma_e=GAMMA_E, kappa=KAPPA, s=S,
          xmin=1500, xmax=2000, npts=501):
    # Print
    title = 'First Stokes lines for N=1, N=2'
    print('\n\nFigure: {}'.format(title))
    xdat = np.linspace(xmin, xmax, npts)

    # Get data for N=1
    ms1 = ModelSpace(num_molecules=1, num_excitations=excitations)
    ham1 = HamiltonianSystem(model_space=ms1, omega_c=omega_c, omega_v=omega_v,
                             omega_e=omega_e, omega_L=omega_L, Omega_p=Omega_p,
                             s=s, g=Omega_R/2/sqrt(1))
    cops1 = c_ops(model_space=ms1, kappa=kappa, gamma_e=gamma_e,
                  gamma_v=gamma_v, gamma_e_phi=gamma_e, gamma_v_phi=gamma_v)

    # First plot: N=1
    ydat1 = spectrum(
        H=ham1(t=0), wlist=-xdat, c_ops=cops1,
        a_op=ms1.total_creator_e, b_op=ms1.total_annihilator_e
    )
    # Print
    lab1 = 'N=1'
    print('\n  BEGIN Plot: {}'.format(lab1))
    for xi, yi in zip(xdat, ydat1):
        print('    {:16.8f}    {:16.8E}'.format(xi, yi))
    print('\n  END Plot: {}'.format(lab1))

    # Get data for N=2
    ms2 = ModelSpace(num_molecules=2, num_excitations=excitations)
    ham2 = HamiltonianSystem(
        model_space=ms2, omega_c=omega_c, omega_v=omega_v, omega_e=omega_e,
        omega_L=omega_L, Omega_p=Omega_p, s=s, g=Omega_R/2/sqrt(2)
    )
    cops2 = c_ops(
        model_space=ms2, kappa=kappa, gamma_e=gamma_e, gamma_v=gamma_v,
        gamma_e_phi=gamma_e, gamma_v_phi=gamma_v
    )

    # Second plot: N=2, coherent
    spec = spectrum(
        H=ham2(t=0), wlist=-xdat, c_ops=cops2,
        a_op=ms2.total_creator_e, b_op=ms2.total_annihilator_e
    )
    assert isinstance(spec, np.ndarray)
    ydat2 = .5 * spec
    # Print
    lab2 = 'N=2, coherent'
    print('\n  BEGIN Plot: {}'.format(lab2))
    for xi, yi in zip(xdat, ydat2):
        print('    {:16.8f}    {:16.8E}'.format(xi, yi))
    print('\n  END Plot: {}'.format(lab2))

    # Third plot: N=2, incoherent
    ce_op = ms2.creator_e
    ae_op = ms2.annihilator_e
    if PARALLEL:
        ydats3 = parallel_map(
            _par_spec_ops, [(ce_op(i), ae_op(i)) for i in range(ms2.n)],
            task_kwargs={'H': ham2(t=0), 'wlist': -xdat, 'c_ops': cops2}
        )
    else:
        ydats3 = []
        for i in range(ms2.n):
            ydats3.append(
                _par_spec_ops(ops=(ce_op(i), ae_op(i)), H=ham2(t=0),
                              wlist=-xdat, c_ops=cops2)
            )
    ydat3 = .5 * sum(ydats3, np.zeros_like(xdat))
    # Print
    lab3 = 'N=2, incoherent'
    print('\n  BEGIN Plot: {}'.format(lab3))
    for xi, yi in zip(xdat, ydat3):
        print('    {:16.8f}    {:16.8E}'.format(xi, yi))
    print('\n  END Plot: {}'.format(lab3))

    # Make and show plots
    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel('omega_L - omega')
    ax.set_ylabel('S(omega)/N')
    ax.set_title(title)
    ax.plot(xdat, ydat1, '--', label=lab1)
    ax.plot(xdat, ydat2, '-', label=lab2)
    ax.plot(xdat, ydat3, '-', label=lab3)
    ax.legend()
    plt.show()
    return xdat, ydat1, ydat2, ydat3


def plot2in():
    pass


if __name__ == '__main__':
    # plot2ab(excitations=(3, 3, 1))
    plot3(excitations=(1, 1, 1))

import numpy as np
import qutip as qt
from itertools import count
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from hamiltonian import ModelSpace, HamiltonianSystem

OMEGA_E = 40000  # [1/cm]
OMEGA_V = 1730  # [1/cm]
OMEGA_L = 20000  # [1/cm]
OMEGA_P = .1 * OMEGA_V
OMEGA_R = 160  # [1/cm]
G = OMEGA_R / 2  # [1/cm]
GAMMA_V = 13  # [1/cm]
GAMMA_E = 50  # [1/cm]
KAPPA = 13  # Cavity losses [1/cm]
S = 2

# Plot-specific parameters
SHOW_PLOT_2_INSET = False
SHOW_PLOT_2 = True
SHOW_PLOT_3 = True

PLOT_2_FOCK_DIMS = {'nfock_cav': 4, 'nfock_vib': 4}
PLOT_3_FOCK_DIMS = {'nfock_cav': 2, 'nfock_vib': 2}


# Construct dipole operator
def dipole(model_space, mu_ge, mu_ev):
    assert isinstance(model_space, ModelSpace)
    n = model_space.n
    c_e = model_space.create_elec
    a_e = model_space.destroy_elec
    c_v = model_space.create_vib
    ge_trans = mu_ge * sum((c_e(i) for i in range(n)), 0)
    ev_trans = mu_ev * sum((c_v(i)*a_e(i) for i in range(n)), 0)
    return ge_trans + ev_trans


def alpha(model_space, dipole_op, hamiltonian_op, omega_L, omega_i):
    div_op = hamiltonian_op - (omega_L+omega_i) * model_space.one()
    mat = div_op.data.toarray()
    newmat = np.linalg.inv(mat)
    mul_op = qt.Qobj(newmat, dims=div_op.dims)
    return dipole_op * mul_op * dipole_op


def sum_R(alpha_op, ground):
    alpha_op2 = alpha_op**2
    return (
        alpha_op2.matrix_element(ground.dag(), ground) -
        alpha_op.matrix_element(ground.dag(), ground)**2
    )


def plot2inset(hamiltonian, xdat=np.linspace(0, .4, 50)):
    """Creates and displays the inset plot of Figure 2.
    Note that the parameter <code>g</code> is mutated throughout this
    function, although it is returned to its initial state
    :param hamiltonian:
    :param xdat:
    """
    print('Making plot Figure 2 inset...')
    g_orig = hamiltonian.g
    ydat = []
    ydat_rwa = []
    for g_div_omega_v in xdat:
        g = hamiltonian.omega_v * g_div_omega_v
        hamiltonian.set_g(g)
        for ydat_i, rwa in zip((ydat_rwa, ydat), (True, False)):
            h_op = hamiltonian.H0_simple(rwa=rwa)
            d_op = hamiltonian.H0_dipole(rwa=rwa)
            omega_g, ket_g = h_op.groundstate()
            alpha_op = alpha(
                model_space=hamiltonian.space,
                dipole_op=d_op, hamiltonian_op=h_op,
                omega_L=hamiltonian.omega_L, omega_i=omega_g
            )
            ydat_i.append(
                qt.expect(alpha_op**2, ket_g) - qt.expect(alpha_op, ket_g)**2
            )
    hamiltonian.set_g(g_orig)
    print('Done')

    # Make plot
    print('Showing plot')
    fig, ax = plt.subplots(1, 1)
    ax.plot(xdat, ydat, '-', label='no rwa')
    ax.plot(xdat, ydat_rwa, '-', label='rwa')
    ax.set_xlabel('g / omega_c')
    ax.set_ylabel('Sigma_R')
    ax.legend()
    plt.show()
    return xdat, ydat, ydat_rwa


def plot2ab(hamiltonian, omega_range=np.linspace(1000, 6000, 501)):
    """Constructs and displays the plot of Figure 2 (a) and (b) of
    the paper "Signatures..."
    :param hamiltonian: <code>HamiltonianSystem</code> object,
    containing the various Hamiltonian operators
    :param omega_range: List or array of frequencies for which to evaluate
    the spectra
    """
    print('Making plot figure 2')
    model_space = hamiltonian.space
    collapse_operators = hamiltonian.collapse_ops()

    # Get spectra
    xdat = omega_range
    print(' Getting spectrum for weak coupling...')
    ydat_weak = qt.spectrum(
        H=hamiltonian.H(coupling=False),
        wlist=-xdat, c_ops=collapse_operators,
        a_op=model_space.create_all_elec(), b_op=model_space.destroy_all_elec()
    ) / model_space.n
    print(' Done')
    print(' Getting spectrum for strong coupling...')
    ydat_strong = qt.spectrum(
        H=hamiltonian.H(coupling=True), wlist=-xdat, c_ops=collapse_operators,
        a_op=model_space.create_all_elec(), b_op=model_space.destroy_all_elec()
    ) / model_space.n
    print(' Done')

    # Print output
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

    # Make plot figure
    print('Displaying plot figure')
    fig, ax1 = plt.subplots(1, 1)
    divider = make_axes_locatable(ax1)
    ax2 = divider.new_vertical(size='100%', pad=0.05)
    fig.add_axes(ax2)
    ax2.plot(xdat, ydat_weak, '-', label=lab1)
    ax1.plot(xdat, ydat_strong, '-', label=lab2)

    # Plot vertical lines
    omega_plus = hamiltonian.omega_up()
    omega_minus = hamiltonian.omega_lp()
    for n in count(1):
        vline = n * hamiltonian.omega_v
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


def plot3(hamiltonian, omega_range=np.linspace(1500, 2000, 501)):
    """Create and display Figure 3 of "Signatures..." based on the given
    Hamiltonian
    :param hamiltonian: <code>HamiltonianSystem</code> object. Note that
    the number of particles will be mutated in order to create the plots
    :param omega_range: Range of frequencies for which to compute the
    spectrum
    """
    print('Making plot figure 3')
    ms = hamiltonian.space
    num_molecules_original = ms.n
    xdat = omega_range

    # Print
    title = 'First Stokes lines for N=1, N=2'
    print('\n\nFigure: {}'.format(title))

    # Get data for N=1
    ms.set_num_molecules(1)

    # First plot: N=1
    ydat1 = qt.spectrum(
        H=hamiltonian.H(), wlist=-xdat, c_ops=hamiltonian.collapse_ops(),
        a_op=ms.create_all_elec(), b_op=ms.destroy_all_elec()
    )

    # Print
    lab1 = ' N=1'
    print('\n  BEGIN Plot: {}'.format(lab1))
    for xi, yi in zip(xdat, ydat1):
        print('    {:16.8f}    {:16.8E}'.format(xi, yi))
    print('\n  END Plot: {}'.format(lab1))

    # Get data for N=2
    ms.set_num_molecules(2)

    # Second plot: N=2, coherent
    spec = qt.spectrum(
        H=hamiltonian.H(), wlist=-xdat, c_ops=hamiltonian.collapse_ops(),
        a_op=ms.create_all_elec(), b_op=ms.destroy_all_elec()
    )
    ydat2 = .5 * spec

    # Print
    lab2 = ' N=2, coherent'
    print('\n  BEGIN Plot: {}'.format(lab2))
    for xi, yi in zip(xdat, ydat2):
        print('    {:16.8f}    {:16.8E}'.format(xi, yi))
    print('\n  END Plot: {}'.format(lab2))

    # Third plot: N=2, incoherent
    ydats3 = []
    for i in range(ms.n):
        ydats3.append(
            qt.spectrum(
                H=hamiltonian.H(), wlist=-xdat,
                c_ops=hamiltonian.collapse_ops(),
                a_op=ms.create_elec(i), b_op=ms.destroy_elec(i)
            )
        )
    ydat3 = .5 * sum(ydats3, np.zeros_like(xdat))

    # Print
    lab3 = ' N=2, incoherent'
    print('\n  BEGIN Plot: {}'.format(lab3))
    for xi, yi in zip(xdat, ydat3):
        print('    {:16.8f}    {:16.8E}'.format(xi, yi))
    print('\n  END Plot: {}'.format(lab3))

    # Reset original num molecules
    ms.set_num_molecules(num_molecules_original)

    # Make and show plots
    print('Displaying plot figure')
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


if __name__ == '__main__':
    # See hamiltonian.py for definitions of ModelSpace and HamiltonianSystem
    model_space = ModelSpace(num_molecules=1)
    hamiltonian = HamiltonianSystem(
        model_space=model_space,
        omega_c=OMEGA_V, omega_v=OMEGA_V, omega_e=OMEGA_E, omega_L=OMEGA_L,
        Omega_p=OMEGA_P, g=G, s=S,
        kappa=KAPPA, gamma_v=GAMMA_V, gamma_e=GAMMA_E,
        gamma_v_phi=GAMMA_V, gamma_e_phi=GAMMA_E,
    )

    # Plot 2 inset
    if SHOW_PLOT_2_INSET:
        plot2inset(hamiltonian=hamiltonian)

    # Plot 2
    if SHOW_PLOT_2:
        model_space.set_nfock(**PLOT_2_FOCK_DIMS)
        plot2ab(hamiltonian=hamiltonian)

    # Plot 3
    if SHOW_PLOT_3:
        model_space.set_nfock(**PLOT_3_FOCK_DIMS)
        plot3(hamiltonian=hamiltonian)

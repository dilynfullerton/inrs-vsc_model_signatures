from __future__ import division, print_function
from numpy import sqrt, exp
from qutip import *


# Constants

PLANK_CONST_H = 4.135667516e-15  # h [eV * s]
C0 = 299792458  # Speed of light in vacuum [m/s]

n_parts = 2  # Number of paticles

omega_v = 1730 * 100  # Vibrational/cavity frequency [1/m]
omega_c = omega_v
omega_e = 5 / (PLANK_CONST_H * C0)  # Electronic frequency [1/m]
omega_l = omega_v * sqrt(omega_e/omega_v)
Omega_p = omega_v**2 / omega_l

gamma_v = 13 * 100  # [1/m]
gamma_e = 50 * 100  # [1/m]
kappa = 13 * 100  # Cavity losses [1/m]
s = 2

# Construct vector space of elements |n_c, n_v^i, n_e^j>,
# where i and j range from 1 to N

gs_c = basis(2, 0)
gs_v = tensor([basis(2, 0)]*n_parts)
gs_e = tensor([basis(2, 0)]*n_parts)
vacuum = tensor([gs_c, gs_v, gs_e])

# Construct identity operator
id_c = qeye(2)
id_v = tensor([qeye(2)]*n_parts)
id_e = tensor([qeye(2)]*n_parts)
one = tensor([id_c, id_v, id_e])

# Construct annihilation operators
a_c = tensor([destroy(2), id_v, id_e])


def a_v(i):
    op_list = [qeye(2)] * n_parts
    op_list.insert(i, destroy(2))
    op_list.pop(i+1)
    return tensor(id_c, tensor(op_list), id_e)


def a_e(i):
    op_list = [qeye(2)] * n_parts
    op_list.insert(i, destroy(2))
    op_list.pop(i+1)
    return tensor(id_c, id_v, tensor(op_list))

# Construct bright state vector
bright = 1/sqrt(n_parts) * sum(
    (a_v(j).dag() * vacuum for j in range(n_parts)),
    0
)

# Construct the polaritons
plus = 1/sqrt(2) * (a_c.dag() * vacuum + bright)
minus = 1/sqrt(2) * (a_c.dag() * vacuum - bright)


# Construct the system hamiltonian
def h_mol(i):
    a_vi = a_v(i)
    n_vi = a_vi.dag() * a_vi
    n_ei = a_e(i).dag() * a_e(i)
    return (
        omega_e * n_ei +
        omega_v * (n_vi + sqrt(s) * n_ei * (a_vi.dag() + a_vi))
    )


h0 = (
         sum((h_mol(i) for i in range(n_parts)), 0) +
         omega_c * a_c.dag() * a_c +
         g * (a_c + a_c.dag()) * sum(
             (a_v(i) + a_v(i).dag() for i in range(n_parts)), 0)
)


def h_d(t):
    return Omega_p * sum(
        (a_e(i) * exp(-1j*omega_l*t) + a_e(i).dag() * exp(1j*omega_l*t)
         for i in range(n_parts)),
        0
    )


def h_sys(t):
    return h0 + h_d(t)

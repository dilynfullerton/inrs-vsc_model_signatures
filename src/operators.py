from __future__ import print_function, division
from numpy import sqrt, exp
from qutip import *


class ModelSpace:
    """The zero, identity, creation, and annihilation operators for the
    Fock space defined:

        F = H(ns_c, omega_c) \otimes
            H(ns_v, omega_v)^{\otimes N} \otimes
            H(ns_e, omega_e)^{\otimes N}

    where H(ns, omega) is the one-dimensional bosonic harmonic oscillator
    space with frequency omega, with ns oscillator orbits.
    """
    def __init__(self, num_molecules, num_excitations):
        """Construct a model space with n particles between 3 states.
        These are the cavity, vibrational, and electronic excited states,
        and they have ne_c, ne_v, ne_e available excitations respectively.
        :param num_molecules: Number of molecules in the system
        :param num_excitations: Either a three-tuple containing
        (ne_c, ne_v, and ne_e), positive integers or a single
        positive integer ne, common to all states.
        """
        self.n = num_molecules
        if isinstance(num_excitations, int):
            self._ne_c, self._ne_v, self._ne_e = (num_excitations,) * 3
        else:
            self._ne_c, self._ne_v, self._ne_e = num_excitations
        # Number of fock states in each mode
        self.nfock_c = self._ne_c + 1
        self.nfock_vib = self._ne_v + 1
        self.nfock_e = self._ne_e + 1

        # Dimensions for cavity, vibrational, and electronic states
        # These are represented as lists, where the ith element of the
        # list represents the number of Fock states for the ith particle
        self.dims_cav = [self.nfock_c]
        self.dims_vib = [self.nfock_vib] * self.n
        self.dims_e = [self.nfock_e] * self.n
        self.dims_all = self.dims_cav + self.dims_vib + self.dims_e

        # Operators
        self.zero = qzero(self.dims_all)
        self.one = qeye(self.dims_all)
        self.annihilator_c = tensor(
            destroy(self.nfock_c), qeye(self.dims_vib + self.dims_e)
        )
        self.creator_c = self.annihilator_c.dag()
        self.total_annihilator_e = sum(
            (self.annihilator_e(i) for i in range(self.n)), 0)
        self.total_creator_e = self.total_annihilator_e.dag()

        # Kets
        self.vacuum = tensor(
            [basis(self.nfock_c, 0)] +
            [basis(self.nfock_vib, 0)] * self.n +
            [basis(self.nfock_e, 0)] * self.n
        )
        self.bright = 1/sqrt(self.n) * sum(
            (self.creator_v(j) * self.vacuum for j in range(self.n)), 0)
        self.polariton_plus = 1/sqrt(2) * (
            self.creator_c * self.vacuum + self.bright)
        self.polariton_minus = 1/sqrt(2) * (
            self.creator_c * self.vacuum - self.bright)

    def _get_op_list(self, i, ns):
        op_list = [qeye(ns)] * self.n
        op_list.insert(i, destroy(ns))
        op_list.pop(i+1)
        return op_list

    def annihilator_v(self, i):
        """Returns the annihilation operator for the ith vibrational mode
        """
        return tensor(
            qeye(self.dims_cav),
            tensor(self._get_op_list(i, self.nfock_vib)),
            qeye(self.dims_e)
        )

    def creator_v(self, i):
        """Returns the creation operator for the ith vibrational mode
        """
        return self.annihilator_v(i).dag()

    def annihilator_e(self, i):
        """Returns the annihilation operator for the ith electronic mode
        """
        return tensor(
            qeye(self.dims_cav + self.dims_vib),
            tensor(self._get_op_list(i, self.nfock_e))
        )

    def creator_e(self, i):
        """Returns the creation operator for the ith electronic mode
        """
        return self.annihilator_e(i).dag()


class HamiltonianSystem:
    def __init__(
            self, model_space, omega_c, omega_v, omega_e, omega_L, Omega_p, s,
            g
    ):
        """Construct the Hamiltonian operator based on "Signatures..."
        :param model_space: Instance of the ModelSpace class, containing
        all of the necessary operators
        :param omega_c: Cavity frequency
        :param omega_v: Vibrational frequency
        :param omega_e: Electronic frequency
        :param omega_L: Laser (radiation) frequency
        :param Omega_p: Driving (probe) power
        :param s: Huang-Rhys parameter
        :param g: Cavity-electron coupling
        """
        self.space = model_space
        self.n = self.space.n
        self.omega_c = omega_c
        self.omega_v = omega_v
        self.omega_e = omega_e
        self.omega_L = omega_L
        self.Omega_p = Omega_p
        self.s = s
        self.g = g
        self.h0c = self._h0_coherent()
        self.h0 = self._h0()
        self.h = self._h()

    def __call__(self, t, *args, **kwargs):
        assert isinstance(self.space, ModelSpace)
        op = self.space.zero
        for hterm in self.h:
            if isinstance(hterm, Qobj):
                op += hterm
            else:
                hterm, f = hterm
                op += f(t) * hterm
        return op

    def __str__(self):
        return str(self.h0)

    def _h0_coherent(self):
        """Hamiltonian based on Equation (1)
        """
        a_c = self.space.annihilator_c
        a_v = self.space.annihilator_v
        a_e = self.space.annihilator_e
        h0c = self.omega_c * a_c.dag() * a_c
        for i in range(self.n):
            h0c += self.omega_v * a_v(i).dag() * a_v(i)
            h0c += self.omega_e * a_e(i).dag() * a_e(i)
            h0c += self.g * (a_c.dag() * a_v(i) + a_v(i).dag() * a_c)
        return h0c

    def _h_mol(self, i):
        """Returns the Hamiltonian based on Equation (4) for the ith molecule
        """
        a_vi = self.space.annihilator_v(i)
        a_ei = self.space.annihilator_e(i)
        n_vi = a_vi.dag() * a_vi
        n_ei = a_ei.dag() * a_ei
        return (
            self.omega_e * n_ei +
            self.omega_v * (n_vi + sqrt(self.s) * n_ei * (a_vi.dag() + a_vi))
        )

    def _h0(self):
        """Returns the Hamiltonian based on Equation (5)
        """
        a_c = self.space.annihilator_c
        a_v = self.space.annihilator_v
        return (
            sum((self._h_mol(i) for i in range(self.n)), 0) +
            self.omega_c * a_c.dag() * a_c +
            self.g * (a_c + a_c.dag()) * sum(
                (a_v(i) + a_v(i).dag() for i in range(self.n)), 0)
        )

    def _h1(self):
        a_e = self.space.annihilator_e
        return self.Omega_p * sum((a_e(i) for i in range(self.n)), 0)

    def _f1(self, t):
        return exp(-1j*self.omega_L*t)

    def _h2(self):
        c_e = self.space.creator_e
        return self.Omega_p * sum((c_e(i) for i in range(self.n)), 0)

    def _f2(self, t):
        return exp(1j*self.omega_L*t)

    def _h(self):
        return [self.h0, [self._h1(), self._f1], [self._h2(), self._f2]]

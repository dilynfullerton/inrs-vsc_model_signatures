from __future__ import print_function, division
from numpy import sqrt, exp
from qutip import *


class ModelSpace:
    def __init__(self, num_molecules, num_excitations):
        """Construct a model space with n particles between 3 states.
        These are the cavity, virtual, and electronic excited states,
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
        self.ns_c = self._ne_c + 1
        self.ns_v = self._ne_v + 1
        self.ns_e = self._ne_e + 1

        # Operators
        self.zero = qzero(
            [self.ns_c] + [self.ns_v]*self.n + [self.ns_e]*self.n)
        self.one = qeye(
            [self.ns_c] + [self.ns_v]*self.n + [self.ns_e]*self.n)
        self.annihilator_c = tensor(
            destroy(self.ns_c),
            qeye([self.ns_v]*self.n + [self.ns_e]*self.n)
        )
        self.creator_c = self.annihilator_c.dag()
        self.total_annihilator_e = sum(
            (self.annihilator_e(i) for i in range(self.n)), 0)
        self.total_creator_e = self.total_annihilator_e.dag()

        # Kets
        self.vacuum = tensor(
            [basis(self.ns_c, 0)] +
            [basis(self.ns_v, 0)] * self.n +
            [basis(self.ns_e, 0)] * self.n
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
        return tensor(
            [qeye(self.ns_c)] +
            self._get_op_list(i, self.ns_v) +
            [qeye(self.ns_e)] * self.n
        )

    def creator_v(self, i):
        return self.annihilator_v(i).dag()

    def annihilator_e(self, i):
        return tensor(
            [qeye(self.ns_c)] +
            [qeye(self.ns_v)] * self.n +
            self._get_op_list(i, self.ns_e)
        )

    def creator_e(self, i):
        return self.annihilator_e(i).dag()


class HamiltonianSystem:
    def __init__(
            self, model_space, omega_c, omega_v, omega_e, omega_L, Omega_p, s,
            g
    ):
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
        a_vi = self.space.annihilator_v(i)
        a_ei = self.space.annihilator_e(i)
        n_vi = a_vi.dag() * a_vi
        n_ei = a_ei.dag() * a_ei
        return (
            self.omega_e * n_ei +
            self.omega_v * (n_vi + sqrt(self.s) * n_ei * (a_vi.dag() + a_vi))
        )

    def _h0(self):
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


from __future__ import print_function, division
from numpy import sqrt, exp
from qutip import *


class ModelSpace:
    def __init__(self, num_molecules):
        self.n = num_molecules

        # Operators
        self.zero = tensor([qzero(2)] * (1 + 2*self.n))
        self.one = tensor([qeye(2)] * (1 + 2*self.n))
        self.annihilator_c = tensor([destroy(2)] + [qeye(2)] * 2*self.n)
        self.creator_c = self.annihilator_c.dag()
        # self.total_annihilator_v = sum(
        #     (self.annihilator_v(i) for i in range(self.n)), 0)
        # self.total_creator_v = self.total_annihilator_v.dag()
        self.total_annihilator_e = sum(
            (self.annihilator_e(i) for i in range(self.n)), 0)
        self.total_creator_e = self.total_annihilator_e.dag()

        # Kets
        self.vacuum = tensor([basis(2, 0)] * (1 + 2*self.n))

        self.bright = 1/sqrt(self.n) * sum(
            (self.creator_v(j) * self.vacuum for j in range(self.n)), 0)
        self.polariton_plus = 1/sqrt(2) * (
            self.creator_c * self.vacuum + self.bright)
        self.polariton_minus = 1/sqrt(2) * (
            self.creator_c * self.vacuum - self.bright)

    def _get_op_list(self, i):
        op_list = [qeye(2)] * self.n
        op_list.insert(i, destroy(2))
        op_list.pop(i+1)
        return op_list

    def annihilator_v(self, i):
        return tensor(
            [qeye(2)] + self._get_op_list(i) + [qeye(2)] * self.n
        )

    def creator_v(self, i):
        return self.annihilator_v(i).dag()

    def annihilator_e(self, i):
        return tensor([qeye(2)] * (1 + self.n) + self._get_op_list(i))

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

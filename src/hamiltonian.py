"""operators.py
This file contains definitions for the operators and Hamiltonian used in
reproducing the results of the paper
"Signature of Vibrational Strong Coupling in Raman Scattering"
"""
import qutip as qt
from numpy import sqrt


class ModelSpace:
    """The zero, identity, creation, and annihilation operators for the
    Fock space defined in occupation representation as:

        F = H(ns_c, omega_c) \otimes
            H(ns_v, omega_v)^{\otimes N} \otimes
            H(ns_e, omega_e)^{\otimes N}

    where H(ns, omega) is the one-dimensional bosonic harmonic oscillator
    space with frequency omega, with ns oscillator orbits.
    """

    def __init__(self, num_molecules, num_excitations=(1, 1, 1)):
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
        self.nfock_cav = self._ne_c + 1
        self.nfock_vib = self._ne_v + 1
        self.nfock_elec = self._ne_e + 1

    def _get_op_list(self, i, ns):
        """Helper function which returns a list n operators of Fock
        dimension ns. All but the ith operator in the list are identity
        operators while the ith operator is an annihilation operator.
        This list can then be acted on with QuTiP's
        <code>tensor</code> function, which returns the corresponding
        <code>Qobj</code>
        :param i: The position in the list to be occupied by the destruction
        operator
        :param ns: The Fock dimension of each operator in the list
        """
        op_list = [qt.qeye(ns)] * (self.n - 1)
        op_list.insert(i, qt.destroy(ns))
        return op_list

    # --- Dimensions ---
    # Dimensions for cavity, vibrational, and electronic states
    # These are represented as lists, where the ith element of the
    # list represents the number of Fock states for the ith particle
    def _dims_cav(self):
        return [self.nfock_cav]

    def _dims_vib(self):
        return [self.nfock_vib] * self.n

    def _dims_elec(self):
        return [self.nfock_elec] * self.n

    def dims(self):
        return self._dims_cav() + self._dims_vib() + self._dims_elec()

    # --- Setters ---
    # Functions which mutate the internal parameters
    def set_num_molecules(self, n):
        if n > 0:
            self.n = int(n)
        else:
            raise RuntimeError('Cannot set molecule number to negative value')

    def set_nfock(self, nfock_cav=None, nfock_vib=None, nfock_elec=None):
        """Sets the Fock dimensions of the cavity, vibrational, or
        electrical modes. If any of these are <code>None</code>, the
        previous value for that is maintained.
        :param nfock_cav: Fock dimension of cavity mode
        :param nfock_vib: Fock dimension of each vibrational mode
        :param nfock_elec: Fock dimension of each electrical mode
        """
        new_dims = []
        for nf_new, nf_old in zip(
                (nfock_cav, nfock_vib, nfock_elec),
                (self.nfock_cav, self.nfock_vib, self.nfock_elec)
        ):
            if nf_new is None:
                new_dims.append(nf_old)
            elif nf_new is not None and nf_new > 1:
                new_dims.append(int(nf_new))
            else:
                raise RuntimeError(
                    'Cannot set fock dimensions to a number less than 2')
        self.nfock_cav, self.nfock_vib, self.nfock_elec = new_dims

    # --- Kets ---
    # Functions for getting relevant Qobj ket objects
    def vacuum(self):
        """Returns the vacuum ket for this space, which is the state wherein
        all of the cavity, vibrational, and electronic modes are unoccupied.
        """
        return qt.tensor(
            [qt.basis(self.nfock_cav, 0)] +
            [qt.basis(self.nfock_vib, 0)] * self.n +
            [qt.basis(self.nfock_elec, 0)] * self.n
        )

    def bright(self):
        """Returns the <code>Qobj</code> representation of the bright
        polariton state, wherein all vibrational modes are excited
        """
        b = 0
        for j in range(self.n):
            b += self.create_vib(j) * self.vacuum() / sqrt(self.n)
        return b

    def polariton_plus(self):
        """Returns the <code>Qobj</code> ket representation of the
        upper polariton state, which is the symmetric linear combination of
        the excited cavity and the bright state
        """
        return 1/sqrt(2) * (self.create_cav * self.vacuum() + self.bright())

    def polariton_minus(self):
        """Returns the <code>Qobj</code> ket representation of the
        lower polariton state, which is the antisymmetric linear combination of
        the excited cavity and the bright state
        """
        return 1/sqrt(2) * (self.create_cav * self.vacuum() - self.bright())

    # --- Operators ---
    # Functions for getting Qobj operators
    def zero(self):
        """Returns the <code>Qobj</code> representation of the zero operator
        for the space, which takes any state identically to zero
        """
        return qt.qzero(self.dims())

    def one(self):
        """Returns the <code>Qobj</code> representation of the identity
        operator for the space, which states any state to itself.
        """
        return qt.qeye(self.dims())

    def destroy_cav(self):
        """Returns the <code>Qobj</code> representation of the destruction
        operator on the cavity mode. This acts as the identity on all other
        modes.
        """
        return qt.tensor(
            qt.destroy(self.nfock_cav),
            qt.qeye(self._dims_vib() + self._dims_elec())
        )

    def create_cav(self):
        return self.destroy_cav().dag()

    def destroy_vib(self, i):
        """Returns the annihilation operator for the ith vibrational mode.
        In the occupation number (Fock) representation, this operator takes
        the form:
            I_{c} \otimes b_{v,i} \otimes I_{e},
        where b_{v,i} acts as the annihilator of the ith vibrational mode
        and the identity on the kth mode for all k \neq i.
        """
        return qt.tensor(
            qt.qeye(self._dims_cav()),
            qt.tensor(self._get_op_list(i, self.nfock_vib)),
            qt.qeye(self._dims_elec())
        )

    def create_vib(self, i):
        """Returns the creation operator for the ith vibrational mode
        """
        return self.destroy_vib(i).dag()

    def destroy_elec(self, i):
        """Returns the annihilation operator for the ith electronic mode.
        In the occupation number (Fock) representation, this operator takes
        the form:
            I_{c} \otimes I_{v} \otimes c_{e, i},
        where c_{e,i} acts as the annihilator of the ith electronic mode
        and the identity on the kth mode for all k \neq i.
        Note that this is NOT a fermionic operator, but a bosonic one.
        Thus it is perhaps better understood as an operator on
        bosonic exitons, not electrons.
        """
        return qt.tensor(
            qt.qeye(self._dims_cav() + self._dims_vib()),
            qt.tensor(self._get_op_list(i, self.nfock_elec))
        )

    def create_elec(self, i):
        """Returns the creation operator for the ith electronic mode
        """
        return self.destroy_elec(i).dag()

    def destroy_all_elec(self):
        """Return the operator which destroys all electronic modes. This is
        the sum of the electronic destruction operator for each mode.
        """
        d = self.zero()
        for i in range(self.n):
            d += self.destroy_elec(i)
        return d

    def create_all_elec(self):
        return self.destroy_all_elec().dag()


class HamiltonianSystem:
    def __init__(
            self, model_space,
            omega_c, omega_v, omega_e, omega_L, Omega_p, s, g,
            kappa, gamma_e, gamma_e_phi, gamma_v, gamma_v_phi,
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
        # Model space
        self.space = model_space

        # Convenience operator references
        self._a = self.space.destroy_cav
        self._b = self.space.destroy_vib
        self._c = self.space.destroy_elec

        # Frequencies
        self.omega_c = omega_c
        self.omega_v = omega_v
        self.omega_e = omega_e
        self.omega_L = omega_L
        self.Omega_p = Omega_p

        # Couplings
        self.s = s
        self.g = g

        # Dissipation
        self.kappa = kappa
        self.gamma_e = gamma_e
        self.gamma_e_phi = gamma_e_phi
        self.gamma_v = gamma_v
        self.gamma_v_phi = gamma_v_phi

    def _g(self):
        return self.g / sqrt(self.space.n)

    def set_g(self, g):
        self.g = g

    # --- Frequencies ---
    def omega_up(self):
        """Returns the frequency of the upper polariton
        """
        return self.omega_v * sqrt(1 + 2 * self._g() / self.omega_v)

    def omega_lp(self):
        """Returns the frequency of the lower polariton
        """
        return self.omega_v * sqrt(1 - 2 * self._g() / self.omega_v)

    # --- Hamiltonian operators ---
    def H(self, coupling=True, drive=True, rwa=False, *args, **kwargs):
        h = self.H_sys(coupling=coupling, rwa=rwa)
        if drive:
            h += self.H_drive()
        return h

    def H0_dipole(self, rwa=False):
        d = 0
        for i in range(self.space.n):
            d = self._g() * (self._a().dag() * self._b(i))
            if not rwa:
                d += self._g() * (self._a() * self._b(i))
        return d + d.dag()

    def H0_simple(self, rwa=False):
        """Hamiltonian based on Equation (1)
        """
        h0c = self.omega_c * self._a().dag() * self._a()
        h0c += self.H0_dipole(rwa=rwa)
        for i in range(self.space.n):
            h0c += self.omega_v * self._b(i).dag() * self._b(i)
            h0c += self.omega_e * self._c(i).dag() * self._c(i)
        return 1/2 * (h0c + h0c.dag())

    def H_mol(self, i):
        """Returns the Hamiltonian based on Equation (4) for the ith molecule
        """
        n_vi = self._b(i).dag() * self._b(i)
        n_ei = self._c(i).dag() * self._c(i)
        hmol = (
            self.omega_e * n_ei +
            self.omega_v * (n_vi + 2 * sqrt(self.s) * n_ei * self._b(i))
        )
        return 1/2 * (hmol + hmol.dag())

    def H_sys(self, coupling=True, rwa=False):
        """Returns the Hamiltonian based on Equation (5)
        """
        h0 = self.omega_c * self._a().dag() * self._a()
        for i in range(self.space.n):
            h0 += self.H_mol(i)
            if coupling and not rwa:
                h0 += 2 * self._g() * (self._a() + self._a().dag()) * self._b(i)
            elif coupling:
                h0 += 2 * self._g() * self._a().dag() * self._b(i)
        return 1/2 * (h0 + h0.dag())

    def H_drive(self):
        """Returns the driving part of the Hamiltonian in the rotating frame
        """
        h_dr = 0
        for i in range(self.space.n):
            h_dr += 2 * self.Omega_p * self._c(i)
            h_dr -= self.omega_L * self._c(i).dag() * self._c(i)
        return 1/2 * (h_dr + h_dr.dag())

    def collapse_ops(self):
        """Returns the collapse operators for the dissipation terms of
        Equation (6)
        """
        cops = [sqrt(self.kappa) * self.space.destroy_cav()]
        for i in range(self.space.n):
            cops.append(sqrt(self.gamma_e) * self._c(i))
            cops.append(sqrt(self.gamma_v) * self._b(i))
            cops.append(sqrt(self.gamma_e_phi) * self._c(i).dag() * self._c(i))
            cops.append(sqrt(self.gamma_v_phi) * self._b(i).dag() * self._b(i))
        return cops

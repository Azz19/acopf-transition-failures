"""Linear AC power flow sensitivities of generator Q and load-bus voltage to the controls.

Controls u are in-service generator Pg (excluding the reference bus) in per unit, then generator-bus
voltage magnitudes. States z are angles at non-reference buses and magnitudes at load buses. With
power flow equations F(z, u) = 0, dz/du = -J^-1 F_u, and output sensitivities follow by the chain rule.
Assumes one in-service generator per bus, which holds for the PGLib cases used here.
"""
import numpy as np
import scipy.sparse as sp
from pypower.dSbus_dV import dSbus_dV

from common import GST


def build_ybus(case):
    bus, br = case["bus"], case["branch"][case["branch"][:, 10] > 0]
    nb, base = bus.shape[0], float(case["baseMVA"])
    ids = {int(b): i for i, b in enumerate(bus[:, 0])}
    f = np.array([ids[int(x)] for x in br[:, 0]])
    t = np.array([ids[int(x)] for x in br[:, 1]])
    ys = 1.0 / (br[:, 2] + 1j * br[:, 3])
    tap = np.where(br[:, 8] == 0, 1.0, br[:, 8]) * np.exp(1j * np.deg2rad(br[:, 9]))
    ytt = ys + 0.5j * br[:, 4]
    y = np.zeros((nb, nb), dtype=complex)
    np.add.at(y, (f, f), ytt / (tap * np.conj(tap)))
    np.add.at(y, (t, t), ytt)
    np.add.at(y, (f, t), -ys / np.conj(tap))
    np.add.at(y, (t, f), -ys / tap)
    y[np.arange(nb), np.arange(nb)] += (bus[:, 4] + 1j * bus[:, 5]) / base
    return sp.csr_matrix(y)


class Sensitivity:
    def __init__(self, case):
        bus, gen = case["bus"], case["gen"]
        nb = bus.shape[0]
        ids = {int(b): i for i, b in enumerate(bus[:, 0])}
        self.base = float(case["baseMVA"])
        self.ybus = build_ybus(case)
        self.gb = np.array([ids[int(x)] for x in gen[:, 0]])
        self.online = np.where(gen[:, GST] > 0)[0]  # in-service generator indices
        gbus = self.gb[self.online]
        if np.unique(gbus).size != gbus.size:
            raise ValueError("More than one in-service generator on a bus is not supported.")
        ref = np.where(bus[:, 1] == 3)[0]
        self.ref = ref
        self.pv = np.setdiff1d(gbus, ref)
        self.pq = np.setdiff1d(np.arange(nb), np.r_[gbus, ref])
        self.pvpq = np.r_[self.pv, self.pq]
        self.p_gen = np.array([g for g in self.online if self.gb[g] not in ref])  # Pg controls
        self.v_gen = self.online  # Vg controls, one per generator bus
        self.nu = self.p_gen.size + self.v_gen.size
        rows = {b: i for i, b in enumerate(self.pvpq)}
        self.p_rows = np.array([rows[self.gb[g]] for g in self.p_gen])

    def compute(self, vm, va_deg):
        """Return (S_Q, S_V): d Qg(online gens) / du and d Vm(load buses) / du, both per unit."""
        v = vm * np.exp(1j * np.deg2rad(va_deg))
        d_vm, d_va = (m.toarray() for m in dSbus_dV(self.ybus, v))
        pvpq, pq = self.pvpq, self.pq
        jac = np.block([[d_va[np.ix_(pvpq, pvpq)].real, d_vm[np.ix_(pvpq, pq)].real],
                        [d_va[np.ix_(pq, pvpq)].imag, d_vm[np.ix_(pq, pq)].imag]])
        vbus = self.gb[self.v_gen]
        fu = np.zeros((jac.shape[0], self.nu))
        fu[self.p_rows, np.arange(self.p_gen.size)] = -1.0
        fu[:, self.p_gen.size:] = np.vstack([d_vm[np.ix_(pvpq, vbus)].real, d_vm[np.ix_(pq, vbus)].imag])
        dz = -np.linalg.solve(jac, fu)  # (n_states, nu)
        gbus = self.gb[self.online]
        s_q = np.hstack([d_va[np.ix_(gbus, pvpq)].imag, d_vm[np.ix_(gbus, pq)].imag]) @ dz
        s_q[:, self.p_gen.size:] += d_vm[np.ix_(gbus, vbus)].imag
        s_v = dz[pvpq.size:]
        return s_q, s_v

    def control_delta(self, pg_hat, vg_hat, pg_opt, vg_opt):
        """Control error vector in per unit, in the same order as the sensitivity columns."""
        return np.r_[(pg_hat - pg_opt)[self.p_gen] / self.base, (vg_hat - vg_opt)[self.v_gen]]

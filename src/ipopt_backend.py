"""Run PYPOWER's AC-OPF with Ipopt in place of PIPS.

The formulation, derivatives, and multiplier scaling stay PYPOWER's, so active
sets remain comparable with the pilot. Only the numerical solver changes.
Call install() once per process before the first runopf.
"""
import numpy as np
import scipy.sparse as sp
import cyipopt
import pypower.pipsopf_solver as _solver

IPOPT_OPTIONS = {
    "tol": 1e-8,
    "acceptable_tol": 1e-6,
    "acceptable_iter": 5,
    "max_iter": 1000,
    "print_level": 0,
    "sb": "yes",
    "mu_strategy": "adaptive",
}
_SUCCESS = {0, 1}  # Solve_Succeeded, Solved_To_Acceptable_Level


def _nnz_pattern(mat):
    coo = sp.coo_matrix(mat)
    return coo.row.astype(int), coo.col.astype(int)


class _Problem:
    """Adapter from PIPS callbacks to the cyipopt interface.

    Constraint order: nonlinear equalities, nonlinear inequalities, linear rows.
    """

    def __init__(self, f_fcn, gh_fcn, hess_fcn, A, x0):
        self.f_fcn, self.gh_fcn, self.hess_fcn = f_fcn, gh_fcn, hess_fcn
        self.A = sp.csr_matrix(A) if A is not None and A.shape[0] > 0 else None
        self.nx = x0.size
        h, g, dh, dg = gh_fcn(x0)
        self.nh, self.ng = np.size(h), np.size(g)
        self._dense_pattern(x0, dh, dg)

    def _stack_jac(self, dh, dg):
        # PIPS returns transposed Jacobians: dg and dh are (nx, m).
        blocks = [sp.csr_matrix(dg).T, sp.csr_matrix(dh).T]
        if self.A is not None:
            blocks.append(self.A)
        return sp.vstack(blocks, format="csr")

    def _dense_pattern(self, x0, dh, dg):
        # Sparsity can shrink at special points, so use a perturbed union.
        rng = np.random.default_rng(0)
        J = self._stack_jac(dh, dg)
        J = abs(J)
        lam0 = {"eqnonlin": np.ones(self.ng), "ineqnonlin": np.ones(self.nh)}
        H = abs(sp.csr_matrix(self.hess_fcn(x0, lam0, 1.0)))
        for _ in range(3):
            xp = x0 * (1 + 1e-3 * rng.standard_normal(self.nx)) + 1e-3 * rng.standard_normal(self.nx)
            _, _, dhp, dgp = self.gh_fcn(xp)
            J = J + abs(self._stack_jac(dhp, dgp))
            H = H + abs(sp.csr_matrix(self.hess_fcn(xp, lam0, 1.0)))
        self.jrow, self.jcol = _nnz_pattern(J)
        H = sp.tril(H, format="coo")
        self.hrow, self.hcol = H.row.astype(int), H.col.astype(int)

    def objective(self, x):
        return float(self.f_fcn(x)[0])

    def gradient(self, x):
        return np.asarray(self.f_fcn(x)[1]).ravel()

    def constraints(self, x):
        h, g, _, _ = self.gh_fcn(x)
        parts = [np.ravel(g), np.ravel(h)]
        if self.A is not None:
            parts.append(self.A @ x)
        return np.concatenate(parts)

    def jacobianstructure(self):
        return self.jrow, self.jcol

    def jacobian(self, x):
        _, _, dh, dg = self.gh_fcn(x)
        J = self._stack_jac(dh, dg).tocsr()
        return np.asarray(J[self.jrow, self.jcol]).ravel()

    def hessianstructure(self):
        return self.hrow, self.hcol

    def hessian(self, x, lagrange, obj_factor):
        lam = {"eqnonlin": lagrange[: self.ng], "ineqnonlin": lagrange[self.ng : self.ng + self.nh]}
        H = sp.csr_matrix(self.hess_fcn(x, lam, obj_factor))
        return np.asarray(H[self.hrow, self.hcol]).ravel()


def ipopt_pips(f_fcn, x0=None, A=None, l=None, u=None, xmin=None, xmax=None,
               gh_fcn=None, hess_fcn=None, opt=None):
    """Drop-in replacement for pypower.pips.pips, returning the same fields."""
    x0 = np.asarray(x0, dtype=float)
    prob = _Problem(f_fcn, gh_fcn, hess_fcn, A, x0)
    nA = 0 if prob.A is None else prob.A.shape[0]
    big = 2e19
    lb = np.where(np.isfinite(xmin), xmin, -big)
    ub = np.where(np.isfinite(xmax), xmax, big)
    cl = np.r_[np.zeros(prob.ng), -big * np.ones(prob.nh),
               np.where(np.isfinite(l), l, -big) if nA else []]
    cu = np.r_[np.zeros(prob.ng), np.zeros(prob.nh),
               np.where(np.isfinite(u), u, big) if nA else []]
    nlp = cyipopt.Problem(n=x0.size, m=cl.size, problem_obj=prob, lb=lb, ub=ub, cl=cl, cu=cu)
    for k, v in IPOPT_OPTIONS.items():
        nlp.add_option(k, v)
    x, info = nlp.solve(np.clip(x0, lb, ub))
    mult = info["mult_g"]
    lam_lin = mult[prob.ng + prob.nh :]
    # Ipopt removes fixed variables, so it reports no bound multipliers for them.
    # Recover them from stationarity, matching what PIPS reports.
    z_l, z_u = np.array(info["mult_x_L"]), np.array(info["mult_x_U"])
    fixed = np.where(lb == ub)[0]
    if fixed.size:
        _, _, dh, dg = gh_fcn(x)
        residual = prob.gradient(x) + prob._stack_jac(dh, dg).T @ mult
        z_l[fixed] = np.maximum(residual[fixed], 0)
        z_u[fixed] = np.maximum(-residual[fixed], 0)
    return {
        "x": x,
        "f": info["obj_val"],
        "eflag": 1 if info["status"] in _SUCCESS else -1,
        "output": {"status": info["status"], "message": info["status_msg"]},
        "lmbda": {
            "eqnonlin": mult[: prob.ng],
            "ineqnonlin": mult[prob.ng : prob.ng + prob.nh],
            "mu_l": np.maximum(-lam_lin, 0),
            "mu_u": np.maximum(lam_lin, 0),
            "lower": z_l,
            "upper": z_u,
        },
    }


def install():
    """Route PYPOWER's OPF solver calls through Ipopt."""
    _solver.pips = ipopt_pips

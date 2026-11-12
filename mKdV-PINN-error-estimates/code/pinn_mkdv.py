"""Physics-informed neural network experiments for the modified KdV equation.

The equation studied is

    v_t + alpha * v^2 * v_x + beta * v_xxx = 0,    x in I, t in (t0, t0 + T],

with the classical focusing case alpha = 6, beta = 1 used for the benchmarks.
The network is a tanh multilayer perceptron with two hidden layers, trained
with Adam followed by L-BFGS.

The training loss is the midpoint quadrature of the generalization error of
the paper, with weights equal to the cell measures:

    L^2 = |Q_T|/M_eq sum r_eq^2 + |I|/M_in sum r_in^2
          + T/M_b sum_{x in {x_L, x_R}} sum_n (r_b^2 + r_bx^2 + r_bxx^2),

where the boundary residuals are measured against the traces of the exact
solution (value, slope and curvature at both endpoints). The collocation points
of the single-solution benchmarks are tensor-product midpoint grids, so the
training loss is exactly the quadrature analysed in the paper. The
parameterized benchmarks use uniformly random points in the extended
(x, t, parameter) domain.

Every experiment of the paper is a named job. A job trains one
network and writes ``code/outputs/<job>.npz``. Jobs are independent and can be
run in parallel worker processes; each worker uses a single thread. All
randomness is seeded.

Usage:
    python3 pinn_mkdv.py --list                    # list the job names
    python3 pinn_mkdv.py --jobs all --workers 4    # run every job
    python3 pinn_mkdv.py --jobs all --outdir DIR   # write results to DIR
    python3 pinn_mkdv.py --jobs single_s0 amp_a2.0_s0
    python3 pinn_mkdv.py --jobs all --quick        # fast smoke run
"""

import argparse
import math
import multiprocessing as mp
import os
import time

import numpy as np
import torch

torch.set_default_dtype(torch.float64)
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTDIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Exact solutions
# ---------------------------------------------------------------------------

def sech(z):
    return 1.0 / torch.cosh(z)


def soliton(x, t, A=1.0, alpha=6.0, beta=1.0, x0=0.0):
    """Soliton A sech(k (x - x0 - c t)) of v_t + alpha v^2 v_x + beta v_xxx = 0.

    k = A sqrt(alpha / (6 beta)) and c = alpha A^2 / 6. For alpha = 6, beta = 1
    and A = a this is the one-soliton a sech(a (x - a^2 t)).
    """
    A = torch.as_tensor(A)
    alpha = torch.as_tensor(alpha)
    beta = torch.as_tensor(beta)
    k = A * torch.sqrt(alpha / (6.0 * beta))
    c = alpha * A**2 / 6.0
    return A * sech(k * (x - x0 - c * t))


def breather(x, t, a=1.0, b=1.0):
    """Breather of v_t + 6 v^2 v_x + v_xxx = 0, written in closed form.

    v = d/dx [2 arctan((b/a) sin(a y1) / cosh(b y2))] with y1 = x + (a^2 - 3 b^2) t
    and y2 = x + (3 a^2 - b^2) t.
    """
    y1 = x + (a * a - 3.0 * b * b) * t
    y2 = x + (3.0 * a * a - b * b) * t
    s, c = torch.sin(a * y1), torch.cos(a * y1)
    ch, sh = torch.cosh(b * y2), torch.sinh(b * y2)
    num = 2.0 * (b / a) * (a * c * ch - b * s * sh)
    den = ch**2 + (b / a) ** 2 * s**2
    return num / den


# ---------------------------------------------------------------------------
# Network and derivatives
# ---------------------------------------------------------------------------

class MLP(torch.nn.Module):
    """tanh MLP with two hidden layers of equal width."""

    def __init__(self, in_dim=2, width=64):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, width), torch.nn.Tanh(),
            torch.nn.Linear(width, width), torch.nn.Tanh(),
            torch.nn.Linear(width, 1),
        )
        for m in self.net:
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, z):
        return self.net(z)

    def output_bound(self):
        """Rigorous bound sup|v_theta| <= sum_j |c_j| + |c_0| (since |tanh| <= 1)."""
        last = self.net[-1]
        return (last.weight.abs().sum() + last.bias.abs().sum()).item()

    def max_abs_weight(self):
        return max(p.abs().max().item() for p in self.parameters())

    def jet(self, z, order=3, need_t=True):
        """Network value and x-derivatives up to ``order`` plus the t-derivative.

        The derivatives are propagated through the two tanh layers in closed
        form (forward mode), which is much cheaper than nested reverse-mode
        differentiation. The input is z = (x, t, extra...), extra inputs being
        constant in x and t.
        """
        l1, l2, l3 = self.net[0], self.net[2], self.net[4]
        a = l1(z)
        wx, wt = l1.weight[:, 0], l1.weight[:, 1]
        h = torch.tanh(a)
        s1 = 1.0 - h**2                      # tanh'
        s2 = -2.0 * h * s1                   # tanh''
        s3 = -2.0 * s1**2 - 2.0 * h * s2     # tanh'''
        hx = [s1 * wx, s2 * wx**2, s3 * wx**3][:order]
        c = l2(h)
        W2 = l2.weight.t()
        cx = [d @ W2 for d in hx]
        g = torch.tanh(c)
        q1 = 1.0 - g**2
        q2 = -2.0 * g * q1
        q3 = -2.0 * q1**2 - 2.0 * g * q2
        gx = []
        if order >= 1:
            gx.append(q1 * cx[0])
        if order >= 2:
            gx.append(q2 * cx[0] ** 2 + q1 * cx[1])
        if order >= 3:
            gx.append(q3 * cx[0] ** 3 + 3.0 * q2 * cx[0] * cx[1] + q1 * cx[2])
        W3 = l3.weight.t()
        out = {"v": l3(g)}
        for name, d in zip(("vx", "vxx", "vxxx"), gx):
            out[name] = d @ W3
        if need_t:
            out["vt"] = (q1 * ((s1 * wt) @ W2)) @ W3
        return out


def net_fields(model, x, t, extra=None, order=3, need_t=True,
               create_graph=True):
    """Network value and derivatives at (x, t) via the closed-form jet."""
    cols = [x, t] if extra is None else [x, t] + list(extra)
    if create_graph:
        return model.jet(torch.cat(cols, dim=1), order, need_t)
    with torch.no_grad():
        return model.jet(torch.cat(cols, dim=1), order, need_t)


def grad(y, x, create_graph=True):
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y),
                               create_graph=create_graph)[0]


def fields(fn, x, t, order=3, need_t=True, create_graph=True):
    """Value and derivatives of fn(x, t) by automatic differentiation.

    Returns a dict with keys v, vx, vxx, vxxx (up to ``order``) and vt.
    """
    x = x.detach().requires_grad_(True) if not x.requires_grad else x
    t = t.detach().requires_grad_(True) if not t.requires_grad else t
    v = fn(x, t)
    out = {"v": v}
    d = v
    names = ["vx", "vxx", "vxxx"]
    for k in range(order):
        d = grad(d, x, create_graph=True)
        out[names[k]] = d
    if need_t:
        out["vt"] = grad(v, t, create_graph=create_graph)
    if not create_graph:
        out = {k: val.detach() for k, val in out.items()}
    return out


def net_fn(model, extra=None):
    """Wrap a network as a function of (x, t), with optional fixed extra inputs."""
    def f(x, t):
        cols = [x, t] if extra is None else [x, t] + list(extra)
        return model(torch.cat(cols, dim=1))
    return f


# ---------------------------------------------------------------------------
# Midpoint grids
# ---------------------------------------------------------------------------

def mid1(lo, hi, n):
    return lo + (hi - lo) * (torch.arange(n, dtype=torch.float64) + 0.5) / n


def mid2(xb, tb, nx, nt):
    X, T = torch.meshgrid(mid1(*xb, nx), mid1(*tb, nt), indexing="ij")
    return X.reshape(-1, 1), T.reshape(-1, 1)


# ---------------------------------------------------------------------------
# Problems with an exact solution
# ---------------------------------------------------------------------------

class Problem:
    """A benchmark on I x (t0, t1] with a known exact solution."""

    def __init__(self, kind, xb, tb, alpha=6.0, beta=1.0, a=1.0, b=1.0):
        self.kind, self.xb, self.tb = kind, xb, tb
        self.alpha, self.beta, self.a, self.b = alpha, beta, a, b
        self.T = tb[1] - tb[0]
        self.L = xb[1] - xb[0]

    def exact(self, x, t):
        if self.kind == "soliton":
            return soliton(x, t, self.a, self.alpha, self.beta)
        return breather(x, t, self.a, self.b)

    def sup_norms(self):
        """Sup-norms of v and v_x over Q_T.

        For the soliton they are a and a^2/2 (the peak stays inside I); for the
        breather they are computed on a fine grid.
        """
        if self.kind == "soliton":
            return self.a, self.a**2 / 2.0
        xs, ts = mid2(self.xb, self.tb, 4000, 400)
        sv, svx = 0.0, 0.0
        for i in range(0, xs.shape[0], 200000):
            f = fields(self.exact, xs[i:i+200000], ts[i:i+200000], order=1,
                       need_t=False, create_graph=False)
            sv = max(sv, f["v"].abs().max().item())
            svx = max(svx, f["vx"].abs().max().item())
        return sv, svx


def build_data(prob, nx, nt, n_in, n_b):
    """Midpoint training sets and exact targets."""
    xe, te = mid2(prob.xb, prob.tb, nx, nt)
    xi = mid1(*prob.xb, n_in).reshape(-1, 1)
    ti = torch.full_like(xi, prob.tb[0])
    vi = prob.exact(xi, ti).detach()
    tbd = mid1(*prob.tb, n_b).reshape(-1, 1)
    bnd = []
    for xv in prob.xb:
        xbd = torch.full_like(tbd, xv)
        g = fields(prob.exact, xbd, tbd, order=2, need_t=False,
                   create_graph=False)
        bnd.append((xbd, tbd, g["v"], g["vx"], g["vxx"]))
    return {"xe": xe, "te": te, "nx": nx, "nt": nt, "xi": xi, "ti": ti,
            "vi": vi, "bnd": bnd,
            "w_eq": prob.L * prob.T / (nx * nt), "w_in": prob.L / n_in,
            "w_b": prob.T / n_b}


def loss_terms(model, prob, data, causal_eps=None):
    """Return (L^2, eq part, in part, b part, per-time-row eq means)."""
    f = net_fields(model, data["xe"], data["te"])
    req = f["vt"] + prob.alpha * f["v"] ** 2 * f["vx"] + prob.beta * f["vxxx"]
    rows = (req**2).view(data["nx"], data["nt"]).mean(0)   # mean per time row
    area = prob.L * prob.T
    if causal_eps is None:
        l_eq = data["w_eq"] * (req**2).sum()
    else:
        cum = torch.cat([torch.zeros(1), torch.cumsum(rows, 0)[:-1]])
        w = torch.exp(-causal_eps * cum).detach()
        l_eq = area * (w * rows).mean()
    vin = model(torch.cat([data["xi"], data["ti"]], dim=1))
    l_in = data["w_in"] * ((vin - data["vi"]) ** 2).sum()
    l_b = 0.0
    for xbd, tbd, g0, g1, g2 in data["bnd"]:
        fb = net_fields(model, xbd, tbd, order=2, need_t=False)
        l_b = l_b + data["w_b"] * (((fb["v"] - g0) ** 2).sum()
                                   + ((fb["vx"] - g1) ** 2).sum()
                                   + ((fb["vxx"] - g2) ** 2).sum())
    total = l_eq + l_in + l_b
    return total, l_eq, l_in, l_b, rows.detach()


# ---------------------------------------------------------------------------
# Error metrics and the quantities of the total-error theorem
# ---------------------------------------------------------------------------

def quick_errors(model, prob, nx=200, nt=50):
    """Relative L2 errors of v, v_x, v_t on a coarse grid (for histories)."""
    xs, ts = mid2(prob.xb, prob.tb, nx, nt)
    fn = net_fields(model, xs, ts, order=1, create_graph=False)
    fe = fields(prob.exact, xs, ts, order=1, create_graph=False)
    out = []
    for k in ("v", "vx", "vt"):
        out.append((torch.norm(fn[k] - fe[k]) / torch.norm(fe[k])).item())
    return out


def assess(model, prob, data, nx=800, nt=200, n1=4000, nb=2000, chunk=20000):
    """Errors, residual norms, G, K, E and the bound of the total-error theorem."""
    xs, ts = mid2(prob.xb, prob.tb, nx, nt)
    cell = prob.L * prob.T / (nx * nt)
    acc = {k: 0.0 for k in ("e2", "ex2", "et2", "v2", "vx2", "vt2", "req2")}
    sup_th, sup_th_xx, sup_v_xx = 0.0, 0.0, 0.0
    for i in range(0, xs.shape[0], chunk):
        x, t = xs[i:i+chunk], ts[i:i+chunk]
        fn = net_fields(model, x, t, order=3, create_graph=False)
        fe = fields(prob.exact, x, t, order=2, create_graph=False)
        req = fn["vt"] + prob.alpha * fn["v"] ** 2 * fn["vx"] \
            + prob.beta * fn["vxxx"]
        acc["e2"] += ((fn["v"] - fe["v"]) ** 2).sum().item() * cell
        acc["ex2"] += ((fn["vx"] - fe["vx"]) ** 2).sum().item() * cell
        acc["et2"] += ((fn["vt"] - fe["vt"]) ** 2).sum().item() * cell
        acc["v2"] += (fe["v"] ** 2).sum().item() * cell
        acc["vx2"] += (fe["vx"] ** 2).sum().item() * cell
        acc["vt2"] += (fe["vt"] ** 2).sum().item() * cell
        acc["req2"] += (req**2).sum().item() * cell
        sup_th = max(sup_th, fn["v"].abs().max().item())
        sup_th_xx = max(sup_th_xx, fn["vxx"].abs().max().item())
        sup_v_xx = max(sup_v_xx, fe["vxx"].abs().max().item())
    # initial residual
    xi = mid1(*prob.xb, n1).reshape(-1, 1)
    ti = torch.full_like(xi, prob.tb[0])
    with torch.no_grad():
        rin = model(torch.cat([xi, ti], 1)) - prob.exact(xi, ti)
    rin2 = (rin**2).sum().item() * prob.L / n1
    # boundary residuals against the exact traces
    tb = mid1(*prob.tb, nb).reshape(-1, 1)
    rb2 = rbx2 = rbxx2 = 0.0
    for xv in prob.xb:
        xbd = torch.full_like(tb, xv)
        fn = net_fields(model, xbd, tb, order=2, need_t=False,
                    create_graph=False)
        fe = fields(prob.exact, xbd, tb, order=2, need_t=False,
                    create_graph=False)
        w = prob.T / nb
        rb2 += ((fn["v"] - fe["v"]) ** 2).sum().item() * w
        rbx2 += ((fn["vx"] - fe["vx"]) ** 2).sum().item() * w
        rbxx2 += ((fn["vxx"] - fe["vxx"]) ** 2).sum().item() * w
        sup_th = max(sup_th, fn["v"].abs().max().item())
    G2 = acc["req2"] + rin2 + rb2 + rbx2 + rbxx2
    sv, svx = prob.sup_norms()
    al, be, T = abs(prob.alpha), prob.beta, prob.T

    def theorem_quantities(sup_theta):
        M = sup_theta + sv
        kappa = be + al * (sv**2 + 4.0 / 3.0 * sv * M + 0.5 * M**2)
        E = 2 * al * sv * svx + 2 * al / 3.0 * svx * (sup_theta + sv) + 1.0
        K = rin2 + acc["req2"] + kappa * (rb2 + rbx2 + rbxx2)
        log10_bound = math.log10(K * T) + T * E / math.log(10.0)
        # sharper form K (exp(T E) - 1) / E
        log10_sharp = (math.log10(K / E) + T * E / math.log(10.0)
                       + math.log10(-math.expm1(-T * E)))
        rho = math.log(acc["e2"] / (K * T)) / T
        return {"kappa_b": kappa, "E": E, "K": K, "log10_bound": log10_bound,
                "log10_bound_sharp": log10_sharp, "rho_eff": rho}

    L2 = loss_terms_nograd(model, prob, data)[0]
    res = {
        "rel_v": math.sqrt(acc["e2"] / acc["v2"]),
        "rel_vx": math.sqrt(acc["ex2"] / acc["vx2"]),
        "rel_vt": math.sqrt(acc["et2"] / acc["vt2"]),
        "e2": acc["e2"], "req2": acc["req2"], "rin2": rin2, "rb2": rb2,
        "rbx2": rbx2, "rbxx2": rbxx2, "G2": G2, "L2": L2,
        "sup_v": sv, "sup_vx": svx, "sup_theta": sup_th,
        "sup_theta_xx": sup_th_xx, "sup_v_xx": sup_v_xx,
        "out_bound": model.output_bound(),
        "max_weight": model.max_abs_weight(), "T": T,
    }
    for k, val in theorem_quantities(sup_th).items():
        res[k] = val
    for k, val in theorem_quantities(model.output_bound()).items():
        res[k + "_w"] = val
    return res


def loss_terms_nograd(model, prob, data):
    """Training loss L(theta, S)^2 evaluated in chunks without a training graph."""
    total = 0.0
    xe, te = data["xe"], data["te"]
    l_eq = 0.0
    for i in range(0, xe.shape[0], 20000):
        f = net_fields(model, xe[i:i+20000], te[i:i+20000],
                   create_graph=False)
        req = f["vt"] + prob.alpha * f["v"] ** 2 * f["vx"] \
            + prob.beta * f["vxxx"]
        l_eq += data["w_eq"] * (req**2).sum().item()
    vin = model(torch.cat([data["xi"], data["ti"]], dim=1))
    l_in = data["w_in"] * ((vin - data["vi"]) ** 2).sum().item()
    l_b = 0.0
    for xbd, tbd, g0, g1, g2 in data["bnd"]:
        fb = net_fields(model, xbd, tbd, order=2, need_t=False,
                    create_graph=False)
        l_b += data["w_b"] * (((fb["v"] - g0) ** 2).sum()
                              + ((fb["vx"] - g1) ** 2).sum()
                              + ((fb["vxx"] - g2) ** 2).sum()).item()
    total = l_eq + l_in + l_b
    return total, l_eq, l_in, l_b, None


# ---------------------------------------------------------------------------
# Checkpoints, so that an interrupted job resumes where it stopped
# ---------------------------------------------------------------------------

CKPT_EVERY = 500


def ckpt_path(name):
    d = os.path.join(OUTDIR, "ckpt")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name + ".pt")


def load_ckpt(name):
    if name is None or not os.path.exists(ckpt_path(name)):
        return None
    return torch.load(ckpt_path(name), weights_only=False)


def save_ckpt(name, state):
    if name is None:
        return
    path = ckpt_path(name)
    torch.save(state, path + ".tmp")
    os.replace(path + ".tmp", path)


def drop_ckpt(name):
    if name is not None and os.path.exists(ckpt_path(name)):
        os.remove(ckpt_path(name))


# ---------------------------------------------------------------------------
# Training of a single-solution benchmark (soliton or breather)
# ---------------------------------------------------------------------------

def train(prob, width=64, nx=400, nt=20, n_in=400, n_b=100, adam_iters=8000,
          lbfgs_iters=500, lr=1e-3, seed=0, record=False, causal=False,
          causal_eps=(1e-2, 1e-1, 1.0, 1e1, 1e2), verbose=True, tag="",
          ckpt=None):
    torch.manual_seed(seed)
    model = MLP(2, width)
    data = build_data(prob, nx, nt, n_in, n_b)
    history = []
    eps_idx = 0
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    start, elapsed = 0, 0.0
    state = load_ckpt(ckpt)
    if state is not None:
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        start, eps_idx = state["it"], state["eps_idx"]
        history, elapsed = state["history"], state["elapsed"]
        if verbose:
            print(f"    [{tag}] resuming at Adam iteration {start}", flush=True)
    t0 = time.time() - elapsed
    for it in range(start, adam_iters):
        opt.zero_grad()
        eps = causal_eps[eps_idx] if causal else None
        loss, l_eq, l_in, l_b, rows = loss_terms(model, prob, data, eps)
        loss.backward()
        opt.step()
        if causal:
            cum = torch.cat([torch.zeros(1), torch.cumsum(rows, 0)[:-1]])
            if torch.exp(-eps * cum).min() > 0.99 and \
                    eps_idx < len(causal_eps) - 1:
                eps_idx += 1
        if record and (it % 100 == 0):
            L2 = loss_terms_nograd(model, prob, data)[0]
            history.append((it, L2, *quick_errors(model, prob)))
        if verbose and it % 1000 == 0:
            print(f"    [{tag}] adam {it:6d}  loss={loss.item():.3e}"
                  + (f"  eps={eps:.0e}" if causal else ""), flush=True)
        if (it + 1) % CKPT_EVERY == 0 or it + 1 == adam_iters:
            save_ckpt(ckpt, {"model": model.state_dict(),
                             "opt": opt.state_dict(), "it": it + 1,
                             "eps_idx": eps_idx, "history": history,
                             "elapsed": time.time() - t0})

    lbfgs = torch.optim.LBFGS(model.parameters(), max_iter=lbfgs_iters,
                              history_size=50, line_search_fn="strong_wolfe",
                              tolerance_grad=1e-12, tolerance_change=1e-15)
    n_eval = [0]

    def closure():
        lbfgs.zero_grad()
        loss = loss_terms(model, prob, data)[0]
        loss.backward()
        n_eval[0] += 1
        if record and n_eval[0] % 25 == 0:
            history.append((adam_iters + n_eval[0], loss.item(),
                            *quick_errors(model, prob)))
        return loss

    lbfgs.step(closure)
    secs = time.time() - t0
    res = assess(model, prob, data)
    res["train_seconds"] = secs
    res["lbfgs_evals"] = n_eval[0]
    res["final_eps"] = causal_eps[eps_idx] if causal else 0.0
    if record:
        history.append((adam_iters + n_eval[0], res["L2"],
                        *quick_errors(model, prob)))
    if verbose:
        print(f"    [{tag}] done in {secs:.0f}s  rel v={res['rel_v']:.3e} "
              f"vx={res['rel_vx']:.3e} vt={res['rel_vt']:.3e}  "
              f"G^2={res['G2']:.3e} L^2={res['L2']:.3e} E={res['E']:.2f}",
              flush=True)
    return model, res, np.array(history)


def snapshots(model, prob, times, n=500):
    xs = mid1(*prob.xb, n).reshape(-1, 1)
    out = {"snap_x": xs.reshape(-1).numpy(), "snap_t": np.array(times)}
    P, E = [], []
    for tv in times:
        t = torch.full_like(xs, tv)
        with torch.no_grad():
            P.append(model(torch.cat([xs, t], 1)).reshape(-1).numpy())
        E.append(prob.exact(xs, t).reshape(-1).numpy())
    out["snap_pred"] = np.array(P)
    out["snap_exact"] = np.array(E)
    return out


def save(name, cfg, res, history=None, extra=None):
    payload = {"cfg_" + k: np.array(v) for k, v in cfg.items()}
    payload.update({"res_" + k: np.array(v) for k, v in res.items()})
    if history is not None and len(history):
        payload["history"] = history
    if extra:
        payload.update(extra)
    np.savez(os.path.join(OUTDIR, name + ".npz"), **payload)


# ---------------------------------------------------------------------------
# Parameterized benchmarks (uncertainty quantification)
# ---------------------------------------------------------------------------

PARAM_CASES = {
    # name: (list of (parameter, (lo, hi))), width
    "param_A": ([("A", (0.85, 1.15))], 64),
    "param_B": ([("alpha", (5.4, 6.6)), ("beta", (0.9, 1.1))], 64),
    "param_C": ([("A", (0.85, 1.15)), ("alpha", (5.4, 6.6)),
                 ("beta", (0.9, 1.1)), ("x0", (-1.0, 1.0))], 64),
    "param_C_wide": ([("A", (0.85, 1.15)), ("alpha", (5.4, 6.6)),
                      ("beta", (0.9, 1.1)), ("x0", (-1.0, 1.0))], 128),
}
PARAM_DEFAULT = {"A": 1.0, "alpha": 6.0, "beta": 1.0, "x0": 0.0}


def param_exact(x, t, P):
    return soliton(x, t, P["A"], P["alpha"], P["beta"], P["x0"])


def train_param(case, xb=(-20.0, 20.0), tb=(-1.0, 1.0), n_eq=8000, n_in=2000,
                n_b=1000, adam_iters=8000, lbfgs_iters=500, lr=1e-3, seed=2,
                verbose=True, width=None, scale_inputs=True):
    """Train one network over a parameter family.

    ``case`` is a key of PARAM_CASES, optionally with the suffix
    ``_unscaled``. With ``scale_inputs`` the parameter inputs are rescaled to
    [-1, 1]; without it they enter as raw values (an ablation of the input
    rescaling).
    """
    base = case[:-len("_unscaled")] if case.endswith("_unscaled") else case
    plist, w_default = PARAM_CASES[base]
    width = width or w_default
    names = [nm for nm, _ in plist]
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    model = MLP(2 + len(names), width)
    L, T = xb[1] - xb[0], tb[1] - tb[0]

    def u(n, lo, hi):
        return lo + (hi - lo) * torch.rand(n, 1, generator=gen)

    def draw(n):
        P = {k: torch.full((n, 1), v) for k, v in PARAM_DEFAULT.items()}
        for nm, (lo, hi) in plist:
            P[nm] = u(n, lo, hi)
        return P

    xe, te, Pe = u(n_eq, *xb), u(n_eq, *tb), draw(n_eq)
    xi, Pi = u(n_in, *xb), draw(n_in)
    ti = torch.full_like(xi, tb[0])
    vi = param_exact(xi, ti, Pi)
    tbd, Pb = u(n_b, *tb), draw(n_b)
    bnd = []
    for xv in xb:
        xbd = torch.full_like(tbd, xv)
        g = fields(lambda x, t: param_exact(x, t, Pb), xbd, tbd, order=2,
                   need_t=False, create_graph=False)
        bnd.append((xbd, g["v"], g["vx"], g["vxx"]))
    # parameter inputs are rescaled to [-1, 1]; this fixed affine map is
    # absorbed by the first layer, so the network class is unchanged
    bounds = dict(plist)

    def scaled(nm, val):
        if not scale_inputs:
            return val
        lo, hi = bounds[nm]
        return (val - 0.5 * (lo + hi)) / (0.5 * (hi - lo))

    ext = lambda P: [scaled(nm, P[nm]) for nm in names]

    def loss_fn():
        f = net_fields(model, xe, te, extra=ext(Pe))
        req = f["vt"] + Pe["alpha"] * f["v"] ** 2 * f["vx"] \
            + Pe["beta"] * f["vxxx"]
        l = L * T * (req**2).mean()
        vin = model(torch.cat([xi, ti] + ext(Pi), 1))
        l = l + L * ((vin - vi) ** 2).mean()
        for xbd, g0, g1, g2 in bnd:
            fb = net_fields(model, xbd, tbd, extra=ext(Pb), order=2,
                        need_t=False)
            l = l + T * (((fb["v"] - g0) ** 2).mean()
                         + ((fb["vx"] - g1) ** 2).mean()
                         + ((fb["vxx"] - g2) ** 2).mean())
        return l

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    start, elapsed = 0, 0.0
    state = load_ckpt(case)
    if state is not None:
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        start, elapsed = state["it"], state["elapsed"]
        if verbose:
            print(f"    [{case}] resuming at Adam iteration {start}", flush=True)
    t0 = time.time() - elapsed
    for it in range(start, adam_iters):
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        opt.step()
        if (it + 1) % CKPT_EVERY == 0 or it + 1 == adam_iters:
            save_ckpt(case, {"model": model.state_dict(),
                             "opt": opt.state_dict(), "it": it + 1,
                             "elapsed": time.time() - t0})
        if verbose and it % 1000 == 0:
            print(f"    [{case}] adam {it:6d} loss={loss.item():.3e}",
                  flush=True)
    lbfgs = torch.optim.LBFGS(model.parameters(), max_iter=lbfgs_iters,
                              history_size=50, line_search_fn="strong_wolfe",
                              tolerance_grad=1e-12, tolerance_change=1e-15)

    def closure():
        lbfgs.zero_grad()
        l = loss_fn()
        l.backward()
        return l

    lbfgs.step(closure)
    final_loss = loss_fn().item()
    secs = time.time() - t0

    # assessment over parameter draws
    g = torch.Generator().manual_seed(7)
    xs, ts = mid2(xb, tb, 400, 50)
    rel = []
    n_draw = 120
    xg = mid1(*xb, 500).reshape(-1, 1)
    preds = {"init": [], "final": []}
    exacts = {"init": [], "final": []}
    for _ in range(n_draw):
        P = dict(PARAM_DEFAULT)
        for nm, (lo, hi) in plist:
            P[nm] = float(lo + (hi - lo) * torch.rand(1, generator=g).item())
        extra = [torch.full_like(xs, scaled(nm, P[nm])) for nm in names]
        with torch.no_grad():
            vp = model(torch.cat([xs, ts] + extra, 1))
        ve = param_exact(xs, ts, P)
        rel.append((torch.norm(vp - ve) / torch.norm(ve)).item())
        for lab, tv in (("init", tb[0]), ("final", tb[1])):
            t = torch.full_like(xg, tv)
            extra = [torch.full_like(xg, scaled(nm, P[nm])) for nm in names]
            with torch.no_grad():
                preds[lab].append(model(torch.cat([xg, t] + extra, 1))
                                  .reshape(-1).numpy())
            exacts[lab].append(param_exact(xg, t, P).reshape(-1).numpy())
    out = {"x": xg.reshape(-1).numpy(), "rel_draws": np.array(rel)}
    stats = {}
    for lab in ("init", "final"):
        p, e = np.array(preds[lab]), np.array(exacts[lab])
        out[f"pred_mean_{lab}"], out[f"pred_std_{lab}"] = p.mean(0), p.std(0)
        out[f"exact_mean_{lab}"], out[f"exact_std_{lab}"] = e.mean(0), e.std(0)
        em, es = e.mean(0), e.std(0)
        stats[f"mean_err_{lab}"] = float(np.linalg.norm(p.mean(0) - em)
                                         / np.linalg.norm(em))
        stats[f"std_err_{lab}"] = float(np.linalg.norm(p.std(0) - es)
                                        / np.linalg.norm(es))
        stats[f"peak_pred_{lab}"] = float(p.mean(0).max())
        stats[f"peak_exact_{lab}"] = float(em.max())
    res = {"rel_mean": float(np.mean(rel)), "rel_max": float(np.max(rel)),
           "rel_std": float(np.std(rel)), "final_loss": final_loss,
           "train_seconds": secs, **stats}
    if verbose:
        print(f"    [{case}] done in {secs:.0f}s  mean rel L2={res['rel_mean']:.3e}"
              f" max={res['rel_max']:.3e}  mean-curve err final="
              f"{stats['mean_err_final']:.3e}", flush=True)
    cfg = {"case": case, "width": width, "n_eq": n_eq, "n_in": n_in,
           "n_b": n_b, "adam_iters": adam_iters, "lbfgs_iters": lbfgs_iters,
           "lr": lr, "seed": seed, "params": ",".join(names),
           "scaled_inputs": int(scale_inputs)}
    return model, cfg, res, out


# ---------------------------------------------------------------------------
# Job definitions
# ---------------------------------------------------------------------------

SOLITON_BASE = dict(width=64, nx=400, nt=20, n_in=400, n_b=100,
                    adam_iters=8000, lbfgs_iters=500, lr=1e-3)
BREATHER_BASE = dict(width=100, nx=320, nt=32, n_in=640, n_b=160,
                     adam_iters=15000, lbfgs_iters=1000, lr=1e-3)
AMPLITUDES = (0.5, 1.5, 2.0, 2.5, 3.0)
MEQ_NT = (4, 6, 9, 13, 28)          # n_x = 20 n_t; n_t = 20 is the benchmark


def job_list():
    jobs = [f"single_s{s}" for s in range(5)]
    jobs += [f"amp_a{a:.1f}_s{s}" for a in AMPLITUDES for s in range(3)]
    jobs += [f"meq_n{n}_s{s}" for n in MEQ_NT for s in range(3)]
    jobs += ["breather_plain", "breather_causal"]
    jobs += list(PARAM_CASES)
    jobs += [f"{case}_unscaled" for case in PARAM_CASES]
    return jobs


def save_weights(name, model):
    """Store the final trained parameters next to the results."""
    d = os.path.join(OUTDIR, "weights")
    os.makedirs(d, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(d, name + ".pt"))


def load_weights(name, outdir=None):
    """Rebuild a trained network from ``<outdir>/weights/<name>.pt``.

    The input width and the hidden width are read from the stored tensors.
    The inputs are (x, t) for the soliton and breather jobs and
    (x, t, p_1, ..., p_m) for the ``param_*`` jobs, with the parameters in
    the order of PARAM_CASES, each mapped affinely from its range to [-1, 1]
    (raw values for the ``*_unscaled`` jobs).
    """
    path = os.path.join(outdir or OUTDIR, "weights", name + ".pt")
    state = torch.load(path, weights_only=True)
    first = state["net.0.weight"]
    model = MLP(first.shape[1], first.shape[0])
    model.load_state_dict(state)
    return model.eval()


def run_job(name, quick=False):
    torch.set_num_threads(1)
    scale = 0.01 if quick else 1.0

    def budget(cfg):
        cfg = dict(cfg)
        cfg["adam_iters"] = max(50, int(cfg["adam_iters"] * scale))
        cfg["lbfgs_iters"] = max(10, int(cfg["lbfgs_iters"] * scale))
        return cfg

    if name.startswith("param_"):
        kw = {} if not quick else {"adam_iters": 50, "lbfgs_iters": 5}
        model, cfg, res, out = train_param(
            name, scale_inputs=not name.endswith("_unscaled"), **kw)
        save(name, cfg, res, extra=out)
        save_weights(name, model)
        drop_ckpt(name)
        return name
    if name.startswith("single_") or name.startswith("amp_") \
            or name.startswith("meq_"):
        cfg = budget(SOLITON_BASE)
        a, seed = 1.0, int(name.split("_s")[-1])
        if name.startswith("amp_"):
            a = float(name.split("_a")[1].split("_")[0])
        if name.startswith("meq_"):
            n = int(name.split("_n")[1].split("_")[0])
            cfg["nt"], cfg["nx"] = n, 20 * n
        prob = Problem("soliton", (-20.0, 20.0), (-1.0, 1.0), a=a)
        record = name == "single_s0"
        model, res, hist = train(prob, seed=seed, record=record, tag=name,
                                 ckpt=name, **cfg)
        cfg.update({"a": a, "seed": seed, "kind": "soliton"})
        save(name, cfg, res, hist,
             snapshots(model, prob, (prob.tb[0], 0.0, prob.tb[1])))
        save_weights(name, model)
        drop_ckpt(name)
        return name
    if name.startswith("breather_"):
        cfg = budget(BREATHER_BASE)
        prob = Problem("breather", (-20.0, 20.0), (-0.8, 0.8), a=1.0, b=1.0)
        causal = name == "breather_causal"
        model, res, hist = train(prob, seed=0, causal=causal, tag=name,
                                 ckpt=name, **cfg)
        cfg.update({"a": 1.0, "b": 1.0, "seed": 0, "kind": "breather",
                    "causal": causal})
        save(name, cfg, res, hist,
             snapshots(model, prob, (-0.8, -0.4, 0.0, 0.4, 0.8)))
        save_weights(name, model)
        drop_ckpt(name)
        return name
    raise ValueError(name)


def _worker(args):
    name, quick = args
    t0 = time.time()
    try:
        run_job(name, quick)
        return name, time.time() - t0, None
    except Exception as exc:  # report and keep the other workers going
        return name, time.time() - t0, repr(exc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", nargs="*", default=["all"])
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--quick", action="store_true",
                    help="reduced budgets for a fast smoke run")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--outdir", default=None,
                    help="write results here instead of code/outputs")
    args = ap.parse_args()
    jobs = job_list() if args.jobs == ["all"] else args.jobs
    if args.list:
        print("\n".join(job_list()))
        return
    global OUTDIR
    if args.quick:
        OUTDIR = os.path.join(os.path.dirname(OUTDIR), "outputs_quick")
    if args.outdir:
        OUTDIR = os.path.abspath(args.outdir)
    os.makedirs(OUTDIR, exist_ok=True)
    if args.skip_existing:
        jobs = [j for j in jobs
                if not os.path.exists(os.path.join(OUTDIR, j + ".npz"))]
    # longest jobs first so the pool stays busy
    order = {"breather": 0, "param_C_wide": 1, "param": 2, "meq_n28": 3}
    jobs.sort(key=lambda j: min([v for k, v in order.items() if k in j] + [9]))
    t0 = time.time()
    print(f"Running {len(jobs)} jobs with {args.workers} workers", flush=True)
    with mp.get_context("fork").Pool(args.workers) as pool:
        for name, secs, err in pool.imap_unordered(
                _worker, [(j, args.quick) for j in jobs]):
            status = "ok" if err is None else f"FAILED {err}"
            print(f"  job {name:<20s} {status}  ({secs:.0f}s)", flush=True)
    print(f"Done in {time.time() - t0:.0f}s. Outputs in {OUTDIR}")


if __name__ == "__main__":
    main()

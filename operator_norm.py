"""Random-plane operator-norm estimation and reproducible spectral experiments.

Only forward products A @ v are required. Real matrices/operators are supported.
The estimate is a lower bound on ||A||_2, not an a posteriori error certificate.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class NormResult:
    estimate: float
    vector: np.ndarray
    rayleigh_history: np.ndarray
    matvec_count: int


def _unit(v):
    v = np.asarray(v)
    if np.iscomplexobj(v):
        raise ValueError("Only real vectors/operators are supported.")
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if v.ndim != 1 or not np.isfinite(n) or n == 0:
        raise ValueError("Expected a finite, nonzero one-dimensional vector.")
    return v / n


def _plane_coefficients(rho, cross, other):
    """Largest eigenvector of one or a batch of symmetric 2x2 matrices."""
    rho, cross, other = np.broadcast_arrays(rho, cross, other)
    H = np.empty(rho.shape + (2, 2), dtype=float)
    H[..., 0, 0], H[..., 1, 1] = rho, other
    H[..., 0, 1] = H[..., 1, 0] = cross
    _, vectors = np.linalg.eigh(H)
    coefficients = vectors[..., :, -1].copy()
    # Explicit convention for a=0: retain v unless the other direction is better.
    zero = cross == 0
    coefficients[..., 0] = np.where(zero, (other <= rho).astype(float), coefficients[..., 0])
    coefficients[..., 1] = np.where(zero, (other > rho).astype(float), coefficients[..., 1])
    return coefficients


def estimate_operator_norm(A, *, n=None, steps=1000, v0=None, seed=None):
    """Estimate the spectral norm of a real dense/sparse matrix or forward operator.

    A may be a matrix, a scipy LinearOperator, or a callable v -> A v.
    Supply n (the input dimension) for callables without a shape attribute.
    Uses one initial forward product and one per iteration; never calls A.T.
    Fixed iteration budgets avoid mistaking one small random step for convergence.
    The returned history includes iteration zero and stores squared norm estimates.
    """
    if isinstance(steps, bool) or int(steps) != steps or steps < 0:
        raise ValueError("steps must be a nonnegative integer.")
    steps = int(steps)
    if n is None:
        if not hasattr(A, 'shape') or len(A.shape) != 2:
            raise ValueError("Supply n for an operator without a matrix shape.")
        n = A.shape[1]
    if isinstance(n, bool) or int(n) != n or n < 1:
        raise ValueError("n must be a positive integer.")
    n = int(n)
    if hasattr(A, 'dtype') and np.issubdtype(A.dtype, np.complexfloating):
        raise ValueError("Only real operators are supported.")
    action = A if callable(A) else lambda v: A @ v

    def matvec(v):
        av = np.asarray(action(v))
        if np.iscomplexobj(av) or av.ndim != 1 or not np.all(np.isfinite(av)):
            raise ValueError("A must return a finite real one-dimensional vector.")
        return np.asarray(av, dtype=float)

    rng = np.random.default_rng(seed)
    v = _unit(rng.normal(size=n) if v0 is None else v0)
    if v.shape != (n,):
        raise ValueError("v0 has the wrong dimension.")
    av = matvec(v)
    history = [float(av @ av)]
    if n == 1:
        return NormResult(float(np.linalg.norm(av)), v, np.array(history), 1)
    for _ in range(steps):
        # A fresh Gaussian and Euclidean tangent projection at every step.
        while True:
            x = rng.normal(size=n)
            x -= (x @ v) * v
            x -= (x @ v) * v  # reorthogonalization for floating-point robustness
            nx = np.linalg.norm(x)
            if nx > 0:
                x /= nx
                break
        ax = matvec(x)
        if ax.shape != av.shape:
            raise ValueError("Operator output dimension changed.")
        c = _plane_coefficients(av @ av, av @ ax, ax @ ax)
        v_new = c[0] * v + c[1] * x
        norm = np.linalg.norm(v_new)
        v = v_new / norm
        av = (c[0] * av + c[1] * ax) / norm
        history.append(float(av @ av))
    return NormResult(float(np.sqrt(max(history[-1], 0))), v, np.array(history), steps + 1)


def controlled_spectrum(d, gap):
    """lambda_1=1, lambda_2=1-gap, fixed lower spectrum in [0,1/2]."""
    if int(d) != d or d < 3 or not 0 < gap <= .5:
        raise ValueError("Require integer d>=3 and 0<gap<=1/2.")
    return np.r_[1., 1. - gap, np.linspace(.5, 0., int(d)-2)] if d > 3 else np.array([1., 1.-gap, 0.])


def local_initial_vectors(lambdas, runs, eta=.5):
    """Identical local starts with epsilon_0=eta*delta/2."""
    if not 0 < eta < 1:
        raise ValueError("eta must be in (0,1).")
    V = np.zeros((runs, len(lambdas)))
    V[:, 0], V[:, 1] = np.sqrt(1-eta/2), np.sqrt(eta/2)
    return V


def stable_errors(V, lambdas):
    """Nonnegative spectral error without subtracting nearly equal quotients."""
    return np.sum(np.square(V) * (lambdas[0] - lambdas), axis=-1)


def diagonal_ensemble(lambdas, *, runs, steps, seed, initial=None, eta=.5):
    """Vectorized independent instances of the same 2x2 plane algorithm.

    A=diag(sqrt(lambdas)). Diagnostics use the known spectrum, but the updates
    use only forward images Av and Ax. Histories include k=0. All runs receive
    the same fixed budget, including those whose errors are already small.
    hit_times=-1 explicitly marks right-censoring; hit_vectors stores first entry.
    """
    lam = np.asarray(lambdas, dtype=float)
    if lam.ndim != 1 or len(lam) < 2 or np.any(~np.isfinite(lam)) or np.any(lam < 0) or np.any(np.diff(lam) > 0):
        raise ValueError("Need a finite, nonnegative, descending spectrum of length >=2.")
    if not 0 < eta < 1 or runs < 1 or int(runs) != runs or steps < 0 or int(steps) != steps:
        raise ValueError("Invalid eta, runs, or steps.")
    runs, steps = int(runs), int(steps)
    rng = np.random.default_rng(seed)
    d = len(lam)
    V = rng.normal(size=(runs, d)) if initial is None else np.array(initial, dtype=float, copy=True)
    if V.shape != (runs, d) or not np.all(np.isfinite(V)) or np.any(np.linalg.norm(V, axis=1) == 0):
        raise ValueError("initial must have shape (runs,d) with finite nonzero rows.")
    V /= np.linalg.norm(V, axis=1)[:, None]
    sqrt_lam = np.sqrt(lam)
    E = np.empty((runs, steps+1))
    E[:, 0] = stable_errors(V, lam)
    hits = np.full(runs, -1, dtype=int)
    hit_vectors = np.full_like(V, np.nan)
    threshold = eta * (lam[0] - lam[1])

    def record_hits(k):
        entered = (hits < 0) & (E[:, k] <= threshold)
        hits[entered] = k
        hit_vectors[entered] = V[entered]

    record_hits(0)
    for k in range(1, steps+1):
        X = rng.normal(size=V.shape)
        X -= np.sum(X*V, axis=1)[:, None] * V
        X -= np.sum(X*V, axis=1)[:, None] * V
        norms = np.linalg.norm(X, axis=1)
        while np.any(norms == 0):
            bad = norms == 0
            X[bad] = rng.normal(size=(bad.sum(), d))
            X[bad] -= np.sum(X[bad]*V[bad], axis=1)[:, None]*V[bad]
            norms = np.linalg.norm(X, axis=1)
        X /= norms[:, None]
        AV, AX = V*sqrt_lam, X*sqrt_lam
        c = _plane_coefficients(np.sum(AV*AV, axis=1), np.sum(AV*AX, axis=1), np.sum(AX*AX, axis=1))
        V = c[:, :1]*V + c[:, 1:]*X
        V /= np.linalg.norm(V, axis=1)[:, None]
        E[:, k] = stable_errors(V, lam)
        record_hits(k)
    return dict(errors=E, hit_times=hits, hit_vectors=hit_vectors, final_vectors=V)


def rate_constants(lambdas, eta=.5):
    from scipy.special import betainc, betaln, hyp2f1
    lam = np.asarray(lambdas)
    d = len(lam)
    delta, width = float(lam[0]-lam[1]), float(lam[0]-lam[-1])
    if d < 3 or delta <= 0 or not 0 < eta < 1:
        raise ValueError("Need d>=3, a simple largest eigenvalue, and 0<eta<1.")
    c = (1-eta)*delta/((d-1)*width)
    u, b = eta*delta/width, (d-2)/2
    # Symmetry: p = I_u(b,1/2). Keep log(p) even if p underflows.
    p = float(betainc(b, .5, u))
    logp = float(b*np.log(u)-np.log(b)-betaln(b,.5)+np.log(hyp2f1(b,.5,b+1,u)))
    return dict(delta=delta, width=width, q=1-c, logq=float(np.log1p(-c)), p=p, logp=logp)


def global_bound(steps, constants, eta=.5):
    """Optimize (12) over integer j using convexity, in O(steps) work."""
    k = np.arange(steps+1)
    D, h = constants['width'], eta*constants['delta']
    a, b = -np.log1p(-constants['p']), -constants['logq']
    if a == 0:  # underflowed p: conservative limiting bound, explicitly reported
        j = np.zeros_like(k)
    else:
        j = np.clip((np.log(D*a/(h*b)) + b*k)/(a+b), 0, k)
    lo, hi = np.floor(j), np.ceil(j)
    def value(jj):
        return D*np.exp(-a*jj)+h*np.exp(-b*(k-jj))
    return np.minimum(value(lo), value(hi))


def bootstrap_mean_band(errors, *, seed=1, resamples=200, max_points=160):
    """Pointwise 95% bootstrap bands, resampling complete independent runs."""
    rng = np.random.default_rng(seed)
    indices = np.unique(np.linspace(0, errors.shape[1]-1, min(max_points, errors.shape[1]), dtype=int))
    sample = errors[:, indices]
    weights = rng.multinomial(len(sample), np.full(len(sample), 1/len(sample)), size=resamples)
    means = weights @ sample / len(sample)
    lower, upper = np.quantile(means, [.025, .975], axis=0)
    return indices, lower, upper


def fitted_mean_rate(errors, *, relative_window=(1e-4, .5)):
    """Descriptive log-mean slope over a declared window; not a rate guarantee."""
    mean = errors.mean(axis=0)
    ratio = mean / mean[0]
    mask = (ratio >= relative_window[0]) & (ratio <= relative_window[1])
    if mask.sum() < 10:
        return np.nan
    return float(-np.polyfit(np.flatnonzero(mask), np.log(mean[mask]), 1)[0])

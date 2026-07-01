"""
Nucleo del modelo propio.
Estado latente X = ln(V/K), MBA absorbido en X=0. DOS derivas :
  * Valoracion  mu_X = 1/2(beta^2-gamma^2): entra en q(X,tau), b(X,tau), inversion, jacobiano.
  * Real        mu_Q = (mu_V-mu_K)+mu_X    : gobierna simulacion y densidad de transicion.
  Volatilidad   sigma_X = beta-gamma (beta>gamma>0).
q,b dependen de (mu_X,sigma_X); la densidad de (mu_Q,sigma_X). gamma,beta nunca
aparecen por separado: el modelo se describe con (mu_X,sigma_X,mu_Q,W).
"""
import numpy as np
from scipy.special import ndtr

_INV_SQRT2PI = 1.0 / np.sqrt(2.0 * np.pi)
def _phi(x): return _INV_SQRT2PI * np.exp(-0.5 * x * x)
def _Phi(x): return ndtr(x)


def valuation_drift(gamma, beta):
    return 0.5 * (beta ** 2 - gamma ** 2)

def sigma_from_gamma_beta(gamma, beta):
    return beta - gamma

def gamma_beta_from_muX_sigma(mu_X, sigma_X):
    s = 2.0 * mu_X / sigma_X
    return 0.5 * (s - sigma_X), 0.5 * (s + sigma_X)


def default_prob(X, tau, mu_X, sigma_X):
    X = np.asarray(X, dtype=float)
    s = sigma_X * np.sqrt(tau)
    z1 = (X + mu_X * tau) / s
    z2 = (X - mu_X * tau) / s
    return _Phi(-z1) + np.exp(-2.0 * mu_X * X / sigma_X ** 2) * _Phi(-z2)

def survival_prob(X, tau, mu_X, sigma_X):
    return 1.0 - default_prob(X, tau, mu_X, sigma_X)

def bond_price(X, tau, mu_X, sigma_X, W, r):
    return np.exp(-r * tau) * (1.0 - W * default_prob(X, tau, mu_X, sigma_X))

def dq_dX(X, tau, mu_X, sigma_X):
    X = np.asarray(X, dtype=float)
    s = sigma_X * np.sqrt(tau)
    z1 = (X + mu_X * tau) / s
    z2 = (X - mu_X * tau) / s
    t1 = -_phi(z1) / s
    t2 = -np.exp(-2.0 * mu_X * X / sigma_X ** 2) * (
        (2.0 * mu_X / sigma_X ** 2) * _Phi(-z2) + _phi(z2) / s)
    return t1 + t2

def db_dX(X, tau, mu_X, sigma_X, W, r):
    return -W * np.exp(-r * tau) * dq_dX(X, tau, mu_X, sigma_X)


def invert_bond(b_obs, tau, mu_X, sigma_X, W, r, x0=None, tol=1e-10, maxit=40):
    b_obs = np.atleast_1d(np.asarray(b_obs, dtype=float))
    tau = np.broadcast_to(np.asarray(tau, dtype=float), b_obs.shape).astype(float).copy()
    disc = np.exp(-r * tau)
    q_obj = np.clip((1.0 - b_obs / disc) / W, 1e-13, 1.0 - 1e-13)
    lo = np.full_like(b_obs, 1e-9); hi = np.full_like(b_obs, 60.0)
    X = np.clip(np.asarray(x0, float).copy(), lo, hi) if x0 is not None else np.full_like(b_obs, 0.5)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for _ in range(maxit):
            f = default_prob(X, tau, mu_X, sigma_X) - q_obj
            pos = f > 0.0
            lo = np.where(pos, X, lo); hi = np.where(~pos, X, hi)
            d = dq_dX(X, tau, mu_X, sigma_X)
            step = np.where(np.abs(d) > 1e-14, f / d, 0.0)
            Xnew = X - step
            bad = (Xnew <= lo) | (Xnew >= hi) | ~np.isfinite(Xnew)
            Xnew = np.where(bad, 0.5 * (lo + hi), Xnew)
            if np.max(np.abs(Xnew - X)) < tol:
                X = Xnew; break
            X = Xnew
    return X

invert_bond_vectorized = invert_bond


def simulate_X_path(X0, mu_Q, sigma_X, dt, n, rng):
    incr = rng.normal(mu_Q * dt, sigma_X * np.sqrt(dt), size=n)
    X = np.empty(n + 1); X[0] = X0; X[1:] = X0 + np.cumsum(incr)
    return X, bool(np.all(X > 0.0))

def simulate_surviving_path(X0, mu_Q, sigma_X, dt, n, rng, max_tries=50000):
    for _ in range(max_tries):
        X, ok = simulate_X_path(X0, mu_Q, sigma_X, dt, n, rng)
        if ok:
            return X
    return None

def simulate_VK_path(V0, K0, mu_V, mu_K, gamma, beta, delta, dt, n, rng):
    z = rng.normal(0.0, np.sqrt(dt), size=n)
    V = np.empty(n + 1); K = np.empty(n + 1); V[0] = V0; K[0] = K0
    for i in range(n):
        V[i + 1] = V[i] * np.exp((mu_V - delta - 0.5 * gamma ** 2) * dt + gamma * z[i])
        K[i + 1] = K[i] * np.exp((mu_K - delta - 0.5 * beta ** 2) * dt + beta * z[i])
    return V, K


def log_transition_absorbed(X_next, X_prev, dt, mu_Q, sigma_X):
    var = sigma_X ** 2 * dt
    main = np.exp(-(X_next - X_prev - mu_Q * dt) ** 2 / (2.0 * var))
    image = np.exp(-2.0 * mu_Q * X_prev / sigma_X ** 2) * \
        np.exp(-(X_next + X_prev - mu_Q * dt) ** 2 / (2.0 * var))
    dens = (main - image) / (sigma_X * np.sqrt(2.0 * np.pi * dt))
    return np.log(np.clip(dens, 1e-300, None))


def recover_state(b_obs, tau, mu_X, sigma_X, W, r, x0=None):
    Xhat = invert_bond(b_obs, tau, mu_X, sigma_X, W, r, x0=x0)
    if np.any(~np.isfinite(Xhat)) or np.any(Xhat <= 0.0):
        return None
    if np.max(np.abs(bond_price(Xhat, tau, mu_X, sigma_X, W, r) - b_obs)) > 1e-7:
        return None
    return Xhat


def loglik(theta, b_obs, tau, dt, r, mu_X, W=None):
    if W is None:
        mu_Q, sigma_X, W = float(theta[0]), float(theta[1]), float(theta[2])
    else:
        mu_Q, sigma_X = float(theta[0]), float(theta[1])
    if sigma_X <= 1e-6 or not (0.0 < W < 1.0):
        return -np.inf
    Xhat = recover_state(b_obs, tau, mu_X, sigma_X, W, r)
    if Xhat is None:
        return -np.inf
    ll = log_transition_absorbed(Xhat[1:], Xhat[:-1], dt, mu_Q, sigma_X)
    jac = db_dX(Xhat[1:], tau[1:], mu_X, sigma_X, W, r)
    if np.any(jac <= 0.0) or np.any(~np.isfinite(jac)):
        return -np.inf
    v = np.sum(ll) - np.sum(np.log(jac))
    return v if np.isfinite(v) else -np.inf

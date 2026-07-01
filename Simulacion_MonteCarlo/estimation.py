"""
EMV por NEWTON-RAPHSON.
Se resuelve la condicion de primer orden grad L(theta)=0 por el algoritmo de
Newton. La deriva real mu_Q es un parametro de la SOLA densidad de transicion y
NO interviene en la recuperacion del estado Xhat=b^{-1}(.) (que depende de
mu_X fijo, sigma_X, W). Se concentra (perfila) mu_Q en forma cerrada mediante un
Newton 1D interno sobre la densidad (con Xhat dado), y el Newton EXTERNO resuelve
la log-verosimilitud concentrada en sigma_X (y, en su caso, W). Asi se evalua la
verosimilitud exacta y cada paso de Newton requiere
una sola inversion del precio.

  fit_newton(..., W_fixed=valor) -> estima (mu_Q, sigma_X)      (W FIJO)
  fit_newton(..., W_fixed=None)  -> estima (mu_Q, sigma_X, W)   (CONJUNTO)

Errores estandar por informacion observada (Hessiano de la log-verosimilitud
COMPLETA en el optimo), usados para la cobertura de los intervalos asintoticos.
"""
import numpy as np
import model as M

_SCALE = {"mu_Q": 0.05, "sigma_X": 0.10, "W": 0.30}


def _dens_negll(mu_Q, Xhat, dt, sigma_X):
    ll = M.log_transition_absorbed(Xhat[1:], Xhat[:-1], dt, mu_Q, sigma_X)
    return -np.sum(ll)


def profile_muQ(Xhat, dt, sigma_X, mu0=0.05, bounds=(-0.6, 0.9)):
    """Newton 1D que maximiza la densidad en mu_Q con Xhat, sigma_X dados."""
    mu = float(np.clip(mu0, *bounds))
    h = 1e-5
    for _ in range(8):
        f0 = _dens_negll(mu, Xhat, dt, sigma_X)
        fp = _dens_negll(mu + h, Xhat, dt, sigma_X)
        fm = _dens_negll(mu - h, Xhat, dt, sigma_X)
        g = (fp - fm) / (2 * h)
        H = (fp - 2 * f0 + fm) / h ** 2
        if not np.isfinite(H) or abs(H) < 1e-12:
            break
        step = g / H
        munew = mu - step
        munew = float(np.clip(munew, *bounds))
        if abs(munew - mu) < 1e-9:
            mu = munew
            break
        mu = munew
    return mu


class _Concentrated:
    """-log L concentrada: recupera Xhat(sigma,W) y perfila mu_Q."""
    def __init__(self, b, tau, dt, r, mu_X, x0_warm):
        self.b = b; self.tau = tau; self.dt = dt; self.r = r; self.mu_X = mu_X
        self._warm = x0_warm.copy()
        self._key = None; self._Xhat = None; self._mu = 0.05

    def state(self, sigma_X, W):
        key = (round(sigma_X, 9), round(W, 9))
        if key == self._key:
            return self._Xhat
        Xhat = M.recover_state(self.b, self.tau, self.mu_X, sigma_X, W, self.r,
                               x0=self._warm)
        if Xhat is not None:
            self._warm = Xhat; self._key = key; self._Xhat = Xhat
        return Xhat

    def __call__(self, sigma_X, W):
        if sigma_X <= 1e-4 or not (0.0 < W < 1.0):
            return 1e12, np.nan
        Xhat = self.state(sigma_X, W)
        if Xhat is None:
            return 1e12, np.nan
        mu = profile_muQ(Xhat, self.dt, sigma_X, mu0=self._mu)
        self._mu = mu
        ll = M.log_transition_absorbed(Xhat[1:], Xhat[:-1], self.dt, mu, sigma_X)
        jac = M.db_dX(Xhat[1:], self.tau[1:], self.mu_X, sigma_X, W, self.r)
        if np.any(jac <= 0.0) or np.any(~np.isfinite(jac)):
            return 1e12, mu
        v = np.sum(ll) - np.sum(np.log(jac))
        return (-v if np.isfinite(v) else 1e12), mu


def _full_negll(theta, b, tau, dt, r, mu_X, W_fixed, x0w=None):
    """-log L COMPLETA (mu_Q explicito), para Hessiano / errores estandar."""
    if W_fixed is None:
        mu_Q, sigma_X, W = theta
    else:
        mu_Q, sigma_X = theta; W = W_fixed
    if sigma_X <= 1e-4 or not (0.0 < W < 1.0):
        return 1e12
    Xhat = M.recover_state(b, tau, mu_X, sigma_X, W, r, x0=x0w)
    if Xhat is None:
        return 1e12
    ll = M.log_transition_absorbed(Xhat[1:], Xhat[:-1], dt, mu_Q, sigma_X)
    jac = M.db_dX(Xhat[1:], tau[1:], mu_X, sigma_X, W, r)
    if np.any(jac <= 0.0) or np.any(~np.isfinite(jac)):
        return 1e12
    v = np.sum(ll) - np.sum(np.log(jac))
    return -v if np.isfinite(v) else 1e12


def _hess(f, x, h):
    k = len(x); H = np.zeros((k, k)); f0 = f(x)
    fp = [f(x + np.eye(k)[i] * h[i]) for i in range(k)]
    fm = [f(x - np.eye(k)[i] * h[i]) for i in range(k)]
    for i in range(k):
        H[i, i] = (fp[i] - 2 * f0 + fm[i]) / h[i] ** 2
    for i in range(k):
        for j in range(i + 1, k):
            ei = np.eye(k)[i] * h[i]; ej = np.eye(k)[j] * h[j]
            H[i, j] = H[j, i] = (f(x+ei+ej) - f(x+ei-ej) - f(x-ei+ej) + f(x-ei-ej)) \
                / (4 * h[i] * h[j])
    return H


def _newton1d(F, x0, lo, hi, h0, maxit=6):
    """Newton 1D amortiguado sobre F(x) (escalar) en [lo,hi]."""
    x = float(np.clip(x0, lo, hi)); fx = F(x); ok = False
    for _ in range(maxit):
        h = h0 * max(abs(x), 1.0)
        fp = F(min(x + h, hi)); fm = F(max(x - h, lo))
        g = (fp - fm) / (2 * h); Hh = (fp - 2 * fx + fm) / h ** 2
        if not np.isfinite(Hh) or Hh <= 1e-9:
            Hh = max(abs(Hh), 1.0)            # salvaguarda (region plana)
        step = g / Hh; t = 1.0; moved = False
        for _ in range(8):
            xn = float(np.clip(x - t * step, lo, hi)); fn = F(xn)
            if fn < fx - 1e-10:
                if abs(xn - x) < 1e-7:
                    x, fx = xn, fn; ok = True; moved = True
                    return x, ok
                x, fx = xn, fn; moved = True; break
            t *= 0.5
        if not moved:
            ok = True; break
    return x, ok


def fit_newton(b_obs, tau, dt, r, mu_X, theta_true, x0_warm, W_fixed=None,
               rng=None, jitter=0.10):
    mu_Q0, sig0, W0 = theta_true
    rng = rng if rng is not None else np.random.default_rng()
    j = jitter
    sig_start = sig0 * (1 + rng.uniform(-j, j))
    conc = _Concentrated(b_obs, tau, dt, r, mu_X, x0_warm)

    if W_fixed is None:
        W_start = float(np.clip(W0 * (1 + rng.uniform(-j, j)), 0.15, 0.9))
        # Newton externo 2D (sigma_X, W) sobre la verosimilitud concentrada,
        # alternando Newton-1D en cada coordenada (Gauss-Seidel/Newton).
        sg, W = sig_start, W_start; ok_all = True
        for sweep in range(3):
            sg_new, ok1 = _newton1d(lambda s: conc(s, W)[0], sg, 0.03, 0.40, 2e-2)
            W_new, ok2 = _newton1d(lambda w: conc(sg_new, w)[0], W, 0.15, 0.92, 2e-2)
            if abs(sg_new - sg) < 1e-5 and abs(W_new - W) < 1e-5:
                sg, W = sg_new, W_new; break
            sg, W = sg_new, W_new
        ok_all = ok1 and ok2
        _, mu_Q = conc(sg, W)
        theta = np.array([mu_Q, sg, W]); names = ("mu_Q", "sigma_X", "W")
    else:
        W = W_fixed
        sg, ok_all = _newton1d(lambda s: conc(s, W)[0], sig_start, 0.03, 0.40, 2e-2)
        _, mu_Q = conc(sg, W)
        theta = np.array([mu_Q, sg]); names = ("mu_Q", "sigma_X")

    # --- errores estandar: Hessiano de la log-verosimilitud COMPLETA --------
    h = np.array([3e-4 * max(abs(theta[i]), _SCALE[names[i]]) for i in range(len(theta))])
    H = _hess(lambda x: _full_negll(x, b_obs, tau, dt, r, mu_X, W_fixed, x0w=x0_warm), theta, h)
    se = np.full(len(theta), np.nan)
    try:
        cov = np.linalg.inv(H); d = np.diag(cov)
        if np.all(d > 0) and np.all(np.isfinite(d)):
            se = np.sqrt(d)
    except np.linalg.LinAlgError:
        pass

    out = dict(success=bool(ok_all), mu_Q=float(theta[0]), sigma_X=float(theta[1]),
               se_mu_Q=float(se[0]), se_sigma_X=float(se[1]))
    if W_fixed is None:
        out["W"] = float(theta[2]); out["se_W"] = float(se[2])
    else:
        out["W"] = float(W_fixed); out["se_W"] = float("nan")
    return out

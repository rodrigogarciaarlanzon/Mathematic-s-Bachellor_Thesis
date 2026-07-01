"""
Experimento de relleno (in-fill) sobre sigma_X.
OBJETIVO. Discernir si el sesgo de sigma_hat es estructural (inconsistencia) o un
artefacto de muestreo discreto. Para AISLAR el efecto de la frecuencia se mantiene
FIJO el horizonte de observacion (1 ano) y se refina el paso temporal,
    n in {252, 504, 1008},  dt = 1/n,
de modo que el numero de observaciones aumenta pero el intervalo calendario es el
mismo. Se emplea una empresa informativa pero holgadamente solvente (X0 alto, sin
seleccion por supervivencia), con W fijado, para que la unica fuente de variacion
sea la finura de la malla. Si el sesgo de sigma_hat decrece monotonamente al
refinar dt, el estimador es consistente en el limite de relleno (dt -> 0) y el
sesgo a frecuencia diaria es un fenomeno de discretizacion.
"""
import os, json, numpy as np, multiprocessing as mp
import model as M, estimation as E, config as C

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "resultados"); os.makedirs(RES, exist_ok=True)

# empresa informativa, holgadamente solvente (sin seleccion por supervivencia)
GAMMA, BETA, W, X0 = 0.20, 0.35, 0.60, 0.30
MU_VMK = 0.02875
SIGMA = BETA - GAMMA
MU_X = M.valuation_drift(GAMMA, BETA)
MU_Q = MU_VMK + MU_X
SIZES = [252, 504, 1008]
REP = 60


def fit_one(args):
    n, k = args
    dt = 1.0 / n
    tau = np.array([C.TAU0 - i * dt for i in range(n + 1)])
    rng = np.random.default_rng(4000 + 13 * n + k)
    X = M.simulate_surviving_path(X0, MU_Q, SIGMA, dt, n, rng)
    if X is None:
        return (n, None)
    b = M.bond_price(X, tau, MU_X, SIGMA, W, C.R)
    fit = E.fit_newton(b, tau, dt, C.R, MU_X, (MU_Q, SIGMA, W),
                       x0_warm=X, W_fixed=W, rng=np.random.default_rng(7 * k + 1))
    return (n, fit["sigma_X"])


def main():
    jobs = [(n, k) for n in SIZES for k in range(REP)]
    res = {n: [] for n in SIZES}
    with mp.Pool(2) as p:
        for n, v in p.map(fit_one, jobs):
            if v is not None and np.isfinite(v):
                res[n].append(v)
    out = []
    for n in SIZES:
        v = np.array(res[n])
        out.append(dict(n=n, dt=1.0/n, sigma_true=SIGMA, mean=float(v.mean()),
                        median=float(np.median(v)), std=float(v.std(ddof=1)),
                        bias=float(v.mean()-SIGMA), nrep=len(v)))
        print("n=%4d dt=1/%d  mean=%.4f bias=%+.4f std=%.4f (N=%d)" % (
            n, n, v.mean(), v.mean()-SIGMA, v.std(ddof=1), len(v)))
    json.dump(out, open(os.path.join(RES, "infill.json"), "w"), indent=2)
    print("OK infill")


if __name__ == "__main__":
    main()

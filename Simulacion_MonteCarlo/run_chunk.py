"""
Experimento Monte Carlo del estimador MV por Newton.
Ejecuta, de forma REANUDABLE y por tramos de tiempo, las 8 celdas del diseño:
  2 escenarios (solvente, barrera) x 2 horizontes (252, 504) x 2 modos
  (W conjunto / W fijo).  Cada fila estimada se anade a resultados/mc_raw.csv.
"""
import os, sys, csv, time
import numpy as np
import multiprocessing as mp
import config as C, model as M, estimation as E

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "resultados"); os.makedirs(RES, exist_ok=True)
RAW = os.path.join(RES, "mc_raw.csv")
FIELDS = ["mode", "scenario", "n", "rep",
          "muQ_true", "sigma_true", "W_true",
          "muQ_hat", "sigma_hat", "W_hat",
          "se_muQ", "se_sigma", "se_W", "success"]
MODES = ["wfix", "joint"]


def jobs():
    out = []
    for mode in MODES:
        m = C.N_REP_JOINT if mode == "joint" else C.N_REP
        for name in C.SCENARIOS:
            for n in C.SAMPLE_SIZES:
                for rep in range(m):
                    out.append((mode, name, n, rep))
    return out


def done_keys():
    if (not os.path.exists(RAW)) or os.path.getsize(RAW) == 0:
        return set()
    return {(r["mode"], r["scenario"], int(r["n"]), int(r["rep"]))
            for r in csv.DictReader(open(RAW))}


def one(args):
    mode, name, n, rep = args
    p = C.scenario_params(name)
    tau = C.tau_grid(n)
    rng = np.random.default_rng(C.SEED + 9973 * n + rep)
    X = M.simulate_surviving_path(p["X0"], p["mu_Q"], p["sigma_X"], C.DT, n, rng)
    if X is None:
        return None
    b = M.bond_price(X, tau, p["mu_X"], p["sigma_X"], p["W"], C.R)
    Wf = None if mode == "joint" else p["W"]
    fit = E.fit_newton(b, tau, C.DT, C.R, p["mu_X"],
                       (p["mu_Q"], p["sigma_X"], p["W"]), x0_warm=X,
                       W_fixed=Wf, rng=np.random.default_rng(13 * rep + 7),
                       jitter=C.START_JITTER)
    return dict(mode=mode, scenario=name, n=n, rep=rep,
                muQ_true=p["mu_Q"], sigma_true=p["sigma_X"], W_true=p["W"],
                muQ_hat=fit["mu_Q"], sigma_hat=fit["sigma_X"], W_hat=fit["W"],
                se_muQ=fit["se_mu_Q"], se_sigma=fit["se_sigma_X"], se_W=fit["se_W"],
                success=int(fit["success"]))


def main(budget=1e12):
    t0 = time.time()
    pend = [j for j in jobs() if (j[0], j[1], j[2], j[3]) not in done_keys()]
    print("pendientes:", len(pend), flush=True)
    if not pend:
        print("COMPLETO", flush=True); return
    new = (not os.path.exists(RAW)) or os.path.getsize(RAW) == 0
    f = open(RAW, "a", newline=""); w = csv.DictWriter(f, fieldnames=FIELDS)
    if new:
        w.writeheader()
    nd = 0
    nproc = max(2, (os.cpu_count() or 2) - 1)
    with mp.Pool(processes=nproc) as pool:
        B = 8; i = 0
        while i < len(pend) and (time.time() - t0) < budget:
            for res in pool.imap_unordered(one, pend[i:i + B]):
                if res is not None:
                    w.writerow(res); nd += 1
            f.flush(); i += B
    f.close()
    print("anadidas %d filas en %.0fs (restan ~%d)" % (nd, time.time()-t0, len(pend)-i), flush=True)


if __name__ == "__main__":
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 1e12
    main(budget)

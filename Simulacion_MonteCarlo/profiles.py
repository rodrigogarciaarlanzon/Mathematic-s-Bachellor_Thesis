"""
Figuras de identificacion.
(1) Verosimilitud perfil en W (perfilando sigma_X y mu_Q): su planitud revela la
    debil identificacion de W a partir de un unico bono.
(2) Contorno de -logL en (mu_Q, sigma_X) con (mu_X, W) fijos: el valle alargado a
    lo largo de mu_Q ilustra la debil identificacion de la deriva real.
"""
import os, numpy as np
import matplotlib.pyplot as plt
import model as M, config as C, estimation as E
from plotstyle import apply_style, PALETTE, BLUE, ORANGE
apply_style()
HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figuras"); os.makedirs(FIG, exist_ok=True)


def _save(fig, name):
    fig.savefig(os.path.join(FIG, name)); plt.close(fig); print("  guardada", name)


def _sample(name, n=504, seed=7):
    p = C.scenario_params(name); rng = np.random.default_rng(seed)
    tau = C.tau_grid(n)
    X = M.simulate_surviving_path(p["X0"], p["mu_Q"], p["sigma_X"], C.DT, n, rng)
    b = M.bond_price(X, tau, p["mu_X"], p["sigma_X"], p["W"], C.R)
    return p, tau, b, X


def profile_W():
    Ws = np.linspace(0.30, 0.92, 22)
    SG = np.linspace(0.05, 0.32, 22)
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    for name, col in [("barrera", ORANGE), ("solvente", BLUE)]:
        p, tau, b, X = _sample(name)
        conc = E._Concentrated(b, tau, C.DT, C.R, p["mu_X"], X)
        prof = []
        for W in Ws:
            best = np.inf
            for sg in SG:
                v, _ = conc(sg, W)
                if v < best: best = v
            prof.append(best)
        prof = np.array(prof)
        prof[prof >= 1e11] = np.nan            # W infactible (q implicito > 1)
        prof -= np.nanmin(prof)
        ax.plot(Ws, prof, color=col, lw=1.9, label="%s" % p["label"])
        ax.axvline(p["W"], color=col, ls=":", lw=1.1)
    ax.set_ylim(-0.5, 12)
    ax.set_xlabel(r"$W$")
    ax.set_ylabel(r"$-\log\mathcal{L}$ perfil (normalizado)")
    from matplotlib.lines import Line2D
    h, lab = ax.get_legend_handles_labels()
    h.append(Line2D([0], [0], color="0.4", ls=":", lw=1.1))
    lab.append(r"$W$ verdadero")
    ax.legend(h, lab)
    fig.tight_layout(); _save(fig, "fig_perfil_W.png")


def contour_muQ_sigma():
    p, tau, b, X = _sample("barrera")
    muX, W = p["mu_X"], p["W"]
    SG = np.linspace(0.08, 0.30, 60); MU = np.linspace(-0.20, 0.40, 70)
    Z = np.full((len(SG), len(MU)), np.nan)
    for i, sg in enumerate(SG):
        Xhat = M.recover_state(b, tau, muX, sg, W, C.R)
        if Xhat is None: continue
        jac = np.sum(np.log(M.db_dX(Xhat[1:], tau[1:], muX, sg, W, C.R)))
        for j, mu in enumerate(MU):
            ll = np.sum(M.log_transition_absorbed(Xhat[1:], Xhat[:-1], C.DT, mu, sg))
            Z[i, j] = -(ll - jac)
    Z = np.ma.masked_invalid(Z); Z = Z - Z.min()
    fig, ax = plt.subplots(figsize=(8.8, 6.0))
    # Escala: cuanto mas cerca de 0 (optimo) mas blanco; -logL alto -> azul oscuro
    cs = ax.contourf(MU, SG, Z, levels=24, cmap="Blues")
    levs = [2, 5, 10, 20, 40]
    cl = ax.contour(MU, SG, Z, levels=levs, colors="black",
                    linewidths=0.7, alpha=0.8)
    # Valor numerico de cada curva de nivel
    ax.clabel(cl, fmt="%d", inline=True, fontsize=14)
    ax.plot(p["mu_Q"], p["sigma_X"], marker="*", color=ORANGE, ms=18,
            mec="black", ls="none")
    ax.set_xlabel(r"$\mu_Q$"); ax.set_ylabel(r"$\sigma_X$")
    fig.colorbar(cs, ax=ax, label=r"$-\log\mathcal{L}$ (normalizado)")
    fig.tight_layout(); _save(fig, "fig_contorno_musigma.png")


if __name__ == "__main__":
    profile_W()
    contour_muQ_sigma()
    print("OK figuras de identificacion")

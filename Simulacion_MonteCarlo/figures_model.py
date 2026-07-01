"""
Panel izquierdo: trayectorias simuladas del log-ratio de solvencia X con la
barrera absorbente en X=0 para ambos escenarios. Panel derecho: precio del bono
b(X,tau0) frente al estado, con la posicion inicial X0 de cada escenario, que
ilustra la sensibilidad al riesgo (informatividad) de cada bono.
Estilo: Times New Roman, paleta por defecto, sin titulo.
"""
import os, numpy as np
import matplotlib.pyplot as plt
import model as M, config as C
from plotstyle import apply_style, BLUE, ORANGE
apply_style()
HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figuras"); os.makedirs(FIG, exist_ok=True)


def _save(fig, name):
    fig.savefig(os.path.join(FIG, name)); plt.close(fig); print("  guardada", name)


def fig_escenarios():
    n = 504; dt = C.DT; t = np.arange(n + 1) * dt
    fig, ax = plt.subplots(1, 2, figsize=(13.0, 5.2))
    for name, col in [("solvente", BLUE), ("barrera", ORANGE)]:
        p = C.scenario_params(name); rng = np.random.default_rng(20 + len(name))
        for k in range(5):
            X, _ = M.simulate_X_path(p["X0"], p["mu_Q"], p["sigma_X"], dt, n, rng)
            ax[0].plot(t, X, color=col, lw=0.9, alpha=0.7,
                       label=p["label"] if k == 0 else None)
    ax[0].axhline(0.0, color="black", lw=1.3)
    ax[0].text(t[-1]*0.45, 0.012, "barrera de quiebra $X=0$", fontsize=14)
    ax[0].set_xlabel("$t$ (años)"); ax[0].set_ylabel(r"$X=\ln(V/K)$")
    ax[0].legend(loc="upper left")
    # precio del bono frente a X (tau = tau0)
    Xg = np.linspace(1e-3, 0.9, 400)
    for name, col in [("solvente", BLUE), ("barrera", ORANGE)]:
        p = C.scenario_params(name)
        b = M.bond_price(Xg, C.TAU0, p["mu_X"], p["sigma_X"], p["W"], C.R)
        ax[1].plot(Xg, b, color=col, lw=1.8, label=p["label"])
        b0 = M.bond_price(np.array([p["X0"]]), C.TAU0, p["mu_X"], p["sigma_X"], p["W"], C.R)[0]
        ax[1].plot([p["X0"]], [b0], marker="o", color=col, ms=7, mec="black", ls="none")
        ax[1].annotate(r"$X_0=%.2f$" % p["X0"], (p["X0"], b0),
                       textcoords="offset points", xytext=(6, -14), fontsize=14)
    ax[1].set_xlabel(r"$X=\ln(V/K)$")
    ax[1].set_ylabel(r"$b(X,\tau_0)$")
    ax[1].legend(loc="lower right")
    fig.tight_layout(); _save(fig, "fig_escenarios.png")


if __name__ == "__main__":
    fig_escenarios()
    print("OK figura de escenarios")

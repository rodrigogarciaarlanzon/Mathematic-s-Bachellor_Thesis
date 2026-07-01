"""
Distribucion muestral de los estimadores EMV
y figura del experimento in-fill. Lee resultados/mc_raw.csv e infill.json.
"""
import os, csv, json, numpy as np
import matplotlib.pyplot as plt
from plotstyle import apply_style, PALETTE, BLUE, ORANGE, GREEN
apply_style()
HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figuras"); os.makedirs(FIG, exist_ok=True)
RAW = os.path.join(HERE, "resultados", "mc_raw.csv")
SCN = {"solvente": "Solvente", "barrera": "Próxima a la barrera"}
NCOL = {"solvente": BLUE, "barrera": ORANGE}


def _save(fig, name):
    fig.savefig(os.path.join(FIG, name)); plt.close(fig); print("  guardada", name)


def load(mode=None):
    rows = list(csv.DictReader(open(RAW)))
    for r in rows:
        r["n"] = int(r["n"])
        for k in ("muQ_hat","sigma_hat","W_hat","muQ_true","sigma_true","W_true"):
            r[k] = float(r[k])
    return [r for r in rows if (mode is None or r["mode"] == mode)]


def _panel(ax, vals, true, col, xlabel):
    vals = vals[np.isfinite(vals)]
    ax.hist(vals, bins=26, color=col, alpha=0.65, density=True,
            edgecolor="white", linewidth=0.4)
    ax.axvline(true, color="black", lw=1.5, label="Verdadero")
    ax.axvline(np.median(vals), color=PALETTE[3], ls="--", lw=1.4, label="Mediana")
    ax.set_xlabel(xlabel); ax.set_ylabel("Densidad")


def hist_param(rows, par, fname, mode_label):
    scens = ["solvente", "barrera"]; ns = [252, 504]
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.0))
    for i, sc in enumerate(scens):
        for j, n in enumerate(ns):
            ax = axes[i][j]
            sub = [r for r in rows if r["scenario"]==sc and r["n"]==n]
            vals = np.array([r[par+"_hat"] for r in sub])
            true = sub[0][par+"_true"]
            sym = {"muQ": r"$\hat{\mu}_Q$", "sigma": r"$\hat{\sigma}_X$", "W": r"$\hat{W}$"}[par]
            _panel(ax, vals, true, NCOL[sc], r"%s  (%s, $n=%d$)" % (sym, SCN[sc], n))
            if i == 0 and j == 0:
                ax.legend(loc="upper right")
    fig.tight_layout(); _save(fig, fname)


def hist_W_joint(rows):
    hist_param(rows, "W", "fig_hist_W.png", "conjunto")


def infill_fig():
    f = os.path.join(HERE, "resultados", "infill.json")
    if not os.path.exists(f): return
    d = json.load(open(f)); ns = [x["n"] for x in d]
    bias = [x["bias"] for x in d]; std = [x["std"] for x in d]; true = d[0]["sigma_true"]
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    mean = [x["mean"] for x in d]
    ax.errorbar(ns, mean, yerr=std, fmt="o-", color=BLUE, capsize=3,
                label=r"$\hat{\sigma}_X$ (media $\pm$ d.t.)")
    ax.axhline(true, color="black", lw=1.4, label=r"$\sigma_X$ verdadero")
    ax.set_xlabel("$n$ (obs., horizonte fijo de 1 año)")
    ax.set_ylabel(r"$\hat{\sigma}_X$"); ax.set_xticks(ns); ax.legend()
    fig.tight_layout(); _save(fig, "fig_infill.png")


if __name__ == "__main__":
    wf = load("wfix"); jo = load("joint")
    hist_param(wf, "sigma", "fig_hist_sigma.png", "W fijo")
    hist_param(wf, "muQ", "fig_hist_muQ.png", "W fijo")
    hist_W_joint(jo)
    infill_fig()
    print("OK figuras de estimadores")

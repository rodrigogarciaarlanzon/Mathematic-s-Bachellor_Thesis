"""
Resumenes y TABLAS LaTeX del experimento Monte Carlo.
Lee resultados/mc_raw.csv y, para cada modo (W conjunto / W fijo) y escenario,
construye una tabla estandar con bloques por horizonte (n=252, n=504):
fila Verdadero (comun) y filas Media, Mediana, Desv. tipica y Cobertura empirica
de los intervalos asintoticos a niveles 25/50/75/95 %.
Escribe los ficheros .tex en tablas y un resumen CSV.
"""
import os, csv, numpy as np
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "resultados", "mc_raw.csv")
TAB = os.path.join(HERE, "tablas"); os.makedirs(TAB, exist_ok=True)
CONF = [0.25, 0.50, 0.75, 0.95]
PARAMS = ["muQ", "sigma", "W"]
PLAB = {"muQ": r"$\mu_Q$", "sigma": r"$\sigma_X$", "W": r"$W$"}
SC_TITLE = {"solvente": "Escenario Solvente",
            "barrera": "Escenario Pr\\'oximo a la Barrera"}


def load():
    rows = list(csv.DictReader(open(RAW)))
    for r in rows:
        r["n"] = int(r["n"])
        for k in ("muQ_true","sigma_true","W_true","muQ_hat","sigma_hat","W_hat",
                  "se_muQ","se_sigma","se_W"):
            r[k] = float(r[k])
    return rows


def cell(rows, mode, scen, n):
    return [r for r in rows if r["mode"]==mode and r["scenario"]==scen and r["n"]==n]


def stats(sub, par):
    hat = np.array([r[par+"_hat"] for r in sub])
    se = np.array([r["se_"+par] for r in sub])
    true = sub[0][par+"_true"]
    fh = np.isfinite(hat)
    h = hat[fh]
    d = dict(true=true, mean=np.mean(h), median=np.median(h),
             std=np.std(h, ddof=1), n=int(fh.sum()))
    cov = {}
    fin = np.isfinite(se) & (se > 0) & fh
    for L in CONF:
        z = norm.ppf((1+L)/2)
        if fin.sum() >= 5:
            inside = np.abs(hat[fin]-true) <= z*se[fin]
            cov[L] = float(np.mean(inside))
        else:
            cov[L] = np.nan
    d["cov"] = cov
    return d


def fmt(x, nd=3):
    if x is None or (isinstance(x,float) and not np.isfinite(x)):
        return "--"
    return ("%.{}f".format(nd) % x).replace(".", ",")


def fmtbias(x, nd=3):
    """Sesgo: positivo sin signo '+', negativo con '-'."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    s = ("%.{}f".format(nd) % abs(x)).replace(".", ",")
    return ("-" if x < 0 else "") + s


def make_table(rows, mode, scen):
    cols = PARAMS if mode == "joint" else ["muQ", "sigma"]
    ncol = len(cols)
    ns = sorted(set(r["n"] for r in rows if r["mode"]==mode and r["scenario"]==scen))
    blocks = {n: {p: stats(cell(rows,mode,scen,n), p) for p in cols} for n in ns}
    nrep = blocks[ns[0]][cols[0]]["n"]
    head = "M\\'axima Verosimilitud --- %s" % SC_TITLE[scen]
    wtxt = "estimaci\\'on conjunta de $(\\mu_Q,\\sigma_X,W)$" if mode=="joint" \
        else "$W$ fijado en su valor verdadero"
    colspec = "l" + ("|" + "c"*ncol)*len(ns)
    L = []
    L.append(r"{\footnotesize")
    L.append(r"\renewcommand{\arraystretch}{1.18}")
    L.append(r"\begin{tabular}{%s}" % colspec)
    L.append(r"\hline")
    # cabecera de bloques por horizonte
    blockhead = " & ".join(r"\multicolumn{%d}{c}{$n=%d$}" % (ncol, n) for n in ns)
    L.append(r"\multicolumn{1}{c}{} & " + blockhead + r" \\")
    parhead = " & ".join(" & ".join(PLAB[p] for p in cols) for _ in ns)
    L.append(r"\textbf{Estad\'istico} & " + parhead + r" \\")
    L.append(r"\hline")
    # fila Verdadero (comun)
    truerow = " & ".join(" & ".join(fmt(blocks[ns[0]][p]["true"]) for p in cols) for _ in ns)
    L.append(r"\textit{Verdadero} & " + truerow + r" \\")
    L.append(r"\hline")
    def rowline(name, key, nd=3):
        vals = " & ".join(" & ".join(fmt(blocks[n][p][key], nd) for p in cols) for n in ns)
        L.append(name + " & " + vals + r" \\")
    rowline("Media", "mean")
    # fila Sesgo = Media - Verdadero (positivo sin signo)
    biasvals = " & ".join(" & ".join(
        fmtbias(blocks[n][p]["mean"] - blocks[n][p]["true"]) for p in cols) for n in ns)
    L.append("Sesgo & " + biasvals + r" \\")
    rowline("Mediana", "median")
    rowline("Desv.\\ t\\'ipica", "std")
    L.append(r"\hline")
    for Lvl in CONF:
        vals = " & ".join(" & ".join(fmt(blocks[n][p]["cov"][Lvl],3) for p in cols) for n in ns)
        L.append(r"Cobertura %d\,\%% & " % int(Lvl*100) + vals + r" \\")
    L.append(r"\hline")
    L.append(r"\end{tabular}}")
    return "\n".join(L)


def main():
    rows = load()
    for mode in ["wfix", "joint"]:
        for scen in ["solvente", "barrera"]:
            tex = make_table(rows, mode, scen)
            fn = os.path.join(TAB, "tabla_%s_%s.tex" % (mode, scen))
            open(fn, "w").write(tex)
            print("== %s %s ==" % (mode, scen))
            # resumen legible
            cols = PARAMS if mode=="joint" else ["muQ","sigma"]
            for n in sorted(set(r["n"] for r in rows if r["mode"]==mode and r["scenario"]==scen)):
                for p in cols:
                    s = stats(cell(rows,mode,scen,n), p)
                    print("  n=%d %-6s true=%.3f mean=%.3f med=%.3f std=%.3f cov95=%.2f (N=%d)"%(
                        n,p,s["true"],s["mean"],s["median"],s["std"],s["cov"][0.95],s["n"]))
    print("Tablas .tex escritas en", TAB)


if __name__ == "__main__":
    main()

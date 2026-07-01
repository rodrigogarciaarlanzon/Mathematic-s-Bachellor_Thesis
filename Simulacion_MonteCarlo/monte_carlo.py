"""
Experimento completo.
Ejecuta de una sola vez:
  1) la simulacion Monte Carlo de las 8 celdas,
  2) el experimento in-fill,
  3) la generacion de tablas LaTeX,
  4) la generacion de todas las figuras.
"""
import time
import run_chunk as RC
import config as C


def main():
    t0 = time.time()
    print(">> Simulacion Monte Carlo (W fijo: %d, conjunto: %d por celda)" % (C.N_REP, C.N_REP_JOINT), flush=True)
    RC.main(budget=1e12)
    print(">> Experimento in-fill", flush=True)
    import infill_run; infill_run.main()
    print(">> Tablas LaTeX", flush=True)
    import tables; tables.main()
    print(">> Figuras", flush=True)
    __import__("runpy").run_path("figures_model.py", run_name="__main__")
    __import__("runpy").run_path("figures_estimators.py", run_name="__main__")
    __import__("runpy").run_path("profiles.py", run_name="__main__")
    print(">> LISTO en %.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()

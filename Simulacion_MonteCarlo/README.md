# Simulación Monte Carlo — TFG Finanzas Cuantitativas

Código de la simulación Monte Carlo del modelo propio (versión simplificada de
Hsu, Saá-Requejo y Santa-Clara) y de su estimador de máxima verosimilitud
**resuelto por el algoritmo de Newton**. Los resultados se exponen en la
**Sección 8** del TFG.

## Modelo y notación
- Estado latente `X = ln(V/K)`, MBA absorbido en `X=0`.
- **Deriva de valoración** `mu_X = ½(β²−γ²)`: entra en `q(X,τ)`, en el precio
  `b(X,τ)`, en su inversión y en el jacobiano. Se mantiene **fija**.
- **Deriva real** `mu_Q = (mu_V−mu_K) + mu_X`: gobierna la simulación y la
  densidad de transición de los incrementos.
- `sigma_X = β−γ`. Vector estimado: **`θ = (mu_Q, sigma_X, W)`**.

## Estructura
- `model.py`      — núcleo: `q`, `b`, `∂b/∂X`, inversión `b⁻¹` (Newton),
                    densidad absorbida y log-verosimilitud.
- `estimation.py` — **EMV por Newton-Raphson** (`fit_newton`). Modo conjunto
                    (`W_fixed=None`) y modo `W` fijo. Arranque en un entorno del
                    verdadero valor; `mu_Q` se concentra con un Newton 1D interno;
                    errores estándar por información observada (Hessiano).
- `config.py`     — escenarios (solvente / próxima a la barrera), horizontes
                    `n∈{252,554}`, `N_REP`, niveles de confianza.
- `run_chunk.py`  — runner Monte Carlo **reanudable** (multiproceso); escribe
                    `resultados/mc_raw.csv`. 8 celdas: 2 escenarios × 2 horizontes
                    × 2 modos (W conjunto / W fijo).
- `monte_carlo.py`— orquestador: ejecuta todo de una vez (simulación + in-fill +
                    tablas + figuras).
- `infill_run.py` — experimento de relleno (in-fill, `Δt→0`).
- `tables.py`     — genera las TABLAS LaTeX (`tablas/tabla_*.tex`) con media,
                    mediana, desv. típica y coberturas 25/50/75/95 %.
- `figures_model.py`      — figura de los dos escenarios (`fig_escenarios.png`).
- `figures_estimators.py` — histogramas de `μ_Q`, `σ_X`, `W` e in-fill.
- `profiles.py`           — verosimilitud perfil en `W` y contorno `(μ_Q,σ_X)`.
- `plotstyle.py`          — estilo de figuras: **Times New Roman**, paleta por
                            defecto de Matplotlib/PyCharm, **sin título** (la
                            descripción va en el pie de la figura).
- `figuras/`     — figuras `.png`.
- `tablas/`      — tablas `.tex` .
- `resultados/`  — `mc_raw.csv`, `infill.json`.

## Reproducir la corrida DEFINITIVA (10 000 réplicas)
En `config.py` ya está `N_REP = 10000`. Entonces:
```bash
pip install numpy scipy matplotlib
python monte_carlo.py        # simulación + in-fill + tablas + figuras
```
(En entornos con límite de tiempo puede usarse `run_chunk.py` por tramos:
`python run_chunk.py 10000 60`, repetido hasta "COMPLETO".)

## Nota sobre las figuras
Las figuras se generan en **Times New Roman** con la paleta por defecto de
Matplotlib/PyCharm y **sin título superior**; la descripción de cada figura va en
su `\caption` dentro del `.tex`.

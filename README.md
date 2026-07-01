# Mathematics Bachelor Thesis — Modelización Estocástica en Finanzas

**Modelos Estructurales de Valoración de Deuda y Estimación de Parámetros**

Trabajo de Fin de Grado en Finanzas Cuantitativas. El trabajo desarrolla los
modelos estructurales de riesgo de crédito (Merton, Black–Scholes, Black–Cox y
una versión propia basada en Hsu, Saá-Requejo y Santa-Clara) y su estimación por
máxima verosimilitud, acompañados de un estudio de simulación Monte Carlo.

## Contenido del repositorio

- **`Modelización Estocástica en Finanzas. Modelos Estructurales de Valoración de Deuda y Estimación de Parámetros.pdf`**
  — memoria final del TFG (versión definitiva).
- **`Simulacion_MonteCarlo/`** — código Python de la simulación Monte Carlo del
  modelo propio y de su estimador de máxima verosimilitud (Newton-Raphson), junto
  con las figuras y tablas que aparecen en la memoria.

## La carpeta de simulación

Consulta `Simulacion_MonteCarlo/README.md` para el detalle de cada módulo. En
resumen:

- `model.py` — núcleo del modelo: probabilidad de impago `q`, precio de la deuda
  `b`, su derivada e inversión (Newton), densidad absorbida y log-verosimilitud.
- `estimation.py` — estimación máximo-verosímil por Newton-Raphson.
- `config.py` — escenarios, horizontes y parámetros de la simulación.
- `run_chunk.py` / `monte_carlo.py` — runners de la simulación Monte Carlo.
- `infill_run.py` — experimento de relleno (in-fill, Δt→0).
- `tables.py`, `figures_model.py`, `figures_estimators.py`, `profiles.py` —
  generación de tablas y figuras.
- `figuras/`, `tablas/`, `resultados/` — salidas de la simulación.

## Reproducir la simulación

```bash
cd Simulacion_MonteCarlo
python monte_carlo.py
```

## Autor

Rodrigo García Arlanzón

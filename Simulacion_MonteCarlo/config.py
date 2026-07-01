"""
Parametros del experimento Monte Carlo.
Cada escenario se calibra con (X0, gamma, beta, W, mu_V-mu_K); de ahí:
    sigma_X = beta-gamma ; mu_X = 1/2(beta^2-gamma^2) ; mu_Q = (mu_V-mu_K)+mu_X
Se estima theta=(mu_Q,sigma_X,W) por Newton; mu_X queda FIJO.
Horizontes diarios (dt=1/252): n=252 (1 ano) y n=504 (2 anos).
"""
import model as M

R = 0.03
DT = 1.0 / 252.0
TAU0 = 5.0
MU_K = 0.03

SCENARIOS = {
    "solvente": dict(label="Escenario solvente",
                     X0=0.35, gamma=0.15, beta=0.25, W=0.50, mu_VmK=0.03),
    "barrera":  dict(label="Escenario próximo a barrera de quiebra",
                     X0=0.15, gamma=0.20, beta=0.35, W=0.60, mu_VmK=0.02875),
}

SAMPLE_SIZES = [252, 504]
N_REP = 10000         # replicas en modo W fijo
N_REP_JOINT = 10000   # replicas en modo conjunto
START_JITTER = 0.10
SEED = 20260612
CONF_LEVELS = [0.25, 0.50, 0.75, 0.95]


def scenario_params(name):
    s = SCENARIOS[name]
    gamma, beta = s["gamma"], s["beta"]
    sigma_X = beta - gamma
    mu_X = M.valuation_drift(gamma, beta)
    mu_Q = s["mu_VmK"] + mu_X
    return dict(label=s["label"], X0=s["X0"], gamma=gamma, beta=beta, W=s["W"],
                mu_VmK=s["mu_VmK"], mu_K=MU_K, mu_V=MU_K + s["mu_VmK"],
                sigma_X=sigma_X, mu_X=mu_X, mu_Q=mu_Q)


def tau_grid(n):
    import numpy as np
    return np.array([TAU0 - i * DT for i in range(n + 1)], dtype=float)

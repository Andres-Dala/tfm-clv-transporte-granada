"""
Centraliza todos los parámetros que pueden cambiar entre runs y que son
compartidos por varios notebooks. Importar desde aquí en lugar de duplicar
valores en cada notebook.

"""
from __future__ import annotations

from pathlib import Path
from typing import Any


# ══════════════════════════════════════════════════════════════════
# Rutas del proyecto
# ══════════════════════════════════════════════════════════════════
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "synthetic"
NOTEBOOKS_DIR = ROOT / "notebooks"
DOCS_DIR = ROOT / "docs"

DATA_DIR.mkdir(parents=True, exist_ok=True)

RFM_PATH = DATA_DIR / "rfm_full.csv"


# ══════════════════════════════════════════════════════════════════
# Semilla maestra (reproducibilidad)
# ══════════════════════════════════════════════════════════════════
SEED = 42


# ══════════════════════════════════════════════════════════════════
# Estadísticas públicas del Metro de Granada
# ══════════════════════════════════════════════════════════════════
STATS: dict[str, Any] = {
    # Viajeros anuales totales del metro (validaciones)
    "viajeros_anuales": {
        2018: 10_207_006,
        2019: 11_721_039,
        2020:  5_881_004,  # caída COVID-19
        2021:  7_986_675,
        2022: 11_067_712,
        2023: 14_180_797,
        2024: 16_243_044,
        2025: 17_329_112,
    },

    # Media diaria de viajeros en días laborables
    "media_diaria_laborable": {
        2018: 32_050,
        2023: 38_851,
        2024: 44_380,
        2025: 53_874,
    },

    # Índice mensual de demanda (octubre = pico, agosto = mínimo)
    "indice_mensual": {
        1: 95,  2: 100, 3: 105, 4: 110,
        5: 112, 6: 108, 7:  85, 8:  65,
        9: 108, 10: 120, 11: 118, 12: 102,
    },

    # Distribución por tipo de título (porcentajes, 2023-2025)
    "pct_titulos": {
        "tarjeta_consorcio": 0.58,
        "monedero_metro":    0.23,
        "bono_30_dias":      0.13,
        "ocasional":         0.06,
    },

    # Usuarios únicos activos estimados — usado en la calibración
    "n_usuarios_activos_2025": 66_650,

    # Retención anual implícita
    "tasa_retencion_anual": 0.85,

    # Factor multiplicativo de reducción de demanda durante COVID-19
    "factor_covid": 0.38,
}


# ══════════════════════════════════════════════════════════════════
# Configuración de la simulación BG/NBD
# ══════════════════════════════════════════════════════════════════
SIM_CONFIG: dict[str, Any] = {
    "n_total":    60000,       
    "semana_fin": 433,          # Semana de cierre del estudio (fin de 2025)

    # Semana de inicio de cada cohorte
    # (week 0 = sept 2017, apertura del metro)
    "semana_inicio_por_año": {
        2017:   0,
        2018:  17,
        2019:  69,
        2020: 121,
        2021: 173,
        2022: 225,
        2023: 277,
        2024: 329,
        2025: 381,
    },

    # Ventana COVID-19 (en semanas locales del usuario)
    "covid_window": (130, 230),

    # Modificadores para usuarios con perfil "Tarjeta Joven"
    "joven_lambda_mult": 1.3,
    "joven_p_mult":      1.8,
    "joven_year_min":    2020,
    "joven_fraction":    0.25,

    # Hiperparámetros de la calibración por método de momentos
    "cv_lambda":     0.7,
    "concentracion": 20.0,
    "T_efectivo":    12,
    "año_calib":     2025,
}


# ══════════════════════════════════════════════════════════════════
# Calibración por método de momentos
# ══════════════════════════════════════════════════════════════════
def calibrar_parametros_momentos(
    stats: dict = STATS,
    cv_lambda: float = SIM_CONFIG["cv_lambda"],
    concentracion: float = SIM_CONFIG["concentracion"],
    año_calib: int = SIM_CONFIG["año_calib"],
    T_efectivo: int = SIM_CONFIG["T_efectivo"],
) -> dict[str, float]:
    """
    Estima los parámetros poblacionales (r, α, a, b) del BG/NBD por
    método de momentos a partir de las estadísticas públicas del metro.

    """
    media_diaria = stats["media_diaria_laborable"][año_calib]
    n_usuarios   = stats[f"n_usuarios_activos_{año_calib}"]
    lambda_media = (media_diaria / n_usuarios) * 5  # semana laboral

    alpha = 1.0 / (cv_lambda ** 2)
    r     = lambda_media * alpha

    p_media = 1 - stats["tasa_retencion_anual"] ** (1 / T_efectivo)
    a = p_media * concentracion
    b = (1 - p_media) * concentracion

    return {
        "r":        r,
        "alpha":    alpha,
        "a":        a,
        "b":        b,
        "E_lambda": lambda_media,
        "E_p":      p_media,
    }


PARAMS_CALIBRATED = calibrar_parametros_momentos()


# ══════════════════════════════════════════════════════════════════
# Distribuciones a priori
#
# Tres juegos en formato pymc-marketing (dist + kwargs):
#   - "elicited":           obtenidos por maxima entropia
#   - "proposal":           débilmente informativos, los del enunciado de la propuesta
#   - "informative":        centrados en el método de momentos (tight)
#   - "weakly_informative": recomendados tras el prior predictive check
# ══════════════════════════════════════════════════════════════════
PRIORS: dict[str, dict[str, dict]] = {
    "elicited": {
    "r"     : {'dist': 'Gamma', 'kwargs': {'alpha': 2.77, 'beta': 0.23}},
    "alpha" : {'dist': 'Gamma', 'kwargs': {'alpha': 2.72, 'beta': 0.71}},
    "a"     : {'dist': 'Gamma', 'kwargs': {'alpha': 2.47, 'beta': 2.65}},
    "b"     : {'dist': 'Gamma', 'kwargs': {'alpha': 2.98, 'beta': 0.10}},
  },
    "proposal": {
        "r":     {"dist": "Gamma",      "kwargs": {"alpha": 2.0, "beta": 0.5}},
        "alpha": {"dist": "Gamma",      "kwargs": {"alpha": 2.0, "beta": 0.5}},
        "a":     {"dist": "HalfNormal", "kwargs": {"sigma": 1.0}},
        "b":     {"dist": "HalfNormal", "kwargs": {"sigma": 1.0}},
    },

    "informative": {
        "r":     {"dist": "Gamma",      "kwargs": {"alpha": PARAMS_CALIBRATED["r"],     "beta": 1.0}},
        "alpha": {"dist": "Gamma",      "kwargs": {"alpha": PARAMS_CALIBRATED["alpha"], "beta": 1.0}},
        "a":     {"dist": "HalfNormal", "kwargs": {"sigma": PARAMS_CALIBRATED["a"] * 2}},
        "b":     {"dist": "HalfNormal", "kwargs": {"sigma": PARAMS_CALIBRATED["b"] * 2}},
    },

    "weakly_informative": {
        "r":     {"dist": "Gamma",      "kwargs": {"alpha": 4.0, "beta": 0.5}},
        "alpha": {"dist": "Gamma",      "kwargs": {"alpha": 2.0, "beta": 1.0}},
        "a":     {"dist": "HalfNormal", "kwargs": {"sigma": 0.5}},
        "b":     {"dist": "HalfNormal", "kwargs": {"sigma": 25.0}},
    },
}

# Prior por defecto para el ajuste MCMC del TFM
PRIORS_DEFAULT_KEY = "elicited"
PRIORS_DEFAULT     = PRIORS[PRIORS_DEFAULT_KEY]


# ══════════════════════════════════════════════════════════════════
# Helpers de conversión entre formatos
# ══════════════════════════════════════════════════════════════════
def prior_to_scipy(spec: dict):
    """
    Convierte una especificación de prior estilo pymc a una distribución
    congelada de scipy.stats — útil para visualizaciones y muestreo
    fuera del modelo bayesiano.

    Equivalencias:
        pymc Gamma(alpha, beta)   ↔  scipy gamma(a=alpha, scale=1/beta)
        pymc HalfNormal(sigma)    ↔  scipy halfnorm(scale=sigma)
        pymc Normal(mu, sigma)    ↔  scipy norm(loc=mu, scale=sigma)
    """
    from scipy import stats
    dist = spec["dist"]
    kw   = spec["kwargs"]
    if dist == "Gamma":
        return stats.gamma(a=kw["alpha"], scale=1.0 / kw["beta"])
    if dist == "HalfNormal":
        return stats.halfnorm(scale=kw["sigma"])
    if dist == "Normal":
        return stats.norm(loc=kw.get("mu", 0), scale=kw["sigma"])
    raise ValueError(f"Distribución no soportada: {dist}")


def priors_to_scipy(priors: dict) -> dict:
    """Convierte un juego completo de priors a sus equivalentes scipy."""
    return {k: prior_to_scipy(v) for k, v in priors.items()}


def priors_to_model_config(priors: dict) -> dict:
    """
    Empaqueta un juego de priors en el formato model_config que espera
    pymc_marketing.clv.BetaGeoModel:
        {"r_prior": {...}, "alpha_prior": {...}, ...}
    """
    return {f"{k}_prior": v for k, v in priors.items()}


# ══════════════════════════════════════════════════════════════════
# Rangos de plausibilidad para Granada
#   Usados en el prior predictive check y en validaciones post-fit
# ══════════════════════════════════════════════════════════════════
PLAUSIBILITY_RANGES: dict[str, tuple] = {
    "E_lambda":       (0.5, 20.0),     # viajes/semana del usuario medio
    "E_p":            (0.001, 0.1),    # prob. abandono semanal
    "monetary_value": (0.5, 20.0),     # viajes/semana activa
    "supervivencia_52sem": (0.5, 0.95),  # fracción viva tras 1 año
}


# ══════════════════════════════════════════════════════════════════
# Configuración del muestreo MCMC
# ══════════════════════════════════════════════════════════════════
MCMC_CONFIG: dict[str, Any] = {
    "draws":         2000,
    "tune":          1500,
    "chains":        4,
    "target_accept": 0.90,
    "nuts_sampler":  "nutpie",
    "progressbar":   True,
    "random_seed":   SEED,
}

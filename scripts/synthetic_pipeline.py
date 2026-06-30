# %% [markdown]
# # Pipeline de generación de datos sintéticos calibrados con estadísticas públicas del Metro de Granada.
#
# **Fuentes de calibración:**
#   - Metropolitano de Granada: Balances anuales 2017–202
#
# **Estructura del pipeline:**
#   1. Fijar estadísticas conocidas (extraídas de informes públicos)
#   2. Calibrar parámetros BG/NBD por Método de Momentos
#   3. Simular historiales individuales de usuarios
#   4. Añadir cohortes, estacionalidad y eventos (COVID, bonificaciones)
#   5. Validar que los datos sintéticos reproducen los agregados reales
#   6. Exportar dataset RFM listo para pymc-marketing

# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")

from config import (
    SEED, STATS, SIM_CONFIG, PARAMS_CALIBRATED,RFM_PATH,
    calibrar_parametros_momentos)

rng = np.random.default_rng(seed=SEED)

# %% [markdown]
# ## Estadísticas Conocidas (extraídas de informes públicos)
# Estas cifras provienen directamente de los balances anuales publicados por el Metropolitano de Granada y el CTAGR.

# %%
# Las estadísticas públicas se cargan desde config.py
print("Estadísticas públicas cargadas desde config.STATS:")
print(f"  Años con datos de viajeros:    {sorted(STATS['viajeros_anuales'].keys())}")
print(f"  Viajeros 2025:                 {STATS['viajeros_anuales'][2025]:,}")
print(f"  Media diaria laborable 2025:   {STATS['media_diaria_laborable'][2025]:,}")
print(f"  Tipos de título:               {list(STATS['pct_titulos'].keys())}")
print(f"  Usuarios activos 2025 (est.):  {STATS['n_usuarios_activos_2025']:,}")
print(f"  Retención anual:               {STATS['tasa_retencion_anual']}")
print(f"  Factor COVID:                  {STATS['factor_covid']}")

# %% [markdown]
# ## Método de Momentos - Calibración de parámetros BG/NBD
#
# El modelo BG/NBD tiene 4 parámetros: $(r,\ \alpha,\ a,\ b)$.
#
# **Relaciones teóricas entre parámetros y momentos observables:**
#
# $$
# \begin{aligned}
# \mathbb{E}[\lambda]   &= \frac{r}{\alpha}                                  &&\text{tasa media de uso} \\[4pt]
# \mathrm{Var}[\lambda] &= \frac{r}{\alpha^{2}}                              &&\text{varianza de tasas} \\[4pt]
# \mathbb{E}[p]         &= \frac{a}{a+b}                                     &&\text{prob.\ media de abandono} \\[4pt]
# \mathrm{Var}[p]       &= \frac{a\,b}{(a+b)^{2}\,(a+b+1)}                   &&\text{varianza del abandono}
# \end{aligned}
# $$
#
# **De las estadísticas conocidas podemos estimar:**
#
# - $\mathbb{E}[\lambda]$: media de viajes por semana activa.
# - $\mathbb{E}[p]$: probabilidad de abandono implícita en la tasa de retención anual.
#

# %%
# La calibración por método de momentos está implementada en
# config.calibrar_parametros_momentos(). El resultado por defecto se
# exporta como config.PARAMS_CALIBRATED.

params = calibrar_parametros_momentos(STATS)

print("── Parámetros BG/NBD calibrados por método de momentos ──────")
print(f"  r     = {params['r']:.4f}  (forma de la Gamma para λ)")
print(f"  α     = {params['alpha']:.4f}  (escala de la Gamma para λ)")
print(f"  E[λ]  = r/α = {params['E_lambda']:.3f} viajes/semana/usuario activo")
print(f"  a     = {params['a']:.4f}  (param. Beta para p, abandono)")
print(f"  b     = {params['b']:.4f}  (param. Beta para p, permanencia)")
print(f"  E[p]  = a/(a+b) = {params['E_p']:.4f} prob. abandono por período")
print("─────────────────────────────────────────────────────────────")
print(f"\nHiperparámetros usados (config.SIM_CONFIG):")
print(f"  CV[λ]         = {SIM_CONFIG['cv_lambda']}")
print(f"  concentración = a + b = {SIM_CONFIG['concentracion']}")
print(f"  T_efectivo    = {SIM_CONFIG['T_efectivo']} períodos/año")
print(f"  año_calib     = {SIM_CONFIG['año_calib']}")


# %% [markdown]
# ## Generador de Cohortes de usuarios
# - Estrategia: simular cohortes anuales de entrada al sistema desde 2017.
# - Cada cohorte tiene un tamaño proporcional al crecimiento real.
# - Usuarios dentro de cada cohorte reciben λ_i y p_i individuales.

# %%
def calcular_tamaños_cohortes(stats: dict | None = None,
                              n_total: int | None = None) -> dict:
    """
    Estima el número de usuarios nuevos por año, calibrado con el
    crecimiento real de viajeros publicado por el CTAGR.

    Por defecto lee de config.STATS y config.SIM_CONFIG["n_total"].
    """
    stats   = stats   if stats   is not None else STATS
    n_total = n_total if n_total is not None else SIM_CONFIG["n_total"]

    viajeros = stats["viajeros_anuales"]
    años     = sorted(viajeros.keys())

    incrementos = {}
    for i, año in enumerate(años):
        if i == 0:
            incrementos[año] = viajeros[año]
        else:
            inc = viajeros[año] - viajeros[años[i-1]]
            incrementos[año] = max(inc, viajeros[año] * 0.05)

    total_inc = sum(incrementos.values())
    return {año: int(n_total * inc / total_inc)
            for año, inc in incrementos.items()}


def simular_usuario(lambda_i: float, p_i: float, T_semanas: int,
                    factor_covid: float | None = None,
                    sim_config: dict | None = None) -> tuple:
    """
    Simula el historial RFM de un único usuario bajo el modelo BG/NBD.

    Parámetros
    ----------
    lambda_i    : tasa de uso individual (viajes/semana)
    p_i         : probabilidad de abandono tras cada semana de uso
    T_semanas   : período total de observación en semanas
    factor_covid: multiplicador opcional de λ durante la ventana COVID
    sim_config  : config de simulación (default: config.SIM_CONFIG)
    """
    sim_config = sim_config or SIM_CONFIG
    covid_lo, covid_hi = sim_config["covid_window"]

    vivo, primera, ultima, activas, viajes = True, None, None, 0, 0
    p_eff = min(max(p_i, 0.0), 1.0)  # clip por si p_i fue multiplicado

    for t in range(T_semanas):
        if not vivo:
            break
        lam = (lambda_i * factor_covid
               if (factor_covid is not None and covid_lo <= t <= covid_hi)
               else lambda_i)
        n_viajes = rng.poisson(lam)
        if n_viajes > 0:
            if primera is None:
                primera = t
            ultima = t
            activas += 1
            viajes += n_viajes
            if rng.random() < p_eff:
                vivo = False

    if primera is None:
        return 0, 0.0, T_semanas, 0.0
    return (activas - 1,
            float(ultima - primera),
            float(T_semanas - primera),
            viajes / max(activas, 1))


def simular_dataset(params: dict,
                    stats: dict | None = None,
                    sim_config: dict | None = None) -> pd.DataFrame:
    """
    Genera el dataset RFM completo. Cohortes, tipos de título, perfil
    Tarjeta Joven y efecto COVID se controlan desde config.SIM_CONFIG.
    """
    stats      = stats      or STATS
    sim_config = sim_config or SIM_CONFIG

    r, alpha_p, a, b = params["r"], params["alpha"], params["a"], params["b"]
    cohortes = calcular_tamaños_cohortes(stats, sim_config["n_total"])

    semana_inicio  = sim_config["semana_inicio_por_año"]
    semana_fin     = sim_config["semana_fin"]
    joven_year_min = sim_config["joven_year_min"]
    joven_frac     = sim_config["joven_fraction"]
    joven_lam      = sim_config["joven_lambda_mult"]
    joven_p        = sim_config["joven_p_mult"]

    titulos_choices = list(stats["pct_titulos"].keys())
    titulos_probs   = list(stats["pct_titulos"].values())

    registros = []
    for año, n_coh in cohortes.items():
        ini   = semana_inicio.get(año, 0)
        T_obs = semana_fin - ini
        if T_obs <= 4:
            continue

        for i in range(n_coh):
            lambda_i = rng.gamma(shape=r, scale=1/alpha_p)
            p_i      = rng.beta(a=a, b=b)

            es_joven = (año >= joven_year_min) and (rng.random() < joven_frac)
            if es_joven:
                lambda_i *= joven_lam
                p_i      *= joven_p

            x, t_x, T, z = simular_usuario(
                lambda_i, p_i, T_obs,
                factor_covid=stats["factor_covid"] if año <= 2019 else None,
                sim_config=sim_config,
            )

            tipo = rng.choice(titulos_choices, p=titulos_probs)
            registros.append({
                "customer_id":    f"USR_{año}_{i:05d}",
                "cohort_year":    año,
                "frequency":      x,
                "recency":        round(t_x, 1),
                "T":              round(T, 1),
                "monetary_value": round(z, 2),
                "tipo_titulo":    tipo,
                "es_joven":       es_joven,
                "lambda_true":    round(lambda_i, 4),
                "p_true":         round(p_i, 6),
            })

    df = pd.DataFrame(registros)
    df = df[df["T"] > 0].copy()
    print(f"Dataset sintético generado: {len(df):,} usuarios, "
          f"{df['cohort_year'].nunique()} cohortes "
          f"(n_total objetivo = {sim_config['n_total']:,})\n")
    return df


df = simular_dataset(params)


# %% [markdown]
# ## Validación de los datos contra los valores reales

# %%
def validar_dataset(df: pd.DataFrame, stats: dict | None = None,
                    params: dict | None = None) -> None:
    """
    Compara estadísticas del dataset sintético con los valores reales.
    Las referencias provienen de config.STATS y config.PARAMS_CALIBRATED.
    """
    stats  = stats  or STATS
    params = params or PARAMS_CALIBRATED

    print("══════════════════════════════════════════════════════════════")
    print("VALIDACIÓN: datos sintéticos vs. informes públicos")
    print("══════════════════════════════════════════════════════════════")

    # 1. Distribución de tipos de título
    print("\n1. Distribución de títulos de transporte:")
    dist_sim  = df["tipo_titulo"].value_counts(normalize=True).round(3)
    dist_real = stats["pct_titulos"]
    for titulo, pct_real in dist_real.items():
        pct_sim = dist_sim.get(titulo, 0)
        ok = "✓" if abs(pct_sim - pct_real) < 0.03 else "✗"
        print(f"   {ok} {titulo:25s}  real={pct_real:.2f}  sim={pct_sim:.2f}")

    # 2. Proporción de usuarios recurrentes
    print("\n2. Proporción de usuarios recurrentes (x > 0):")
    pct_rec_sim  = (df["frequency"] > 0).mean()
    pct_rec_real = 1 - stats["pct_titulos"]["ocasional"]
    ok = "✓" if abs(pct_rec_sim - pct_rec_real) < 0.05 else "✗"
    print(f"   {ok} Real: {pct_rec_real:.2f}  Simulado: {pct_rec_sim:.2f}")

    # 3. Tasa de viajes por semana activa (vs E[λ] calibrada)
    print("\n3. Tasa media de viajes por semana activa:")
    rec = df[df["frequency"] > 0]
    lambda_sim  = rec["monetary_value"].mean()
    lambda_real = params["E_lambda"]
    ok = "✓" if abs(lambda_sim - lambda_real) / lambda_real < 0.15 else "✗"
    print(f"   {ok} E[λ] calibrado: {lambda_real:.2f}  Simulado: {lambda_sim:.2f}")

    # 4. Distribución de cohortes
    print("\n4. Distribución de cohortes:")
    coh_sim = df["cohort_year"].value_counts(normalize=True).sort_index()
    for año, pct in coh_sim.items():
        print(f"   Cohorte {año}: {pct:.3f}  ({int(pct*len(df)):,} usuarios)")

    # 5. Recencia / dormancia
    print("\n5. Estadísticas de recencia (semanas desde última validación):")
    dormidos = (df["T"] - df["recency"] > 26).mean()
    print(f"   Usuarios con >26 semanas inactivos: {dormidos:.1%}")
    print(f"   (usuarios potencialmente 'muertos' según el modelo)")

    print("\n══════════════════════════════════════════════════════════════\n")


validar_dataset(df, STATS, params)


# %% [markdown]
# ## Exportar datos sintéticos

# %%
def exportar_dataset_rfm(df: pd.DataFrame,
                         incluir_validacion: bool = False) -> pd.DataFrame:
    """
    Prepara el DataFrame final en el formato que espera
    pymc_marketing.clv.BetaGeoModel.
    """
    cols_modelo = ["customer_id", "frequency", "recency", "T",
                   "monetary_value", "tipo_titulo", "cohort_year", "es_joven"]

    if incluir_validacion:
        cols_modelo += ["lambda_true", "p_true"]

    df_out = df[cols_modelo].copy()

    assert (df_out["recency"] <= df_out["T"]).all(), \
        "Error: recency > T en algunos usuarios"
    assert (df_out["frequency"] >= 0).all(), \
        "Error: frecuencias negativas"

    print("═" * 60)
    print("DATASET RFM FINAL")
    print("═" * 60)
    print(f"  Usuarios totales:          {len(df_out):,}")
    print(f"  Usuarios con x>0:          {(df_out['frequency'] > 0).sum():,}")
    print(f"  Media frecuencia (x>0):    {df_out.loc[df_out['frequency'] > 0, 'frequency'].mean():.2f} semanas")
    print(f"  Media antigüedad (T):      {df_out['T'].mean():.1f} semanas")
    print(f"  Media valor monetario:     {df_out['monetary_value'].mean():.2f} viajes/semana")
    print(f"  Cohortes incluidas:        {sorted(df_out['cohort_year'].unique())}")
    print()
    print(df_out.head(5).to_string(index=False))
    print("═" * 60)

    return df_out


rfm = exportar_dataset_rfm(df, incluir_validacion=True)
rfm.to_csv(RFM_PATH, index=False, encoding="utf-8")

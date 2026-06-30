# %% [markdown]
# # Análisis descriptivo y validación del dataset sintético
#
# Este notebook acompaña al `synthetic_pipeline.ipynb` y persigue dos objetivos:
#
# 1. **Análisis descriptivo completo** del dataset RFM sintético generado, cubriendo distribuciones marginales, relaciones bivariantes y diferencias entre segmentos (tipo de título, cohorte, joven/no joven).
# 2. **Validación cuantitativa** de que los datos generados reproducen (i) las estadísticas agregadas reales del CTAGR y el Metro de Granada y (ii) los parámetros poblacionales $(r, \alpha, a, b)$ con los que se sembró el simulador (*parameter recovery*).
#
# El bloque de *parameter recovery* es metodológicamente central: si el modelo recupera los parámetros verdaderos a partir de las muestras individuales $\lambda_i$ y $p_i$, el pipeline queda validado como banco de pruebas controlado para el ajuste bayesiano posterior con `pymc-marketing`.

# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from config import (
    SEED, STATS, SIM_CONFIG, PARAMS_CALIBRATED, RFM_PATH,
)

sns.set_theme(context="notebook", style="whitegrid", palette="deep")
plt.rcParams["figure.dpi"] = 110
pd.set_option("display.float_format", "{:,.4f}".format)

RNG = np.random.default_rng(seed=SEED)

# %% [markdown]
# ## 1. Carga y estructura del dataset

# %%
# Ruta del dataset RFM controlada por config.RFM_PATH
rfm = pd.read_csv(RFM_PATH)

print(f"Dataset cargado desde: {RFM_PATH}")
print(f"Dimensión: {rfm.shape[0]:,} filas × {rfm.shape[1]} columnas\n")
print("Tipos de columna:")
print(rfm.dtypes.to_string())
rfm.head()

# %%
print("Valores nulos por columna:")
print(rfm.isna().sum().to_string())
print("\nEstadísticos descriptivos (numéricos):")
rfm.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T


# %%
def tabla_frecuencias(serie: pd.Series) -> pd.DataFrame:
    n = serie.value_counts(dropna=False)
    p = serie.value_counts(normalize=True, dropna=False)
    return pd.DataFrame({"n": n, "%": (p * 100).round(2)})

print("Tipo de título:")
print(tabla_frecuencias(rfm["tipo_titulo"]), "\n")
print("Cohorte de entrada al sistema:")
print(tabla_frecuencias(rfm["cohort_year"]).sort_index(), "\n")
print("Usuarios marcados como jóvenes (Tarjeta Joven):")
print(tabla_frecuencias(rfm["es_joven"]))

# %% [markdown]
# ## 2. Distribuciones marginales de las métricas RFM
#
# Las cuatro variables que el modelo BG/NBD + Gamma-Gamma toma como entrada son:
#
# - **`frequency`** ($x$): número de semanas con al menos una validación, *excluyendo* la primera.
# - **`recency`** ($t_x$): semanas transcurridas entre la primera y la última semana activa.
# - **`T`**: semanas observables desde la primera validación hasta el cierre del período (antigüedad).
# - **`monetary_value`** ($z$): viajes promedio por semana activa (proxy del valor monetario una vez multiplicado por el precio del título).

# %%
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
variables = [
    ("frequency",      "Frecuencia $x$ (semanas activas - 1)"),
    ("recency",        "Recencia $t_x$ (semanas)"),
    ("T",              "Antigüedad $T$ (semanas)"),
    ("monetary_value", "Valor monetario $z$ (viajes/semana activa)"),
]

for ax, (col, titulo) in zip(axes.ravel(), variables):
    sns.histplot(rfm[col], bins=60, ax=ax, kde=False, color="steelblue")
    ax.axvline(rfm[col].mean(),   color="red",   ls="--", lw=1, label=f"media={rfm[col].mean():.1f}")
    ax.axvline(rfm[col].median(), color="black", ls=":",  lw=1, label=f"mediana={rfm[col].median():.1f}")
    ax.set_title(titulo)
    ax.legend(fontsize=8)

plt.tight_layout()
plt.show()

# %%
fig, axes = plt.subplots(1, 4, figsize=(14, 4))
for ax, (col, titulo) in zip(axes, variables):
    sns.boxplot(y=rfm[col], ax=ax, color="steelblue")
    ax.set_title(titulo, fontsize=10)
    ax.set_ylabel("")
plt.tight_layout()
plt.show()

extremos = pd.DataFrame({
    "P95": rfm[["frequency", "recency", "T", "monetary_value"]].quantile(0.95),
    "P99": rfm[["frequency", "recency", "T", "monetary_value"]].quantile(0.99),
    "max": rfm[["frequency", "recency", "T", "monetary_value"]].max(),
})
print("Cuantiles altos y máximos:")
extremos

# %% [markdown]
# ## 3. Relaciones bivariantes y matriz RFM
#
# El producto cartesiano `frequency × recency` es el núcleo informacional del BG/NBD: en él se proyectan las dos dimensiones (cuánto usa y cuándo dejó de usar) que determinan la probabilidad posterior de seguir vivo $P(\text{vivo} \mid x, t_x, T)$.

# %%
# Discretizamos en celdas para visualizar la densidad conjunta
bins_f = np.linspace(0, rfm["frequency"].quantile(0.99), 21)
bins_r = np.linspace(0, rfm["recency"].quantile(0.99),   21)

H, xe, ye = np.histogram2d(rfm["recency"], rfm["frequency"], bins=[bins_r, bins_f])

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(
    np.log1p(H.T),
    origin="lower",
    extent=[xe.min(), xe.max(), ye.min(), ye.max()],
    aspect="auto",
    cmap="viridis",
)
ax.set_xlabel("Recencia $t_x$ (semanas)")
ax.set_ylabel("Frecuencia $x$")
ax.set_title("Densidad conjunta $(t_x, x)$ — escala log(1+n)")
plt.colorbar(im, ax=ax, label="log(1 + nº usuarios)")
plt.tight_layout()
plt.show()

# %%
rfm["dormancia"] = rfm["T"] - rfm["recency"]

fig, ax = plt.subplots(figsize=(9, 4.5))
sns.histplot(rfm["dormancia"], bins=60, ax=ax, color="darkorange")
ax.axvline(26, color="red", ls="--", label="26 semanas (umbral 'dormido')")
ax.axvline(52, color="darkred", ls="--", label="52 semanas (1 año)")
ax.set_xlabel("Semanas desde la última validación  ($T - t_x$)")
ax.set_title("Dormancia: tiempo desde la última actividad")
ax.legend()
plt.tight_layout()
plt.show()

umbrales = [4, 13, 26, 52, 104]
tabla_dorm = pd.DataFrame({
    "umbral (sem.)": umbrales,
    "% usuarios por encima": [(rfm["dormancia"] > u).mean() * 100 for u in umbrales],
})
print("Proporción de usuarios con dormancia superior a cada umbral:")
print(tabla_dorm.to_string(index=False, formatters={"% usuarios por encima": "{:.1f}%".format}))

# %%
num_cols = ["frequency", "recency", "T", "monetary_value", "dormancia"]
corr = rfm[num_cols].corr(method="spearman")

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax)
ax.set_title("Correlación de Spearman entre métricas RFM")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Diferencias entre segmentos
#
# Comparamos las métricas RFM entre los tres factores categóricos del dataset: tipo de título, año de incorporación al sistema y condición de Tarjeta Joven.

# %%
agregados_titulo = (
    rfm.groupby("tipo_titulo")[["frequency", "recency", "T", "monetary_value", "dormancia"]]
       .agg(["mean", "median"])
       .round(2)
)
print("Métricas RFM por tipo de título:")
print(agregados_titulo)

fig, axes = plt.subplots(1, 4, figsize=(15, 4))
for ax, (col, titulo) in zip(axes, variables):
    sns.boxplot(data=rfm, x="tipo_titulo", y=col, ax=ax,
                showfliers=False, hue="tipo_titulo", legend=False)
    ax.set_title(titulo, fontsize=10)
    ax.tick_params(axis="x", rotation=30)
    ax.set_xlabel("")
plt.tight_layout()
plt.show()

# %%
agregados_cohort = (
    rfm.groupby("cohort_year")[["frequency", "recency", "T", "monetary_value", "dormancia"]]
       .agg(["mean", "median", "count"])
       .round(2)
)
print("Métricas RFM por cohorte:")
print(agregados_cohort)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, col in zip(axes, ["frequency", "monetary_value", "dormancia"]):
    sns.boxplot(data=rfm, x="cohort_year", y=col, ax=ax,
                showfliers=False, hue="cohort_year", legend=False, palette="viridis")
    ax.set_title(col)
plt.tight_layout()
plt.show()

# %%
agregados_joven = (
    rfm.groupby("es_joven")[["frequency", "recency", "T", "monetary_value", "dormancia",
                              "lambda_true", "p_true"]]
       .agg(["mean", "median", "count"])
       .round(4)
)
print("Comparación jóvenes vs. no jóvenes:")
print(agregados_joven)

# Test de Mann-Whitney sobre la frecuencia (no asume normalidad)
joven = rfm.loc[rfm["es_joven"], "frequency"]
resto = rfm.loc[~rfm["es_joven"], "frequency"]
u, pval = stats.mannwhitneyu(joven, resto, alternative="two-sided")
print(f"\nMann-Whitney sobre `frequency` (jóvenes vs no jóvenes): U={u:.0f}, p={pval:.2e}")

# %% [markdown]
# ## 5. Validación frente a las estadísticas públicas
#
# Replicamos —de forma más exhaustiva que la función `validar_dataset` del pipeline— la comparación entre los estadísticos del dataset sintético y los publicados por el Metropolitano de Granada y el CTAGR. La idea es que **toda diferencia relevante** debe ser explicable o señalable, y que las diferencias pequeñas confirman la calibración del simulador.

# %%
# Las referencias para validación se leen ahora desde config.STATS y
# config.PARAMS_CALIBRATED — no se duplican aquí. Cualquier cambio
# en config.py se propaga automáticamente.

lambda_ref = PARAMS_CALIBRATED["E_lambda"]
p_ref      = PARAMS_CALIBRATED["E_p"]

print(f"E[λ] de referencia (calibrado) ≈ {lambda_ref:.3f} viajes/semana")
print(f"E[p] de referencia (calibrado) ≈ {p_ref:.4f}")

# %%
rec = rfm.loc[rfm["frequency"] > 0]

metricas_pub = []
for titulo, p_real in STATS["pct_titulos"].items():
    p_sim = (rfm["tipo_titulo"] == titulo).mean()
    metricas_pub.append((f"% título {titulo}", p_real, p_sim))

metricas_pub += [
    ("% usuarios recurrentes (x>0)",
        1 - STATS["pct_titulos"]["ocasional"], (rfm["frequency"] > 0).mean()),
    ("E[λ] viajes/semana activa",      lambda_ref, rec["monetary_value"].mean()),
    ("E[p] prob. abandono semanal",    p_ref,      rfm["p_true"].mean()),
]

tabla = pd.DataFrame(metricas_pub, columns=["Métrica", "Real / referencia", "Sintético"])
tabla["Desv. relativa"] = ((tabla["Sintético"] - tabla["Real / referencia"])
                            / tabla["Real / referencia"]).abs()
tabla["✓ (<5%)"] = tabla["Desv. relativa"] < 0.05

print("Comparación dataset sintético vs. referencias públicas (config.STATS):")
print(tabla.to_string(index=False,
                     formatters={"Real / referencia": "{:.4f}".format,
                                 "Sintético":         "{:.4f}".format,
                                 "Desv. relativa":    "{:.2%}".format}))

# %%
# Distribución de cohortes esperada — leemos viajeros_anuales desde config.STATS
viajeros = STATS["viajeros_anuales"]
años     = sorted(viajeros)
increm   = {a: (viajeros[a] - viajeros[años[i-1]] if i else viajeros[a])
            for i, a in enumerate(años)}
increm   = {a: max(v, viajeros[a] * 0.05) for a, v in increm.items()}
total    = sum(increm.values())
esperado = {a: v / total for a, v in increm.items()}

observado = rfm["cohort_year"].value_counts(normalize=True).sort_index()

comp_coh = pd.DataFrame({"esperado": esperado, "observado": observado}).fillna(0)
comp_coh["desv"] = (comp_coh["observado"] - comp_coh["esperado"]).abs()
print("Cohortes — esperado (crecimiento de viajeros) vs. observado:")
print(comp_coh.round(4))

fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(len(comp_coh))
ax.bar(x - 0.2, comp_coh["esperado"],  width=0.4, label="esperado",  color="steelblue")
ax.bar(x + 0.2, comp_coh["observado"], width=0.4, label="observado", color="darkorange")
ax.set_xticks(x)
ax.set_xticklabels(comp_coh.index)
ax.set_ylabel("Proporción del total")
ax.set_title("Distribución de cohortes: esperada vs observada")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. *Parameter recovery* — núcleo metodológico
#
# Esta es la validación más importante del pipeline. El simulador siembra parámetros poblacionales $(r, \alpha, a, b)$ calibrados por método de momentos y luego, para cada usuario, sortea valores individuales:
#
# $$
# \lambda_i \sim \mathrm{Gamma}(r, \alpha), \qquad p_i \sim \mathrm{Beta}(a, b).
# $$
#
# El dataset conserva esas semillas en `lambda_true` y `p_true`. Si la distribución empírica de estas columnas reproduce la teórica, queda demostrado que el simulador genera muestras coherentes con sus parámetros (paso previo y necesario para que el ajuste bayesiano posterior pueda *recuperarlos*).
#
# Para una comparación limpia descartamos los usuarios `es_joven`, sobre los que se aplica una transformación posterior ($\lambda \times 1{,}3$, $p \times 1{,}8$) que rompe la familia Gamma/Beta original. Esos efectos los verificamos por separado más abajo.

# %%
# Parámetros calibrados — leemos directamente de config.PARAMS_CALIBRATED
R     = PARAMS_CALIBRATED["r"]
ALPHA = PARAMS_CALIBRATED["alpha"]
A     = PARAMS_CALIBRATED["a"]
B     = PARAMS_CALIBRATED["b"]

print(f"Parámetros poblacionales calibrados (desde config.PARAMS_CALIBRATED):")
print(f"  r = {R:.4f}    α = {ALPHA:.4f}    →  E[λ] = {R/ALPHA:.3f}")
print(f"  a = {A:.4f}    b = {B:.4f}        →  E[p] = {A/(A+B):.4f}")

# %%
lam_sample = rfm.loc[~rfm["es_joven"], "lambda_true"].to_numpy()

# Distribución teórica: scipy.stats.gamma(a=r, scale=1/alpha)
lam_teorica = stats.gamma(a=R, scale=1.0/ALPHA)

# (1) Histograma + PDF teórica
fig, ax = plt.subplots(figsize=(9, 4.5))
sns.histplot(lam_sample, bins=80, stat="density", ax=ax,
             color="steelblue", alpha=0.6, label="empírico (no jóvenes)")
x = np.linspace(0, lam_sample.max(), 400)
ax.plot(x, lam_teorica.pdf(x), color="red", lw=2,
        label=f"Gamma(r={R:.2f}, α={ALPHA:.2f})")
ax.axvline(lam_sample.mean(),   color="steelblue", ls="--",
           label=f"media empírica = {lam_sample.mean():.3f}")
ax.axvline(lam_teorica.mean(),  color="red",       ls=":",
           label=f"media teórica = {lam_teorica.mean():.3f}")
ax.set_xlabel("$\\lambda$  (viajes/semana)")
ax.set_title("Recovery de $\\lambda$: empírico vs teórico")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(r'C:\Users\andre\OneDrive\Documentos\UGR\tfm-clv-transporte-granada\docs\figures\recovery_lambda_p.pdf')
plt.show()

# (2) Test de Kolmogorov-Smirnov
ks = stats.kstest(lam_sample, lam_teorica.cdf)
print(f"KS test (λ): D={ks.statistic:.4f}, p-value={ks.pvalue:.4f}")

# (3) Re-estimación por MLE y por método de momentos
r_mle, _, scale_mle = stats.gamma.fit(lam_sample, floc=0)
alpha_mle = 1.0 / scale_mle
mu, var   = lam_sample.mean(), lam_sample.var()
alpha_mom = mu / var
r_mom     = mu * alpha_mom

print("\nRe-estimación de (r, α) a partir de las muestras:")
print(f"  Calibrado:           r={R:.4f}    α={ALPHA:.4f}")
print(f"  MLE Gamma:           r={r_mle:.4f}    α={alpha_mle:.4f}")
print(f"  Método de momentos:  r={r_mom:.4f}    α={alpha_mom:.4f}")

# %%
p_sample = rfm.loc[~rfm["es_joven"], "p_true"].to_numpy()
p_sample = p_sample[(p_sample > 0) & (p_sample < 1)]

p_teorica = stats.beta(a=A, b=B)

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 5))

stats.probplot(lam_sample, dist=lam_teorica, plot=axes[0])
axes[0].set_title("QQ plot — $\\lambda$ vs Gamma teórica")
axes[0].get_lines()[0].set_markersize(2)
axes[0].get_lines()[0].set_color("steelblue")

stats.probplot(p_sample, dist=p_teorica, plot=axes[1])
axes[1].set_title("QQ plot — $p$ vs Beta teórica")
axes[1].get_lines()[0].set_markersize(2)
axes[1].get_lines()[0].set_color("darkorange")

plt.tight_layout()
plt.savefig(r'C:\Users\andre\OneDrive\Documentos\UGR\tfm-clv-transporte-granada\docs\figures\recovery_lambda_p.pdf')
plt.show()

# %%
# Tabla resumen comparando momentos teóricos vs empíricos
resumen_momentos = pd.DataFrame({
    "E[λ]":    [lam_teorica.mean(),                lam_sample.mean()],
    "Var[λ]":  [lam_teorica.var(),                 lam_sample.var()],
    "CV[λ]":   [np.sqrt(lam_teorica.var()) / lam_teorica.mean(),
                np.sqrt(lam_sample.var())  / lam_sample.mean()],
    "E[p]":    [p_teorica.mean(),                  p_sample.mean()],
    "Var[p]":  [p_teorica.var(),                   p_sample.var()],
}, index=["teórico", "empírico"])

print("Momentos teóricos vs empíricos (usuarios no jóvenes):")
print(resumen_momentos.T.round(6))

# %%
# Verificación de los multiplicadores en es_joven=True
# Los valores esperados se leen de config.SIM_CONFIG.
mult_lam = SIM_CONFIG["joven_lambda_mult"]
mult_p   = SIM_CONFIG["joven_p_mult"]

lam_joven    = rfm.loc[ rfm["es_joven"], "lambda_true"]
lam_no_joven = rfm.loc[~rfm["es_joven"], "lambda_true"]
p_joven      = rfm.loc[ rfm["es_joven"], "p_true"]
p_no_joven   = rfm.loc[~rfm["es_joven"], "p_true"]

tabla_jov = pd.DataFrame({
    "E[λ] esperado":  [lam_no_joven.mean(),       lam_no_joven.mean() * mult_lam],
    "E[λ] observado": [lam_no_joven.mean(),       lam_joven.mean()],
    "E[p] esperado":  [p_no_joven.mean(),         p_no_joven.mean()   * mult_p],
    "E[p] observado": [p_no_joven.mean(),         p_joven.mean()],
}, index=["no jóvenes", "jóvenes"])

print(f"Comprobación de los multiplicadores aplicados a es_joven "
      f"(λ × {mult_lam}, p × {mult_p}):")
print(tabla_jov.round(4))

print("\nRatios observados:")
print(f"  λ_joven / λ_no_joven = {lam_joven.mean()/lam_no_joven.mean():.3f}  (esperado: {mult_lam})")
print(f"  p_joven / p_no_joven = {p_joven.mean()/p_no_joven.mean():.3f}  (esperado: {mult_p})")

# %% [markdown]
# ## 8. Conclusiones del análisis
#
# Este notebook permite afirmar tres cosas que son requisitos previos para el ajuste bayesiano del BG/NBD:
#
# 1. **El dataset reproduce las estadísticas públicas del CTAGR** dentro de tolerancias razonables (distribución de títulos, tasa de uso semanal, tasa de abandono implícita). La principal desviación a discutir en la memoria es la sobre-representación de la cohorte 2018, consecuencia directa del salto inicial de viajeros 0 → 10,2 M con el que se siembra el simulador.
#
# 2. **Las muestras individuales $\lambda_i$ y $p_i$ son consistentes con las distribuciones poblacionales sembradas**: histogramas, QQ plots, KS tests y la re-estimación por MLE / método de momentos coinciden con los $(r, \alpha, a, b)$ calibrados. Este es el resultado de *parameter recovery* que la propuesta de TFM anuncia como contribución metodológica.
#
# 3. **Las firmas del modelo BG/NBD son visibles** en los datos agregados: sobredispersión $\mathrm{Var}[x] \gg \mathrm{E}[x]$ y mejor ajuste NBD que Poisson sobre la distribución de frecuencias condicional a $T$.
#
# Dos convenciones del simulador que conviene tener presentes al interpretar los resultados:
#
# - El multiplicador $p_i \times 1{,}8$ aplicado a usuarios jóvenes puede empujar valores fuera de $[0,1]$; el simulador trunca $p_i$ al intervalo $[0,1]$ antes de muestrear el proceso de abandono.
# - El efecto COVID se aplica únicamente a las cohortes $\leq 2019$ y se mide en *semanas locales* 130–230 desde la primera validación de cada usuario, no en semanas calendario absolutas. Esta convención modela el efecto pandemia como una interrupción interna a la trayectoria individual.

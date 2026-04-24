# TFM: CLV en el Transporte Público de Granada mediante los modelos BG/NBD y Gamma-Gamma

> Bayesian estimation of Client Lifetime Value (CLV) for public transport in Granada using the BG/NBD and Gamma-Gamma models with MCMC.

**Trabajo de Fin de Máster — Máster en Física y Matemáticas, Universidad de Granada (2025–2026)**

---

## Descripción

Este repositorio contiene el código, datos sintéticos y análisis del TFM cuyo objetivo es calcular el **Valor de Vida del Cliente** (CLV) del sistema de transporte público del Área Metropolitana de Granada. Para ello se aplica el modelo probabilístico **BG/NBD** (*Beta-Geometric/Negative Binomial Distribution*), complementado con el modelo **Gamma-Gamma**, dentro de un marco de **inferencia bayesiana completa con MCMC**.

El modelo, originalmente desarrollado para comercio electrónico, se reinterpreta formalmente en el dominio del transporte público: los usuarios validan su tarjeta de forma voluntaria y su abandono del sistema es silencioso e inobservable, lo que constituye un entorno no contractual y continuo matemáticamente equivalente.

Dado que los datos individuales de validación no son públicos, se desarrolla un **pipeline de datos sintéticos calibrados** con las estadísticas publicadas en los balances anuales del Metropolitano de Granada.

---

## Modelos utilizados

| Modelo | Descripción |
|---|---|
| **BG/NBD** | Estima la frecuencia de uso futura y la probabilidad de que cada usuario siga activo |
| **Gamma-Gamma** | Estima el valor monetario esperado por período activo |
| **BG/NBD + Gamma-Gamma** | CLV completo a 1, 3 y 5 años por usuario |

El ajuste se realiza con el algoritmo **NUTS** (*No-U-Turn Sampler*) a través de `pymc-marketing`, produciendo distribuciones posteriores completas e intervalos de alta densidad (HDI) sobre todas las predicciones.

---

## Datos

Los datos individuales de validación de tarjetas no son públicos. Este proyecto utiliza un **dataset RFM sintético** generado a partir de estadísticas agregadas publicadas por:

- [Metropolitano de Granada](https://www.metropolitanogranada.es) — Balances anuales 2017–2024
- [CTAGR](https://ctagr.es) — Estadísticas del sistema multimodal
- [Red de Consorcios de Andalucía](https://api.ctan.es) — Portal de datos abiertos GTFS

El pipeline de generación estima los parámetros del modelo generativo mediante el **Método de Momentos** y los refina con ***Approximate Bayesian Computation*** (ABC).

---

## Dependencias principales

| Paquete | Versión | Uso |
|---|---|---|
| `pymc-marketing` | ≥ 0.7 | Modelos BG/NBD y Gamma-Gamma |
| `pymc` | ≥ 5.0 | Muestreo MCMC con NUTS |
| `arviz` | ≥ 0.17 | Diagnósticos y visualización |
| `scipy` | ≥ 1.11 | Método de Momentos y ABC |
| `pandas` | ≥ 2.0 | Manipulación de datos |
| `numpy` | ≥ 1.25 | Cálculo numérico |

---

## Referencia principal

Fader, P. S., Hardie, B. G. S., & Lee, K. L. (2005).
*"Counting your customers" the easy way: An alternative to the Pareto/NBD model.*
Marketing Science, 24(2), 275–284.

---

## Autor

**Daniel Andrés Dala**
Máster en Física y Matemáticas — Universidad de Granada
danieldala@correo.ugr.es

Tutor: Osvaldo Antonio Martin, Aalto University

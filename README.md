# TFM: Valor de Vida del Usuario del Metropolitano en Granada: un enfoque bayesiano mediante los modelos BG/NBD y Gamma–Gamma con MCMC

**Trabajo de Fin de Máster — Máster en Física y Matemáticas, Universidad de Granada (2025–2026)**

---

## Descripción

Este repositorio contiene el código, datos sintéticos y análisis del TFM cuyo objetivo es calcular el **Valor de Vida del Cliente** (CLV) del Metro de Granada. Para ello se aplica el modelo probabilístico **BG/NBD** (*Beta-Geometric/Negative Binomial Distribution*), complementado con el modelo **Gamma-Gamma**, dentro de un marco de **inferencia bayesiana completa con MCMC**.

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

El pipeline de generación estima los parámetros del modelo generativo mediante el **Método de Momentos**.

---

## Dependencias principales

> Generado con Python 3.13.7. Para reproducir exactamente los resultados,
> se recomienda fijar estas versiones (ver `requirements.txt`).

| Paquete | Versión | Uso |
|---|---|---|
| `pymc-marketing` | ≥ 0.19 | Modelos BG/NBD y Gamma-Gamma |
| `pymc` | ≥ 5.28 | Modelo probabilístico subyacente |
| `nutpie` | ≥ 0.16 | Muestreador NUTS (backend en Rust) |
| `preliz` | ≥ 0.24 | Elicitación de priors por máxima entropía |
| `arviz` | ≥ 0.23 | Diagnósticos de convergencia y visualización |
| `scipy` | ≥ 1.17 | Método de los momentos y funciones especiales |
| `pandas` | ≥ 3.0 | Manipulación de datos |
| `numpy` | ≥ 2.4 | Cálculo numérico |
| `matplotlib` | ≥ 3.10 | Generación de figuras |
| `seaborn` | ≥ 0.13 | Gráficos estadísticos |

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

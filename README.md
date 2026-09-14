# Optimización de fronteras en espacios geométricos de alta asimetría

Código reproducible del experimento de la **Actividad de Evaluación Unidad 1 — Reconocimiento de Patrones (Maestría)**.

El proyecto estudia el efecto de **Tomek Links** sobre un espacio multiclase altamente desbalanceado y evalúa si la limpieza geométrica se traduce en cambios de desempeño de un clasificador paramétrico de **mínima distancia (D-min)**.

## Objetivo experimental

Se simula un conjunto de `N=2000` observaciones, `d=10` características y tres clases:

- clase 0: referencia/mayoritaria, 98.2 %
- clase 1: patrón intermedio, 1.0 %
- clase 2: patología rara, 0.8 %

La evaluación clínica se realiza como:

`clase 2 vs {clase 0, clase 1}`

El pipeline principal es:

1. generación del conjunto sintético;
2. inyección explícita de artefactos geométricos;
3. validación estratificada repetida;
4. normalización robusta Mediana/MAD ajustada solo con entrenamiento;
5. caracterización geométrica;
6. Tomek Links solo sobre entrenamiento y eliminando únicamente mayoría;
7. control `random_matched`;
8. D-min multiclase;
9. evaluación rare-vs-rest;
10. Wilcoxon pareado + Holm;
11. bootstrap descriptivo;
12. auditoría geométrica de los puntos eliminados.

## Fundamentación matemática principal

### Escalado robusto

\[
\tilde{x}_j=\operatorname{mediana}(x_{1j},...,x_{nj})
\]

\[
MAD_j=\operatorname{mediana}_i|x_{ij}-\tilde{x}_j|
\]

\[
z^{(R)}_{ij}=
\frac{x_{ij}-\tilde{x}_j}{1.4826\,MAD_j}
\]

### Geometría de la clase rara

\[
D_{\mathrm{intra,rara}}
=
\frac{1}{n_r}
\sum_{i:y_i=r}d(x_i,\mu_r)
\]

\[
D_{\mathrm{inter,rara}}
=
\frac{1}{C-1}
\sum_{j\neq r}d(\mu_r,\mu_j)
\]

\[
G_{\mathrm{rara}}
=
\frac{D_{\mathrm{inter,rara}}}
{D_{\mathrm{intra,rara}}}
\]

### D-min

\[
\hat y(x)=\arg\min_k d(x,\mu_k)
\]

La distancia Euclidiana es la métrica confirmatoria por su relación con MAP bajo clases gaussianas equiprobables y covarianza común isotrópica.

### Score rare-vs-rest

\[
s_r(x)=
\min_{j\neq r}d(x,\mu_j)-d(x,\mu_r)
\]

### kDN

\[
kDN(x_i)=
\frac{
|\{x_j\in N_k(x_i):y_j\neq y_i\}|
}{k}
\]

## Estructura del repositorio

```text
tomek_dmin_tesina/
├── README.md
├── requirements.txt
├── .gitignore
├── run_experiment.py
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── preprocessing.py
│   ├── geometry.py
│   ├── classifier.py
│   ├── evaluation.py
│   ├── audit.py
│   ├── plots.py
│   └── experiment.py
├── tests/
│   └── test_smoke.py
├── notebooks/
│   └── Experimento_Final_Tomek_Dmin_Multiclase.ipynb
└── results/
    └── .gitkeep
```

## Instalación

Python 3.10 o superior.

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Instalar dependencias:

```bash
pip install -r requirements.txt
```

## Ejecución principal

```bash
python run_experiment.py
```

Para indicar otra carpeta:

```bash
python run_experiment.py --output results/experimento_01
```

El script genera CSV, PNG, JSON, TXT y un ZIP final con todos los resultados.




## Validaciones automáticas

El pipeline verifica que:

- todos los folds contengan la clase rara;
- Tomek no elimine casos raros;
- el test no participe en el escalado;
- las métricas confirmatorias tengan exactamente `R` pares;
- no existan valores `NaN` inesperados en las variables confirmatorias.

## Vectorización

La matriz de distancias se calcula con `scipy.spatial.distance.cdist`. No se utilizan bucles iterativos para calcular distancias entre matrices, en concordancia con la asignación.

Los bucles existentes recorren folds, métodos, métricas o clases; no reemplazan operaciones matriciales de distancia.

## Resultados principales reproducibles con la semilla 42

Con la configuración oficial del repositorio se espera observar un patrón equivalente al documentado en la tesina:

- mejora geométrica pequeña pero consistente en `G_rare`;
- sensibilidad prácticamente sin cambio;
- BA y ROC-AUC sin mejora significativa bajo Euclidiana;
- reducción pequeña de kDN global;
- kDN de la clase rara prácticamente sin cambio;
- Tomek elimina observaciones mayoritarias geométricamente ambiguas, pero no necesariamente los artefactos sintéticos inyectados.

Los valores exactos pueden variar ligeramente entre versiones de bibliotecas.

## Referencias metodológicas centrales

- Tomek, I. (1976). *Two modifications of CNN*.
- Batista, G. E. A. P. A., Prati, R. C., & Monard, M. C. (2004).
- Rousseeuw, P. J., & Croux, C. (1993).
- Ledoit, O., & Wolf, M. (2004).
- Duda, Hart, & Stork (2001).
- Theodoridis et al. (2010).
- Brodersen et al. (2010).
- Saito & Rehmsmeier (2015).
- Varma & Simon (2006).
- Weiss (2013).

## Alcance

El experimento constituye **validación interna metodológica** sobre datos sintéticos. No representa validación clínica externa ni demuestra superioridad universal de Tomek Links.

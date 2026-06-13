"""
================================================================================
 Comparacion_ML.py
 Comparacion sistematica de modelos de aprendizaje automatico para la
 identificacion inversa de parametros elasticos: caso isotropico (E, nu)
 y caso ortotropico (E1, E2, E3, nu12, nu13, nu23)
================================================================================
 Proyecto de Investigacion - IF4702 Fisica Computacional I - TEC

 FLUJO DEL ALGORITMO
 -------------------
 [1] Configuracion y reproducibilidad (semilla fija, parametros, versiones)
 [2] PROBLEMA ISOTROPICO
     a. Carga de los CSV FEM (Datos_COMSOL/<Material>/F####N.csv)
     b. Dataset de entrenamiento: muestreo del espacio (E, nu, F) con el
        modelo directo de compresion uniaxial + ruido relativo del 1 %
     c. Features (5): sigma_yy, eps_yy, eps_xx, eps_zz, |v|max
        Targets (2): E [GPa], nu
 [3] PROBLEMA ORTOTROPICO
     a. Ley de Hooke ortotropica (matriz de compliance S simetrica)
     b. Protocolo de 3 ensayos uniaxiales independientes (cargas en X, Y, Z):
        un unico ensayo uniaxial solo permite identificar 3 de las 9
        constantes (problema de identificabilidad); con 3 ensayos se
        identifican E1, E2, E3, nu12, nu13, nu23 (los modulos cortantes
        G_ij requieren ensayos de cortante y quedan fuera del alcance)
     c. Muestreo con rechazo: solo tensores de compliance definidos
        positivos (estabilidad termodinamica):
            |nu_ij| < sqrt(E_i/E_j)   y
            1 - nu12*nu21 - nu23*nu32 - nu13*nu31 - 2*nu21*nu32*nu13 > 0
     d. Features (12): 9 deformaciones (3 por ensayo) + 3 esfuerzos aplicados
        Targets (6): E1, E2, E3 [GPa], nu12, nu13, nu23
 [4] MODELOS COMPARADOS (mismo preprocesamiento para todos:
     StandardScaler en entradas y salidas)
        Lineal          : regresion lineal multivariada (linea base)
        Polinomial-2    : caracteristicas polinomiales grado 2 + Ridge
        KNN             : k vecinos mas cercanos (k=8, pesos por distancia)
        Bosque Aleatorio: 200 arboles
        Grad. Boosting  : HistGradientBoosting (un regresor por salida)
        MLP (32)        : perceptron multicapa de 1 capa oculta
        MLP (64,64)     : 2 capas ocultas
        MLP (128,64,32) : 3 capas ocultas
 [5] EVALUACION
     a. Particion 80/20 entrenamiento/prueba (hold-out)
     b. Validacion cruzada de 3 particiones sobre el conjunto de
        entrenamiento (media y desviacion del RMSE)
     c. Metricas por modelo: RMSE y MAPE por parametro, R2 global,
        tiempo de entrenamiento y de inferencia
     d. Caso isotropico: identificacion fisica sobre los casos FEM reales
        (100 fuerzas x 4 materiales) y error relativo contra los valores
        nominales de cada material
 [6] EXPORTACION: figuras PDF, tablas CSV y LaTeX, metricas.txt

 PARAMETROS (reproducibilidad)
 -----------------------------
 Semilla global : 2026
 Isotropico     : N = 8000, E ~ U(50, 280) GPa, nu ~ U(0.20, 0.40),
                  F ~ U(10, 1000) N, ruido relativo gaussiano 1 %
 Ortotropico    : N = 12000 (tras rechazo), E_i ~ U(60, 250) GPa,
                  nu_ij ~ U(0.10, 0.45), sigma ~ U(0.1, 10) MPa por ensayo,
                  ruido relativo gaussiano 1 %
 Geometria FEM  : cubo L = 10 mm, area A = 1e-4 m^2, filtro de
                  Saint-Venant 0.25L <= y <= 0.95L

 EJECUCION
 ---------
     pip install numpy pandas scipy scikit-learn matplotlib
     python Comparacion_ML.py

 Estructura esperada:  Comparacion_ML.py junto a Datos_COMSOL/<Material>/
 Salidas en            resultados_ml/
================================================================================
"""

import sys
import time
import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.base import clone

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ==============================================================================
# [1] CONFIGURACION
# ==============================================================================
SEED      = 2026
BASE_DIR  = Path(__file__).resolve().parent
DATA_DIR  = BASE_DIR / 'Datos_COMSOL'
OUT_DIR   = BASE_DIR / 'resultados_ml'
OUT_DIR.mkdir(exist_ok=True)

L     = 10e-3
AREA  = L * L
Y_MIN, Y_MAX = 0.25 * L, 0.95 * L

MATERIALES = {
    'AISI1020':       {'E': 200e9, 'nu': 0.29},
    'HierroDuctil':   {'E': 120e9, 'nu': 0.31},
    'Aluminio6061':   {'E':  69e9, 'nu': 0.33},
    'TitanioTi6Al4V': {'E': 114e9, 'nu': 0.34},
}

COLS = ['x', 'y', 'z',
        'sx', 'sy', 'sz', 'sxy', 'sxz', 'syz',
        'eX', 'eY', 'eZ', 'eXY', 'eXZ', 'eYZ',
        'u', 'v', 'w']

N_ISO       = 8000
E_RANGO     = (50e9, 280e9)
NU_RANGO    = (0.20, 0.40)
F_RANGO     = (10.0, 1000.0)

N_ORTO      = 12000
EI_RANGO    = (60e9, 250e9)
NUIJ_RANGO  = (0.10, 0.45)
SIG_RANGO   = (0.1e6, 10e6)

RUIDO_REL   = 0.01
CV_FOLDS    = 3

rng = np.random.default_rng(SEED)

C1, C2, C3, C4 = '#7aaa97', '#05242f', '#c46d5e', '#9a8c98'
plt.rcParams.update({
    'font.family': 'serif', 'mathtext.fontset': 'cm',
    'axes.grid': True, 'grid.alpha': 0.3, 'figure.dpi': 110,
    'axes.spines.top': False, 'axes.spines.right': False,
})

# ==============================================================================
# LaTeX typeset. Si no se tiene instalado de forma nativa, comentar esta sección; la parte anterior da un formato relativamente similar

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": "Computer Modern Roman",
})
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsfonts}\usepackage{amsmath}\usepackage{amssymb}\usepackage{siunitx}')

# ==============================================================================

print('=' * 72)
print('Comparacion de modelos ML: identificacion isotropica y ortotropica')
print('=' * 72)
print(f'numpy {np.__version__} | pandas {pd.__version__} | '
      f'scipy {scipy.__version__} | scikit-learn {sklearn.__version__}')
print(f'Semilla = {SEED}\n')

# ==============================================================================
# [2] PROBLEMA ISOTROPICO
# ==============================================================================
def cargar_csv(path):
    df = pd.read_csv(path, comment='%', header=None, sep=',', engine='python')
    if df.shape[1] == 1:
        df = pd.read_csv(path, comment='%', header=None,
                         sep=r'\s+', engine='python')
    df = df.iloc[:, :len(COLS)]
    df.columns = COLS[:df.shape[1]]
    return df.apply(pd.to_numeric, errors='coerce').dropna()


def extraer_features_iso(df, F):
    """La carga aplicada F es un dato conocido del ensayo, por lo que
    sigma_yy = -F/A se usa directamente en lugar del esfuerzo postprocesado
    (el promedio nodal de esfuerzos en tetraedros lineales esta sesgado)."""
    nucleo = df[(df['y'] >= Y_MIN) & (df['y'] <= Y_MAX)]
    if len(nucleo) < 20:
        nucleo = df
    return np.array([
        -F / AREA,
        nucleo['eY'].mean(),
        nucleo['eX'].mean(),
        nucleo['eZ'].mean(),
        df['v'].abs().max(),
    ])


def modelo_directo_iso(E, nu, F):
    """Compresion uniaxial isotropica. Vectorizado -> (N, 5)."""
    E, nu, F = np.broadcast_arrays(np.atleast_1d(E),
                                   np.atleast_1d(nu),
                                   np.atleast_1d(F))
    sy   = -F / AREA
    eY   = sy / E
    eX   = -nu * eY
    vmax = np.abs(eY) * L
    return np.column_stack([sy, eY, eX, eX, vmax])


print('[Isotropico] Cargando datos FEM...')
datos_fem = {}
for mat in MATERIALES:
    carpeta = DATA_DIR / mat
    F_list, X_list = [], []
    if carpeta.is_dir():
        for f in sorted(carpeta.glob('*.csv')):
            m = re.search(r'F(\d+)N', f.name)
            if not m:
                continue
            F_val = float(m.group(1))
            F_list.append(F_val)
            X_list.append(extraer_features_iso(cargar_csv(f), F_val))
    if F_list:
        orden = np.argsort(F_list)
        datos_fem[mat] = {'F': np.asarray(F_list)[orden],
                          'X': np.vstack(X_list)[orden]}
        print(f'  {mat:15s}: {len(F_list):3d} casos')
hay_fem = len(datos_fem) > 0
if not hay_fem:
    print('  Aviso: no se encontraron CSV en', DATA_DIR)

print('\n[Isotropico] Generando dataset de entrenamiento...')
E_s  = rng.uniform(*E_RANGO,  N_ISO)
nu_s = rng.uniform(*NU_RANGO, N_ISO)
F_s  = rng.uniform(*F_RANGO,  N_ISO)
X_iso = modelo_directo_iso(E_s, nu_s, F_s)
X_iso = X_iso * (1.0 + rng.normal(0.0, RUIDO_REL, X_iso.shape))
y_iso = np.column_stack([E_s / 1e9, nu_s])
TARGETS_ISO = ['E [GPa]', 'nu']
print(f'  N = {N_ISO} | features = {X_iso.shape[1]} | targets = {y_iso.shape[1]}')


# ==============================================================================
# [3] PROBLEMA ORTOTROPICO
# ==============================================================================
def muestrear_ortotropico(n, rng):
    """Muestreo con rechazo de tensores ortotropicos termodinamicamente
    estables (compliance definida positiva)."""
    acc_E, acc_nu = [], []
    while sum(len(a) for a in acc_E) < n * 3:
        m = 2 * n
        E1, E2, E3 = (rng.uniform(*EI_RANGO, m) for _ in range(3))
        n12, n13, n23 = (rng.uniform(*NUIJ_RANGO, m) for _ in range(3))
        n21 = n12 * E2 / E1
        n31 = n13 * E3 / E1
        n32 = n23 * E3 / E2
        ok = ((np.abs(n12) < np.sqrt(E1 / E2)) &
              (np.abs(n13) < np.sqrt(E1 / E3)) &
              (np.abs(n23) < np.sqrt(E2 / E3)) &
              (1 - n12 * n21 - n23 * n32 - n13 * n31
                 - 2 * n21 * n32 * n13 > 0))
        acc_E.append(np.column_stack([E1, E2, E3])[ok])
        acc_nu.append(np.column_stack([n12, n13, n23])[ok])
        if sum(a.shape[0] for a in acc_E) >= n:
            break
    E_arr  = np.vstack(acc_E)[:n]
    nu_arr = np.vstack(acc_nu)[:n]
    return E_arr, nu_arr


def modelo_directo_orto(E_arr, nu_arr, sig_arr):
    """Protocolo de 3 ensayos uniaxiales (X, Y, Z) sobre material
    ortotropico con ejes alineados. Devuelve (N, 12):
        [eps_x^X, eps_y^X, eps_z^X,
         eps_x^Y, eps_y^Y, eps_z^Y,
         eps_x^Z, eps_y^Z, eps_z^Z,
         sig^X, sig^Y, sig^Z]
    donde el superindice indica la direccion de carga."""
    E1, E2, E3 = E_arr[:, 0], E_arr[:, 1], E_arr[:, 2]
    n12, n13, n23 = nu_arr[:, 0], nu_arr[:, 1], nu_arr[:, 2]
    sX, sY, sZ = sig_arr[:, 0], sig_arr[:, 1], sig_arr[:, 2]

    S11, S22, S33 = 1 / E1, 1 / E2, 1 / E3
    S12, S13, S23 = -n12 / E1, -n13 / E1, -n23 / E2

    ens_X = np.column_stack([S11 * sX, S12 * sX, S13 * sX])
    ens_Y = np.column_stack([S12 * sY, S22 * sY, S23 * sY])
    ens_Z = np.column_stack([S13 * sZ, S23 * sZ, S33 * sZ])
    return np.hstack([ens_X, ens_Y, ens_Z,
                      np.column_stack([sX, sY, sZ])])


print('\n[Ortotropico] Muestreo con rechazo del espacio de parametros...')
E_o, nu_o = muestrear_ortotropico(N_ORTO, rng)
sig_o = rng.uniform(*SIG_RANGO, (N_ORTO, 3))
X_orto = modelo_directo_orto(E_o, nu_o, sig_o)
X_orto = X_orto * (1.0 + rng.normal(0.0, RUIDO_REL, X_orto.shape))
y_orto = np.hstack([E_o / 1e9, nu_o])
TARGETS_ORTO = ['E1 [GPa]', 'E2 [GPa]', 'E3 [GPa]', 'nu12', 'nu13', 'nu23']
print(f'  N = {N_ORTO} | features = {X_orto.shape[1]} | '
      f'targets = {y_orto.shape[1]}')


# ==============================================================================
# [4] MODELOS
# ==============================================================================
def fabrica_modelos(seed=SEED):
    defs = {
        'Lineal': LinearRegression(),
        'Polinomial-2': Pipeline([
            ('poly', PolynomialFeatures(degree=2, include_bias=False)),
            ('reg', Ridge(alpha=1.0, random_state=seed))]),
        'KNN': KNeighborsRegressor(n_neighbors=8, weights='distance'),
        'Bosque Aleatorio': RandomForestRegressor(
            n_estimators=200, n_jobs=-1, random_state=seed),
        'Grad. Boosting': MultiOutputRegressor(
            HistGradientBoostingRegressor(random_state=seed)),
        'MLP (32)': MLPRegressor(
            hidden_layer_sizes=(32,), activation='tanh', solver='adam',
            max_iter=600, early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=25, random_state=seed),
        'MLP (64,64)': MLPRegressor(
            hidden_layer_sizes=(64, 64), activation='tanh', solver='adam',
            max_iter=600, early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=25, random_state=seed),
        'MLP (128,64,32)': MLPRegressor(
            hidden_layer_sizes=(128, 64, 32), activation='tanh',
            solver='adam', max_iter=600, early_stopping=True,
            validation_fraction=0.15, n_iter_no_change=25,
            random_state=seed),
    }
    modelos = {}
    for nombre, est in defs.items():
        pipe = Pipeline([('escala', StandardScaler()), ('modelo', est)])
        modelos[nombre] = TransformedTargetRegressor(
            regressor=pipe, transformer=StandardScaler())
    return modelos


# ==============================================================================
# [5] EVALUACION
# ==============================================================================
def evaluar_problema(nombre_problema, X, y, target_names):
    print(f'\n{"="*72}\n  PROBLEMA {nombre_problema.upper()}\n{"="*72}')
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, random_state=SEED)

    filas, predicciones, ajustados = [], {}, {}
    for nombre, modelo in fabrica_modelos().items():
        t0 = time.time()
        modelo.fit(X_tr, y_tr)
        t_fit = time.time() - t0

        t0 = time.time()
        y_pred = modelo.predict(X_te)
        t_pred = (time.time() - t0) / len(X_te) * 1e3

        rmse = np.sqrt(mean_squared_error(y_te, y_pred,
                                          multioutput='raw_values'))
        mape = np.mean(np.abs((y_pred - y_te) / y_te), axis=0) * 100
        r2   = r2_score(y_te, y_pred)

        cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
        rmse_cv = []
        for itr, iva in cv.split(X_tr):
            m = clone(modelo)
            m.fit(X_tr[itr], y_tr[itr])
            p = m.predict(X_tr[iva])
            rmse_cv.append(np.sqrt(mean_squared_error(y_tr[iva], p)))
        rmse_cv = np.asarray(rmse_cv)

        fila = {'modelo': nombre, 'R2': r2,
                'RMSE_CV_media': rmse_cv.mean(),
                'RMSE_CV_std': rmse_cv.std(ddof=1),
                't_fit_s': t_fit, 't_pred_ms': t_pred}
        for j, tn in enumerate(target_names):
            fila[f'RMSE {tn}'] = rmse[j]
            fila[f'MAPE {tn} [%]'] = mape[j]
        filas.append(fila)
        predicciones[nombre] = y_pred
        ajustados[nombre] = modelo

        print(f'  {nombre:16s}: R2 = {r2:.5f} | '
              f'RMSE_CV = {rmse_cv.mean():.4f} +- {rmse_cv.std(ddof=1):.4f} | '
              f't_fit = {t_fit:6.1f} s | t_pred = {t_pred:.4f} ms/muestra')

    tabla = pd.DataFrame(filas)
    mejor = tabla.loc[tabla['R2'].idxmax(), 'modelo']
    print(f'\n  Mejor modelo ({nombre_problema}): {mejor} '
          f'(R2 = {tabla["R2"].max():.5f})')
    return tabla, predicciones, ajustados, mejor, (X_te, y_te)


tabla_iso, pred_iso, mod_iso, mejor_iso, (Xte_i, yte_i) = \
    evaluar_problema('isotropico', X_iso, y_iso, TARGETS_ISO)
tabla_orto, pred_orto, mod_orto, mejor_orto, (Xte_o, yte_o) = \
    evaluar_problema('ortotropico', X_orto, y_orto, TARGETS_ORTO)


# --- Identificacion fisica sobre los casos FEM reales (isotropico) ----------
ident_fem = []
if hay_fem:
    print(f'\n{"="*72}\n  IDENTIFICACION SOBRE DATOS FEM (isotropico)\n{"="*72}')
    for nombre, modelo in mod_iso.items():
        for mat, d in datos_fem.items():
            E_nom = MATERIALES[mat]['E'] / 1e9
            nu_nom = MATERIALES[mat]['nu']
            pred = modelo.predict(d['X'])
            ident_fem.append({
                'modelo': nombre, 'material': mat,
                'E_GPa': pred[:, 0].mean(),
                'err_E_pct': abs(pred[:, 0].mean() - E_nom) / E_nom * 100,
                'nu': pred[:, 1].mean(),
                'err_nu_pct': abs(pred[:, 1].mean() - nu_nom) / nu_nom * 100,
            })
    ident_fem = pd.DataFrame(ident_fem)
    resumen_fem = ident_fem.groupby('modelo')[
        ['err_E_pct', 'err_nu_pct']].mean().sort_values('err_E_pct')
    print(resumen_fem.round(3).to_string())
    mejor_fem = resumen_fem.index[0]
    print(f'\n  Mejor sobre datos FEM: {mejor_fem}')


# ==============================================================================
# [6] FIGURAS Y EXPORTACION
# ==============================================================================
print('\n[Figuras] Generando PDFs en', OUT_DIR.name, '...')
orden_modelos = list(fabrica_modelos().keys())


def fig_barras_rmse(tabla, target_names, fname, titulo):
    n_t = len(target_names)
    fig, ax = plt.subplots(1, n_t, figsize=(4.2 * n_t, 4.2), squeeze=False)
    ax = ax[0]
    xpos = np.arange(len(orden_modelos))
    for j, tn in enumerate(target_names):
        vals = [tabla.loc[tabla['modelo'] == m, f'RMSE {tn}'].iloc[0]
                for m in orden_modelos]
        ax[j].bar(xpos, vals, color=C1, edgecolor=C2, lw=0.6)
        ax[j].set_xticks(xpos)
        ax[j].set_xticklabels(orden_modelos, rotation=60, ha='right',
                              fontsize=7)
        ax[j].set_ylabel(f'RMSE {tn}')
        ax[j].set_yscale('log')
    fig.suptitle(titulo, y=1.0)
    fig.tight_layout()
    fig.savefig(OUT_DIR / fname, bbox_inches='tight')
    plt.close(fig)


fig_barras_rmse(tabla_iso, TARGETS_ISO, 'fig_rmse_isotropico.pdf',
                'RMSE en prueba por modelo: problema isotropico')
fig_barras_rmse(tabla_orto, TARGETS_ORTO[:3], 'fig_rmse_orto_E.pdf',
                'RMSE en prueba por modelo: ortotropico, modulos de Young')
fig_barras_rmse(tabla_orto, TARGETS_ORTO[3:], 'fig_rmse_orto_nu.pdf',
                'RMSE en prueba por modelo: ortotropico, razones de Poisson')

# --- Pareto tiempo vs error --------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.4))
for a, (tabla, titulo) in zip(
        ax, [(tabla_iso, 'Isotrópico'), (tabla_orto, 'Ortotrópico')]):
    a.scatter(tabla['t_fit_s'], tabla['RMSE_CV_media'],
              s=45, color=C1, edgecolor=C2, zorder=3)
    for _, r in tabla.iterrows():
        a.annotate(r['modelo'], (r['t_fit_s'], r['RMSE_CV_media']),
                   fontsize=6.5, xytext=(4, 4), textcoords='offset points')
    a.set_xscale('log'); a.set_yscale('log')
    a.set_xlabel('Tiempo de entrenamiento [s]')
    a.set_ylabel('RMSE (validación cruzada)')
    a.set_title(titulo)
fig.suptitle('Compromiso costo computacional vs precisión', y=1.02)
fig.tight_layout()
fig.savefig(OUT_DIR / 'fig_pareto_tiempo.pdf', bbox_inches='tight')
plt.close(fig)

# --- Paridad del mejor modelo ------------------------------------------------
def fig_paridad(y_true, y_pred, target_names, fname, titulo):
    n_t = len(target_names)
    ncol = min(n_t, 3)
    nrow = int(np.ceil(n_t / ncol))
    fig, ax = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.6 * nrow),
                           squeeze=False)
    for j, tn in enumerate(target_names):
        a = ax[j // ncol][j % ncol]
        a.plot(y_true[:, j], y_pred[:, j], '.', color=C1, ms=2, alpha=0.35)
        lim = [y_true[:, j].min(), y_true[:, j].max()]
        a.plot(lim, lim, '-', color=C2, lw=1)
        r2j = r2_score(y_true[:, j], y_pred[:, j])
        a.set_xlabel(f'{tn} real'); a.set_ylabel(f'{tn} predicho')
        a.set_title(f'$R^2$ = {r2j:.4f}', fontsize=9)
    for j in range(n_t, nrow * ncol):
        ax[j // ncol][j % ncol].axis('off')
    fig.suptitle(titulo, y=1.0)
    fig.tight_layout()
    fig.savefig(OUT_DIR / fname, bbox_inches='tight')
    plt.close(fig)


fig_paridad(yte_i, pred_iso[mejor_iso], TARGETS_ISO,
            'fig_paridad_isotropico.pdf',
            f'Paridad en prueba: {mejor_iso} (isotrópico)')
fig_paridad(yte_o, pred_orto[mejor_orto], TARGETS_ORTO,
            'fig_paridad_ortotropico.pdf',
            f'Paridad en prueba: {mejor_orto} (ortotrópico)')

# --- Identificacion FEM por material y modelo --------------------------------
if hay_fem and len(ident_fem):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    mats = list(datos_fem.keys())
    ancho = 0.8 / len(mats)
    xpos = np.arange(len(orden_modelos))
    colores = [C1, C2, C3, C4]
    for k, mat in enumerate(mats):
        sub = ident_fem[ident_fem['material'] == mat].set_index('modelo')
        ax[0].bar(xpos + k * ancho,
                  [sub.loc[m, 'err_E_pct'] for m in orden_modelos],
                  ancho, color=colores[k % 4], label=mat)
        ax[1].bar(xpos + k * ancho,
                  [sub.loc[m, 'err_nu_pct'] for m in orden_modelos],
                  ancho, color=colores[k % 4], label=mat)
    for a, lbl in zip(ax, ['$E$', r'$\nu$']):
        a.set_xticks(xpos + ancho * (len(mats) - 1) / 2)
        a.set_xticklabels(orden_modelos, rotation=60, ha='right', fontsize=7)
        a.set_ylabel(f'Error relativo en {lbl} [\%]')
        a.set_yscale('log')
    ax[0].legend(fontsize=7)
    fig.suptitle('Error de identificación sobre los casos FEM reales', y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig_identificacion_fem.pdf', bbox_inches='tight')
    plt.close(fig)

# --- Tablas -------------------------------------------------------------------
def fmt_coma(x, dec):
    return f'{x:.{dec}f}'.replace('.', ',')


def tabla_latex(tabla, target_names, fname, caption, label):
    filas = []
    for m in orden_modelos:
        r = tabla[tabla['modelo'] == m].iloc[0]
        rmse_cols = ' & '.join(fmt_coma(r[f'RMSE {tn}'], 4)
                               for tn in target_names)
        filas.append(
            f"{m} & {rmse_cols} & {fmt_coma(r['R2'], 4)} & "
            f"{fmt_coma(r['RMSE_CV_media'], 4)} $\\pm$ "
            f"{fmt_coma(r['RMSE_CV_std'], 4)} & "
            f"{fmt_coma(r['t_fit_s'], 1)} \\\\")
    enc = ' & '.join(f'RMSE {tn}' for tn in target_names)
    cols = 'l' + 'c' * (len(target_names) + 3)
    tex = (
        f"\\begin{{table}}[H]\n\\centering\n\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n\\begin{{tabular}}{{{cols}}}\n\\toprule\n"
        f"Modelo & {enc} & $R^2$ & RMSE CV & $t$ [s] \\\\\n\\midrule\n"
        + '\n'.join(filas) +
        "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    (OUT_DIR / fname).write_text(tex, encoding='utf-8')


tabla_iso.to_csv(OUT_DIR / 'comparacion_isotropico.csv', index=False)
tabla_orto.to_csv(OUT_DIR / 'comparacion_ortotropico.csv', index=False)
tabla_latex(tabla_iso, TARGETS_ISO, 'tabla_isotropico.tex',
            'Comparación de modelos: problema isotrópico '
            '($E$, $\\nu$; conjunto de prueba sintético).',
            'tab:ml_isotropico')
tabla_latex(tabla_orto, TARGETS_ORTO, 'tabla_ortotropico.tex',
            'Comparación de modelos: problema ortotrópico '
            '(6 parámetros; conjunto de prueba sintético).',
            'tab:ml_ortotropico')
if hay_fem and len(ident_fem):
    ident_fem.to_csv(OUT_DIR / 'identificacion_fem.csv', index=False)

lineas = [
    'COMPARACION DE MODELOS ML', '=' * 64,
    f'Versiones: numpy {np.__version__}, pandas {pd.__version__}, '
    f'scipy {scipy.__version__}, scikit-learn {sklearn.__version__}',
    f'Semilla: {SEED} | N_iso: {N_ISO} | N_orto: {N_ORTO} | '
    f'ruido: {RUIDO_REL*100:.1f} % | CV: {CV_FOLDS} folds',
    '',
    f'Mejor modelo isotropico  (R2 en test): {mejor_iso}',
    f'Mejor modelo ortotropico (R2 en test): {mejor_orto}',
]
if hay_fem and len(ident_fem):
    lineas.append(f'Mejor sobre datos FEM reales: {mejor_fem}')
lineas += ['', '--- Tabla isotropico ---', tabla_iso.round(5).to_string(),
           '', '--- Tabla ortotropico ---', tabla_orto.round(5).to_string()]
(OUT_DIR / 'metricas.txt').write_text('\n'.join(lineas), encoding='utf-8')

print('\nListo. Archivos generados en', OUT_DIR)
for f in sorted(OUT_DIR.iterdir()):
    print('  -', f.name)

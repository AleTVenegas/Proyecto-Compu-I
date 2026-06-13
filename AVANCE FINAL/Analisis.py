"""
================================================================================
 Analisis.py
 Identificacion inversa de parametros elasticos (E, nu) mediante red neuronal
 y FEMU, con generacion Monte Carlo del dataset y propagacion de incertidumbre
================================================================================
 Proyecto de Investigacion - IF4702 Fisica Computacional I - TEC

 FLUJO DEL ALGORITMO
 -------------------
 [1] Configuracion y reproducibilidad (semilla fija, parametros, versiones)
 [2] Carga de los CSV exportados por COMSOL (datos/AISI1020, datos/HierroDuctil)
 [3] Validacion fisica de los datos FEM contra la solucion analitica uniaxial:
         sigma_yy = -F/A,   eps_yy = sigma_yy/E,   eps_xx = eps_zz = -nu*eps_yy
 [4] Generacion del dataset de entrenamiento por muestreo Monte Carlo del
     espacio de parametros (E, nu, F) + ruido gaussiano relativo (robustez)
 [5] Entrenamiento de la red neuronal (MLP) para el mapeo inverso
         (sigma_yy, eps_yy, eps_xx, eps_zz, |v|max)  ->  (E, nu)
 [6] Analisis de convergencia y estabilidad:
         - curva de perdida del entrenamiento
         - error de generalizacion vs tamano del dataset (escala ~ N^-1/2)
         - sensibilidad del error al nivel de ruido en la entrada
 [7] Metodo inverso clasico FEMU (minimos cuadrados no lineales sobre el
     modelo directo) como referencia de comparacion
 [8] Evaluacion de ambos metodos sobre los 200 casos FEM de COMSOL
 [9] Propagacion de incertidumbre Monte Carlo sobre las predicciones
     (B replicas con ruido; intervalos de cobertura 95.45 %:
      percentiles 2.275 y 97.725) y convergencia del estimador MC
 [10] Exportacion: figuras PDF, resumen.csv, tabla LaTeX y metricas.txt

 PARAMETROS DEL ESTUDIO (reproducibilidad)
 -----------------------------------------
 Geometria      : cubo L = 10 mm, area cargada A = L^2 = 1e-4 m^2
 Cargas FEM     : F = 10, 20, ..., 1000 N (100 casos por material)
 Materiales     : AISI 1020 (E=200 GPa, nu=0.29), Hierro Ductil (E=120 GPa, nu=0.31)
 Filtro espacial: nodos con 0.25L <= y <= 0.95L (principio de Saint-Venant,
                  excluye la capa de borde de la cara empotrada)
 Dataset MC     : N = 8000 muestras, E ~ U(80, 260) GPa, nu ~ U(0.20, 0.40),
                  F ~ U(10, 1000) N, ruido relativo gaussiano del 1 %
 Red neuronal   : MLP (64, 64), tanh, Adam, early stopping (15 % validacion),
                  entradas y salidas estandarizadas
 FEMU           : scipy.optimize.least_squares (TRF), residuo relativo,
                  x0 = (150 GPa, 0.30), cotas E in [50,400] GPa, nu in [0.05,0.49]
 Monte Carlo    : B = 400 replicas por caso, ruido relativo 1 %
 Semilla global : 2026

 EJECUCION
 ---------
     pip install numpy pandas scipy scikit-learn matplotlib
     python Analisis.py

 Estructura esperada:   P1/Analisis.py,  P1/datos/<Material>/F####N.csv
 Salidas en             P1/resultados/
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

from scipy.optimize import least_squares
from scipy.stats import linregress
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.base import clone
import joblib

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ==============================================================================
# [1] CONFIGURACION
# ==============================================================================
SEED        = 2026
BASE_DIR    = Path(__file__).resolve().parent
DATA_DIR    = BASE_DIR / 'Datos_COMSOL'
OUT_DIR     = BASE_DIR / 'resultados'
OUT_DIR.mkdir(exist_ok=True)

L     = 10e-3
AREA  = L * L
Y_MIN, Y_MAX = 0.25 * L, 0.95 * L

MATERIALES = {
    'AISI1020':     {'E': 200e9, 'nu': 0.29},
    'HierroDuctil': {'E': 120e9, 'nu': 0.31},
}

COLS = ['x', 'y', 'z',
        'sx', 'sy', 'sz', 'sxy', 'sxz', 'syz',
        'eX', 'eY', 'eZ', 'eXY', 'eXZ', 'eYZ',
        'u', 'v', 'w']

FEATURES = ['sigma_yy', 'eps_yy', 'eps_xx', 'eps_zz', 'v_max']

N_TRAIN     = 8000
E_RANGO     = (80e9, 260e9)
NU_RANGO    = (0.20, 0.40)
F_RANGO     = (10.0, 1000.0)
RUIDO_REL   = 0.01
B_MC        = 400

rng = np.random.default_rng(SEED)

C1, C2, C3 = '#7aaa97', '#05242f', '#c46d5e'
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

print('=' * 70)
print('Identificacion inversa de (E, nu): red neuronal MLP vs FEMU')
print('=' * 70)
print(f'numpy {np.__version__} | pandas {pd.__version__} | '
      f'scipy {scipy.__version__} | scikit-learn {sklearn.__version__}')
print(f'Semilla = {SEED}\n')


# ==============================================================================
# [2] CARGA DE DATOS COMSOL
# ==============================================================================
def cargar_csv(path):
    """Lee un export 'Data' de COMSOL (lineas de encabezado con %)."""
    df = pd.read_csv(path, comment='%', header=None, sep=',', engine='python')
    if df.shape[1] == 1:
        df = pd.read_csv(path, comment='%', header=None,
                         sep=r'\s+', engine='python')
    df = df.iloc[:, :len(COLS)]
    df.columns = COLS[:df.shape[1]]
    return df.apply(pd.to_numeric, errors='coerce').dropna()


def extraer_features(df):
    """Estadisticos del campo en el sub-volumen de Saint-Venant -> vector (5,)."""
    nucleo = df[(df['y'] >= Y_MIN) & (df['y'] <= Y_MAX)]
    if len(nucleo) < 20:
        nucleo = df
    return np.array([
        nucleo['sy'].mean(),
        nucleo['eY'].mean(),
        nucleo['eX'].mean(),
        nucleo['eZ'].mean(),
        df['v'].abs().max(),
    ])


def cargar_material(nombre):
    """Carga todos los F####N.csv de un material. Devuelve (F, X) ordenados."""
    carpeta = DATA_DIR / nombre
    if not carpeta.is_dir():
        return np.array([]), np.empty((0, 5))
    F, X = [], []
    for f in sorted(carpeta.glob('*.csv')):
        m = re.search(r'F(\d+)N', f.name)
        if not m:
            continue
        F.append(float(m.group(1)))
        X.append(extraer_features(cargar_csv(f)))
    if not F:
        return np.array([]), np.empty((0, 5))
    F = np.asarray(F)
    X = np.vstack(X)
    orden = np.argsort(F)
    return F[orden], X[orden]


datos = {}
for mat in MATERIALES:
    F_mat, X_mat = cargar_material(mat)
    datos[mat] = {'F': F_mat, 'X': X_mat}
    print(f'[Datos] {mat:13s}: {len(F_mat):3d} simulaciones cargadas')

hay_datos = any(len(d['F']) > 0 for d in datos.values())
if not hay_datos:
    print('\nAviso: no se encontraron CSV en', DATA_DIR)
    print('Se ejecutara unicamente la parte sintetica (secciones 4-7).\n')


# ==============================================================================
# [3] VALIDACION FISICA: FEM vs SOLUCION ANALITICA
# ==============================================================================
def modelo_directo(E, nu, F):
    """Solucion analitica de compresion uniaxial. Vectorizado.
    Devuelve las mismas 5 caracteristicas que extraer_features()."""
    E, nu, F = np.broadcast_arrays(np.atleast_1d(E),
                                   np.atleast_1d(nu),
                                   np.atleast_1d(F))
    sy   = -F / AREA
    eY   = sy / E
    eX   = -nu * eY
    vmax = np.abs(eY) * L
    return np.column_stack([sy, eY, eX, eX, vmax])


validacion = {}
if hay_datos:
    print('\n[Validacion FEM vs analitica]')
    for mat, props in MATERIALES.items():
        F_mat, X_mat = datos[mat]['F'], datos[mat]['X']
        if len(F_mat) == 0:
            continue
        ana = modelo_directo(props['E'], props['nu'], F_mat)
        err_sigma = np.abs(X_mat[:, 0] - ana[:, 0]) / np.abs(ana[:, 0]) * 100
        err_eps   = np.abs(X_mat[:, 1] - ana[:, 1]) / np.abs(ana[:, 1]) * 100
        reg = linregress(F_mat, X_mat[:, 1])
        E_lin = (-1 / AREA) / reg.slope
        validacion[mat] = {
            'err_sigma_max': err_sigma.max(),
            'err_eps_max': err_eps.max(),
            'r2_linealidad': reg.rvalue ** 2,
            'E_pendiente': E_lin,
        }
        print(f'  {mat:13s}: err max sigma_yy = {err_sigma.max():.3f} % | '
              f'err max eps_yy = {err_eps.max():.3f} % | '
              f'R2 linealidad = {reg.rvalue**2:.6f} | '
              f'E (pendiente) = {E_lin/1e9:.2f} GPa')


# ==============================================================================
# [4] DATASET DE ENTRENAMIENTO POR MUESTREO MONTE CARLO
# ==============================================================================
print('\n[Monte Carlo] Generando dataset sintetico de entrenamiento...')
E_s  = rng.uniform(*E_RANGO,  N_TRAIN)
nu_s = rng.uniform(*NU_RANGO, N_TRAIN)
F_s  = rng.uniform(*F_RANGO,  N_TRAIN)

X_sint = modelo_directo(E_s, nu_s, F_s)
X_sint = X_sint * (1.0 + rng.normal(0.0, RUIDO_REL, X_sint.shape))
y_sint = np.column_stack([E_s / 1e9, nu_s])

X_tr, X_te, y_tr, y_te = train_test_split(
    X_sint, y_sint, test_size=0.20, random_state=SEED)
print(f'  N_train = {len(X_tr)} | N_test = {len(X_te)} | '
      f'ruido relativo = {RUIDO_REL*100:.1f} %')


# ==============================================================================
# [5] RED NEURONAL (MLP)
# ==============================================================================
def construir_red(seed=SEED):
    mlp = MLPRegressor(hidden_layer_sizes=(64, 64), activation='tanh',
                       solver='adam', max_iter=3000, tol=1e-8,
                       early_stopping=True, validation_fraction=0.15,
                       n_iter_no_change=50, random_state=seed)
    pipe = Pipeline([('escala', StandardScaler()), ('red', mlp)])
    return TransformedTargetRegressor(regressor=pipe,
                                      transformer=StandardScaler())


print('\n[Red neuronal] Entrenando MLP (64, 64) tanh + Adam...')
t0 = time.time()
modelo_nn = construir_red()
modelo_nn.fit(X_tr, y_tr)
t_fit = time.time() - t0

y_pred_te = modelo_nn.predict(X_te)
r2_E   = r2_score(y_te[:, 0], y_pred_te[:, 0])
r2_nu  = r2_score(y_te[:, 1], y_pred_te[:, 1])
rmse_E  = np.sqrt(mean_squared_error(y_te[:, 0], y_pred_te[:, 0]))
rmse_nu = np.sqrt(mean_squared_error(y_te[:, 1], y_pred_te[:, 1]))
mape_E  = np.mean(np.abs((y_pred_te[:, 0] - y_te[:, 0]) / y_te[:, 0])) * 100
mape_nu = np.mean(np.abs((y_pred_te[:, 1] - y_te[:, 1]) / y_te[:, 1])) * 100

red_interna = modelo_nn.regressor_.named_steps['red']
print(f'  Entrenamiento: {t_fit:.1f} s | iteraciones = {red_interna.n_iter_}')
print(f'  Test  E : R2 = {r2_E:.5f} | RMSE = {rmse_E:.3f} GPa | MAPE = {mape_E:.3f} %')
print(f'  Test  nu: R2 = {r2_nu:.5f} | RMSE = {rmse_nu:.5f}     | MAPE = {mape_nu:.3f} %')

joblib.dump(modelo_nn, OUT_DIR / 'modelo_nn.joblib')


# ==============================================================================
# [6] CONVERGENCIA Y ESTABILIDAD
# ==============================================================================
print('\n[Convergencia] Error de generalizacion vs tamano del dataset...')
tamanos = np.array([250, 500, 1000, 2000, 4000, len(X_tr)])
rmse_vs_N = []
for n in tamanos:
    m = clone(modelo_nn)
    m.fit(X_tr[:n], y_tr[:n])
    p = m.predict(X_te)
    rmse_vs_N.append([np.sqrt(mean_squared_error(y_te[:, 0], p[:, 0])),
                      np.sqrt(mean_squared_error(y_te[:, 1], p[:, 1]))])
    print(f'  N = {n:5d}: RMSE_E = {rmse_vs_N[-1][0]:.3f} GPa | '
          f'RMSE_nu = {rmse_vs_N[-1][1]:.5f}')
rmse_vs_N = np.asarray(rmse_vs_N)

print('\n[Estabilidad] Sensibilidad al nivel de ruido en la entrada...')
niveles = np.array([0.0, 0.005, 0.01, 0.02, 0.05])
X_te_limpio = modelo_directo(y_te[:, 0] * 1e9, y_te[:, 1],
                             rng.uniform(*F_RANGO, len(y_te)))
rmse_vs_ruido = []
for s in niveles:
    Xp = X_te_limpio * (1.0 + rng.normal(0.0, s, X_te_limpio.shape))
    p = modelo_nn.predict(Xp)
    rmse_vs_ruido.append([np.sqrt(mean_squared_error(y_te[:, 0], p[:, 0])),
                          np.sqrt(mean_squared_error(y_te[:, 1], p[:, 1]))])
    print(f'  ruido = {s*100:4.1f} %: RMSE_E = {rmse_vs_ruido[-1][0]:.3f} GPa | '
          f'RMSE_nu = {rmse_vs_ruido[-1][1]:.5f}')
rmse_vs_ruido = np.asarray(rmse_vs_ruido)


# ==============================================================================
# [7] FEMU: MINIMOS CUADRADOS NO LINEALES SOBRE EL MODELO DIRECTO
# ==============================================================================
def femu(obs, F):
    """Identifica (E [GPa], nu) minimizando el residuo relativo."""
    def residuo(p):
        mod = modelo_directo(p[0] * 1e9, p[1], F)[0]
        return (mod - obs) / (np.abs(obs) + 1e-30)
    sol = least_squares(residuo, x0=[150.0, 0.30],
                        bounds=([50.0, 0.05], [400.0, 0.49]), method='trf')
    return sol.x


# ==============================================================================
# [8] EVALUACION SOBRE LOS DATOS FEM DE COMSOL
# ==============================================================================
resultados = []
if hay_datos:
    print('\n[Evaluacion sobre datos COMSOL]')
    for mat, props in MATERIALES.items():
        F_mat, X_mat = datos[mat]['F'], datos[mat]['X']
        if len(F_mat) == 0:
            continue
        E_nom, nu_nom = props['E'] / 1e9, props['nu']

        t0 = time.time()
        pred_nn = modelo_nn.predict(X_mat)
        t_nn = time.time() - t0

        t0 = time.time()
        pred_fe = np.array([femu(X_mat[i], F_mat[i]) for i in range(len(F_mat))])
        t_fe = time.time() - t0

        datos[mat]['pred_nn'] = pred_nn
        datos[mat]['pred_fe'] = pred_fe
        datos[mat]['t_nn'], datos[mat]['t_fe'] = t_nn, t_fe

        for met, pred, t_met in (('NN', pred_nn, t_nn), ('FEMU', pred_fe, t_fe)):
            E_m, nu_m = pred[:, 0].mean(), pred[:, 1].mean()
            resultados.append({
                'material': mat, 'metodo': met,
                'E_GPa': E_m, 'E_std': pred[:, 0].std(ddof=1),
                'err_E_pct': abs(E_m - E_nom) / E_nom * 100,
                'nu': nu_m, 'nu_std': pred[:, 1].std(ddof=1),
                'err_nu_pct': abs(nu_m - nu_nom) / nu_nom * 100,
                't_total_s': t_met,
            })
            print(f'  {mat:13s} [{met:4s}]: '
                  f'E = {E_m:7.2f} GPa (err {abs(E_m-E_nom)/E_nom*100:.3f} %) | '
                  f'nu = {nu_m:.4f} (err {abs(nu_m-nu_nom)/nu_nom*100:.3f} %) | '
                  f't = {t_met:.3f} s')


# ==============================================================================
# [9] PROPAGACION DE INCERTIDUMBRE MONTE CARLO
# ==============================================================================
mc = {}
if hay_datos:
    print(f'\n[Monte Carlo] Propagacion de incertidumbre '
          f'(B = {B_MC} replicas/caso, ruido {RUIDO_REL*100:.0f} %)...')
    for mat, props in MATERIALES.items():
        X_mat = datos[mat]['X']
        if len(X_mat) == 0:
            continue
        X_rep = np.repeat(X_mat, B_MC, axis=0)
        X_rep = X_rep * (1.0 + rng.normal(0.0, RUIDO_REL, X_rep.shape))
        pred  = modelo_nn.predict(X_rep)
        q_E   = np.percentile(pred[:, 0], [2.275, 97.725])
        q_nu  = np.percentile(pred[:, 1], [2.275, 97.725])
        mc[mat] = {'pred': pred, 'q_E': q_E, 'q_nu': q_nu}
        print(f'  {mat:13s}: E  = {pred[:,0].mean():7.2f} GPa, '
              f'IC 95.45 % = [{q_E[0]:.2f}, {q_E[1]:.2f}] GPa')
        print(f'  {"":13s}  nu = {pred[:,1].mean():7.4f},   '
              f'IC 95.45 % = [{q_nu[0]:.4f}, {q_nu[1]:.4f}]')


# ==============================================================================
# [10] FIGURAS Y EXPORTACION
# ==============================================================================
print('\n[Figuras] Generando PDFs en', OUT_DIR.name, '...')

# --- Fig 1: validacion FEM vs analitica -------------------------------------
if hay_datos:
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for mat, props in MATERIALES.items():
        F_mat, X_mat = datos[mat]['F'], datos[mat]['X']
        if len(F_mat) == 0:
            continue
        c = C1 if mat == 'AISI1020' else C2
        ax[0].plot(F_mat, -F_mat / AREA / 1e6, '-', color=c, lw=1)
        ax[0].plot(F_mat, X_mat[:, 0] / 1e6, 'o', color=c, ms=3,
                   label=f'{mat} (FEM)')
        ax[1].plot(F_mat, -F_mat / AREA / props['E'] * 1e6, '-', color=c, lw=1)
        ax[1].plot(F_mat, X_mat[:, 1] * 1e6, 'o', color=c, ms=3,
                   label=f'{mat} (FEM)')
    ax[0].set_xlabel('$F$ [N]'); ax[0].set_ylabel(r'$\bar\sigma_{yy}$ [MPa]')
    ax[1].set_xlabel('$F$ [N]'); ax[1].set_ylabel(r'$\bar\varepsilon_{yy}\times 10^{6}$')
    ax[0].set_title('Esfuerzo axial: FEM vs $-F/A$')
    ax[1].set_title('Deformación axial: FEM vs $-F/(AE)$')
    ax[0].legend(fontsize=8); ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig_validacion_fem.pdf'); plt.close(fig)

# --- Fig 2: convergencia del entrenamiento ----------------------------------
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].semilogy(red_interna.loss_curve_, color=C2, lw=1.2)
ax[0].set_xlabel('Iteración'); ax[0].set_ylabel('Pérdida (MSE escalado)')
ax[0].set_title('Curva de entrenamiento')
ax[1].loglog(tamanos, rmse_vs_N[:, 0], 'o-', color=C1, label='$E$ [GPa]')
ref = rmse_vs_N[0, 0] * (tamanos / tamanos[0]) ** -0.5
ax[1].loglog(tamanos, ref, '--', color=C3, lw=1, label=r'$\propto N^{-1/2}$')
ax[1].set_xlabel('$N$ entrenamiento'); ax[1].set_ylabel('RMSE en test')
ax[1].set_title('Convergencia con el tamaño de muestra')
ax[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT_DIR / 'fig_convergencia_entrenamiento.pdf'); plt.close(fig)

# --- Fig 3: paridad en el conjunto de prueba sintetico ----------------------
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(y_te[:, 0], y_pred_te[:, 0], '.', color=C1, ms=2, alpha=0.4)
lim = [y_te[:, 0].min(), y_te[:, 0].max()]
ax[0].plot(lim, lim, '-', color=C2, lw=1)
ax[0].set_xlabel(r'$E$ real [GPa]'); ax[0].set_ylabel(r'$E$ predicho [GPa]')
ax[0].set_title(f'$E$: $R^2$ = {r2_E:.4f}')
ax[1].plot(y_te[:, 1], y_pred_te[:, 1], '.', color=C1, ms=2, alpha=0.4)
lim = [y_te[:, 1].min(), y_te[:, 1].max()]
ax[1].plot(lim, lim, '-', color=C2, lw=1)
ax[1].set_xlabel(r'$\nu$ real'); ax[1].set_ylabel(r'$\nu$ predicho')
ax[1].set_title(rf'$\nu$: $R^2$ = {r2_nu:.4f}')
fig.tight_layout()
fig.savefig(OUT_DIR / 'fig_paridad_test.pdf'); plt.close(fig)

# --- Fig 4: estabilidad frente al ruido -------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(niveles * 100, rmse_vs_ruido[:, 0], 'o-', color=C1)
ax[0].set_xlabel('Ruido relativo [%]'); ax[0].set_ylabel('RMSE $E$ [GPa]')
ax[1].plot(niveles * 100, rmse_vs_ruido[:, 1], 'o-', color=C2)
ax[1].set_xlabel('Ruido relativo [%]'); ax[1].set_ylabel(r'RMSE $\nu$')
fig.suptitle('Estabilidad de la red frente a ruido en la entrada', y=1.02)
fig.tight_layout()
fig.savefig(OUT_DIR / 'fig_estabilidad_ruido.pdf', bbox_inches='tight')
plt.close(fig)

# --- Fig 5: evaluacion sobre COMSOL (NN vs FEMU) ----------------------------
if hay_datos and resultados:
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for mat, props in MATERIALES.items():
        if 'pred_nn' not in datos[mat]:
            continue
        F_mat = datos[mat]['F']
        c = C1 if mat == 'AISI1020' else C2
        ax[0].plot(F_mat, datos[mat]['pred_nn'][:, 0], 'o', color=c, ms=3,
                   label=f'{mat} NN')
        ax[0].plot(F_mat, datos[mat]['pred_fe'][:, 0], 'x', color=c, ms=4,
                   alpha=0.6, label=f'{mat} FEMU')
        ax[0].axhline(props['E'] / 1e9, color=c, ls='--', lw=0.8)
        ax[1].plot(F_mat, datos[mat]['pred_nn'][:, 1], 'o', color=c, ms=3)
        ax[1].plot(F_mat, datos[mat]['pred_fe'][:, 1], 'x', color=c, ms=4,
                   alpha=0.6)
        ax[1].axhline(props['nu'], color=c, ls='--', lw=0.8)
    ax[0].set_xlabel('$F$ [N]'); ax[0].set_ylabel('$\\hat{E}$ [GPa]')
    ax[1].set_xlabel('$F$ [N]'); ax[1].set_ylabel(r'$\hat{\nu}$')
    ax[0].legend(fontsize=7)
    fig.suptitle('Parámetros identificados en los 100 casos FEM por material',
                 y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig_evaluacion_comsol.pdf', bbox_inches='tight')
    plt.close(fig)

# --- Fig 6: histogramas Monte Carlo -----------------------------------------
if mc:
    fig, ax = plt.subplots(2, 2, figsize=(10, 7))
    for j, (mat, props) in enumerate(MATERIALES.items()):
        if mat not in mc:
            continue
        pred, q_E, q_nu = mc[mat]['pred'], mc[mat]['q_E'], mc[mat]['q_nu']
        ax[j, 0].hist(pred[:, 0], bins=80, color=C1, edgecolor='none')
        ax[j, 0].axvline(props['E'] / 1e9, color=C3, lw=1.4, label='nominal')
        for q in q_E:
            ax[j, 0].axvline(q, color=C2, ls='--', lw=1)
        ax[j, 0].set_xlabel('$\\hat{E}$ [GPa]'); ax[j, 0].set_title(mat)
        ax[j, 0].legend(fontsize=8)
        ax[j, 1].hist(pred[:, 1], bins=80, color=C1, edgecolor='none')
        ax[j, 1].axvline(props['nu'], color=C3, lw=1.4)
        for q in q_nu:
            ax[j, 1].axvline(q, color=C2, ls='--', lw=1)
        ax[j, 1].set_xlabel(r'$\hat{\nu}$'); ax[j, 1].set_title(mat)
    fig.suptitle('Monte Carlo: distribuciones con percentiles 2.275 / 97.725 %',
                 y=1.0)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig_mc_incertidumbre.pdf'); plt.close(fig)

# --- Fig 7: convergencia del estimador Monte Carlo --------------------------
if mc:
    fig, ax = plt.subplots(figsize=(6, 4))
    for mat, props in MATERIALES.items():
        if mat not in mc:
            continue
        c = C1 if mat == 'AISI1020' else C2
        serie = mc[mat]['pred'][:20000, 0]
        acum = np.cumsum(serie) / np.arange(1, len(serie) + 1)
        ax.plot(acum, color=c, lw=1, label=mat)
        ax.axhline(props['E'] / 1e9, color=c, ls='--', lw=0.8)
    ax.set_xlabel('Número de réplicas MC')
    ax.set_ylabel('Media acumulada de $\\hat{E}$ [GPa]')
    ax.set_title('Convergencia del estimador Monte Carlo')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig_mc_convergencia.pdf'); plt.close(fig)


# --- Tablas y metricas -------------------------------------------------------
def fmt_coma(x, dec):
    return f'{x:.{dec}f}'.replace('.', ',')


if resultados:
    df_res = pd.DataFrame(resultados)
    df_res.to_csv(OUT_DIR / 'resumen.csv', index=False)

    filas = []
    for r in resultados:
        filas.append(
            f"{r['material']} & {r['metodo']} & "
            f"{fmt_coma(r['E_GPa'], 2)} & {fmt_coma(r['err_E_pct'], 3)} & "
            f"{fmt_coma(r['nu'], 4)} & {fmt_coma(r['err_nu_pct'], 3)} & "
            f"{fmt_coma(r['t_total_s'], 3)} \\\\")
    tabla = (
        "\\begin{table}[H]\n\\centering\n"
        "\\caption{Parámetros elásticos identificados sobre los 100 casos FEM "
        "por material (valores medios).}\n"
        "\\label{tab:resultados_identificacion}\n"
        "\\begin{tabular}{llccccc}\n\\toprule\n"
        "Material & Método & $\\hat{E}$ [GPa] & $e_{E}$ [\\%] & "
        "$\\hat{\\nu}$ & $e_{\\nu}$ [\\%] & $t$ [s] \\\\\n\\midrule\n"
        + "\n".join(filas) +
        "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    (OUT_DIR / 'tabla_resultados.tex').write_text(tabla, encoding='utf-8')

lineas = [
    'METRICAS DEL ESTUDIO', '=' * 60,
    f'Versiones: numpy {np.__version__}, pandas {pd.__version__}, '
    f'scipy {scipy.__version__}, scikit-learn {sklearn.__version__}',
    f'Semilla: {SEED} | N_train: {len(X_tr)} | ruido: {RUIDO_REL*100:.1f} % | '
    f'B_MC: {B_MC}',
    f'Red: MLP (64,64) tanh, Adam, early stopping | iteraciones: '
    f'{red_interna.n_iter_} | t_fit: {t_fit:.1f} s',
    '',
    f'Test sintetico  E : R2 = {r2_E:.5f}, RMSE = {rmse_E:.3f} GPa, '
    f'MAPE = {mape_E:.3f} %',
    f'Test sintetico  nu: R2 = {r2_nu:.5f}, RMSE = {rmse_nu:.5f}, '
    f'MAPE = {mape_nu:.3f} %',
    '',
]
for mat, v in validacion.items():
    lineas.append(
        f'Validacion FEM {mat}: err max sigma = {v["err_sigma_max"]:.3f} %, '
        f'err max eps = {v["err_eps_max"]:.3f} %, '
        f'R2 lin = {v["r2_linealidad"]:.6f}, '
        f'E pendiente = {v["E_pendiente"]/1e9:.2f} GPa')
for r in resultados:
    lineas.append(
        f'{r["material"]} [{r["metodo"]}]: E = {r["E_GPa"]:.2f} GPa '
        f'(err {r["err_E_pct"]:.3f} %), nu = {r["nu"]:.4f} '
        f'(err {r["err_nu_pct"]:.3f} %), t = {r["t_total_s"]:.3f} s')
for mat, v in mc.items():
    lineas.append(
        f'MC {mat}: IC95.45 E = [{v["q_E"][0]:.2f}, {v["q_E"][1]:.2f}] GPa, '
        f'IC95.45 nu = [{v["q_nu"][0]:.4f}, {v["q_nu"][1]:.4f}]')
(OUT_DIR / 'metricas.txt').write_text('\n'.join(lineas), encoding='utf-8')

print('\nListo. Archivos generados en', OUT_DIR)
for f in sorted(OUT_DIR.iterdir()):
    print('  -', f.name)

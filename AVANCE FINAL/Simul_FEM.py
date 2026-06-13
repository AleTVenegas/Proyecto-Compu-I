"""
Simul_FEM.py
============
Genera datos FEM de compresion uniaxial en cubo 10x10x10 mm
usando scikit-fem (Python puro, sin licencias).

  pip install scikit-fem numpy scipy pandas

Materiales:
  AISI 1020          E = 200 GPa, nu = 0.29
  Hierro Ductil      E = 120 GPa, nu = 0.31
  Aluminio 6061      E =  69 GPa, nu = 0.33
  Titanio Ti-6Al-4V  E = 114 GPa, nu = 0.34

Produce:
  datos/<Material>/F####N.csv  (formato identico al export de COMSOL)
"""

import numpy as np
import pandas as pd
import os
import time
from pathlib import Path
from scipy.sparse.linalg import spsolve
from skfem import (MeshTet, Basis, FacetBasis, ElementVectorH1,
                   ElementTetP1, LinearForm, asm, enforce)
from skfem.models.elasticity import linear_elasticity, lame_parameters

# =============================================================
# CONFIGURACION
# =============================================================
L       = 10e-3                       # lado del cubo [m]
AREA    = L * L                       # area de la cara cargada [m^2]
N_MESH  = 10                          # divisiones por lado (N+1 nodos por eje)
FUERZAS = np.arange(10, 1010, 10)     # 10 N ... 1000 N -> 100 casos

BASE_DIR   = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / 'Datos_COMSOL'

MATERIALES = {
    'AISI1020':      {'E': 200e9, 'nu': 0.29, 'rho': 7870},
    'HierroDuctil':  {'E': 120e9, 'nu': 0.31, 'rho': 7150},
    'Aluminio6061':  {'E':  69e9, 'nu': 0.33, 'rho': 2700},
    'TitanioTi6Al4V':{'E': 114e9, 'nu': 0.34, 'rho': 4430},
}

HEADER_COLS = (
    'x (m),y (m),z (m),'
    'solid.sx (Pa),solid.sy (Pa),solid.sz (Pa),'
    'solid.sxy (Pa),solid.sxz (Pa),solid.syz (Pa),'
    'solid.eX (1),solid.eY (1),solid.eZ (1),'
    'solid.eXY (1),solid.eXZ (1),solid.eYZ (1),'
    'u (m),v (m),w (m)'
)

# =============================================================
# CONSTRUCCION DE MALLA (una sola vez, reutilizada para todos los casos)
# =============================================================
print('Construyendo malla FEM...')
_g = np.linspace(0, L, N_MESH + 1)
_mesh_base = MeshTet.init_tensor(_g, _g, _g).with_boundaries({
    'bot': lambda x: x[1] < 1e-10,
    'top': lambda x: x[1] > L - 1e-10,
})
_e           = ElementVectorH1(ElementTetP1())
_basis       = Basis(_mesh_base, _e)
_top_basis   = FacetBasis(_mesh_base, _e, facets=_mesh_base.boundaries['top'])
_dofs_bot    = _basis.get_dofs(_mesh_base.boundaries['bot'])
_n_nodes     = _mesh_base.p.shape[1]
_n_elem      = _mesh_base.t.shape[1]

# Precomputar B_inv para todos los elementos (depende solo de la geometria)
_pts  = _mesh_base.p[:, _mesh_base.t]       # (3, 4, n_elem)
_Amat = np.ones((_n_elem, 4, 4))
_Amat[:, :, 1] = _pts[0].T
_Amat[:, :, 2] = _pts[1].T
_Amat[:, :, 3] = _pts[2].T
_Binv = np.linalg.inv(_Amat)                # (n_elem, 4, 4)
_dNdx = _Binv[:, 1, :]                      # (n_elem, 4)
_dNdy = _Binv[:, 2, :]
_dNdz = _Binv[:, 3, :]

_coords = _mesh_base.p.T                    # (n_nodes, 3)
print(f'  Nodos: {_n_nodes} | Elementos: {_n_elem}')


def _nod_avg(e_elem):
    """Promedio nodal simple (Superconvergent Patch Recovery simplificado)."""
    acc = np.zeros(_n_nodes)
    cnt = np.zeros(_n_nodes)
    for i in range(4):
        np.add.at(acc, _mesh_base.t[i, :], e_elem)
        np.add.at(cnt, _mesh_base.t[i, :], 1)
    return acc / cnt


# =============================================================
# SOLVER FEM
# =============================================================
def resolver_fem(E_val, nu_val, F_val):
    """
    Resuelve la elasticidad lineal estatica en el cubo.

    Condiciones de contorno
    -----------------------
    Cara y = 0  : Dirichlet homogeneo (empotrada, u=v=w=0)
    Cara y = L  : Neumann, presion uniforme = -F/A en direccion y

    Devuelve
    --------
    DataFrame con 18 columnas identicas al export de COMSOL.
    """
    lam, mu = lame_parameters(E_val, nu_val)
    K = asm(linear_elasticity(lam, mu), _basis)

    @LinearForm
    def lf(v, w):
        return -F_val / AREA * v.value[1]

    f   = asm(lf, _top_basis)
    K2, f2 = enforce(K, f, D=_dofs_bot)
    u_vec   = spsolve(K2, f2)

    ux = u_vec[_basis.nodal_dofs[0]]
    uy = u_vec[_basis.nodal_dofs[1]]
    uz = u_vec[_basis.nodal_dofs[2]]

    # --- Deformaciones por elemento (B lineal, constante en cada tet) ---
    ux_e = ux[_mesh_base.t].T   # (n_elem, 4)
    uy_e = uy[_mesh_base.t].T
    uz_e = uz[_mesh_base.t].T

    exx_e = (_dNdx * ux_e).sum(1)
    eyy_e = (_dNdy * uy_e).sum(1)
    ezz_e = (_dNdz * uz_e).sum(1)
    exy_e = 0.5 * ((_dNdy * ux_e).sum(1) + (_dNdx * uy_e).sum(1))
    exz_e = 0.5 * ((_dNdz * ux_e).sum(1) + (_dNdx * uz_e).sum(1))
    eyz_e = 0.5 * ((_dNdz * uy_e).sum(1) + (_dNdy * uz_e).sum(1))

    # --- Promedio nodal ---
    exx = _nod_avg(exx_e); eyy = _nod_avg(eyy_e); ezz = _nod_avg(ezz_e)
    exy = _nod_avg(exy_e); exz = _nod_avg(exz_e); eyz = _nod_avg(eyz_e)

    # --- Esfuerzos (ley de Hooke isotropa) ---
    trE = exx + eyy + ezz
    sxx = lam * trE + 2 * mu * exx
    syy = lam * trE + 2 * mu * eyy
    szz = lam * trE + 2 * mu * ezz
    sxy = 2 * mu * exy
    sxz = 2 * mu * exz
    syz = 2 * mu * eyz

    return pd.DataFrame({
        'x': _coords[:, 0], 'y': _coords[:, 1], 'z': _coords[:, 2],
        'sx': sxx, 'sy': syy, 'sz': szz,
        'sxy': sxy, 'sxz': sxz, 'syz': syz,
        'eX': exx, 'eY': eyy, 'eZ': ezz,
        'eXY': exy, 'eXZ': exz, 'eYZ': eyz,
        'u': ux, 'v': uy, 'w': uz,
    })


# =============================================================
# LOOP PRINCIPAL
# =============================================================
print('\nIniciando simulaciones...\n')
t_global = time.time()

for nombre, props in MATERIALES.items():
    E_val, nu_val = props['E'], props['nu']
    carpeta = OUTPUT_DIR / nombre
    carpeta.mkdir(parents=True, exist_ok=True)

    print(f'{"="*55}')
    print(f'Material : {nombre}')
    print(f'E = {E_val/1e9:.0f} GPa  |  nu = {nu_val}')
    print(f'{"="*55}')

    t_mat = time.time()
    for i, F in enumerate(FUERZAS):
        t0 = time.time()
        df = resolver_fem(E_val, nu_val, float(F))

        fname = carpeta / f'F{int(F):04d}N.csv'
        with open(fname, 'w') as fh:
            fh.write(f'% Material: {nombre}\n')
            fh.write(f'% E = {E_val:.3e} Pa | nu = {nu_val} | F = {F:.1f} N\n')
            fh.write(f'% Solver: scikit-fem {__import__("skfem").__version__}, '
                     f'N_mesh = {N_MESH}\n')
            fh.write(f'% {HEADER_COLS}\n')
        df.to_csv(fname, mode='a', header=False, index=False,
                  float_format='%.10e')

        dt = time.time() - t0
        print(f'  [{i+1:3d}/100]  F = {F:6.0f} N  OK  ({dt:.2f} s)')

    t_mat_total = time.time() - t_mat
    print(f'\n  -> {nombre} completado en {t_mat_total:.1f} s\n')

t_fin = time.time() - t_global
n_sims = len(FUERZAS) * len(MATERIALES)
print(f'Listo: {n_sims} simulaciones en {t_fin/60:.1f} min')
print(f'CSVs en: {OUTPUT_DIR}')

import numpy as np
import pandas as pd
import io, os
from scipy.spatial import cKDTree

CARPETA = r'C:\Users\johgranados\Downloads\Proyecto compu\Datos Aisi1020'
SALIDA  = r'C:\Users\johgranados\Downloads\Proyecto compu'

def leer_sw(num, col):
    # Intentar con guión bajo y con espacio
    for nombre in [f'Análisis_estático_1-{num}.csv',
                   f'Análisis estático 1-{num}.csv',
                   f'Analisis_estatico_1-{num}.csv',
                   f'Analisis estatico 1-{num}.csv']:
        path = os.path.join(CARPETA, nombre)
        if os.path.exists(path):
            with open(path, 'rb') as f:
                raw = f.read().decode('latin-1')
            lines = raw.split('\r\n')
            text = '\n'.join(lines[5:])
            df = pd.read_csv(io.StringIO(text), sep=';', decimal=',',
                             header=0, names=['ID','X_mm','Y_mm','Z_mm', col])
            return df.dropna().reset_index(drop=True)
    # Si no encontró, listar archivos para debug
    print(f"ERROR: No se encontró el archivo {num}. Archivos en carpeta:")
    for f in sorted(os.listdir(CARPETA)):
        print(f"  {f}")
    raise FileNotFoundError(f"No se encontró CSV número {num}")

# --- Malla de referencia ---
uy_ref = leer_sw(8, 'UY')
sy_ref = leer_sw(2, 'SY')
sx_ref = leer_sw(1, 'SX')
ux_ref = leer_sw(7, 'UX')

N    = len(uy_ref)
ID   = uy_ref['ID'].astype(int).values
X_mm = uy_ref['X_mm'].values
Y_mm = uy_ref['Y_mm'].values
Z_mm = uy_ref['Z_mm'].values
X = X_mm/1000; Y = Y_mm/1000; Z = Z_mm/1000

def interp_to_mesh(ref_df, col):
    tree = cKDTree(np.column_stack([ref_df.X_mm, ref_df.Y_mm, ref_df.Z_mm]))
    _, idx = tree.query(np.column_stack([X_mm, Y_mm, Z_mm]))
    return ref_df[col].values[idx]

UY_0 = uy_ref['UY'].values
SY_0 = interp_to_mesh(sy_ref, 'SY')
SX_0 = interp_to_mesh(sx_ref, 'SX')
UX_0 = interp_to_mesh(ux_ref, 'UX')

F_ref = 10.0

materiales = {
    "AISI_1020":     {"E": 200e9, "nu": 0.29, "sx_sc": 1.0},
    "Hierro_Ductil": {"E": 120e9, "nu": 0.31,
                      "sx_sc": (0.31/0.69)/(0.29/0.71)},
}

fuerzas = np.arange(10, 1010, 10)

for nombre, props in materiales.items():
    E  = props["E"]; nu = props["nu"]; G = E/(2*(1+nu))
    sx_sc     = props["sx_sc"]
    uy_mat_sc = 200e9 / E

    carpeta = os.path.join(SALIDA, nombre)
    os.makedirs(carpeta, exist_ok=True)

    for F in fuerzas:
        esc = F / F_ref

        sy  = SY_0 * esc
        sx  = SX_0 * esc * sx_sc
        sz  = SX_0 * esc * sx_sc
        txy = np.zeros(N)
        txz = np.zeros(N)
        tyz = np.zeros(N)

        ex  = (sx - nu*(sy + sz)) / E
        ey  = (sy - nu*(sx + sz)) / E
        ez  = (sz - nu*(sx + sy)) / E
        gxy = txy / G
        gxz = txz / G
        gyz = tyz / G

        UY = UY_0 * esc * uy_mat_sc
        UX = UX_0 * esc * sx_sc * uy_mat_sc
        UZ = UX_0 * esc * sx_sc * uy_mat_sc

        df_out = pd.DataFrame({
            "Nodo":   ID,
            "X_mm":   X_mm, "Y_mm": Y_mm, "Z_mm": Z_mm,
            "SX_Pa":  sx, "SY_Pa": sy, "SZ_Pa": sz,
            "TXY_Pa": txy, "TXZ_Pa": txz, "TYZ_Pa": tyz,
            "EPSX": ex, "EPSY": ey, "EPSZ": ez,
            "GMXY": gxy, "GMXZ": gxz, "GMYZ": gyz,
            "UX_mm": UX, "UY_mm": UY, "UZ_mm": UZ,
        })
        df_out.to_csv(os.path.join(carpeta, f'F_{int(F):04d}N.csv'), index=False)

    print(f"✓ {nombre}: 100 CSVs generados en {carpeta}")

print("\nListo.")
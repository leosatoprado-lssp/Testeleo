# ============================================================================
# VEGETAÇÃO NATIVA NA VIZINHANÇA DE CADA MUNICÍPIO (ANÉIS A PARTIR DA BORDA)
# ============================================================================
# 🎯 Objetivo: variáveis de vizinhança para a pergunta secundária (escala
# espacial da influência da vegetação de fora do município).
#
# Entrada: grade de 5 km com a fração de nativa por célula e ano
#          (geo/grade_nativa_5km_1985_2025.npz, de gee_grade_vegetacao.py)
# Saídas:
#   dados/vizinhanca_aneis_1985_2025.csv  (município × ano)
#       viz_nat_0_25_pct, viz_nat_25_50_pct, viz_nat_50_100_pct ... % de nativa
#       viz_flo_*_pct ............................................. % florestal
#       (% sobre a área mapeada pelo MapBiomas dentro do anel)
#   dados/vizinhanca_faixas_5km.npz
#       área de nativa (km²) por município × faixa de 5 km (0-200 km) × ano,
#       e área mapeada por faixa. Base para o índice de acesso Σ Nativa × f(d)
#       com o parâmetro de decaimento ESTIMADO (basta trocar f(d), sem EE).
#
# Distância: da borda do município ao centro de cada célula de 5 km. Medir da
# borda (e não do centroide) é essencial porque os municípios vão de 3 km² a
# 160 mil km²; com centroides, o "anel de 25 km" de Altamira cairia dentro do
# próprio município.
# Rodar da raiz do repositório:  python scripts/vizinhanca_municipios.py
# ============================================================================

import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from rasterio import features
from rasterio.transform import from_origin
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
import gee_lst_modis_municipios as base
from gee_grade_vegetacao import AEA_PROJ4, SAIDA as GRADE

FAIXA_KM = 5                       # largura de cada faixa de distância
DMAX_KM = 200                      # alcance máximo
N_FAIXAS = DMAX_KM // FAIXA_KM     # 40 faixas
ANEIS = {'0_25': (0, 25), '25_50': (25, 50), '50_100': (50, 100)}
DENSIFICA_M = 1000                 # pontos da borda a cada 1 km (erro ≤ 0,5 km)


# ============================================================================
# 1. GRADE E MALHA NA MESMA PROJEÇÃO (ALBERS, METROS)
# ============================================================================
t0 = time.time()
z = np.load(GRADE)
bandas = list(z['bandas'])
grade = z['grade'].astype(np.float32) / 10000          # volta para fração 0-1
X0, Y0, LADO = float(z['x0']), float(z['y0']), float(z['lado'])
NY, NX = grade.shape[:2]
anos = sorted({int(b[-4:]) for b in bandas if b.startswith('nat_')})
i_nat = [bandas.index(f'nat_{a}') for a in anos]
i_flo = [bandas.index(f'flo_{a}') for a in anos]
area_cel = (LADO / 1000) ** 2                          # 25 km²
nat = grade[:, :, i_nat] * area_cel                    # km² de nativa por célula e ano
flo = grade[:, :, i_flo] * area_cel
val = grade[:, :, bandas.index('val')] * area_cel      # km² mapeados (dentro do Brasil)
del grade

malha = (gpd.GeoDataFrame(base.carregar_malha(), geometry='geometry', crs='EPSG:4674')
         .to_crs(AEA_PROJ4).reset_index(drop=True))
malha['geocodigo'] = malha['CD_MUN'].astype(int)
print(f'📊 Grade {NY}×{NX} células, {len(anos)} anos; {len(malha)} municípios ({time.time() - t0:.0f}s)')

# Centro de cada célula
xs = X0 + (np.arange(NX) + 0.5) * LADO
ys = Y0 - (np.arange(NY) + 0.5) * LADO

# Rasterizo os municípios: cada célula recebe o índice do município que
# contém o seu centro. Células "de dentro" não entram na vizinhança (a
# vegetação própria já está no painel, medida a 30 m).
transf = from_origin(X0, Y0, LADO, LADO)
dono = features.rasterize(((g, i + 1) for i, g in enumerate(malha.geometry)),
                          out_shape=(NY, NX), transform=transf, fill=0, dtype='int32')


# ============================================================================
# 2. FAIXAS DE DISTÂNCIA POR MUNICÍPIO
# ============================================================================
# Para cada município: pego as células numa janela de ±200 km em torno dele,
# calculo a distância de cada centro até a borda (árvore KD sobre pontos da
# borda a cada 1 km, bem mais rápido que a distância exata ao polígono) e somo
# a vegetação por faixa de 5 km.
NAT = np.zeros((len(malha), N_FAIXAS, len(anos)), dtype=np.float32)
FLO = np.zeros_like(NAT)
VAL = np.zeros((len(malha), N_FAIXAS), dtype=np.float32)
dmax = DMAX_KM * 1000
for m, geom in enumerate(malha.geometry):
    borda = shapely.segmentize(geom.boundary, DENSIFICA_M)
    pts = shapely.get_coordinates(borda)
    arvore = cKDTree(pts)
    xmin, ymin, xmax, ymax = geom.bounds
    j0, j1 = np.searchsorted(xs, [xmin - dmax, xmax + dmax])
    i0, i1 = np.searchsorted(-ys, [-(ymax + dmax), -(ymin - dmax)])
    jj, ii = np.meshgrid(np.arange(j0, j1), np.arange(i0, i1))
    jj, ii = jj.ravel(), ii.ravel()
    d, _ = arvore.query(np.column_stack([xs[jj], ys[ii]]), distance_upper_bound=dmax)
    fora = (dono[ii, jj] != m + 1) & np.isfinite(d) & (d < dmax)
    faixa = (d[fora] // (FAIXA_KM * 1000)).astype(int)
    ii, jj = ii[fora], jj[fora]
    # Soma por faixa (np.add.at acumula células repetidas na mesma faixa)
    np.add.at(NAT[m], faixa, nat[ii, jj])
    np.add.at(FLO[m], faixa, flo[ii, jj])
    np.add.at(VAL[m], faixa, val[ii, jj])
    if (m + 1) % 1000 == 0:
        print(f'   {m + 1}/{len(malha)} municípios ({time.time() - t0:.0f}s)')

np.savez_compressed('dados/vizinhanca_faixas_5km.npz', nat_km2=NAT, flo_km2=FLO, val_km2=VAL,
                    geocodigo=malha['geocodigo'].values, anos=np.array(anos),
                    faixa_km=np.arange(N_FAIXAS) * FAIXA_KM)
print('💾 dados/vizinhanca_faixas_5km.npz')


# ============================================================================
# 3. ANÉIS (0-25, 25-50, 50-100 KM) EM % DA ÁREA MAPEADA
# ============================================================================
# ⚠️ ATENÇÃO METODOLÓGICA: o MapBiomas só cobre o Brasil. Em municípios de
# fronteira, a parte estrangeira do anel fica de fora (o % é calculado só sobre
# a área mapeada). Mudanças de vegetação no Paraguai, na Bolívia etc. não
# entram. O mar também fica de fora do denominador.
# ⚠️ ATENÇÃO METODOLÓGICA: células com centro fora do município, mas muito
# próximas da borda, cobrem parte do próprio município; com células de 5 km o
# efeito fica restrito aos primeiros ~3 km do anel interno.
linhas = []
for nome, (a, b) in ANEIS.items():
    fa, fb = a // FAIXA_KM, b // FAIXA_KM
    v = VAL[:, fa:fb].sum(axis=1)                      # km² mapeados no anel
    for tipo, arr in [('nat', NAT), ('flo', FLO)]:
        pct = 100 * arr[:, fa:fb, :].sum(axis=1) / np.where(v > 0, v, np.nan)[:, None]
        linhas.append(pd.DataFrame(pct, columns=anos).assign(geocodigo=malha['geocodigo'].values)
                      .melt(id_vars='geocodigo', var_name='ano', value_name=f'viz_{tipo}_{nome}_pct')
                      .set_index(['geocodigo', 'ano']))
    linhas.append(pd.DataFrame({'geocodigo': malha['geocodigo'].values, f'viz_area_{nome}_km2': v})
                  .set_index('geocodigo'))
aneis = pd.concat([l for l in linhas if l.index.nlevels == 2], axis=1).reset_index()
areas = pd.concat([l for l in linhas if l.index.nlevels == 1], axis=1).reset_index()
aneis = aneis.merge(areas, on='geocodigo').round(3).sort_values(['geocodigo', 'ano'])
aneis.to_csv('dados/vizinhanca_aneis_1985_2025.csv', index=False)
print(f'💾 dados/vizinhanca_aneis_1985_2025.csv: {aneis.shape}')


# ============================================================================
# 4. CONFERÊNCIA DA GRADE: VEGETAÇÃO PRÓPRIA (GRADE) × PAINEL (ESTATÍSTICA 30 M)
# ============================================================================
# Se a grade estiver certa, o % de nativa das células de dentro do município
# deve bater com o pct_nativa do painel (que vem da tabela oficial do MapBiomas).
proprio = np.zeros((len(malha), len(anos)), dtype=np.float64)
area_p = np.zeros(len(malha))
idx = dono.ravel() - 1
ok = idx >= 0
np.add.at(area_p, idx[ok], val.ravel()[ok])
for k in range(len(anos)):
    np.add.at(proprio[:, k], idx[ok], nat[:, :, k].ravel()[ok])
grade_pct = pd.DataFrame(100 * proprio / np.where(area_p > 0, area_p, np.nan)[:, None], columns=anos)
grade_pct['geocodigo'] = malha['geocodigo'].values
grade_pct = grade_pct.melt(id_vars='geocodigo', var_name='ano', value_name='pct_grade')
painel = pd.read_csv('dados/painel_municipios_1985_2025.csv', usecols=['geocodigo', 'ano', 'pct_nativa', 'area_total_ha'])
cmp = painel.merge(grade_pct, on=['geocodigo', 'ano']).dropna()
grandes = cmp[cmp['area_total_ha'] > 50_000]           # > 500 km²: várias células
print('\n📈 Conferência da grade (% nativa dentro do município):')
print(f'   correlação (todos): {cmp["pct_nativa"].corr(cmp["pct_grade"]):.3f}  '
      f'| municípios > 500 km²: {grandes["pct_nativa"].corr(grandes["pct_grade"]):.3f}')
print(f'   diferença média (grade − painel), > 500 km²: {(grandes["pct_grade"] - grandes["pct_nativa"]).mean():+.2f} pp')
# Variação no tempo: a grade capta a mesma mudança que o painel?
g5 = grandes.sort_values(['geocodigo', 'ano']).copy()
g5['dp'] = g5.groupby('geocodigo')['pct_nativa'].diff(5)
g5['dg'] = g5.groupby('geocodigo')['pct_grade'].diff(5)
print(f'   correlação das variações de 5 anos, > 500 km²: {g5["dp"].corr(g5["dg"]):.3f}')
print(f'\n✅ Concluído em {(time.time() - t0) / 60:.1f} min')

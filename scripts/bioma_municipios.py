# ============================================================================
# BIOMA DE CADA MUNICÍPIO (IBGE, BIOMAS 1:250.000, 2019)
# ============================================================================
# 🎯 Objetivo: variável fixa para (a) heterogeneidade por bioma, pergunta
# secundária da pesquisa, e (b) efeito fixo de bioma × ano na robustez.
# Saída: dados/bioma_municipios.csv com o bioma predominante e a fração da
# área do município em cada bioma.
# Fonte: https://geoftp.ibge.gov.br/informacoes_ambientais/estudos_ambientais/biomas/vetores/Biomas_250mil.zip
# Rodar da raiz do repositório:  python scripts/bioma_municipios.py
# ============================================================================

import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import gee_lst_modis_municipios as base
from gee_grade_vegetacao import AEA_PROJ4

# Cruzamento em projeção de área igual, para que as frações sejam de área real
malha = gpd.GeoDataFrame(base.carregar_malha(), geometry='geometry', crs='EPSG:4674').to_crs(AEA_PROJ4)
biomas = gpd.read_file('geo/biomas/lm_bioma_250.shp').to_crs(AEA_PROJ4)

inter = gpd.overlay(malha[['CD_MUN', 'geometry']], biomas[['Bioma', 'geometry']], how='intersection')
inter['area'] = inter.area
frac = inter.pivot_table(index='CD_MUN', columns='Bioma', values='area', aggfunc='sum', fill_value=0)
frac = frac.div(frac.sum(axis=1), axis=0)

# ⚠️ ATENÇÃO METODOLÓGICA: 636 municípios (11%) têm menos de 90% da área num só
# bioma. O predominante é o de maior área; a coluna frac_bioma_principal
# permite restringir a heterogeneidade aos municípios "puros" (ex.: > 0,9).
saida = pd.DataFrame({
    'geocodigo': frac.index.astype(int),
    'bioma': frac.idxmax(axis=1).values,
    'frac_bioma_principal': frac.max(axis=1).round(3).values,
})
nomes = {'Amazônia': 'amazonia', 'Caatinga': 'caatinga', 'Cerrado': 'cerrado',
         'Mata Atlântica': 'mata_atlantica', 'Pampa': 'pampa', 'Pantanal': 'pantanal'}
for b, n in nomes.items():
    saida[f'frac_{n}'] = frac[b].round(3).values if b in frac else 0.0

faltando = set(malha['CD_MUN'].astype(int)) - set(saida['geocodigo'])
print('=' * 70)
print(f'📊 {len(saida)} municípios com bioma; sem cruzamento: {sorted(faltando)}')
print(saida['bioma'].value_counts().to_string())
print(f'   Municípios em mais de um bioma (principal < 0,9): {(saida["frac_bioma_principal"] < 0.9).sum()}')
saida.to_csv('dados/bioma_municipios.csv', index=False)
print('💾 dados/bioma_municipios.csv')

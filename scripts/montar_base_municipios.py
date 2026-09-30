# ============================================================================
# BASE CONSOLIDADA DE VARIÁVEIS MUNICIPAIS
# ============================================================================
# 🎯 Objetivo: juntar, pelo código IBGE de 7 dígitos, todas as variáveis
# municipais numa única tabela:
#   1. Cobertura de vegetação nativa (MapBiomas Col. 11, 2025) -> script anterior
#   2. População: Censo 2022 e estimativa 2025 (IBGE, via repositório
#      lsbastos/popBR_mun no GitHub)
#   3. Área urbanizada (MapBiomas Col. 11, classe 24 "Área Urbanizada")
#   4. Latitude e longitude da sede (repositório kelvins/municipios-brasileiros)
#   5. Distância da sede ao mar (linha de costa Natural Earth 1:10m)
#   6. Altitude e clima (Google Earth Engine), quando os CSVs estiverem prontos
#
# Uso: python montar_base_municipios.py   (com os arquivos na mesma pasta)
# ============================================================================

import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box, MultiLineString
from shapely.ops import nearest_points
from pyproj import Geod

PASTA_MAPBIOMAS = 'mb'      # xlsx do MapBiomas + CSV de cobertura gerado antes
PASTA_GEO = 'geo'           # bases baixadas do GitHub
ARQ_MAPBIOMAS = os.path.join(PASTA_MAPBIOMAS, 'MAPBIOMAS_BRAZIL-COL.11-BIOME_STATE_MUNICIPALITY.xlsx')

# Lagoas Mirim e dos Patos: têm código IBGE mas não são municípios
GEOCODIGOS_NAO_MUNICIPIOS = [4300001, 4300002]


# ============================================================================
# 1. LISTA-MESTRE DE MUNICÍPIOS E COORDENADAS DA SEDE
# ============================================================================
# Uso a tabela do kelvins como lista-mestre porque ela tem os 5.571
# municípios atuais (inclui Boa Esperança do Norte/MT, instalado em 2025, e
# Fernando de Noronha/PE, que o MapBiomas não mapeia).
sedes = pd.read_csv(os.path.join(PASTA_GEO, 'kelvins.csv'))
sedes = sedes.rename(columns={'codigo_ibge': 'geocodigo', 'nome': 'municipio'})
print(f"✓ Sedes municipais: {len(sedes):,}")


# ============================================================================
# 2. DISTÂNCIA DA SEDE AO MAR
# ============================================================================
# ⚠️ ATENÇÃO METODOLÓGICA: a linha de costa do Natural Earth trata como "mar"
# as lagoas costeiras ligadas ao oceano (Lagoa dos Patos, Lagoa Mirim, Baía
# de Guanabara...). Na primeira tentativa, Porto Alegre ficava a 3 km do mar,
# o que não faz sentido para medir continentalidade. Para corrigir, aplico um
# "fechamento morfológico" no polígono de terra (expande 2,5 km e contrai
# 2,5 km): canais e barras com menos de ~5 km de largura se fecham, as lagoas
# viram "buracos" e passo a usar só o contorno externo, que é o oceano aberto.
terra = gpd.read_file(os.path.join(PASTA_GEO, 'ne_land.geojson')).explode(index_parts=False)
recorte = box(-82, -57, -26, 14)   # América do Sul e Atlântico próximo
terra = terra[terra.intersects(recorte)].to_crs(5880)   # SIRGAS 2000 Policônica (metros)
terra = terra.clip(box(*gpd.GeoSeries([recorte], crs=4326).to_crs(5880).total_bounds))

RAIO = 2500  # metros
contornos = []
for geom in terra.geometry:
    fechado = geom.buffer(RAIO).buffer(-RAIO)
    for parte in getattr(fechado, 'geoms', [fechado]):
        if not parte.is_empty:
            contornos.append(parte.exterior)
costa = MultiLineString(contornos)

# Ponto da costa mais próximo de cada sede (em projeção métrica) e, depois,
# distância geodésica real no elipsoide WGS84 entre a sede e esse ponto
pts = gpd.GeoSeries(gpd.points_from_xy(sedes.longitude, sedes.latitude), crs=4326).to_crs(5880)
mais_proximo = gpd.GeoSeries([nearest_points(p, costa)[1] for p in pts], crs=5880).to_crs(4326)
_, _, dist_m = Geod(ellps='WGS84').inv(sedes.longitude.values, sedes.latitude.values,
                                        mais_proximo.x.values, mais_proximo.y.values)
sedes['dist_mar_km'] = np.round(dist_m / 1000, 2)
print(f"✓ Distância ao mar calculada (mediana: {sedes['dist_mar_km'].median():.0f} km)")


# ============================================================================
# 3. POPULAÇÃO (estimativa oficial do IBGE para 2025)
# ============================================================================
# ⚠️ ATENÇÃO METODOLÓGICA: o repositório diz que o valor de 2022 é o do
# Censo, mas conferindo vi que não é: o total dá 210,9 milhões, contra os
# 203,1 milhões do Censo 2022 (São Paulo aparece com 11,95 mi em vez de
# 11,45 mi). É a série revisada pelo IBGE, que corrige o sub-registro do
# Censo. Por isso uso só a estimativa de 2025, que confere com a divulgação
# oficial (ex.: São Carlos = 266.427 hab.), e não chamo nada de "Censo".
# A tabela usa o código IBGE de 6 dígitos (sem o dígito verificador), então
# crio essa chave a partir do código de 7 dígitos.
pop = pd.read_csv(os.path.join(PASTA_GEO, 'pop_long.csv'))
pop = (pop[pop['Ano'] == 2025][['MUNCOD', 'POP']]
       .rename(columns={'POP': 'populacao_estimada_2025'}))
sedes['cod6'] = sedes['geocodigo'] // 10
sedes = sedes.merge(pop, left_on='cod6', right_on='MUNCOD', how='left')
print(f"✓ População: {sedes['populacao_estimada_2025'].notna().sum():,} municípios com estimativa 2025")

# Censo 2022 (contagem final), se o CSV da tabela 4714 do SIDRA tiver sido baixado
arq_censo = os.path.join(PASTA_GEO, 'censo2022_sidra4714.csv')
if os.path.exists(arq_censo):
    censo = pd.read_csv(arq_censo)
    censo.columns = ['geocodigo', 'populacao_censo_2022'] + list(censo.columns[2:])
    censo = censo[['geocodigo', 'populacao_censo_2022']]
    censo['geocodigo'] = pd.to_numeric(censo['geocodigo'], errors='coerce')
    censo['populacao_censo_2022'] = pd.to_numeric(censo['populacao_censo_2022'], errors='coerce')
    sedes = sedes.merge(censo.dropna(), on='geocodigo', how='left')


# ============================================================================
# 4. ÁREA URBANIZADA (MapBiomas, classe 24)
# ============================================================================
# Uso o mesmo arquivo do MapBiomas da cobertura vegetal, o que garante que
# área urbana e vegetação venham da mesma classificação e do mesmo ano
mb = pd.read_excel(ARQ_MAPBIOMAS, sheet_name='COVERAGE_11', usecols=['geocode', 'class', 'y2025'])
mb['geocode'] = pd.to_numeric(mb['geocode'], errors='coerce')
urbana = (mb[mb['class'] == 24].groupby('geocode')['y2025'].sum()
          .rename('area_urbana_ha').reset_index()
          .rename(columns={'geocode': 'geocodigo'}))
urbana['geocodigo'] = urbana['geocodigo'].astype(int)
sedes = sedes.merge(urbana, on='geocodigo', how='left')


# ============================================================================
# 5. COBERTURA DE VEGETAÇÃO NATIVA (gerada pelo script anterior)
# ============================================================================
veg = pd.read_csv(os.path.join(PASTA_MAPBIOMAS, 'cobertura_nativa_municipios_2025.csv'))
veg = veg.drop(columns=['municipio', 'ano']).rename(columns={'uf': 'uf_mb'})
base = sedes.merge(veg, on='geocodigo', how='left')

# UF a partir do código (os 2 primeiros dígitos), para não depender de nomes
UF_POR_CODIGO = {11: 'RO', 12: 'AC', 13: 'AM', 14: 'RR', 15: 'PA', 16: 'AP', 17: 'TO',
                 21: 'MA', 22: 'PI', 23: 'CE', 24: 'RN', 25: 'PB', 26: 'PE', 27: 'AL',
                 28: 'SE', 29: 'BA', 31: 'MG', 32: 'ES', 33: 'RJ', 35: 'SP', 41: 'PR',
                 42: 'SC', 43: 'RS', 50: 'MS', 51: 'MT', 52: 'GO', 53: 'DF'}
base['uf'] = (base['geocodigo'] // 100000).map(UF_POR_CODIGO)

# Percentual da área do município ocupado por área urbanizada
base['pct_area_urbana'] = np.round(100 * base['area_urbana_ha'] / base['area_total_ha'], 3)
base['area_urbana_ha'] = base['area_urbana_ha'].round(2)


# ============================================================================
# 6. ALTITUDE E CLIMA (Google Earth Engine), se os CSVs já existirem
# ============================================================================
arq_alt = os.path.join(PASTA_GEO, 'altitude_municipios.csv')
arq_clima = os.path.join(PASTA_GEO, 'clima_municipios.csv')
if os.path.exists(arq_alt) and os.path.exists(arq_clima):
    alt = pd.read_csv(arq_alt).rename(columns={
        'CD_MUN': 'geocodigo', 'AREA_KM2': 'area_ibge_km2',
        'mean': 'altitude_media_m', 'min': 'altitude_min_m', 'max': 'altitude_max_m'})
    alt = alt[['geocodigo', 'area_ibge_km2', 'altitude_media_m', 'altitude_min_m', 'altitude_max_m']]
    clima = pd.read_csv(arq_clima).rename(columns={'CD_MUN': 'geocodigo'})
    for t in (alt, clima):
        t['geocodigo'] = pd.to_numeric(t['geocodigo'], errors='coerce').astype('Int64')
    base = base.merge(alt, on='geocodigo', how='left').merge(clima, on='geocodigo', how='left')
    print("✓ Altitude e clima do GEE incorporados")
else:
    print("⏳ CSVs do GEE ainda não encontrados: base gerada sem altitude e clima")


# ============================================================================
# 7. ORGANIZAÇÃO, CONFERÊNCIAS E EXPORTAÇÃO
# ============================================================================
base = base[~base['geocodigo'].isin(GEOCODIGOS_NAO_MUNICIPIOS)]
colunas = ['geocodigo', 'municipio', 'uf', 'latitude', 'longitude',
           'populacao_censo_2022', 'populacao_estimada_2025',
           'area_total_ha', 'area_ibge_km2', 'area_urbana_ha', 'pct_area_urbana',
           'dist_mar_km',
           'altitude_media_m', 'altitude_min_m', 'altitude_max_m',
           'temp_media_c', 'precip_anual_mm', 'umidade_rel_pct',
           'pct_nativa', 'nativa_ha', 'nativa_florestal_ha', 'nativa_nao_florestal_ha',
           'agua_ha', 'pct_nativa_sem_agua']
base = base[[c for c in colunas if c in base.columns]].sort_values('geocodigo')

print("=" * 70)
print(f"📊 Municípios na base: {len(base):,}")
print("📊 Valores ausentes por coluna:")
print(base.isna().sum()[base.isna().sum() > 0].to_string())

base.to_csv('base_municipios.csv', index=False, encoding='utf-8-sig')
print("💾 Salvo: base_municipios.csv")

# ============================================================================
# PAINEL ANUAL DE VARIÁVEIS MUNICIPAIS (1985-2025)
# ============================================================================
# 🎯 Objetivo: uma linha por município e ano, de 1985 a 2025, com:
#   - vegetação nativa (total, florestal, não florestal), água e área total
#     (MapBiomas Coleção 11, 1985-2025)
#   - área urbanizada (MapBiomas Coleção 11, classe 24, 1985-2025)
#   - população (estimativas RIPSA/Ministério da Saúde, 2000-2025)
#   - variáveis fixas repetidas em todos os anos: latitude, longitude e
#     distância da sede ao mar
#
# Pré-requisito: ter rodado cobertura_nativa_municipios.py (gera o painel de
# vegetação) e montar_base_municipios.py (gera as variáveis fixas).
# ============================================================================

import os
import numpy as np
import pandas as pd

PASTA_MAPBIOMAS = 'mb'
PASTA_GEO = 'geo'
ARQ_MAPBIOMAS = os.path.join(PASTA_MAPBIOMAS, 'MAPBIOMAS_BRAZIL-COL.11-BIOME_STATE_MUNICIPALITY.xlsx')
ANOS = range(1985, 2026)


# ============================================================================
# 1. ESQUELETO DO PAINEL: TODOS OS MUNICÍPIOS × TODOS OS ANOS
# ============================================================================
# Parto das variáveis fixas (5.571 municípios) e faço o produto cartesiano
# com os anos. Assim o painel fica balanceado: todo município tem 41 linhas,
# mesmo quando alguma fonte não cobre aquele ano (o valor fica vazio).
fixas = pd.read_csv('base_municipios.csv',
                    usecols=['geocodigo', 'municipio', 'uf', 'latitude', 'longitude', 'dist_mar_km'])
anos = pd.DataFrame({'ano': list(ANOS)})
painel = fixas.merge(anos, how='cross')
print(f"✓ Esqueleto: {fixas['geocodigo'].nunique():,} municípios × {len(anos)} anos = {len(painel):,} linhas")


# ============================================================================
# 2. VEGETAÇÃO NATIVA, ÁGUA E ÁREA TOTAL (MapBiomas 1985-2025)
# ============================================================================
# O MapBiomas calcula todos os anos com a malha municipal ATUAL. Isso é bom
# para o painel: um município criado em 1997 tem valores desde 1985,
# referentes ao território que hoje é dele, então a série é comparável.
veg = pd.read_csv(os.path.join(PASTA_MAPBIOMAS, 'cobertura_nativa_municipios_painel.csv'))
veg = veg.drop(columns=['municipio', 'uf'])
painel = painel.merge(veg, on=['geocodigo', 'ano'], how='left')


# ============================================================================
# 3. ÁREA URBANIZADA (MapBiomas 1985-2025, classe 24)
# ============================================================================
colunas_ano = [f'y{a}' for a in ANOS]
mb = pd.read_excel(ARQ_MAPBIOMAS, sheet_name='COVERAGE_11', usecols=['geocode', 'class'] + colunas_ano)
mb['geocode'] = pd.to_numeric(mb['geocode'], errors='coerce')
urbana = (mb[mb['class'] == 24]
          .groupby('geocode')[colunas_ano].sum()            # soma os biomas do município
          .reset_index()
          .melt(id_vars='geocode', var_name='ano', value_name='area_urbana_ha'))
urbana['ano'] = urbana['ano'].str.lstrip('y').astype(int)
urbana = urbana.rename(columns={'geocode': 'geocodigo'})
urbana['geocodigo'] = urbana['geocodigo'].astype(int)
painel = painel.merge(urbana, on=['geocodigo', 'ano'], how='left')

# ⚠️ ATENÇÃO METODOLÓGICA: se o município não tinha nenhum pixel urbano num
# ano, a classe simplesmente não aparece na tabela. Nesses casos a área
# urbana é zero, não ausente. Só preencho com zero onde o MapBiomas tem dado
# (área total preenchida), para não inventar valor em Fernando de Noronha.
tem_mapbiomas = painel['area_total_ha'].notna()
painel.loc[tem_mapbiomas, 'area_urbana_ha'] = painel.loc[tem_mapbiomas, 'area_urbana_ha'].fillna(0)
painel['area_urbana_ha'] = painel['area_urbana_ha'].round(2)
painel['pct_area_urbana'] = np.round(100 * painel['area_urbana_ha'] / painel['area_total_ha'], 3)


# ============================================================================
# 4. POPULAÇÃO (RIPSA/Ministério da Saúde, 2000-2025)
# ============================================================================
# Essa série foi feita para o DATASUS e já vem compatibilizada com a malha
# atual (municípios criados depois de 2000 têm população desde 2000), o que
# é exatamente o que um painel precisa.
# ⚠️ ATENÇÃO METODOLÓGICA: de 1985 a 1999 a população fica vazia de
# propósito. As estimativas do IBGE dessa época usam a divisão municipal da
# própria época (cerca de 1.500 municípios foram criados entre 1985 e 2001),
# então não são comparáveis com a malha atual sem uma compatibilização, por
# exemplo pelas Áreas Mínimas Comparáveis (AMC) do IPEA. Preferi deixar
# vazio a interpolar ou extrapolar valores sem base.
pop = pd.read_csv(os.path.join(PASTA_GEO, 'pop_long.csv'))
pop = pop.rename(columns={'Ano': 'ano', 'POP': 'populacao'})[['MUNCOD', 'ano', 'populacao']]
painel['cod6'] = painel['geocodigo'] // 10
painel = (painel.merge(pop, left_on=['cod6', 'ano'], right_on=['MUNCOD', 'ano'], how='left')
                .drop(columns=['cod6', 'MUNCOD']))


# ============================================================================
# 5. ORGANIZAÇÃO, CONFERÊNCIAS E EXPORTAÇÃO
# ============================================================================
colunas = ['geocodigo', 'municipio', 'uf', 'ano',
           'latitude', 'longitude', 'dist_mar_km',
           'populacao', 'area_total_ha', 'area_urbana_ha', 'pct_area_urbana',
           'pct_nativa', 'nativa_ha', 'nativa_florestal_ha', 'nativa_nao_florestal_ha',
           'agua_ha', 'pct_nativa_sem_agua']
painel = painel[colunas].sort_values(['geocodigo', 'ano']).reset_index(drop=True)

print("=" * 70)
print(f"📊 Linhas: {len(painel):,} | Municípios: {painel['geocodigo'].nunique():,} | "
      f"Anos: {painel['ano'].min()}-{painel['ano'].max()}")
print("📊 Cobertura por variável (% de linhas preenchidas):")
print((painel.notna().mean() * 100).round(1).to_string())

# Conferências de sanidade: área total constante no tempo (malha fixa) e
# percentuais dentro de 0-100
var_area = painel.groupby('geocodigo')['area_total_ha'].agg(lambda s: s.max() - s.min())
print(f"✓ Maior variação da área total de um município ao longo dos anos: {var_area.max():.2f} ha")
assert painel['pct_nativa'].dropna().between(0, 100.01).all()
assert painel['pct_area_urbana'].dropna().between(0, 100.01).all()
print("✓ Percentuais entre 0 e 100")

painel.to_csv('painel_municipios_1985_2025.csv', index=False, encoding='utf-8-sig')
print("💾 Salvo: painel_municipios_1985_2025.csv")

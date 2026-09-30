# ============================================================================
# JUNÇÃO DA TEMPERATURA DE SUPERFÍCIE (MODIS LST) AO PAINEL MUNICIPAL
# ============================================================================
# 🎯 Objetivo: acrescentar ao painel 1985-2025 as colunas de LST extraídas no
# Earth Engine (gee_lst_modis_municipios.py) e rodar conferências de faixa,
# cobertura e coerência com o BR-DWGD.
# Rodar da raiz do repositório:  python scripts/juntar_lst_painel.py
# ============================================================================

import numpy as np
import pandas as pd

painel = pd.read_csv('dados/painel_municipios_1985_2025.csv')
lst = pd.read_csv('dados/lst_modis_municipios_2001_2025.csv')

# Se o painel já tiver colunas de LST (execução anterior), removo antes de juntar
cols_lst = [c for c in lst.columns if c not in ('geocodigo', 'ano')]
painel = painel.drop(columns=[c for c in cols_lst if c in painel.columns])


# ============================================================================
# 1. CONFERÊNCIA DE CHAVES
# ============================================================================
# Todo município da LST precisa existir no painel e vice-versa (a malha IBGE
# 2025 e o painel usam a mesma divisão territorial).
mun_painel, mun_lst = set(painel['geocodigo']), set(lst['geocodigo'])
print('=' * 70)
print(f'📊 Municípios no painel: {len(mun_painel)} | na LST: {len(mun_lst)}')
print(f'   Só no painel: {sorted(mun_painel - mun_lst)}')
print(f'   Só na LST:    {sorted(mun_lst - mun_painel)}')
assert not lst.duplicated(['geocodigo', 'ano']).any(), 'chave duplicada na LST'


# ============================================================================
# 2. JUNÇÃO
# ============================================================================
# Junção à esquerda: anos 1985-2000 ficam sem LST (o MODIS começa em 2000) e
# o Aqua só tem ano completo a partir de 2003.
painel = painel.merge(lst, on=['geocodigo', 'ano'], how='left')
painel.to_csv('dados/painel_municipios_1985_2025.csv', index=False)
print(f'💾 Painel: {painel.shape[0]} linhas × {painel.shape[1]} colunas')


# ============================================================================
# 3. CONFERÊNCIAS
# ============================================================================
p = painel[painel['ano'] >= 2001]

# 3.1 Faixa de valores: LST diurna anual no Brasil deve ficar entre ~15 e ~45 °C
print('\n' + '=' * 70)
print('📈 Faixa das variáveis principais (2001-2025)')
print('=' * 70)
print(p[[c for c in cols_lst if c.endswith('_c') or c.endswith('_c_rig')]].describe().T.round(2).to_string())

# 3.2 Cobertura: municípios × anos sem valor e nº de observações válidas
print('\n📊 Valores ausentes por variável (linhas de 2001-2025):')
print(p[cols_lst].isna().sum().to_string())
print('\n📊 Nº médio de compostos válidos por ano (de 46), por região (1º dígito):')
reg = p['geocodigo'] // 1_000_000
print(p.groupby(reg)[[c for c in cols_lst if c.startswith(('terra_n_obs', 'aqua_n_obs'))]]
      .mean().round(1).rename(index={1: 'N', 2: 'NE', 3: 'SE', 4: 'S', 5: 'CO'}).T.to_string())

# 3.3 Coerência entre fontes: correlações no mesmo município-ano
# ⚠️ ATENÇÃO METODOLÓGICA: LST e temperatura do ar medem coisas diferentes
# (Winckler et al., 2019); espera-se correlação alta entre municípios, mas não
# igualdade de níveis. A correlação "dentro do município" (desvios da média de
# cada um) é a que importa para um modelo com efeito fixo de município.
pares = [('aqua_lst_dia_c', 'tmax_media_c'), ('terra_lst_dia_c', 'tmax_media_c'),
         ('aqua_lst_noite_c', 'tmin_media_c'), ('terra_lst_dia_c', 'aqua_lst_dia_c'),
         ('aqua_lst_dia_c', 'aqua_lst_dia_c_rig')]
print('\n📈 Correlações (geral | dentro do município):')
for a, b in pares:
    q = p[['geocodigo', a, b]].dropna()
    geral = q[a].corr(q[b])
    dm = q[[a, b]] - q.groupby('geocodigo')[[a, b]].transform('mean')
    print(f'   {a:>22} × {b:<20} {geral:6.3f} | {dm[a].corr(dm[b]):6.3f}')

# 3.4 Deriva orbital do Terra: diferença média Terra − Aqua (dia) por ano.
# ⚠️ ATENÇÃO METODOLÓGICA: o Terra passou a cruzar o equador cada vez mais cedo
# (antes das 10h30) desde ~2020, com o fim das manobras de órbita. Uma tendência
# nessa diferença indica deriva; o efeito fixo de UF × ano absorve a parte
# comum, mas convém usar o Aqua como Y principal se a deriva for grande.
dif = (p['terra_lst_dia_c'] - p['aqua_lst_dia_c']).groupby(p['ano']).mean().round(2)
print('\n📈 Terra − Aqua (LST diurna, °C) por ano:')
print(dif.to_string())

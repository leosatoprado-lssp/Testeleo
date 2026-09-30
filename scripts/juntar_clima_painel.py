# ============================================================================
# JUNÇÃO DO CLIMA (BR-DWGD) AO PAINEL MUNICIPAL 1985-2025
# ============================================================================
# 🎯 Objetivo: acrescentar ao painel as variáveis climáticas anuais geradas no
# Colab (clima_anual_municipios.csv), tratando:
#   - municípios criados depois da malha usada pelo BR-DWGD (sem clima)
#   - municípios sem célula da grade (Bombinhas/SC e Fernando de Noronha/PE)
#   - degraus artificiais nas séries (mudanças na rede de estações)
# ============================================================================

import pandas as pd
import numpy as np

painel = pd.read_csv('painel_municipios_1985_2025.csv')
clima = pd.read_csv('clima_anual_municipios.csv')
cols_clima = [c for c in clima.columns if c not in ('geocodigo', 'ano')]

# As lagoas Mirim e dos Patos (4300001 e 4300002) vêm no clima, mas não são municípios
clima = clima[~clima['geocodigo'].isin([4300001, 4300002])]


# ============================================================================
# 1. MUNICÍPIOS SEM CLIMA: USAR O MUNICÍPIO DE ORIGEM
# ============================================================================
# Seis municípios foram criados depois da malha usada nas estatísticas zonais
# e Bombinhas é pequena demais para conter uma célula da grade de ~10 km.
# Como o clima varia pouco nessa escala, uso os valores do município do qual
# cada um foi desmembrado (que é vizinho e contém a área de origem).
# ⚠️ ATENÇÃO METODOLÓGICA: é uma imputação. A coluna clima_imputado permite
# excluir esses 7 municípios num teste de robustez. Fernando de Noronha fica
# sem clima: é uma ilha oceânica e nenhum vizinho a representa.
ORIGEM = {
    1504752: 1506807,  # Mojuí dos Campos/PA   <- Santarém
    4212650: 4209409,  # Pescaria Brava/SC     <- Laguna
    4220000: 4207007,  # Balneário Rincão/SC   <- Içara
    4314548: 4302105,  # Pinto Bandeira/RS     <- Bento Gonçalves
    5006275: 5003256,  # Paraíso das Águas/MS  <- Costa Rica
    5101837: 5107925,  # Boa Esperança do Norte/MT <- Sorriso
    4202453: 4213500,  # Bombinhas/SC          <- Porto Belo
}
clima = clima[~clima['geocodigo'].isin(ORIGEM.keys())]   # remove as linhas vazias de Bombinhas
imputados = []
for novo, origem in ORIGEM.items():
    bloco = clima[clima['geocodigo'] == origem].copy()
    bloco['geocodigo'] = novo
    imputados.append(bloco)
clima['clima_imputado'] = 0
imputados = pd.concat(imputados).assign(clima_imputado=1)
clima = pd.concat([clima, imputados], ignore_index=True)
print(f"✓ Clima imputado pelo município de origem: {imputados['geocodigo'].nunique()} municípios")


# ============================================================================
# 2. MARCAÇÃO DE DEGRAUS ARTIFICIAIS NAS SÉRIES
# ============================================================================
# O BR-DWGD é interpolado a partir de estações. Quando uma estação entra, sai
# ou muda de lugar, a série dos municípios próximos pode dar um "degrau" que
# não é clima real (ex.: Esperantina/PI, onde a Tmín sobe 2 °C de 2018 para
# 2019 e a Tmáx não muda).
# Detecção: para cada município, comparo a média de 5 anos depois com a de 5
# anos antes de cada ano possível, na anomalia em relação à média da UF
# naquele ano (o que remove a variação climática regional). Guardo o maior
# degrau encontrado na Tmín e na Tmáx.
# ⚠️ ATENÇÃO METODOLÓGICA: parte desses degraus pode ser real (uma mudança
# forte de uso do solo, justamente o que queremos medir). Por isso só
# marco, não apago; a decisão fica para os testes de robustez do modelo.
base_uf = painel[['geocodigo', 'uf']].drop_duplicates()
tmp = clima[clima['clima_imputado'] == 0].merge(base_uf, on='geocodigo').dropna(subset=['temp_media_c'])


def maior_degrau(df, var):
    df = df.copy()
    df['rel'] = df[var] - df.groupby(['uf', 'ano'])[var].transform('mean')
    tab = df.pivot(index='ano', columns='geocodigo', values='rel')
    melhor = pd.Series(0.0, index=tab.columns)
    for ano in range(tab.index.min() + 5, tab.index.max() - 3):
        dif = tab.loc[ano:ano + 4].mean() - tab.loc[ano - 5:ano - 1].mean()
        melhor = melhor.where(melhor.abs() >= dif.abs(), dif)
    return melhor


degraus = pd.DataFrame({'degrau_tmin_c': maior_degrau(tmp, 'tmin_media_c'),
                        'degrau_tmax_c': maior_degrau(tmp, 'tmax_media_c')}).round(2)
degraus['serie_com_degrau'] = ((degraus['degrau_tmin_c'].abs() > 1.5) |
                               (degraus['degrau_tmax_c'].abs() > 1.5)).astype(int)
degraus = degraus.reset_index()
# Imputados herdam a marcação do município de origem
herdados = degraus.set_index('geocodigo').loc[list(ORIGEM.values())].reset_index()
herdados['geocodigo'] = list(ORIGEM.keys())
degraus = pd.concat([degraus, herdados], ignore_index=True)
print(f"✓ Municípios com degrau > 1,5 °C na Tmín ou na Tmáx: {degraus['serie_com_degrau'].sum()}")


# ============================================================================
# 3. JUNÇÃO E EXPORTAÇÃO
# ============================================================================
painel = (painel.merge(clima, on=['geocodigo', 'ano'], how='left')
                .merge(degraus, on='geocodigo', how='left'))

# O clima vai até 2023; em 2024-2025 as colunas ficam vazias
print("=" * 70)
print(f"📊 Linhas: {len(painel):,} | Municípios: {painel['geocodigo'].nunique():,}")
cobertura = painel[painel['ano'] <= 2023]['temp_media_c'].notna().mean()
print(f"📊 Temperatura preenchida em 1985-2023: {cobertura:.2%}")
sem = painel[(painel['ano'] <= 2023) & painel['temp_media_c'].isna()]['municipio'].unique()
print(f"📊 Sem temperatura: {list(sem)}")

painel.to_csv('painel_municipios_1985_2025.csv', index=False, encoding='utf-8-sig')
print("💾 Salvo: painel_municipios_1985_2025.csv")

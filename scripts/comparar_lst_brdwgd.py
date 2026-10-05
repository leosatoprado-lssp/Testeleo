# ============================================================================
# CONTRAPONTO: VERSÕES DA LST (MODIS) × TEMPERATURA DO AR (BR-DWGD)
# ============================================================================
# 🎯 Objetivo: verificar, de forma descritiva, quanto do sinal ano a ano da
# LST diurna do Aqua pode ser viés de céu limpo, comparando as versões:
#   bruta ......... aqua_lst_dia_c          (média dos compostos válidos)
#   rigorosa ...... aqua_lst_dia_c_rig      (QC com erro ≤ 1 K)
#   dessaz ........ aqua_lst_dia_dessaz_c   (anomalias por semana do ano)
#   seca .......... aqua_lst_dia_seca_c     (3 meses mais secos do pixel)
#   zhang ......... zhang_lst_dia_c         (base preenchida, 2003-2020)
#   zhang_seca .... zhang_lst_dia_seca_c
# com a Tmáx do ar do BR-DWGD, que não depende de céu limpo.
#
# Toda comparação usa a variação que sobra depois de tirar os mesmos efeitos
# fixos do modelo planejado (município e UF × ano). É essa variação que vai
# identificar o efeito da vegetação, então é nela que o viés importa.
# ⚠️ ATENÇÃO METODOLÓGICA: não é um teste de qual versão está "certa". LST e
# temperatura do ar respondem de forma diferente à vegetação (Winckler et al.,
# 2019). A Tmáx do ar serve como referência livre de nuvem, não como gabarito.
#
# Rodar da raiz do repositório:  python scripts/comparar_lst_brdwgd.py
# ============================================================================

from pathlib import Path

import numpy as np
import pandas as pd

VERSOES = {
    'bruta': 'aqua_lst_dia_c',
    'rigorosa': 'aqua_lst_dia_c_rig',
    'dessaz': 'aqua_lst_dia_dessaz_c',
    'seca': 'aqua_lst_dia_seca_c',
    'zhang': 'zhang_lst_dia_c',
    'zhang_seca': 'zhang_lst_dia_seca_c',
}
NOBS = {'bruta': 'aqua_n_obs_dia', 'rigorosa': 'aqua_n_obs_dia_rig',
        'dessaz': 'aqua_n_obs_dia', 'seca': 'aqua_n_obs_seca',
        'zhang': 'aqua_n_obs_dia', 'zhang_seca': 'aqua_n_obs_seca'}
ANOS = (2003, 2020)   # período comum a todas as versões (Zhang termina em 2020)
SAIDA = Path('resultados')


# ============================================================================
# 1. DADOS E AMOSTRA
# ============================================================================
painel = pd.read_csv('dados/painel_municipios_1985_2025.csv')
# Amostra robusta do BR-DWGD: sem clima imputado e sem degrau artificial.
# Aqui ela é obrigatória, porque um degrau na Tmáx contaminaria a comparação.
df = painel[painel['ano'].between(*ANOS)
            & (painel['clima_imputado'] == 0) & (painel['serie_com_degrau'] == 0)].copy()
df['regiao'] = (df['geocodigo'] // 1_000_000).map({1: 'N', 2: 'NE', 3: 'SE', 4: 'S', 5: 'CO'})
df['uf_ano'] = df['uf'] + '_' + df['ano'].astype(str)
print('=' * 70)
print(f'📊 Amostra: {df["geocodigo"].nunique()} municípios, {ANOS[0]}-{ANOS[1]}, {len(df)} linhas')
print('=' * 70)


# ============================================================================
# 2. RESÍDUO DOS EFEITOS FIXOS (MUNICÍPIO e UF × ANO)
# ============================================================================
# Projeções alternadas: tiro a média por município, depois por UF × ano, e
# repito até estabilizar. É o que um modelo de dois efeitos fixos faz com
# cada variável antes de estimar (teorema de Frisch-Waugh-Lovell).
def residualizar(d, col, iteracoes=20):
    r = d[col] - d[col].mean()
    for _ in range(iteracoes):
        r = r - r.groupby(d['geocodigo']).transform('mean')
        r = r - r.groupby(d['uf_ano']).transform('mean')
    return r


cols = list(VERSOES.values()) + list(set(NOBS.values())) + ['tmax_media_c', 'chuva_anual_mm', 'pct_nativa']
# Linhas completas em todas as variáveis, para que todas as versões sejam
# comparadas exatamente nos mesmos municípios-ano
base = df.dropna(subset=cols).copy()
for c in cols:
    base[f'r_{c}'] = residualizar(base, c)
print(f'   Linhas completas usadas: {len(base)}')


def corr(a, b):
    return np.corrcoef(a, b)[0, 1]


def corr_parcial(y, x, z):
    # Correlação entre y e x depois de tirar de ambos a parte explicada por z
    ey = y - np.polyfit(z, y, 1)[0] * z
    ex = x - np.polyfit(z, x, 1)[0] * z
    return corr(ey, ex)


# ============================================================================
# 3. TABELA PRINCIPAL DO CONTRAPONTO
# ============================================================================
# Colunas:
#   dp_resid ........ desvio-padrão da variação usada pelo modelo (°C)
#   corr_tmax ....... correlação com a Tmáx do ar (BR-DWGD): quanto da variação
#                     ano a ano da LST acompanha a do ar
#   corr_nobs|tmax .. correlação com o nº de observações válidas, descontada a
#                     Tmáx. Se a LST sobe só porque o satélite viu mais semanas
#                     (secas), essa correlação fica positiva: é a "assinatura"
#                     do viés de céu limpo
#   corr_chuva|tmax . o mesmo com a chuva anual: anos chuvosos têm menos dias
#                     claros; uma LST livre de viés não deveria responder à
#                     chuva além do que o ar já responde
linhas = []
for nome, col in VERSOES.items():
    for reg, g in [('Brasil', base)] + list(base.groupby('regiao')):
        y, t = g[f'r_{col}'].values, g['r_tmax_media_c'].values
        linhas.append({
            'versao': nome, 'regiao': reg,
            'dp_resid': g[f'r_{col}'].std(),
            'corr_tmax': corr(y, t),
            'corr_nobs|tmax': corr_parcial(y, g[f'r_{NOBS[nome]}'].values, t),
            'corr_chuva|tmax': corr_parcial(y, g['r_chuva_anual_mm'].values, t),
        })
tab = pd.DataFrame(linhas).round(3)
print('\n📈 Contraponto com o BR-DWGD (variação dentro de município, após UF × ano)')
print(tab[tab['regiao'] == 'Brasil'].drop(columns='regiao').to_string(index=False))
print('\n📈 Por região: correlação com a Tmáx | assinatura do viés (n_obs | Tmáx)')
print(tab.pivot(index='versao', columns='regiao', values='corr_tmax').round(2).to_string())
print(tab.pivot(index='versao', columns='regiao', values='corr_nobs|tmax').round(2).to_string())


# ============================================================================
# 4. TENDÊNCIA NACIONAL (°C POR DÉCADA)
# ============================================================================
# Média simples dos municípios por ano e inclinação linear. Uma versão com
# tendência muito diferente da Tmáx do ar pode estar carregando mudança na
# frequência de céu limpo ao longo dos anos.
# ⚠️ ATENÇÃO METODOLÓGICA: aqui não tiro o efeito de UF × ano (ele absorveria
# a própria tendência); é só uma conferência de consistência temporal.
medias = base.groupby('ano')[list(VERSOES.values()) + ['tmax_media_c']].mean()
tend = {c: np.polyfit(medias.index, medias[c], 1)[0] * 10 for c in medias.columns}
print('\n📈 Tendência da média nacional, 2003-2020 (°C/década):')
for c, v in tend.items():
    print(f'   {c:<24} {v:+.2f}')


# ============================================================================
# 5. A VEGETAÇÃO MUDA A FREQUÊNCIA DE CÉU LIMPO? (fonte 2 do viés)
# ============================================================================
# Se municípios que perdem vegetação passam a ter mais (ou menos) dias claros,
# o viés de céu limpo fica correlacionado com o tratamento. Medida descritiva:
# correlação entre % de vegetação nativa e nº de observações válidas, na
# variação dentro do município após UF × ano.
# ⚠️ ATENÇÃO METODOLÓGICA: é uma correlação contemporânea, sem defasagens e
# sem controles; serve de alerta, não de estimativa causal.
print('\n📈 Correlação % nativa × nº de observações válidas (dentro do município):')
for reg, g in [('Brasil', base)] + list(base.groupby('regiao')):
    print(f'   {reg:<7} anual: {corr(g["r_pct_nativa"], g["r_aqua_n_obs_dia"]):+.3f}   '
          f'estação seca: {corr(g["r_pct_nativa"], g["r_aqua_n_obs_seca"]):+.3f}')

SAIDA.mkdir(exist_ok=True)
tab.to_csv(SAIDA / 'contraponto_lst_brdwgd.csv', index=False)
print(f'\n💾 {SAIDA / "contraponto_lst_brdwgd.csv"}')

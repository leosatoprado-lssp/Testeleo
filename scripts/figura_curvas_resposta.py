# ============================================================================
# FIGURA: CURVAS DE RESPOSTA DA TEMPERATURA À PERDA E AO GANHO DE VEGETAÇÃO
# ============================================================================
# 🎯 Objetivo: mostrar a resposta ano a ano (com antecipações t−3 a t−1 como
# teste de placebo) para o Brasil e para a Amazônia, que passa no placebo.
# Modelo: o mesmo de modelo_defasagens.py (efeitos fixos de município e de
# UF × ano, Conley 200 km), Y = LST diurna dessazonalizada do Aqua, 2003-2022
# (as antecipações exigem vegetação até 2025).
# Saídas: resultados/fig_curvas_resposta.png e resultados/curvas_resposta.csv
# Rodar da raiz do repositório:  python scripts/figura_curvas_resposta.py
# ============================================================================

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import modelo_defasagens as md

d = md.montar_base()
coords = md.coordenadas()
d['uf_ano'] = d['uf'] + '_' + d['ano'].astype(str)
d = d[d['ano'].between(2003, 2022)]
xs = ([f'{v}_a{k}' for v in ('perda', 'ganho') for k in (3, 2, 1)]
      + [f'{v}_l{k}' for v in ('perda', 'ganho') for k in range(md.K_MAX + 1)]
      + [f'{v}_l{md.K_MAX + 1}mais' for v in ('perda', 'ganho')])
ctrl = ['pct_area_urbana', 'pct_agua']

linhas = []
for nome, amostra in [('Brasil', d), ('Amazônia', d[d['bioma'] == 'Amazônia'])]:
    res, _, n, nm = md.estimar(amostra, 'aqua_lst_dia_dessaz_c', xs + ctrl, coords)
    for v in ('perda', 'ganho'):
        for rel, col in ([(-k, f'{v}_a{k}') for k in (3, 2, 1)]
                         + [(k, f'{v}_l{k}') for k in range(md.K_MAX + 1)]
                         + [(md.K_MAX + 2, f'{v}_l{md.K_MAX + 1}mais')]):
            linhas.append({'amostra': nome, 'tipo': v, 'ano_rel': rel,
                           'coef': res.loc[col, 'coef'], 'ep': res.loc[col, 'ep_conley'], 'n_mun': nm})
curvas = pd.DataFrame(linhas)
curvas.to_csv('resultados/curvas_resposta.csv', index=False)

# ----------------------------------------------------------------------------
# Gráfico: dois painéis (perda, ganho), mesma escala; Brasil × Amazônia com
# cor + marcador diferentes (identidade não depende só da cor) e IC de 95%.
# ----------------------------------------------------------------------------
COR = {'Brasil': '#2a78d6', 'Amazônia': '#eb6834'}
MARC = {'Brasil': 'o', 'Amazônia': 's'}
TINTA, TINTA2, GRADE = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.size': 12, 'axes.edgecolor': TINTA2, 'axes.labelcolor': TINTA2,
                     'xtick.color': TINTA2, 'ytick.color': TINTA2})
fig, eixos = plt.subplots(1, 2, figsize=(12, 5.6), sharey=True, facecolor='#fcfcfb')
titulos = {'perda': 'Perda de 1 pp de vegetação nativa', 'ganho': 'Ganho de 1 pp (regeneração)'}
rotulos_x = [-3, -2, -1] + list(range(0, md.K_MAX + 1)) + [md.K_MAX + 2]
for ax, v in zip(eixos, ('perda', 'ganho')):
    ax.set_facecolor('#fcfcfb')
    ax.axhline(0, color=TINTA2, lw=1)
    ax.axvspan(-3.5, -0.5, color=GRADE, alpha=0.6, lw=0)
    ax.text(-2, 0.235, 'placebo\n(antes)', ha='center', va='top', fontsize=10, color=TINTA2)
    for nome in ('Brasil', 'Amazônia'):
        c = curvas[(curvas['amostra'] == nome) & (curvas['tipo'] == v)]
        x = c['ano_rel'].to_numpy() + (0.12 if nome == 'Amazônia' else -0.12)
        ax.errorbar(x, c['coef'], yerr=1.96 * c['ep'], fmt=MARC[nome] + '-', color=COR[nome],
                    lw=2, ms=6, elinewidth=1.2, capsize=0, label=nome)
    ax.set_xticks(rotulos_x)
    ax.set_xticklabels([str(k) for k in rotulos_x[:-1]] + ['11+'])
    ax.set_xlabel('Anos desde a mudança da vegetação')
    ax.set_title(titulos[v], fontsize=14, fontweight='bold', color=TINTA, loc='left')
    ax.grid(True, axis='y', color=GRADE, lw=0.8)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
eixos[0].set_ylabel('Diferença na temperatura de superfície (°C por pp)')
eixos[0].set_ylim(-0.32, 0.24)
eixos[1].legend(frameon=False, loc='lower left', labelcolor=TINTA)
fig.suptitle('Resposta da temperatura de superfície (MODIS Aqua, dia) à vegetação nativa, 2003-2022',
             fontsize=13, color=TINTA, x=0.01, ha='left')
fig.text(0.01, 0.005, 'Efeitos fixos de município e UF × ano; barras = IC 95% (Conley, 200 km). '
         'Pontos antes de 0 deveriam ser zero (teste de placebo).', fontsize=9.5, color=TINTA2)
fig.tight_layout(rect=(0, 0.03, 1, 0.95))
fig.savefig('resultados/fig_curvas_resposta.png', dpi=160)
print('💾 resultados/fig_curvas_resposta.png')

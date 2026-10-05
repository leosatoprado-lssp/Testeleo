# ============================================================================
# MODELO DE DEFASAGENS DISTRIBUÍDAS: VEGETAÇÃO NATIVA → TEMPERATURA
# ============================================================================
# 🎯 Pergunta principal: quanto tempo a temperatura leva para responder à
# mudança da vegetação nativa do município, e a resposta difere entre PERDA e
# GANHO (regeneração)?
#
# Especificação (decidida com o Léo):
#   Y_it = Σ_k β⁻_k · Perda_i,t−k + Σ_k β⁺_k · Ganho_i,t−k + controles + α_i + δ_UF,t + ε_it
#   - Perda/Ganho: parte negativa/positiva da variação anual de % nativa (pp)
#   - k = 0 a 10 anos, mais um termo final com tudo o que aconteceu há 11+ anos
#     (endpoint acumulado, como em Schmidheiny e Siegloch, 2023)
#   - Duas versões: defasagens em FAIXAS (0-2, 3-5, 6-10) e ANO A ANO
#   - α_i: efeito fixo de município; δ_UF,t: efeito fixo de UF × ano
#   - Controles: % área urbana e % água
#   - Sem ponderação (cada município conta igual)
#   - Erros-padrão de Conley (200 km, Bartlett) + correlação serial no município
#
# Como ler os coeficientes: β_k é a diferença de temperatura (°C) k anos depois
# de uma perda (ou ganho) de 1 pp de vegetação nativa, com a vegetação ficando
# no novo nível. É a "curva de resposta" ao longo do tempo.
#
# Rodar da raiz do repositório:  python scripts/modelo_defasagens.py
# ============================================================================

import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import scipy.sparse as sp
from pyfixest.estimation import demean
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
import gee_lst_modis_municipios as base
from gee_grade_vegetacao import AEA_PROJ4

K_MAX = 10                                  # janela de defasagens (anos)
FAIXAS = {'0_2': (0, 2), '3_5': (3, 5), '6_10': (6, 10)}
CORTE_CONLEY_KM = 200
SAIDA = Path('resultados')


# ============================================================================
# 1. DADOS: PERDA E GANHO DE VEGETAÇÃO COM DEFASAGENS
# ============================================================================
def montar_base():
    p = pd.read_csv('dados/painel_municipios_1985_2025.csv')
    p = p[p['pct_nativa'].notna()].sort_values(['geocodigo', 'ano']).copy()   # sai Fernando de Noronha
    p['pct_agua'] = 100 * p['agua_ha'] / p['area_total_ha']

    # Variação anual de % nativa (pp) separada em perda e ganho (ambos ≥ 0)
    # ⚠️ ATENÇÃO METODOLÓGICA: perda e ganho entram separados porque a
    # literatura acha respostas assimétricas (Zhang et al., 2024): a perda
    # aquece mais do que o ganho esfria. Com uma variável única, o modelo
    # imporia a mesma resposta (com sinal trocado) aos dois.
    dN = p.groupby('geocodigo')['pct_nativa'].diff()
    p['perda'] = (-dN).clip(lower=0)
    p['ganho'] = dN.clip(lower=0)

    g = p.groupby('geocodigo')
    for v in ('perda', 'ganho'):
        for k in range(K_MAX + 1):
            p[f'{v}_l{k}'] = g[v].shift(k)
        # Antecipações (t+1 a t+3): teste de placebo. A vegetação de AMANHÃ não
        # pode causar a temperatura de HOJE; coeficientes grandes aqui indicariam
        # tendências prévias ou causalidade reversa (o "pré-tendência" do estudo
        # de evento).
        for k in range(1, 4):
            p[f'{v}_a{k}'] = g[v].shift(-k)
        # Endpoint: soma de tudo o que aconteceu há mais de K_MAX anos (desde 1986)
        acum = g[v].cumsum()
        p[f'{v}_l{K_MAX + 1}mais'] = acum.groupby(p['geocodigo']).shift(K_MAX + 1)
        for nome, (a, b) in FAIXAS.items():
            p[f'{v}_f{nome}'] = p[[f'{v}_l{k}' for k in range(a, b + 1)]].sum(axis=1, min_count=b - a + 1)

    # Variáveis de LST e o bioma (fixo)
    p = p.merge(pd.read_csv('dados/bioma_municipios.csv')[['geocodigo', 'bioma']], on='geocodigo', how='left')
    viz = Path('dados/vizinhanca_aneis_1985_2025.csv')
    if viz.exists():
        aneis = pd.read_csv(viz)
        cols = [c for c in aneis.columns if c.startswith('viz_nat_') and c.endswith('_pct')]
        aneis = aneis.sort_values(['geocodigo', 'ano'])
        # Vizinhança em t−1 (a vegetação de fora já estava lá no ano anterior)
        for c in cols:
            aneis[f'{c}_t1'] = aneis.groupby('geocodigo')[c].shift(1)
        p = p.merge(aneis[['geocodigo', 'ano'] + [f'{c}_t1' for c in cols]], on=['geocodigo', 'ano'], how='left')
    return p


# ============================================================================
# 2. ESTIMAÇÃO COM EFEITOS FIXOS E ERRO-PADRÃO DE CONLEY
# ============================================================================
# Pela lógica de Frisch-Waugh-Lovell, tiro os dois efeitos fixos de Y e de
# cada X (projeções alternadas do pyfixest) e rodo MQO nos resíduos.
# Conley: a covariância dos escores soma (a) pares de municípios no MESMO ano
# a até 200 km, com peso decrescente com a distância (Bartlett), e (b) todos
# os pares de anos do MESMO município (correlação serial). Segue Conley (1999)
# e Hsiang (2010). O pyfixest ainda não tem Conley, por isso a implementação.
# ⚠️ ATENÇÃO METODOLÓGICA: a distância é entre centroides; para municípios
# muito grandes, isso subestima a proximidade das bordas. O corte de 200 km
# compensa em parte; vale testar 100 e 500 km.
def coordenadas():
    malha = gpd.GeoDataFrame(base.carregar_malha(), geometry='geometry', crs='EPSG:4674').to_crs(AEA_PROJ4)
    c = malha.geometry.centroid
    return pd.DataFrame({'geocodigo': malha['CD_MUN'].astype(int), 'x': c.x / 1000, 'y': c.y / 1000})


def estimar(d, y, xs, coords, corte_km=CORTE_CONLEY_KM, fe_tempo='uf_ano'):
    d = d.dropna(subset=[y] + xs).copy()
    d[fe_tempo] = d[fe_tempo].astype('category').cat.codes
    fl = np.column_stack([d['geocodigo'].astype('category').cat.codes, d[fe_tempo]]).astype(np.uint64)
    m, _ = demean(d[[y] + xs].to_numpy(np.float64), fl, np.ones(len(d)))
    yd, X = m[:, 0], m[:, 1:]
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ yd
    e = yd - X @ beta
    U = X * e[:, None]                                   # escores

    # (a) parte espacial, ano a ano (matriz esparsa de pesos de Bartlett)
    xy = d[['geocodigo']].merge(coords, on='geocodigo', how='left')[['x', 'y']].to_numpy()
    anos = d['ano'].to_numpy()
    S = np.zeros((X.shape[1], X.shape[1]))
    for a in np.unique(anos):
        ix = np.flatnonzero(anos == a)
        arv = cKDTree(xy[ix])
        dist = arv.sparse_distance_matrix(arv, corte_km, output_type='coo_matrix')
        w = sp.coo_matrix((1 - dist.data / corte_km, (dist.row, dist.col)), shape=(len(ix), len(ix))).tocsr()
        w = w + sp.identity(len(ix), format='csr')        # a diagonal (d = 0) some do coo
        Ua = U[ix]
        S += Ua.T @ (w @ Ua)
    # (b) parte serial: todos os pares de anos diferentes dentro do município
    soma_mun = pd.DataFrame(U).groupby(d['geocodigo'].to_numpy()).sum().to_numpy()
    S += soma_mun.T @ soma_mun - U.T @ U
    V = XtX_inv @ S @ XtX_inv

    # Cluster por município (comparação)
    Vc = XtX_inv @ (soma_mun.T @ soma_mun) @ XtX_inv
    n_mun = d['geocodigo'].nunique()
    Vc *= n_mun / (n_mun - 1)

    res = pd.DataFrame({'coef': beta, 'ep_conley': np.sqrt(np.diag(V)), 'ep_cluster': np.sqrt(np.diag(Vc))}, index=xs)
    res['t'] = res['coef'] / res['ep_conley']
    return res, V, len(d), n_mun


# ============================================================================
# 3. LEITURA DA CURVA: EFEITO DE LONGO PRAZO E DEFASAGEM MÉDIA
# ============================================================================
# A curva β_0, ..., β_10 é a resposta em NÍVEL. O incremento de cada ano é
# b_k = β_k − β_(k−1); a defasagem média é Σ k·b_k / Σ b_k (só faz sentido se
# a curva for monotônica; com curva que sobe e desce, o número engana).
# ⚠️ ATENÇÃO METODOLÓGICA: com ano a ano, os β_k são ruidosos; a defasagem
# média herda esse ruído. Por isso a versão em faixas é a principal.
def defasagem_media(curva):
    inc = np.diff(np.concatenate([[0.0], curva]))
    return float((np.arange(len(curva)) * inc).sum() / inc.sum()) if abs(inc.sum()) > 1e-12 else np.nan


if __name__ == '__main__':
    t0 = time.time()
    d = montar_base()
    coords = coordenadas()
    d['uf_ano'] = d['uf'] + '_' + d['ano'].astype(str)
    d['bioma_ano'] = d['bioma'] + '_' + d['ano'].astype(str)
    controles = ['pct_area_urbana', 'pct_agua']
    f_vars = [f'{v}_f{n}' for v in ('perda', 'ganho') for n in FAIXAS] + [f'{v}_l{K_MAX + 1}mais' for v in ('perda', 'ganho')]
    a_vars = [f'{v}_l{k}' for v in ('perda', 'ganho') for k in range(K_MAX + 1)] + [f'{v}_l{K_MAX + 1}mais' for v in ('perda', 'ganho')]
    lead_vars = [f'{v}_a{k}' for v in ('perda', 'ganho') for k in (3, 2, 1)]
    viz_vars = [c for c in d.columns if c.startswith('viz_nat_') and c.endswith('_t1')]
    SAIDA.mkdir(exist_ok=True)

    # Especificações: (nome, Y, período, amostra, regressores, efeito fixo de tempo)
    principal = 'aqua_lst_dia_dessaz_c'
    robusta = (d['clima_imputado'] == 0) & (d['serie_com_degrau'] == 0)
    specs = [
        ('A_faixas',            principal, (2003, 2025), None,    f_vars, 'uf_ano'),
        ('A_ano_a_ano',         principal, (2003, 2025), None,    a_vars, 'uf_ano'),
        ('A_ano_a_ano_antecip', principal, (2003, 2022), None,    lead_vars + a_vars, 'uf_ano'),
        ('B_faixas_vizinhos',   principal, (2003, 2025), None,    f_vars + viz_vars, 'uf_ano'),
        ('C_bruta',             'aqua_lst_dia_c', (2003, 2025), None, f_vars, 'uf_ano'),
        ('C_zhang',             'zhang_lst_dia_c', (2003, 2020), None, f_vars, 'uf_ano'),
        ('C_seca',              'aqua_lst_dia_seca_c', (2003, 2025), None, f_vars, 'uf_ano'),
        ('C_bioma_ano',         principal, (2003, 2025), None,    f_vars, 'bioma_ano'),
        ('D_tmax_brdwgd',       'tmax_media_c', (2003, 2023), robusta, f_vars, 'uf_ano'),
        ('D_tmax_brdwgd_1996',  'tmax_media_c', (1996, 2023), robusta, f_vars, 'uf_ano'),
    ]
    tabelas = []
    for nome, y, (a0, a1), filtro, xs, fe in specs:
        if nome.startswith('B') and not viz_vars:
            continue
        amostra = d[d['ano'].between(a0, a1) & (filtro if filtro is not None else True)]
        res, V, n, nm = estimar(amostra, y, xs + controles, coords, fe_tempo=fe)
        res = res.assign(modelo=nome, y=y, n_obs=n, n_mun=nm)
        tabelas.append(res.reset_index(names='variavel'))
        print(f'\n⚙️ {nome}: {y}, {a0}-{a1}, {n} obs, {nm} municípios ({time.time() - t0:.0f}s)')
        print(res[['coef', 'ep_conley', 'ep_cluster', 't']].round(4).to_string())

        # Curva ano a ano: defasagem média e efeito no 10º ano
        if nome == 'A_ano_a_ano':
            for v in ('perda', 'ganho'):
                curva = res.loc[[f'{v}_l{k}' for k in range(K_MAX + 1)], 'coef'].to_numpy()
                inc = np.diff(np.concatenate([[0.0], curva]))
                monot = np.all(inc >= 0) or np.all(inc <= 0)
                dm = f'{defasagem_media(curva):.1f} anos' if monot else 'não se aplica (curva sobe e desce)'
                print(f'   {v}: defasagem média {dm} | pico no ano {int(np.argmax(np.abs(curva)))} | '
                      f'efeito no ano 10: {curva[-1]:+.4f} °C/pp | 11+ anos: {res.loc[f"{v}_l{K_MAX + 1}mais", "coef"]:+.4f}')

    out = pd.concat(tabelas, ignore_index=True)
    out.to_csv(SAIDA / 'modelo_defasagens.csv', index=False)
    print(f'\n💾 {SAIDA / "modelo_defasagens.csv"} ({(time.time() - t0) / 60:.1f} min)')

# ============================================================================
# TEMPERATURA DE SUPERFÍCIE (MODIS LST) POR MUNICÍPIO E ANO VIA EARTH ENGINE
# ============================================================================
# 🎯 Objetivo: gerar a variável resposta principal do painel (temperatura de
# superfície terrestre, LST) para os 5.571 municípios, 2001-2025, a partir de:
#   - MOD11A2 v061 (satélite Terra, passagem ~10h30 e ~22h30)
#   - MYD11A2 v061 (satélite Aqua,  passagem ~13h30 e ~01h30; mais perto da Tmáx)
# Ambos são compostos de 8 dias com pixel de ~1 km.
#
# Variáveis por município × ano × satélite (prefixo terra_ ou aqua_):
#   lst_dia_c / lst_noite_c ......... média anual da LST (°C)
#   lst_dia_jas_c / lst_noite_jas_c . média de julho a setembro (estação seca)
#   lst_dia_max_c ................... maior composto de 8 dias do ano (°C)
#   n_obs_dia / n_obs_noite ......... nº médio de compostos válidos por pixel
#                                     (máximo possível: 46 por ano)
#   frac_pix_dia .................... fração da área com ≥ 1 composto válido
#   *_rig ........................... média e n_obs com critério de QC rigoroso
#                                     (erro ≤ 1 K), para robustez; ver seção 3
#
# Como rodar:  python gee_lst_modis_municipios.py
# (precisa de `earthengine authenticate` feito antes; o projeto é o número
#  do projeto Cloud do Léo, registrado para uso não comercial)
#
# A extração é feita por lotes de municípios com getInfo (sem tarefas de
# exportação). Cada lote é salvo em disco, então se cair no meio basta
# rodar de novo que ele continua de onde parou.
# ============================================================================

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import ee
import geopandas as gpd
import pandas as pd
from google.oauth2.credentials import Credentials

# ----------------------------------------------------------------------------
# Configurações
# ----------------------------------------------------------------------------
PROJETO_EE = '136238277956'          # número do projeto Cloud (serve no lugar do ID)
MALHA = Path('geo/BR_Municipios_2025.shp')   # malha municipal IBGE 2025
PASTA_LOTES = Path('lst_lotes')              # cache dos lotes já extraídos
SAIDA = Path('dados/lst_modis_municipios_2001_2025.csv')

ANOS = list(range(2001, 2026))       # 2025 já está completo nas duas coleções
SATELITES = {
    'terra': 'MODIS/061/MOD11A2',
    'aqua':  'MODIS/061/MYD11A2',    # começa em jul/2002 → anos completos a partir de 2003
}
ANO_INICIAL = {'terra': 2001, 'aqua': 2003}

TOLERANCIA_GRAUS = 0.002             # simplificação das geometrias (~200 m)
MAX_MUN_LOTE = 120                   # nº máximo de municípios por requisição
MAX_AREA_LOTE_KM2 = 120_000          # área máxima por requisição (Amazônia pesa)
N_THREADS = 8                        # requisições simultâneas ao Earth Engine


# ============================================================================
# 1. CONEXÃO COM O EARTH ENGINE
# ============================================================================
# Uso as credenciais salvas pelo `earthengine authenticate`. Tiro a lista de
# escopos antes de montar a credencial: se algum escopo pedido não foi
# concedido na tela de autorização (ex.: Drive), a renovação do token falha
# com "invalid_scope"; sem a lista, o Google devolve os escopos concedidos.
def conectar():
    args = ee.oauth.get_credentials_arguments()
    args.pop('scopes', None)
    ee.Initialize(Credentials(None, **args), project=PROJETO_EE)


# ============================================================================
# 2. MALHA MUNICIPAL: LEITURA, LIMPEZA E SIMPLIFICAÇÃO
# ============================================================================
def carregar_malha():
    malha = gpd.read_file(MALHA, columns=['CD_MUN', 'NM_MUN', 'SIGLA_UF', 'AREA_KM2'])
    # As lagoas Mirim e dos Patos vêm na malha com código próprio, mas não são municípios
    malha = malha[~malha['CD_MUN'].isin(['4300001', '4300002'])].copy()

    # Simplifico para reduzir o tamanho da requisição (a malha original tem
    # 320 MB). Com tolerância de 0,002° (~200 m), bem menor que o pixel de
    # 1 km, a perda de área é desprezível; preserve_topology evita polígonos
    # inválidos ou vazios em municípios pequenos.
    # ⚠️ ATENÇÃO METODOLÓGICA: municípios vizinhos passam a ter bordas levemente
    # diferentes (pequenas sobreposições ou buracos). Para uma média zonal de
    # pixels de 1 km o efeito é mínimo, mas não use esta geometria para áreas.
    malha['geometry'] = malha.geometry.simplify(TOLERANCIA_GRAUS, preserve_topology=True)
    malha['geometry'] = malha.geometry.set_precision(1e-5)

    # Ordeno os municípios espacialmente (células de 2°) para que cada lote
    # cubra uma região compacta: o Earth Engine só processa os blocos de
    # pixels que tocam as geometrias do lote, então lotes compactos custam menos.
    centro = malha.geometry.representative_point()
    malha['celula'] = ((centro.y // 2).astype(int) * 1000 + (centro.x // 2).astype(int))
    malha = malha.sort_values(['celula', 'CD_MUN']).reset_index(drop=True)
    return malha


def montar_lotes(malha):
    # Agrupo por contagem e por área somada: um lote com vários municípios
    # gigantes do Amazonas estouraria o limite de memória/tempo do getInfo.
    lotes, atual, area = [], [], 0.0
    for i, a in zip(malha.index, malha['AREA_KM2']):
        if atual and (len(atual) >= MAX_MUN_LOTE or area + a > MAX_AREA_LOTE_KM2):
            lotes.append(atual)
            atual, area = [], 0.0
        atual.append(i)
        area += a
    if atual:
        lotes.append(atual)
    return lotes


def para_feature_collection(malha, idx):
    # GeoJSON → ee.FeatureCollection, levando só o geocódigo como propriedade
    gj = json.loads(malha.loc[idx, ['CD_MUN', 'geometry']].to_json())
    feats = [ee.Feature(ee.Geometry(f['geometry'], geodesic=False), {'geocodigo': f['properties']['CD_MUN']})
             for f in gj['features']]
    return ee.FeatureCollection(feats)


# ============================================================================
# 3. MÁSCARA DE QUALIDADE (QC) E CONVERSÃO PARA °C
# ============================================================================
# Bits do QC do MOD11/MYD11 (Wan, 2014, guia do usuário da coleção 6):
#   bits 0-1: qualidade obrigatória (00 = boa; 01 = produzida, ver outros bits;
#             10 = não produzida por nuvem; 11 = não produzida por outro motivo)
#   bits 6-7: erro médio da LST (00 = ≤ 1 K; 01 = ≤ 2 K; 10 = ≤ 3 K; 11 = > 3 K)
# Dois critérios, testados numa amostra de municípios (Aqua, 2020):
#   'amplo' (PRINCIPAL): qualidade 00 ou 01 com erro ≤ 2 K
#   'rig'   (ROBUSTEZ):  qualidade 00, ou 01 com erro ≤ 1 K
# ⚠️ ATENÇÃO METODOLÓGICA: o critério "só QC = 00" não serve aqui: na coleção
# 6.1 praticamente nenhum pixel noturno recebe QC = 00, e a noite ficaria vazia.
# Com o critério rigoroso, Belém e Manaus ficam com 4 a 5 compostos válidos por
# ano (de 46), e a média diurna de São Paulo cai 3 °C em relação ao amplo, sinal
# de que o filtro seleciona dias de uma época só (viés de céu limpo). Com o
# amplo sobram ~30 compostos por ano mesmo na Amazônia. A versão rigorosa sai
# com sufixo _rig para teste de robustez.
CRITERIOS = {'amplo': '', 'rig': '_rig'}


def mascarar(img, banda, banda_qc, criterio):
    qc = img.select(banda_qc)
    qualidade = qc.bitwiseAnd(3)
    erro = qc.rightShift(6).bitwiseAnd(3)
    if criterio == 'amplo':
        ok = qualidade.lte(1).And(erro.lte(1))
    else:
        ok = qualidade.eq(0).Or(qualidade.eq(1).And(erro.eq(0)))
    # Escala do produto: 0,02 K por unidade; converto para °C
    lst = img.select(banda).multiply(0.02).subtract(273.15)
    # copyProperties mantém a data (system:time_start), usada no filtro de julho-setembro
    return lst.updateMask(ok).copyProperties(img, ['system:time_start'])


def imagem_anual(colecao, ano):
    # Uma imagem com as bandas do ano para um satélite.
    # ⚠️ ATENÇÃO METODOLÓGICA: o composto de 8 dias é indexado pelo dia de
    # início; o último do ano (início em 27/12) inclui dias de janeiro do ano
    # seguinte. O efeito sobre a média anual é desprezível.
    ic = ee.ImageCollection(colecao).filterDate(f'{ano}-01-01', f'{ano + 1}-01-01')
    jas = ee.Filter.calendarRange(7, 9, 'month')
    bandas = []
    for criterio, suf in CRITERIOS.items():
        dia = ic.map(lambda i: mascarar(i, 'LST_Day_1km', 'QC_Day', criterio))
        noite = ic.map(lambda i: mascarar(i, 'LST_Night_1km', 'QC_Night', criterio))

        # ⚠️ ATENÇÃO METODOLÓGICA: a média anual é a média dos compostos válidos
        # de cada pixel. Se as falhas por nuvem se concentram na estação chuvosa,
        # a média fica puxada para a estação seca (viés de céu limpo). Por isso
        # sai junto o n_obs, que permite controlar ou filtrar por cobertura.
        # O count() deixa mascarados os pixels sem nenhuma observação; o
        # unmask(0) faz esses pixels entrarem como 0 na média municipal (sem
        # isso, o n_obs seria a média só dos pixels com dado, o que engana).
        n_dia = dia.count().unmask(0)
        n_noite = noite.count().unmask(0)
        bandas += [
            dia.mean().rename(f'lst_dia_c{suf}_{ano}'),
            noite.mean().rename(f'lst_noite_c{suf}_{ano}'),
            n_dia.rename(f'n_obs_dia{suf}_{ano}'),
            n_noite.rename(f'n_obs_noite{suf}_{ano}'),
        ]
        if criterio == 'amplo':
            bandas += [
                dia.filter(jas).mean().rename(f'lst_dia_jas_c_{ano}'),
                noite.filter(jas).mean().rename(f'lst_noite_jas_c_{ano}'),
                dia.max().rename(f'lst_dia_max_c_{ano}'),
                n_dia.gt(0).rename(f'frac_pix_dia_{ano}'),
            ]
    return ee.Image.cat(bandas)


def imagem_satelite(sat):
    colecao = SATELITES[sat]
    anos = [a for a in ANOS if a >= ANO_INICIAL[sat]]
    img = ee.Image.cat([imagem_anual(colecao, a) for a in anos])
    # Trabalho na projeção nativa do MODIS (sinusoidal, ~926 m) para não
    # reamostrar os pixels.
    proj = ee.ImageCollection(colecao).first().select('LST_Day_1km').projection()
    return img, proj


# ============================================================================
# 4. EXTRAÇÃO POR LOTE (reduceRegions + getInfo, com cache e novas tentativas)
# ============================================================================
# O redutor mean() do Earth Engine pondera cada pixel pela fração dele que cai
# dentro do polígono, então municípios menores que um pixel (ex.: Santa Cruz
# de Minas/MG, 3,4 km²) também recebem valor.
def extrair_lote(sat, n_lote, idx, malha, img, proj, tentativas=5):
    arq = PASTA_LOTES / f'{sat}_{n_lote:04d}.csv'
    if arq.exists():
        return arq, 'cache'
    fc = para_feature_collection(malha, idx)
    for t in range(tentativas):
        try:
            # tileScale maior divide o processamento em blocos menores
            # (evita "User memory limit exceeded" em lotes pesados)
            res = img.reduceRegions(collection=fc, reducer=ee.Reducer.mean(),
                                    crs=proj, scale=proj.nominalScale(),
                                    tileScale=4 if t < 2 else 16)
            feats = res.getInfo()['features']
            df = pd.DataFrame([f['properties'] for f in feats])
            df.to_csv(arq, index=False)
            return arq, f'ok ({t + 1}ª tentativa)'
        except Exception as e:
            espera = 2 ** (t + 2)
            print(f'   ⚠️ {sat} lote {n_lote}: {str(e)[:120]} → nova tentativa em {espera}s')
            time.sleep(espera)
    raise RuntimeError(f'{sat} lote {n_lote} falhou {tentativas} vezes')


# ============================================================================
# 5. REORGANIZAÇÃO: DE "LARGO" (banda_ano) PARA PAINEL (município × ano)
# ============================================================================
def para_longo(sat):
    df = pd.concat([pd.read_csv(a, dtype={'geocodigo': str}) for a in sorted(PASTA_LOTES.glob(f'{sat}_*.csv'))])
    longo = df.melt(id_vars='geocodigo', var_name='banda', value_name='valor')
    longo['ano'] = longo['banda'].str[-4:].astype(int)
    longo['variavel'] = f'{sat}_' + longo['banda'].str[:-5]
    return longo.pivot_table(index=['geocodigo', 'ano'], columns='variavel', values='valor').reset_index()


# ============================================================================
# 6. EXECUÇÃO
# ============================================================================
if __name__ == '__main__':
    t0 = time.time()
    conectar()
    PASTA_LOTES.mkdir(exist_ok=True)

    malha = carregar_malha()
    lotes = montar_lotes(malha)
    print('=' * 70)
    print(f'📊 {len(malha)} municípios em {len(lotes)} lotes por satélite')
    print('=' * 70)

    for sat in SATELITES:
        img, proj = imagem_satelite(sat)
        print(f'⏳ {sat}: {img.bandNames().size().getInfo()} bandas')
        with ThreadPoolExecutor(N_THREADS) as ex:
            futuros = {ex.submit(extrair_lote, sat, n, idx, malha, img, proj): n for n, idx in enumerate(lotes)}
            for k, fut in enumerate(as_completed(futuros), 1):
                arq, status = fut.result()
                if k % 10 == 0 or k == len(lotes):
                    print(f'   {sat}: {k}/{len(lotes)} lotes ({time.time() - t0:.0f}s)')

    # Junta os dois satélites numa tabela única município × ano
    painel_lst = para_longo('terra').merge(para_longo('aqua'), on=['geocodigo', 'ano'], how='left')
    painel_lst = painel_lst.round(3).sort_values(['geocodigo', 'ano'])
    painel_lst.to_csv(SAIDA, index=False)
    print(f'💾 {SAIDA}: {painel_lst.shape[0]} linhas × {painel_lst.shape[1]} colunas '
          f'({(time.time() - t0) / 60:.1f} min)')

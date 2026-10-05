# ============================================================================
# CORREÇÕES DO VIÉS DE CÉU LIMPO NA LST DIURNA DO AQUA (EARTH ENGINE)
# ============================================================================
# 🎯 Objetivo: gerar três versões alternativas da variável resposta principal
# (LST diurna do Aqua, MYD11A2 v061) para tratar o viés de céu limpo:
#
#   1. aqua_lst_dia_anom_c / aqua_lst_dia_dessaz_c  (2003-2025)
#      Média anual de ANOMALIAS: cada composto de 8 dias é comparado com a
#      média daquele mesmo período do ano (ex.: 1ª semana de agosto) no mesmo
#      pixel em 2003-2025. Um ano em que só as semanas secas foram vistas não
#      parece mais quente só por isso. A versão "dessaz" soma de volta o nível
#      médio do pixel, para ficar em °C comparáveis.
#
#   2. aqua_lst_dia_seca_c / aqua_n_obs_seca  (2003-2025)
#      Média só nos 3 meses consecutivos mais secos de cada pixel, como em
#      Butt et al. (2023). Os meses secos vêm da climatologia de chuva CHIRPS
#      2003-2025. Também sai mes_inicio_seca (moda no município).
#
#   3. zhang_lst_dia_c / zhang_lst_dia_seca_c  (2003-2020)
#      LST diária preenchida de Zhang et al. (2022), ~13h30, sem falhas.
#      ⚠️ ATENÇÃO METODOLÓGICA: os próprios autores dizem que os valores
#      preenchidos representam condição de céu claro. Essa base corrige a
#      amostragem sazonal (fonte 1 do viés), mas NÃO a diferença real entre
#      dias claros e nublados.
#
# Reaproveita a malha, os lotes e a máscara de QC de gee_lst_modis_municipios.py.
# Rodar da raiz do repositório:  python scripts/gee_lst_correcoes_municipios.py
# ============================================================================

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import ee
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import gee_lst_modis_municipios as base   # conectar, carregar_malha, montar_lotes, mascarar...

PASTA_LOTES = Path('lst_lotes_correcoes')
SAIDA = Path('dados/lst_correcoes_municipios_2003_2025.csv')
SAIDA_FIXA = Path('dados/mes_inicio_seca_municipios.csv')

ANOS = list(range(2003, 2026))          # anos completos do Aqua
ANOS_ZHANG = list(range(2003, 2021))    # cobertura da base de Zhang et al. (2022)
AQUA = 'MODIS/061/MYD11A2'
ZHANG = 'projects/sat-io/open-datasets/gap-filled-lst/gf_day_1km'
CHIRPS = 'UCSB-CHG/CHIRPS/PENTAD'


# ============================================================================
# 1. COLEÇÃO DO AQUA COM QC E "DIA DO ANO" DE CADA COMPOSTO
# ============================================================================
# Os compostos de 8 dias começam sempre nos mesmos dias do ano (1, 9, 17, ...,
# 361), então o dia de início identifica "qual semana do ano" é cada imagem.
def colecao_aqua():
    def preparar(img):
        lst = base.mascarar(img, 'LST_Day_1km', 'QC_Day', 'amplo')
        doy = ee.Date(img.get('system:time_start')).getRelative('day', 'year').add(1).int()
        return lst.set('doy', doy)
    return (ee.ImageCollection(AQUA)
            .filterDate(f'{ANOS[0]}-01-01', f'{ANOS[-1] + 1}-01-01')
            .map(preparar))


# ============================================================================
# 2. ITEM 1: ANOMALIAS EM RELAÇÃO À CLIMATOLOGIA DE CADA SEMANA DO ANO
# ============================================================================
# ⚠️ ATENÇÃO METODOLÓGICA: tirar a média de cada pixel × semana do ano equivale
# a um efeito fixo de pixel × semana. Com isso, a média anual deixa de depender
# de QUAIS semanas foram observadas. O que sobra de viés é só o de "dias claros
# dentro da mesma semana", que nenhuma base óptica resolve.
# A climatologia usa todo o período 2003-2025 (inclusive o próprio ano). Com
# 23 anos, o peso de cada ano na sua própria referência é pequeno (~4%).
def imagens_anomalia(col):
    semanas = ee.List.sequence(1, 361, 8)
    clima = ee.ImageCollection(semanas.map(
        lambda d: col.filter(ee.Filter.eq('doy', ee.Number(d).int())).mean()
                     .set('doy', ee.Number(d).int())))

    # Junta cada composto à climatologia da sua semana e subtrai
    juncao = ee.Join.saveFirst('clima').apply(
        col, clima, ee.Filter.equals(leftField='doy', rightField='doy'))
    anom = ee.ImageCollection(juncao).map(
        lambda img: ee.Image(img).subtract(ee.Image(ee.Image(img).get('clima')))
                    .copyProperties(img, ['system:time_start']))

    # Nível médio do pixel = média das 46 climatologias semanais (todas as
    # semanas com o mesmo peso, então não depende da cobertura de nuvens)
    nivel = clima.mean()
    bandas = []
    for ano in ANOS:
        a = anom.filterDate(f'{ano}-01-01', f'{ano + 1}-01-01').mean()
        bandas += [a.rename(f'aqua_lst_dia_anom_c_{ano}'),
                   a.add(nivel).rename(f'aqua_lst_dia_dessaz_c_{ano}')]
    return bandas


# ============================================================================
# 3. ITEM 2: TRÊS MESES MAIS SECOS DE CADA PIXEL (CHIRPS 2003-2025)
# ============================================================================
# Climatologia mensal de chuva → soma móvel de 3 meses consecutivos (circular,
# então nov-dez-jan também conta) → mês de início da janela mais seca.
# ⚠️ ATENÇÃO METODOLÓGICA: a janela é fixa no tempo (climatologia), e não
# recalculada a cada ano. Se fosse anual, um ano mais seco mudaria a própria
# janela, e a seleção de meses passaria a depender do clima daquele ano.
# ⚠️ ATENÇÃO METODOLÓGICA: quando a janela atravessa o ano (ex.: dez-jan-fev
# no norte de Roraima), o "ano" da estação seca junta jan-fev do ano t com dez
# do ano t. Afeta poucos municípios, todos no extremo norte.
def inicio_estacao_seca():
    chirps = ee.ImageCollection(CHIRPS).filterDate(f'{ANOS[0]}-01-01', f'{ANOS[-1] + 1}-01-01')
    n_anos = len(ANOS)
    mensal = [chirps.filter(ee.Filter.calendarRange(m, m, 'month')).sum().divide(n_anos)
              for m in range(1, 13)]
    janelas = []
    for s in range(12):
        soma = mensal[s].add(mensal[(s + 1) % 12]).add(mensal[(s + 2) % 12])
        # qualityMosaic escolhe, por pixel, a imagem com MAIOR valor da banda de
        # qualidade; uso a chuva negativa para pegar a janela MAIS SECA
        janelas.append(ee.Image.cat([soma.multiply(-1).rename('q'),
                                     ee.Image.constant(s).rename('inicio')]).toFloat())
    return ee.ImageCollection(janelas).qualityMosaic('q').select('inicio')   # 0 = janeiro


def no_periodo_seco(img, inicio):
    # Verdadeiro se o mês da imagem (0-11) está dentro da janela de 3 meses
    mes = ee.Number(ee.Date(img.get('system:time_start')).get('month')).subtract(1)
    return ee.Image.constant(mes).subtract(inicio).add(12).mod(12).lt(3)


def imagens_seca(col, inicio):
    bandas = []
    for ano in ANOS:
        sec = (col.filterDate(f'{ano}-01-01', f'{ano + 1}-01-01')
                  .map(lambda img: img.updateMask(no_periodo_seco(img, inicio))))
        bandas += [sec.mean().rename(f'aqua_lst_dia_seca_c_{ano}'),
                   sec.count().unmask(0).rename(f'aqua_n_obs_seca_{ano}')]
    return bandas


# ============================================================================
# 4. ITEM 3: BASE PREENCHIDA DE ZHANG ET AL. (2022)
# ============================================================================
# Valores em décimos de °C, uma imagem por dia (13h30), sem 29 de fevereiro.
def imagens_zhang(inicio):
    gf = ee.ImageCollection(ZHANG)
    bandas = []
    for ano in ANOS_ZHANG:
        dias = gf.filterDate(f'{ano}-01-01', f'{ano + 1}-01-01').map(
            lambda img: img.multiply(0.1).copyProperties(img, ['system:time_start']))
        seca = dias.map(lambda img: img.updateMask(no_periodo_seco(img, inicio)))
        bandas += [dias.mean().rename(f'zhang_lst_dia_c_{ano}'),
                   seca.mean().rename(f'zhang_lst_dia_seca_c_{ano}')]
    return bandas


# ============================================================================
# 5. EXTRAÇÃO POR LOTE (mesma lógica do script principal: cache + novas tentativas)
# ============================================================================
def extrair(nome, n_lote, idx, malha, img, proj, reducer, tentativas=5):
    arq = PASTA_LOTES / f'{nome}_{n_lote:04d}.csv'
    if arq.exists():
        return arq
    fc = base.para_feature_collection(malha, idx)
    for t in range(tentativas):
        try:
            res = img.reduceRegions(collection=fc, reducer=reducer, crs=proj,
                                    scale=proj.nominalScale(), tileScale=4 if t < 2 else 16)
            pd.DataFrame([f['properties'] for f in res.getInfo()['features']]).to_csv(arq, index=False)
            return arq
        except Exception as e:
            espera = 2 ** (t + 2)
            print(f'   ⚠️ {nome} lote {n_lote}: {str(e)[:120]} → nova tentativa em {espera}s')
            time.sleep(espera)
    raise RuntimeError(f'{nome} lote {n_lote} falhou {tentativas} vezes')


def rodar(nome, img, reducer, malha, lotes, proj):
    with ThreadPoolExecutor(base.N_THREADS) as ex:
        futuros = [ex.submit(extrair, nome, n, idx, malha, img, proj, reducer) for n, idx in enumerate(lotes)]
        for k, fut in enumerate(as_completed(futuros), 1):
            fut.result()
            if k % 15 == 0 or k == len(lotes):
                print(f'   {nome}: {k}/{len(lotes)} lotes')
    return pd.concat([pd.read_csv(a, dtype={'geocodigo': str})
                      for a in sorted(PASTA_LOTES.glob(f'{nome}_*.csv'))])


# ============================================================================
# 6. EXECUÇÃO
# ============================================================================
if __name__ == '__main__':
    t0 = time.time()
    base.conectar()
    PASTA_LOTES.mkdir(exist_ok=True)
    malha = base.carregar_malha()
    lotes = base.montar_lotes(malha)
    proj = ee.ImageCollection(AQUA).first().select('LST_Day_1km').projection()

    col = colecao_aqua()
    inicio = inicio_estacao_seca()

    # Três imagens separadas (uma por item): se uma falhar, as outras ficam no cache
    blocos = {
        'anom': ee.Image.cat(imagens_anomalia(col)),
        'seca': ee.Image.cat(imagens_seca(col, inicio)),
        'zhang': ee.Image.cat(imagens_zhang(inicio)),
    }
    largos = []
    for nome, img in blocos.items():
        print(f'⏳ {nome} ({time.time() - t0:.0f}s)')
        largos.append(rodar(nome, img, ee.Reducer.mean(), malha, lotes, proj).set_index('geocodigo'))

    # Mês de início da estação seca: moda dentro do município (é um mês, não faz
    # sentido tirar média). +1 para ficar 1 = janeiro.
    print('⏳ mês de início da estação seca')
    mes = rodar('mes', inicio.rename('mes_inicio_seca'), ee.Reducer.mode(), malha, lotes, proj)
    # Com uma banda só, o reduceRegions chama a coluna pelo nome do redutor ('mode')
    mes = mes.rename(columns={'mode': 'mes_inicio_seca'})
    mes['mes_inicio_seca'] = mes['mes_inicio_seca'].astype(int) + 1
    mes[['geocodigo', 'mes_inicio_seca']].to_csv(SAIDA_FIXA, index=False)

    # Largo (variavel_ano) → painel (município × ano)
    largo = pd.concat(largos, axis=1).reset_index()
    longo = largo.melt(id_vars='geocodigo', var_name='banda', value_name='valor')
    longo['ano'] = longo['banda'].str[-4:].astype(int)
    longo['variavel'] = longo['banda'].str[:-5]
    painel = (longo.pivot_table(index=['geocodigo', 'ano'], columns='variavel', values='valor')
                   .reset_index().round(3).sort_values(['geocodigo', 'ano']))
    painel.to_csv(SAIDA, index=False)
    print(f'💾 {SAIDA}: {painel.shape[0]} linhas × {painel.shape[1]} colunas '
          f'({(time.time() - t0) / 60:.1f} min)')

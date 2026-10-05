# ============================================================================
# GRADE DE 5 KM COM A FRAÇÃO DE VEGETAÇÃO NATIVA (MAPBIOMAS COL. 11, 1985-2025)
# ============================================================================
# 🎯 Objetivo: base para as variáveis de VIZINHANÇA (anéis de distância e
# índice de acesso à vegetação). Em vez de calcular cada anel no Earth Engine
# (5.571 municípios × 3 anéis × 41 anos, a 30 m), reduzo o MapBiomas uma vez a
# uma grade de 5 km e faço as contas de distância localmente
# (vizinhanca_municipios.py). Com a grade local, dá também para recalcular o
# índice de acesso para qualquer parâmetro de decaimento sem voltar ao EE,
# o que é necessário para ESTIMAR esse parâmetro.
#
# Bandas exportadas (todas como fração da área da célula, 0 a 1):
#   nat_AAAA ... vegetação nativa (florestal + não florestal), mesmas classes do painel
#   flo_AAAA ... só a nativa florestal
#   val ........ área com classificação do MapBiomas (terra e água dentro do Brasil)
#
# Projeção: cônica equivalente de Albers para a América do Sul (área igual),
# para que todas as células tenham 25 km² e as distâncias fiquem em metros.
# Download direto em blocos de 40 × 40 células (200 km) com computePixels.
# ⚠️ Tentei exportar para um asset, mas o caminho de asset exige o ID textual
# do projeto (o número não serve) e não temos esse ID; o download em blocos
# evita a exportação. Blocos maiores estouram o limite de pixels a 30 m.
# Saída (fora do git, ~100 MB): geo/grade_nativa_5km_1985_2025.npz
# Rodar da raiz do repositório:  python scripts/gee_grade_vegetacao.py
# ============================================================================

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import ee
import geopandas as gpd
import numpy as np
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).parent))
import gee_lst_modis_municipios as base

MAPBIOMAS = 'projects/mapbiomas-public/assets/brazil/lulc/collection11/mapbiomas_brazil_collection11_coverage_v3'
ANOS = list(range(1985, 2026))

# Mesmas classes de cobertura_nativa_municipios.py (ver comentários lá)
CLASSES_FLORESTAIS = [3, 4, 5, 6, 7, 49]
CLASSES_NAO_FLORESTAIS = [11, 12, 13, 50, 84]
CLASSES_NATIVAS = CLASSES_FLORESTAIS + CLASSES_NAO_FLORESTAIS

# Albers equivalente para a América do Sul (paralelos -5° e -42°)
WKT_ALBERS = ('PROJCS["SA_Albers",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],'
              'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],PROJECTION["Albers_Conic_Equal_Area"],'
              'PARAMETER["standard_parallel_1",-5],PARAMETER["standard_parallel_2",-42],'
              'PARAMETER["latitude_of_center",-32],PARAMETER["longitude_of_center",-60],'
              'PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1]]')
AEA_PROJ4 = '+proj=aea +lat_1=-5 +lat_2=-42 +lat_0=-32 +lon_0=-60 +x_0=0 +y_0=0 +datum=WGS84 +units=m'
LADO = 5000                      # tamanho da célula (m)
X0, Y0 = -1_750_000, 4_450_000   # canto superior esquerdo: limites do Brasil + 200 km
NX, NY = 1010, 980               # 5.050 km × 4.900 km


BLOCO = 40                        # células por lado em cada requisição
PASTA_BLOCOS = Path('geo/grade_blocos')
SAIDA = Path('geo/grade_nativa_5km_1985_2025.npz')


def fracao(binaria):
    # Fração da célula de 5 km coberta pela classe. O unmask(0) faz pixel fora
    # do Brasil contar como 0, então o resultado é fração da ÁREA DA CÉLULA. A
    # grade de saída (5 km, Albers) vem da própria requisição do computePixels.
    return binaria.unmask(0).reduceResolution(ee.Reducer.mean(), maxPixels=30000)


def imagem():
    mb = ee.Image(MAPBIOMAS)
    bandas = []
    for ano in ANOS:
        c = mb.select(f'classification_{ano}')
        bandas.append(fracao(c.remap(CLASSES_NATIVAS, [1] * len(CLASSES_NATIVAS), 0)).rename(f'nat_{ano}'))
        bandas.append(fracao(c.remap(CLASSES_FLORESTAIS, [1] * len(CLASSES_FLORESTAIS), 0)).rename(f'flo_{ano}'))
    # Área mapeada: qualquer pixel com classe > 0 no último ano
    bandas.append(fracao(mb.select('classification_2025').gt(0)).rename('val'))
    return ee.Image.cat(bandas).toFloat()


def pedir(img, x0, y0, n, tentativas=4):
    # Uma requisição de n × n células com canto superior esquerdo em (x0, y0).
    # Se o EE estourar a memória, divido em 4 sub-blocos (recursivo).
    req = {'expression': img, 'fileFormat': 'NUMPY_NDARRAY',
           'grid': {'dimensions': {'width': n, 'height': n},
                    'affineTransform': {'scaleX': LADO, 'shearX': 0, 'translateX': x0,
                                        'shearY': 0, 'scaleY': -LADO, 'translateY': y0},
                    'crsWkt': WKT_ALBERS}}
    for t in range(tentativas):
        try:
            a = ee.data.computePixels(req)
            # array estruturado (uma "coluna" por banda) → array 3D comum
            return np.stack([a[nm] for nm in a.dtype.names], axis=-1).astype(np.float32)
        except Exception as e:
            if 'memory' in str(e).lower() and n > 5:
                h = n // 2
                q = [[pedir(img, x0 + dj * h * LADO, y0 - di * h * LADO, h) for dj in (0, 1)] for di in (0, 1)]
                return np.vstack([np.hstack(linha) for linha in q])
            print(f'   ⚠️ bloco em ({x0:.0f}, {y0:.0f}), {n} células: {str(e)[:90]} → nova tentativa')
            time.sleep(2 ** (t + 2))
    raise RuntimeError(f'bloco em ({x0}, {y0}) falhou')


def baixar_bloco(img, bi, bj):
    arq = PASTA_BLOCOS / f'b_{bi:02d}_{bj:02d}.npy'
    if not arq.exists():
        np.save(arq, pedir(img, X0 + bj * BLOCO * LADO, Y0 - bi * BLOCO * LADO, BLOCO))
    return arq


if __name__ == '__main__':
    t0 = time.time()
    base.conectar()
    PASTA_BLOCOS.mkdir(parents=True, exist_ok=True)

    # Só baixo blocos que tocam o Brasil (malha + 10 km); o resto é zero
    malha = gpd.GeoDataFrame(base.carregar_malha(), geometry='geometry', crs='EPSG:4674').to_crs(AEA_PROJ4)
    # Índice espacial em vez de unir os 5.571 polígonos (a união falha por
    # pequenos erros de topologia que a simplificação introduz)
    nbi, nbj = int(np.ceil(NY / BLOCO)), int(np.ceil(NX / BLOCO))
    blocos = [(bi, bj) for bi in range(nbi) for bj in range(nbj)
              if len(malha.sindex.query(box(X0 + bj * BLOCO * LADO, Y0 - (bi + 1) * BLOCO * LADO,
                                            X0 + (bj + 1) * BLOCO * LADO, Y0 - bi * BLOCO * LADO).buffer(10_000),
                                        predicate='intersects')) > 0]
    print(f'📊 {len(blocos)} blocos de {BLOCO}×{BLOCO} células')

    img = imagem()
    nomes = img.bandNames().getInfo()
    # ⚠️ No modo restrito do EE (cota mensal esgotada) a concorrência cai muito;
    # rodar com N_THREADS=3 nesse caso
    with ThreadPoolExecutor(int(os.environ.get('N_THREADS', base.N_THREADS))) as ex:
        futuros = [ex.submit(baixar_bloco, img, bi, bj) for bi, bj in blocos]
        for k, f in enumerate(as_completed(futuros), 1):
            f.result()
            if k % 10 == 0 or k == len(blocos):
                print(f'   {k}/{len(blocos)} blocos ({time.time() - t0:.0f}s)')

    # Monta a grade completa (blocos fora do Brasil ficam com zero)
    grade = np.zeros((nbi * BLOCO, nbj * BLOCO, len(nomes)), dtype=np.float32)
    for bi, bj in blocos:
        grade[bi * BLOCO:(bi + 1) * BLOCO, bj * BLOCO:(bj + 1) * BLOCO] = np.load(PASTA_BLOCOS / f'b_{bi:02d}_{bj:02d}.npy')
    grade = grade[:NY, :NX]
    # Guardo como inteiro (fração × 10.000) para reduzir o arquivo
    np.savez_compressed(SAIDA, grade=np.round(grade * 10000).astype(np.uint16), bandas=np.array(nomes),
                        x0=X0, y0=Y0, lado=LADO)
    print(f'💾 {SAIDA}: {grade.shape} ({(time.time() - t0) / 60:.1f} min)')

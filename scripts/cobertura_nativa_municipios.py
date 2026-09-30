# ============================================================================
# BASE DE COBERTURA DE VEGETAÇÃO NATIVA POR MUNICÍPIO (MapBiomas Coleção 11)
# ============================================================================
# 🎯 Objetivo: a partir da tabela oficial do MapBiomas "Biomas, Estados e
# Municípios | Cobertura 30m - Coleção 11" (área em ha por classe, 1985-2025),
# montar uma base com, para cada município e ano:
#   - área de vegetação nativa (ha), separada em florestal e não florestal
#   - área total mapeada do município (ha)
#   - % de vegetação nativa em relação à área do município
#
# Fonte: MapBiomas (https://brasil.mapbiomas.org/downloads/estatisticas/),
# dados abertos sob licença CC BY 4.0 (exige citação).
#
# Uso:  python cobertura_nativa_municipios.py caminho/para/arquivo.xlsx
#       (também aceita .csv, caso a tabela tenha sido exportada)
# ============================================================================

import sys
import re
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ============================================================================
# 1. DEFINIÇÃO DAS CLASSES DE VEGETAÇÃO NATIVA
# ============================================================================
# Pensei em NÃO usar a coluna "Natural/Antrópico" (class_level_0) diretamente,
# porque ela considera natural também água, praias e outras áreas não
# vegetadas. Como a pergunta é sobre *vegetação* nativa, filtro pelos códigos
# de legenda do MapBiomas (Coleção 10/11):
#
#   Floresta (nível 1 = 1):
#     3  Formação Florestal      4  Formação Savânica
#     5  Mangue                  6  Floresta Alagável     49 Restinga Arbórea
#   Vegetação herbácea e arbustiva (nível 1 = 10):
#     11 Campo Alagado e Área Pantanosa   12 Formação Campestre
#     50 Restinga Herbácea
#
# ⚠️ ATENÇÃO METODOLÓGICA: deixei de fora 29 (Afloramento Rochoso) e
# 32 (Apicum), que o MapBiomas agrupa em "herbácea e arbustiva" mas que não
# são propriamente vegetação. Se quiser seguir o agrupamento oficial à risca,
# basta incluí-los em CLASSES_NAO_FLORESTAIS.
#
# ⚠️ ATENÇÃO METODOLÓGICA: a Coleção 11 criou classes nativas novas que não
# existiam na legenda da Coleção 10 e que precisam entrar na conta, senão a
# cobertura fica subestimada (principalmente no Pantanal e na costa):
#     7  Savana Alagável (grupo Floresta, 1.3)
#     84 Marisma (grupo herbácea e arbustiva, 2.2)
#     13 Mosaico Herbáceo-Arbustivo (2.4)
# Conferi os códigos na aba LEGEND_CODE e na coluna class_level_4 do arquivo.
CLASSES_FLORESTAIS = [3, 4, 5, 6, 7, 49]
CLASSES_NAO_FLORESTAIS = [11, 12, 13, 50, 84]

# Unidades que o IBGE registra com geocódigo mas que NÃO são municípios: as
# lagoas Mirim e dos Patos (RS). Removo para a base ter os 5.570 municípios.
GEOCODIGOS_NAO_MUNICIPIOS = [4300001, 4300002]
CLASSES_NATIVAS = CLASSES_FLORESTAIS + CLASSES_NAO_FLORESTAIS

# Classe de água (rio, lago e oceano). Uso para calcular também um % que
# desconta a água do denominador, útil em municípios costeiros ou com
# grandes reservatórios, onde a água "dilui" o percentual de vegetação.
CLASSES_AGUA = [33]


# ============================================================================
# 2. FUNÇÕES AUXILIARES PARA RECONHECER A ESTRUTURA DA TABELA
# ============================================================================
# O MapBiomas muda nomes de colunas e de abas entre coleções, então em vez de
# fixar nomes, eu detecto as colunas pelo conteúdo. Assim o script continua
# funcionando se a Coleção 12 renomear algo.

def eh_coluna_ano(nome):
    """Verdadeiro se o nome da coluna for um ano entre 1985 e 2035.
    Aceita '2025' e também 'y2025', que é o padrão da Coleção 11."""
    s = str(nome).strip().lower().lstrip('y')
    return bool(re.fullmatch(r'(19|20)\d{2}', s)) and 1985 <= int(s) <= 2035


# Siglas das UFs: a Coleção 11 traz só o nome do estado por extenso, e a sigla
# facilita cruzar com outras bases (IBGE, SEEG, PRODES)
SIGLAS_UF = {
    'Acre': 'AC', 'Alagoas': 'AL', 'Amapá': 'AP', 'Amazonas': 'AM', 'Bahia': 'BA',
    'Ceará': 'CE', 'Distrito Federal': 'DF', 'Espírito Santo': 'ES', 'Goiás': 'GO',
    'Maranhão': 'MA', 'Mato Grosso': 'MT', 'Mato Grosso do Sul': 'MS',
    'Minas Gerais': 'MG', 'Pará': 'PA', 'Paraíba': 'PB', 'Paraná': 'PR',
    'Pernambuco': 'PE', 'Piauí': 'PI', 'Rio de Janeiro': 'RJ',
    'Rio Grande do Norte': 'RN', 'Rio Grande do Sul': 'RS', 'Rondônia': 'RO',
    'Roraima': 'RR', 'Santa Catarina': 'SC', 'São Paulo': 'SP', 'Sergipe': 'SE',
    'Tocantins': 'TO',
}


def achar_coluna(df, candidatos):
    """Devolve a primeira coluna cujo nome (minúsculo) contém algum candidato."""
    for c in df.columns:
        nome = str(c).lower()
        if any(k in nome for k in candidatos):
            return c
    return None


def achar_coluna_geocodigo(df):
    """Procura a coluna com o código IBGE de 7 dígitos do município."""
    col = achar_coluna(df, ['geocod', 'geocode', 'cod_ibge', 'ibge', 'municipality_id', 'city_id'])
    if col is not None:
        return col
    # Plano B: qualquer coluna numérica em que quase todos os valores têm 7 dígitos
    for c in df.columns:
        v = pd.to_numeric(df[c], errors='coerce').dropna()
        if len(v) > 0 and (v.between(1_000_000, 9_999_999).mean() > 0.95):
            return c
    return None


def achar_coluna_classe(df):
    """Procura a coluna com o código numérico da classe de cobertura."""
    # Preferência por nomes explícitos de ID da classe (nível mais detalhado)
    for c in df.columns:
        nome = str(c).lower()
        if nome in ('class_id', 'classe_id', 'class', 'id_classe', 'level_4_id', 'class_level_4_id'):
            v = pd.to_numeric(df[c], errors='coerce')
            if v.notna().mean() > 0.95:
                return c
    # Plano B: coluna numérica cujos valores são majoritariamente códigos da legenda
    legenda = {3, 4, 5, 6, 9, 11, 12, 15, 20, 21, 23, 24, 25, 29, 30, 31, 32, 33,
               35, 36, 39, 40, 41, 46, 47, 48, 49, 50, 62, 27}
    for c in df.columns:
        v = pd.to_numeric(df[c], errors='coerce').dropna()
        if len(v) > 0 and v.isin(list(legenda)).mean() > 0.9:
            return c
    return None


# ============================================================================
# 3. CARREGAMENTO DA TABELA
# ============================================================================

def carregar(caminho):
    """Lê o arquivo do MapBiomas e devolve o DataFrame da aba de cobertura."""
    if caminho.lower().endswith('.csv'):
        # Tento vírgula e, se vier tudo numa coluna só, ponto e vírgula
        df = pd.read_csv(caminho)
        if df.shape[1] == 1:
            df = pd.read_csv(caminho, sep=';')
        return df

    # No .xlsx a tabela municipal costuma ter várias abas (cobertura,
    # transições, legenda...). Escolho a aba de cobertura: a que tem colunas de
    # ano e o maior número de linhas.
    abas = pd.read_excel(caminho, sheet_name=None)
    print(f"📊 Abas encontradas: {list(abas.keys())}")
    candidatas = {n: d for n, d in abas.items()
                  if any(eh_coluna_ano(c) for c in d.columns) or
                  achar_coluna(d, ['year', 'ano']) is not None}
    # Evito abas de transição (têm classe "de" e "para")
    candidatas = {n: d for n, d in candidatas.items() if 'transi' not in n.lower()} or candidatas
    nome = max(candidatas, key=lambda n: len(candidatas[n]))
    print(f"✓ Usando a aba: {nome} ({len(candidatas[nome]):,} linhas)")
    return candidatas[nome]


# ============================================================================
# 4. TRANSFORMAÇÃO PARA FORMATO LONGO (município × ano × classe)
# ============================================================================

def para_formato_longo(df):
    """Padroniza a tabela para colunas: geocodigo, municipio, uf, classe, ano, area_ha."""
    col_geo = achar_coluna_geocodigo(df)
    col_classe = achar_coluna_classe(df)
    col_mun = achar_coluna(df, ['municipality', 'municipio', 'município', 'city', 'name_muni'])
    col_uf = achar_coluna(df, ['state_acronym', 'uf', 'sigla'])
    if col_uf is None:
        col_uf = achar_coluna(df, ['state', 'estado'])

    print(f"🔎 Colunas detectadas -> geocódigo: {col_geo} | classe: {col_classe} | "
          f"município: {col_mun} | UF: {col_uf}")
    if col_geo is None or col_classe is None:
        raise ValueError("Não consegui identificar geocódigo ou classe. Colunas: "
                         f"{list(df.columns)}")

    ids = [c for c in [col_geo, col_mun, col_uf, col_classe] if c is not None]
    anos = [c for c in df.columns if eh_coluna_ano(c)]

    if anos:
        # Formato largo (uma coluna por ano), que é o padrão das estatísticas do MapBiomas
        longo = df[ids + anos].melt(id_vars=ids, var_name='ano', value_name='area_ha')
    else:
        # Formato já longo (colunas "year" e "area")
        col_ano = achar_coluna(df, ['year', 'ano'])
        col_area = achar_coluna(df, ['area', 'área'])
        longo = df[ids + [col_ano, col_area]].rename(columns={col_ano: 'ano', col_area: 'area_ha'})

    longo = longo.rename(columns={col_geo: 'geocodigo', col_classe: 'classe',
                                  **({col_mun: 'municipio'} if col_mun else {}),
                                  **({col_uf: 'uf'} if col_uf else {})})

    # Conversão de tipos; área ausente significa que a classe não ocorre no ano
    longo['geocodigo'] = pd.to_numeric(longo['geocodigo'], errors='coerce')
    longo['classe'] = pd.to_numeric(longo['classe'], errors='coerce')
    # Os anos vêm como 'y1985' na Coleção 11, então removo o prefixo antes de converter
    longo['ano'] = pd.to_numeric(longo['ano'].astype(str).str.lower().str.lstrip('y'), errors='coerce')
    longo['area_ha'] = pd.to_numeric(longo['area_ha'], errors='coerce').fillna(0)

    n_antes = len(longo)
    longo = longo.dropna(subset=['geocodigo', 'classe', 'ano'])
    print(f"✓ Formato longo: {len(longo):,} linhas ({n_antes - len(longo):,} descartadas por "
          f"geocódigo/classe/ano ausentes)")
    longo['geocodigo'] = longo['geocodigo'].astype(int)
    longo['classe'] = longo['classe'].astype(int)
    longo['ano'] = longo['ano'].astype(int)

    # Retiro as lagoas que têm geocódigo mas não são municípios
    n_lagoas = longo['geocodigo'].isin(GEOCODIGOS_NAO_MUNICIPIOS).sum()
    longo = longo[~longo['geocodigo'].isin(GEOCODIGOS_NAO_MUNICIPIOS)]
    print(f"✓ Removidas {n_lagoas:,} linhas das lagoas Mirim e dos Patos (não são municípios)")
    return longo


# ============================================================================
# 5. AGREGAÇÃO POR MUNICÍPIO E ANO
# ============================================================================

def agregar(longo):
    """Soma as áreas por município e ano e calcula os percentuais."""
    # ⚠️ ATENÇÃO METODOLÓGICA: a tabela vem cruzada com bioma, então um
    # município que está em dois biomas (ex.: Cerrado e Mata Atlântica)
    # aparece em linhas separadas. Por isso agrego por geocódigo somando
    # TODOS os biomas; agrupar por nome do município também seria arriscado,
    # porque há nomes repetidos entre estados (ex.: vários "Bom Jesus").
    longo = longo.copy()
    longo['nativa_florestal'] = np.where(longo['classe'].isin(CLASSES_FLORESTAIS), longo['area_ha'], 0)
    longo['nativa_nao_florestal'] = np.where(longo['classe'].isin(CLASSES_NAO_FLORESTAIS), longo['area_ha'], 0)
    longo['agua'] = np.where(longo['classe'].isin(CLASSES_AGUA), longo['area_ha'], 0)

    chaves = ['geocodigo', 'ano']
    base = (longo.groupby(chaves)
                 .agg(area_total_ha=('area_ha', 'sum'),
                      nativa_florestal_ha=('nativa_florestal', 'sum'),
                      nativa_nao_florestal_ha=('nativa_nao_florestal', 'sum'),
                      agua_ha=('agua', 'sum'))
                 .reset_index())

    # Nome e UF: pego a primeira ocorrência de cada geocódigo
    info = [c for c in ['municipio', 'uf'] if c in longo.columns]
    if info:
        nomes = longo.drop_duplicates('geocodigo')[['geocodigo'] + info]
        base = nomes.merge(base, on='geocodigo', how='right')

    # Se a UF veio por extenso, troco pela sigla (mantém o valor original se não achar)
    if 'uf' in base.columns:
        base['uf'] = base['uf'].map(SIGLAS_UF).fillna(base['uf'])

    base['nativa_ha'] = base['nativa_florestal_ha'] + base['nativa_nao_florestal_ha']

    # % em relação à área total mapeada do município
    base['pct_nativa'] = 100 * base['nativa_ha'] / base['area_total_ha']
    # % em relação à área terrestre (sem água), menos sensível a represas e mar
    area_terrestre = (base['area_total_ha'] - base['agua_ha']).replace(0, np.nan)
    base['pct_nativa_sem_agua'] = 100 * base['nativa_ha'] / area_terrestre

    cols_num = [c for c in base.columns if c.endswith('_ha') or c.startswith('pct_')]
    base[cols_num] = base[cols_num].round(2)

    # Ordem das colunas: identificação, depois os indicadores principais
    ordem = [c for c in ['geocodigo', 'municipio', 'uf', 'ano', 'pct_nativa', 'nativa_ha',
                         'area_total_ha', 'nativa_florestal_ha', 'nativa_nao_florestal_ha',
                         'agua_ha', 'pct_nativa_sem_agua'] if c in base.columns]
    return base[ordem].sort_values(chaves).reset_index(drop=True)


# ============================================================================
# 6. EXECUÇÃO, CONFERÊNCIAS E EXPORTAÇÃO
# ============================================================================

if __name__ == '__main__':
    caminho = sys.argv[1] if len(sys.argv) > 1 else 'MAPBIOMAS_BRAZIL-COL.11-BIOME_STATE_MUNICIPALITY.xlsx'
    saida = sys.argv[2] if len(sys.argv) > 2 else 'cobertura_nativa_municipios'

    print("=" * 70)
    print("⏳ Lendo a tabela do MapBiomas...")
    df = carregar(caminho)
    print(df.head(5).to_string())

    longo = para_formato_longo(df)
    base = agregar(longo)

    ultimo_ano = base['ano'].max()
    base_ultimo = base[base['ano'] == ultimo_ano]

    # 📊 Conferências de sanidade
    print("=" * 70)
    print(f"📊 Municípios: {base['geocodigo'].nunique():,} (o IBGE lista 5.570)")
    print(f"📊 Anos: {base['ano'].min()} a {ultimo_ano}")
    print(f"📊 % de vegetação nativa em {ultimo_ano}:")
    print(base_ultimo['pct_nativa'].describe().round(2).to_string())
    # Nenhum percentual pode passar de 100; se passar, há problema de agregação
    assert (base['pct_nativa'] <= 100.01).all(), "Percentual acima de 100%: revisar agregação"
    print("✓ Nenhum município com percentual acima de 100%")

    # 💾 Exportação: painel completo (longo) e recorte do último ano
    base.to_csv(f'{saida}_painel.csv', index=False, encoding='utf-8-sig')
    base_ultimo.to_csv(f'{saida}_{ultimo_ano}.csv', index=False, encoding='utf-8-sig')
    print(f"💾 Salvo: {saida}_painel.csv ({len(base):,} linhas) e {saida}_{ultimo_ano}.csv")

    # ⚠️ MELHORIA: para análises de desmatamento, vale derivar a variação da
    # área nativa entre anos (ex.: base.groupby('geocodigo')['nativa_ha'].diff())
    # e cruzar com o PRODES/INPE como verificação de robustez, lembrando que as
    # duas fontes usam definições diferentes de vegetação nativa.

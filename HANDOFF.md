# Pacote de transição: vegetação nativa × temperatura nos municípios brasileiros

Leia este arquivo inteiro antes de qualquer ação. Ele resume uma conversa longa
com o Léo (mestrando em Engenharia de Produção, EESC-USP) em que a base de dados
foi montada e a estratégia de pesquisa foi definida.

## 1. Como trabalhar com o Léo

- Tudo em português. Ele costuma estar no **celular**: prefira soluções que ele
  não precise executar; se precisar, que sejam poucos passos.
- Código Python: seguir a skill **estilo-ml-leo** (carregar antes de escrever
  código): comentários abundantes em português, seções numeradas com banners,
  marcadores `# ⚠️ ATENÇÃO METODOLÓGICA:` e `# ⚠️ MELHORIA:`.
- Textos acadêmicos: citações e referências em **ABNT**; evitar texto entre
  travessões e escrita com "cara de IA".
- **Não montar regressões nem código de inferência/ML sem ele pedir.** Ele quis
  primeiro fechar a estratégia. Construir variáveis e bases é permitido.
- Ele gosta de ser alertado sobre problemas metodológicos e de ver alternativas
  com prós e contras antes de decidir.

## 2. Pergunta de pesquisa (definida com ele)

- **Principal:** qual é a defasagem temporal (tempo médio) entre a mudança da
  vegetação nativa e a resposta da temperatura nos municípios, e ela difere entre
  **perda** e **ganho (regeneração)** de vegetação? Ele gostou especialmente da
  parte de regeneração e redução da temperatura.
- **Secundária:** qual é a escala espacial da influência da vegetação dos
  municípios vizinhos (decaimento com a distância), e como varia entre biomas?

Levantamento de literatura feito (resumo): o efeito do desmatamento sobre a
temperatura já está estabelecido (COHN et al., 2019; BUTT et al., 2023;
RODRIGUES et al., 2022; FRANCO et al., 2025; REDDINGTON et al., 2025). Lacunas
que a proposta ocupa: (a) quase ninguém estima a distribuição das defasagens nos
trópicos (o mais próximo é SU et al., 2026, na Europa: >82% do aquecimento de
distúrbios some em 10 anos); (b) distância tratada com anéis fixos, sem
parâmetro de decaimento estimado; (c) foco em Amazônia/Cerrado e em MODIS
(2001+); (d) nunca em escala municipal nacional. Franco et al. (2025) acharam
resposta logarítmica (efeito concentrado nos primeiros 10–40% de perda).
Winckler et al. (2019): temperatura do ar responde ~metade da de superfície no
local desmatado.

## 3. Estratégia de modelagem (acordada, ainda não implementada)

- Painel com **efeitos fixos de município e de UF × ano**.
- **Y:** temperatura média e temperatura máxima (ele pediu as duas).
- **Vegetação própria:** % nativa em t-1 (área absoluta só como interação
  % × área, porque com efeito fixo de município absoluto ∝ percentual).
- **Vizinhos:** (1) anéis de distância (ex.: 0–25, 25–50, 50–100 km) e
  (2) índice de "acesso à vegetação" Σⱼ Nativaⱼ × f(dᵢⱼ), com kernel de potência
  ou exponencial e **parâmetro de decaimento estimado** (SLX de Halleck Vega e
  Elhorst, 2015; "market access" de Donaldson e Hornbeck, 2016). Vizinhança
  calculada com TODOS os municípios, mesmo os excluídos da amostra de Y.
- **Memória:** defasagens da variação de vegetação, **janela ≥ 10 anos**
  (ele pediu). Métodos: defasagens distribuídas com defasagem média
  Σk·βₖ/Σβₖ; projeções locais (JORDÀ, 2005); estudo de evento
  (CALLAWAY; SANT'ANNA, 2021; equivalência com DL em SCHMIDHEINY; SIEGLOCH, 2023).
  **DLNM foi adiado** (ele não domina o método); não linearidade testada com
  faixas de tamanho da perda.
- **Controles:** % área urbana, % água. População só em robustez (existe de 2000
  em diante). Chuva só robustez (mediador). Uso do solo (pastagem, agricultura...)
  num segundo modelo. **Não usar emissões nem CO₂** (ele concordou).
- Erros-padrão com correção espacial (Conley ou cluster).
- Amostra principal: **robusta** (ver seção 5).

## 4. Temperatura de superfície (MODIS) via Earth Engine: FEITO

Decisão: **MODIS LST como Y principal**; BR-DWGD (já no painel) como complementar.
Estações do INMET descartadas como fonte principal (poucas séries longas, trocas
de equipamento, viés urbano). As defasagens são da vegetação (1985+), então
temperatura de 2001–2025 ainda permite 10–15 anos de defasagem.

Acesso ao Earth Engine:
- Projeto Cloud do Léo, registrado como uso não comercial. O **número do projeto
  (136238277956) funciona no lugar do ID** em `ee.Initialize(project=...)`.
- Autenticação pelo fluxo "notebook" (`earthengine authenticate
  --auth_mode=notebook --quiet`, ele abre o link no celular e cola o código).
  Ele não concedeu todos os escopos pedidos: ao montar a credencial, retirar a
  lista `scopes` (senão a renovação falha com `invalid_scope`); ver `conectar()`
  no script. Credencial fica só no container (`~/.config/earthengine`); numa
  sessão nova, repetir o fluxo. Lembrá-lo de revogar em myaccount.google.com.

O que foi extraído (`scripts/gee_lst_modis_municipios.py`, ~5 min de execução):
- MOD11A2 (Terra, 2001–2025) e MYD11A2 (Aqua, 2003–2025), v061, 8 dias, 1 km,
  malha IBGE 2025 simplificada a 0,002°, média zonal ponderada por fração de
  pixel, na projeção nativa. Resultado: `dados/lst_modis_municipios_2001_2025.csv`
  (139.275 linhas), já juntado ao painel por `scripts/juntar_lst_painel.py`.
- Colunas por satélite (`terra_` / `aqua_`): `lst_dia_c`, `lst_noite_c`,
  `lst_dia_jas_c`, `lst_noite_jas_c`, `lst_dia_max_c` (maior composto de 8 dias),
  `n_obs_dia`, `n_obs_noite` (compostos válidos por pixel, de 46),
  `frac_pix_dia`, e versões `_rig` (QC rigoroso) de média e n_obs.
- QC principal ("amplo"): qualidade 00 ou 01 com erro ≤ 2 K. Rigoroso: 00, ou
  01 com erro ≤ 1 K. Motivo: só QC = 00 zera a noite (quase nenhum pixel noturno
  recebe 00 na coleção 6.1) e o rigoroso deixa 8–18 compostos por ano de dia na
  região Norte, com seleção sazonal forte.

Achados das conferências (importantes para o modelo):
1. **Deriva orbital do Terra confirmada**: Terra − Aqua (dia) fica estável em
   −2,7 °C até 2020 e vai a −3,3 (2022), −3,9 (2024) e −4,4 °C (2025). Proposta:
   **Aqua como Y principal** (2003–2025, e mais perto da Tmáx), Terra só até 2020
   em robustez. Efeito de UF × ano absorve a parte comum, mas a deriva pode
   variar com a cobertura do solo (é justamente o regressor), então não confiar
   nisso.
2. **LST noturna é fraca**: mesmo no critério amplo, só 5–13 compostos por ano
   (N e CO ~5–7). A noite com QC rigoroso é inutilizável (quase toda vazia).
   Sugerir LST diurna como Y principal e noturna só como complemento. Vale
   investigar se o filtro de erro noturno (bits 6–7) está sendo severo demais.
3. Correlação com o BR-DWGD: LST dia × Tmáx do ar 0,78 (Aqua) no geral e 0,62
   dentro do município; LST noite × Tmín 0,92 e 0,33. Coerente com a literatura
   (LST e ar respondem diferente).
4. Viés de céu limpo: o `n_obs_dia` médio vai de 32–35 (Norte) a 43 (Sul);
   usar como controle ou filtro de cobertura.

Validações futuras (opcionais): marcar municípios com estação do INMET dentro
do território e repetir o modelo nesse subconjunto; comparar decaimento espacial
entre MODIS e BR-DWGD; média sazonal balanceada (média das médias mensais) para
reduzir o viés de céu limpo.

## 5. Estado da base (arquivos em `dados/`)

`painel_municipios_1985_2025.csv`: 5.571 municípios × 41 anos (228.411 linhas),
63 colunas (39 originais + 24 de MODIS LST, 2001–2025). Chave: `geocodigo` (IBGE, 7 dígitos) + `ano`.

| Grupo | Colunas | Fonte e observações |
|---|---|---|
| Identificação | geocodigo, municipio, uf, ano | |
| Fixas | latitude, longitude (sede), dist_mar_km | kelvins/municipios-brasileiros; distância geodésica da sede à costa (Natural Earth 1:10m, com fechamento morfológico de 2,5 km para excluir Lagoa dos Patos, Guanabara etc.) |
| População | populacao | Estimativas RIPSA/MS (lsbastos/popBR_mun), 2000–2025, compatibilizadas à malha atual. **Não é contagem censitária** (série ajustada; 2000 soma 174,7 mi). 1985–1999 vazio de propósito |
| Uso da terra | area_total_ha, area_urbana_ha, pct_area_urbana, pct_nativa, nativa_ha, nativa_florestal_ha, nativa_nao_florestal_ha, agua_ha, pct_nativa_sem_agua | MapBiomas Coleção 11 (1985–2025). Nativa florestal = classes 3, 4, 5, 6, 7, 49; não florestal = 11, 12, 13, 50, 84 (29 afloramento e 32 apicum excluídos). Urbana = classe 24. Água = 33. Soma de todos os biomas do município |
| Clima (BR-DWGD) | temp_media_c, tmax_media_c, tmin_media_c, amplitude_termica_c, tmax_jfm/amj/jas/ond_c, tmax_dia_mais_quente_c, tmax_p95_c, dias_quentes_p90, noites_quentes_p90, dias_tmax_35c, chuva_anual_mm, chuva_jas_mm, dias_com_chuva, umidade_media_pct, umidade_jas_pct | Estatísticas zonais diárias do BR-DWGD v3.2.3 (Saldanha et al., Zenodo 10.5281/zenodo.13906834), agregadas por ano, **1985–2023** (2024–2025 vazios). Tmed = (Tmáx+Tmín)/2. p90 relativo a 1985–2014 por município. `dias_com_chuva` não é confiável em municípios grandes da Amazônia |
| Controle de qualidade | clima_imputado, degrau_tmin_c, degrau_tmax_c, serie_com_degrau | 7 municípios com clima do município de origem (6 criados após a malha do BR-DWGD + Bombinhas). Degraus = maior salto de 5 anos contra a média da UF; `serie_com_degrau = 1` se > 1,5 °C (207 municípios, concentrados em SC e norte do PI; ex.: Esperantina/PI +2,4 °C na Tmín em 2019) |

- **Amostra robusta:** `clima_imputado == 0` e `serie_com_degrau == 0` →
  **5.357 municípios**, 208.923 linhas em 1985–2023. Sensibilidade com degrau
  > 1 °C exclui 964 municípios.
- Fernando de Noronha: sem clima e sem MapBiomas (só população).
- Lagoas Mirim e dos Patos (4300001, 4300002) removidas.

Outros arquivos em `dados/`: `base_municipios.csv` (retrato 2025),
`cobertura_nativa_municipios_2025.csv`, `kelvins.csv`, `pop_long.csv`.

Não incluídos (grandes, baixar se precisar):
- MapBiomas Col. 11, tabela municipal (necessária para o modelo com uso do solo):
  https://drive.google.com/uc?id=1otOqymHuixvkRGVl65zTTNyfaHo46Gqk&export=download
  (aba COVERAGE_11; anos como colunas y1985…y2025; página:
  https://brasil.mapbiomas.org/downloads/estatisticas/)
- Linha de costa/terra Natural Earth (raw.githubusercontent.com/nvkelso/natural-earth-vector).

## 6. Scripts (em `scripts/`)

- `cobertura_nativa_municipios.py`: MapBiomas → vegetação nativa por município/ano.
- `montar_base_municipios.py`: sedes, distância ao mar, população, área urbana.
- `montar_painel_municipios.py`: painel 1985–2025.
- `juntar_clima_painel.py`: junta o clima do BR-DWGD, imputação e degraus.
- `clima_anual_municipios_colab.ipynb`: extração do BR-DWGD (rodado pelo Léo no Colab).
- `gee_lst_modis_municipios.py`: extração do MODIS LST no Earth Engine (seção 4).
- `juntar_lst_painel.py`: junta a LST ao painel e roda as conferências.
- `gee_altitude_clima_municipios.js`: rascunho antigo para o GEE (altitude e
  TerraClimate); não usado, pode servir de referência.

Os scripts esperam as pastas `mb/` e `geo/` do ambiente antigo; ajuste caminhos.

## 7. Referências principais (ABNT)

BUTT, E. W. et al. Amazon deforestation causes strong regional warming. **PNAS**, v. 120, n. 45, e2309123120, 2023.
CALLAWAY, B.; SANT'ANNA, P. H. C. Difference-in-differences with multiple time periods. **Journal of Econometrics**, v. 225, n. 2, p. 200-230, 2021.
COHN, A. S. et al. Forest loss in Brazil increases maximum temperatures within 50 km. **Environmental Research Letters**, v. 14, n. 8, 084047, 2019.
DONALDSON, D.; HORNBECK, R. Railroads and American economic growth: a "market access" approach. **The Quarterly Journal of Economics**, v. 131, n. 2, p. 799-858, 2016.
FRANCO, M. A. et al. How climate change and deforestation interact in the transformation of the Amazon rainforest. **Nature Communications**, v. 16, 7944, 2025.
HALLECK VEGA, S.; ELHORST, J. P. The SLX model. **Journal of Regional Science**, v. 55, n. 3, p. 339-363, 2015.
JORDÀ, Ò. Estimation and inference of impulse responses by local projections. **American Economic Review**, v. 95, n. 1, p. 161-182, 2005.
REDDINGTON, C. L. et al. Tropical deforestation is associated with considerable heat-related mortality. **Nature Climate Change**, v. 15, p. 992-999, 2025.
RODRIGUES, A. A. et al. Cerrado deforestation threatens regional climate and water availability for agriculture and ecosystems. **Global Change Biology**, 2022.
SCHMIDHEINY, K.; SIEGLOCH, S. On event studies and distributed-lags in two-way fixed effects models. **Journal of Applied Econometrics**, v. 38, n. 5, p. 695-713, 2023.
SU, Y. et al. [estudo sobre recuperação térmica após distúrbios florestais na Europa]. **Nature Geoscience**, 2026. (conferir título e autores)
WINCKLER, J. et al. Different response of surface temperature and air temperature to deforestation in climate models. **Earth System Dynamics**, v. 10, p. 473-484, 2019.
XAVIER, A. C. et al. New improved Brazilian daily weather gridded data (1961–2020). **International Journal of Climatology**, v. 42, n. 16, p. 8390-8404, 2022.

// ============================================================================
// ALTITUDE E CLIMA POR MUNICÍPIO (Google Earth Engine - Code Editor)
// ============================================================================
// 🎯 Objetivo: para cada município brasileiro, calcular
//   - altitude média, mínima e máxima (SRTM, 30 m)
//   - temperatura média anual, precipitação média anual e umidade relativa
//     média anual (TerraClimate, normal climatológica 1991-2020, ~4 km)
// e exportar dois CSVs para o Google Drive, que depois serão juntados à base
// de cobertura vegetal pelo código IBGE (CD_MUN).
//
// Pré-requisito: subir a malha municipal do IBGE 2025 como asset
//   (Assets > NEW > Shape files > selecionar o .zip BR_Municipios_2025.zip)
// e colar o caminho do asset abaixo.
// ============================================================================


// ============================================================================
// 1. PARÂMETROS
// ============================================================================

// ⚠️ TROCAR pelo caminho do seu asset (aparece na aba Assets ao clicar nele)
var ASSET_MUNICIPIOS = 'projects/SEU_PROJETO/assets/BR_Municipios_2025';

// Campos da malha do IBGE (padrão desde 2021). Confira com print() se mudar.
var CAMPO_CODIGO = 'CD_MUN';
var CAMPO_NOME = 'NM_MUN';
var CAMPO_UF = 'SIGLA_UF';
var CAMPO_AREA = 'AREA_KM2';

// Escala do cálculo de altitude. O SRTM é de 30 m, mas rodar o Brasil todo a
// 30 m pode levar muitas horas. Com 90 m a média praticamente não muda; mínimo
// e máximo ficam levemente suavizados (picos isolados podem perder alguns
// metros). Se quiser a máxima precisão, troque para 30 e tenha paciência.
var ESCALA_DEM = 90;

// Pasta do Google Drive onde os CSVs serão salvos
var PASTA_DRIVE = 'GEE_municipios';


// ============================================================================
// 2. MALHA MUNICIPAL
// ============================================================================

var municipios = ee.FeatureCollection(ASSET_MUNICIPIOS);

// Conferência rápida: nº de feições e campos do primeiro município.
// A malha do IBGE tem os municípios mais as lagoas Mirim e dos Patos,
// que também têm código; elas serão descartadas na junção em Python.
print('Nº de feições na malha:', municipios.size());
print('Exemplo de feição:', municipios.first());


// ============================================================================
// 3. ALTITUDE (SRTM 30 m)
// ============================================================================

// Pensei em usar o SRTM (NASA, v3) por ser a referência mais usada em
// trabalhos acadêmicos no Brasil. O Copernicus GLO-30 é uma alternativa com
// menos ruído em áreas de floresta densa (o SRTM mede o topo do dossel em
// parte da Amazônia, então a altitude pode ficar alguns metros acima do solo).
// ⚠️ ATENÇÃO METODOLÓGICA: por esse motivo, na Amazônia a altitude mínima e a
// média podem estar superestimadas em alguns metros. Para altitude isso é
// irrelevante na maioria das análises, mas vale registrar como limitação.
var dem = ee.Image('USGS/SRTMGL1_003').select('elevation');

// Um único reduceRegions com três redutores combinados (média, mínimo e
// máximo), em vez de três passagens separadas, para economizar processamento
var redutorAltitude = ee.Reducer.mean()
  .combine({reducer2: ee.Reducer.min(), sharedInputs: true})
  .combine({reducer2: ee.Reducer.max(), sharedInputs: true});

var altitude = dem.reduceRegions({
  collection: municipios,
  reducer: redutorAltitude,
  scale: ESCALA_DEM,
  tileScale: 8   // divide o processamento em blocos menores e evita erro de memória
});


// ============================================================================
// 4. CLIMA (TerraClimate, normal 1991-2020)
// ============================================================================

// Escolhi o TerraClimate porque tem resolução de ~4 km, melhor que o
// ERA5-Land (~11 km), o que importa para municípios pequenos. Uso o período
// 1991-2020, que é a normal climatológica padrão da OMM atualmente.
var terra = ee.ImageCollection('IDAHO_EPSCOR/TERRACLIMATE')
  .filterDate('1991-01-01', '2021-01-01');

// Para cada mês, calculo as variáveis já nas unidades físicas.
// Fatores de escala do TerraClimate: tmmx e tmmn = 0,1 °C;
// vap = 0,001 kPa; vpd = 0,01 kPa; pr já está em mm.
var mensal = terra.map(function(img) {
  var tmax = img.select('tmmx').multiply(0.1);
  var tmin = img.select('tmmn').multiply(0.1);

  // ⚠️ ATENÇÃO METODOLÓGICA: o TerraClimate não traz a temperatura média
  // diretamente, então uso (Tmáx + Tmín) / 2, que é a aproximação usada
  // pelo INMET nas normais climatológicas.
  var tmed = tmax.add(tmin).divide(2).rename('temp_media_c');

  // Umidade relativa = pressão de vapor real / pressão de vapor de saturação.
  // Como VPD = es - ea, a saturação é es = ea + VPD, logo UR = ea / (ea + VPD).
  var ea = img.select('vap').multiply(0.001);
  var vpd = img.select('vpd').multiply(0.01);
  var ur = ea.divide(ea.add(vpd)).multiply(100).rename('umidade_rel_pct');

  var pr = img.select('pr').rename('precip_mm');
  return tmed.addBands(ur).addBands(pr);
});

// Normais: média dos 360 meses para temperatura e umidade. Para a chuva,
// a média mensal vezes 12 dá o total anual médio (mm/ano).
var normais = mensal.select(['temp_media_c', 'umidade_rel_pct']).mean()
  .addBands(mensal.select('precip_mm').mean().multiply(12).rename('precip_anual_mm'));

// Escala nativa do TerraClimate (~4,6 km). O redutor mean() do GEE pondera
// pela fração do pixel dentro do polígono, então até municípios menores que
// um pixel recebem valor.
var clima = normais.reduceRegions({
  collection: municipios,
  reducer: ee.Reducer.mean(),
  scale: 4638.3,
  tileScale: 4
});


// ============================================================================
// 5. CONFERÊNCIA VISUAL (opcional)
// ============================================================================

// Mostro os mapas para conferir se as camadas fazem sentido antes de exportar
Map.setCenter(-52, -14, 4);
Map.addLayer(dem, {min: 0, max: 1500, palette: ['006633', 'E5FFCC', '662A00', 'D8D8D8', 'FFFFFF']}, 'Altitude SRTM', false);
Map.addLayer(normais.select('temp_media_c'), {min: 10, max: 30, palette: ['blue', 'yellow', 'red']}, 'Temperatura média', false);
Map.addLayer(normais.select('precip_anual_mm'), {min: 500, max: 3000, palette: ['white', 'blue']}, 'Precipitação anual', false);
Map.addLayer(municipios.style({color: '333333', fillColor: '00000000', width: 0.5}), {}, 'Municípios');


// ============================================================================
// 6. EXPORTAÇÃO 💾
// ============================================================================
// Rodar o script cria duas tarefas na aba "Tasks" (canto superior direito).
// É preciso clicar em RUN em cada uma. A de altitude é a mais demorada.

Export.table.toDrive({
  collection: altitude,
  description: 'altitude_municipios',
  folder: PASTA_DRIVE,
  fileFormat: 'CSV',
  // selectors define as colunas e evitam exportar a geometria (arquivo leve)
  selectors: [CAMPO_CODIGO, CAMPO_NOME, CAMPO_UF, CAMPO_AREA, 'mean', 'min', 'max']
});

Export.table.toDrive({
  collection: clima,
  description: 'clima_municipios',
  folder: PASTA_DRIVE,
  fileFormat: 'CSV',
  selectors: [CAMPO_CODIGO, 'temp_media_c', 'umidade_rel_pct', 'precip_anual_mm']
});

// ⚠️ MELHORIA: se quiser comparar com outra fonte climática, dá para trocar
// o TerraClimate pelo ERA5-Land ('ECMWF/ERA5_LAND/MONTHLY_AGGR'), que traz a
// temperatura média real e o ponto de orvalho, ao custo de resolução menor.

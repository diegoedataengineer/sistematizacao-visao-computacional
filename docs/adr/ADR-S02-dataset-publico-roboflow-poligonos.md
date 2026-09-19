# ADR-S02 — Dataset público do Roboflow Universe com anotações em polígono; splits fixos com seed; classe única

**Status:** Aceito
**Base:** requisitos da Sistematização (≥ 300 imagens anotadas; público com fonte citada ou anotado pelo grupo; treino/validação/teste); ADR-011, ADR-018

## Contexto

Anotar 300+ imagens à mão com polígonos consumiria o prazo inteiro. O enunciado permite dataset público com fonte citada (Roboflow Universe, Kaggle, subconjunto do COCO). Para servir à detecção **e** à segmentação com um só download, a anotação precisa ser em **polígono** (instance segmentation): o Ultralytics deriva as caixas dos polígonos, e o formato de exportação "YOLOv11 segmentation" gera `labels/*.txt` prontos.

Candidatos encontrados (todos CC BY 4.0, instance segmentation):

| Dataset | Imagens | Observação |
|---|---|---|
| [Potholes and Roads Instance Segmentation](https://universe.roboflow.com/pothole-vsmtu/potholes-and-roads-instance-segmentation) | 1 355 | usado em artigo sobre veículos autônomos; **candidato principal** |
| [Pothole Segmentation (GP)](https://universe.roboflow.com/gp-grakz/pothole-segmentation-g6hbh) | 1 600 | 5 classes (manhole, pothole, potholes, unmarked, bump) — exige filtrar/mesclar |
| [PotholeDetection (Projects)](https://universe.roboflow.com/projects-hra06/potholedetection-cxudc) | 1 900 | variações de classe — verificar |
| [Pothole Segmentation (pathole)](https://universe.roboflow.com/pathole-w3cem/pothole-segmentation-nr0lb) | 802 | com modelo pré-treinado de referência |
| [Pothole_Segmentation_YOLOv8 (Farzad)](https://universe.roboflow.com/farzad/pothole_segmentation_yolov8) | 300 | **mínimo exato** — só como fallback |

Alternativas só com caixas (Kaggle: [Road Damage Dataset](https://www.kaggle.com/datasets/lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes), 2 009 imagens, Itália; [Potholes-Detection-YOLOv8](https://www.kaggle.com/datasets/anggadwisunarto/potholes-detection-yolov8)) serviriam à detecção, mas obrigariam a anotar máscaras ou a usar SAM para gerá-las — mais tempo e mais risco.

## Decisão

1. Usar **um** dataset do Roboflow Universe com polígonos, escolhido pelos critérios da SPEC-S01: ≥ 600 imagens, classe `pothole` majoritária, licença CC BY 4.0, imagens variadas (dia/noite, seco/molhado se possível). Primeira opção: *Potholes and Roads Instance Segmentation*; se a inspeção (SPEC-S01 §2) reprovar, o próximo da tabela.
2. **Reduzir a classe única `pothole`** quando houver classes auxiliares (manhole, crack): remapear ou descartar no `data.yaml` — mantém o problema limpo e a matriz de confusão legível (`pothole` × `background`).
3. **Splits fixos:** usar os splits do Roboflow se existirem os três; senão, re-dividir 70/15/15 com `seed=0`, estratificando por número de instâncias por imagem. O `test` é gerado uma vez e **não é aberto** até a Fase 4.
4. Citar no relatório: nome, autor, URL, versão, licença, contagens por split, e a data do download.
5. Augmentation: só o padrão do Ultralytics (mosaic, flips, HSV, escala) — documentado. Sem dados sintéticos (ADR-018 não é necessário no prazo).

## Alternativas rejeitadas

- **Anotar o próprio dataset** (CVAT/Label Studio): inviável em 2 dias com polígonos.
- **Caixas + SAM para máscaras:** pipeline agêntico válido (ADR-013), mas acrescenta uma etapa de revisão humana que não cabe.
- **Vários datasets mesclados:** heterogeneidade de anotação e risco de duplicatas entre splits (vazamento).

## Achado na inspeção (sexta 18/09, após o download)

O dataset escolhido foi *Potholes and Roads Instance Segmentation* v5 (1 076 / 138 / 141 imagens; 6 264 instâncias `pothole` após descartar `road`). Na inspeção das labels:

- **55% das instâncias (3 430) estão anotadas apenas como caixa**, não como polígono; 45% (2 834) são polígonos. Os dois formatos aparecem misturados em 1 270 arquivos.
- O Ultralytics **rejeita arquivos que misturam caixa e polígono** ("labels mix") — o primeiro treino descartou 1 017 das 1 076 imagens de treino sem avisar de forma visível. Correção: `scripts/filtrar_classes.py` converte cada caixa em polígono retangular de 4 pontos e apaga os caches.
- Para a **detecção** a conversão é neutra. Para a **segmentação**, mais da metade das máscaras de referência são retângulos: a métrica de máscara vai premiar retângulos e a "qualidade das máscaras" (20% do barema) precisa ser discutida com honestidade (ver ADR-S04 para o refinamento com SAM).

## Consequências

- A qualidade das anotações é herdada: polígonos grosseiros — e, aqui, retângulos em 55% dos casos — limitam o mAP de máscara e o que ele significa; registrar em "limitações" e na comparação caixas × máscaras.
- Possível vazamento interno do dataset original (mesma rua em splits diferentes): mencionar como limitação; não há como auditar em 2 dias.
- Fonte estrangeira (asfalto, sinalização diferentes): o vídeo próprio em vias brasileiras vira também um **teste de generalização de domínio** — bom argumento para a análise crítica.

## Relacionados

ADR-S01, ADR-S05, SPEC-S01.

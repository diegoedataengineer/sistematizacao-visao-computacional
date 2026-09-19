# Proposta — Detecção e segmentação de buracos em vias urbanas

**Grupo:** Diego Nunes de Morais <!-- adicionar os demais -->
**Cenário:** Cidades Inteligentes — buracos em vias
**Disciplina:** Visão Computacional e Reconhecimento de Padrões (CEUB) · Prof. Romes Heriberto
**Data:** 18/09/2026

## Problema

Prefeituras recebem reclamações de buracos de forma dispersa e sem medida do dano. Propomos um sistema que, a partir de imagens capturadas por veículos em circulação (celular no para-brisa ou dashcam), **detecta** buracos e **delimita sua extensão** em nível de pixel. A caixa localiza; a máscara permite estimar a área relativa de cada buraco e, no vídeo, contar buracos distintos por trecho — insumos para priorizar ordens de serviço.

## Classes-alvo

`pothole` (classe única). Rachaduras, remendos e tampas de bueiro não são alvo; esperamos que apareçam na análise de erros como confusores.

## Dados

*Potholes and Roads Instance Segmentation*, workspace `pothole-vsmtu`, Roboflow Universe, versão 5, licença CC BY 4.0 — https://universe.roboflow.com/pothole-vsmtu/potholes-and-roads-instance-segmentation/dataset/5 (download em 18/09/2026).

O dataset original tem duas classes (`pothole`, `road`); mantivemos apenas `pothole`. Splits do próprio dataset, mantidos fixos:

| split | imagens | instâncias | inst./imagem | imagens sem buraco | luminância média | instâncias < 32² px |
|---|---|---|---|---|---|---|
| treino | 1 076 | 5 097 | 4,74 | 29 | 129,7 | 2 016 (40%) |
| validação | 138 | 627 | 4,54 | 6 | 126,8 | 255 (41%) |
| teste | 141 | 540 | 3,83 | 11 | 131,5 | 200 (37%) |

Resoluções dominantes: 1227×920 (714 imagens) e 720×720 (280); há um grupo pequeno de imagens em ~400×300. Luminância concentrada entre 100 e 160 (cenas diurnas; sem noite). Cerca de 40% das instâncias são pequenas (< 32² px), o que deve pressionar a revocação em buracos distantes.

**Formato das anotações.** Na inspeção, 2 834 instâncias (45%) estão anotadas em polígono e 3 430 (55%) apenas em caixa, misturadas nos mesmos arquivos (1 270 arquivos). Convertemos as caixas em polígonos retangulares para uniformizar o formato. Para a detecção isso é neutro (as caixas são derivadas dos polígonos); para a segmentação, significa que mais da metade das máscaras de referência são retângulos — limitação que trataremos explicitamente na avaliação e na comparação caixas × máscaras.

## Ferramenta de anotação

Dataset já anotado (Roboflow Annotate). Não prevemos re-anotação manual no prazo. Como possível refinamento, avaliaremos gerar máscaras para as instâncias anotadas só por caixa usando um segmentador promptável (SAM) com a caixa como prompt, com revisão amostral — o que também será registrado no relatório.

## Métodos

- **Detecção:** YOLO11s (Ultralytics), fine-tuning a partir de pesos COCO; 640 px, 50 épocas com parada antecipada; baseline YOLO11n para comparação.
- **Segmentação:** YOLO11s-seg (instâncias), mesmo protocolo e mesmos dados; comparação caixas × máscaras e razão área da máscara / área da caixa.
- **Avaliação:** validação para escolher modelo e limiar (curva precisão × revocação); teste avaliado uma única vez com mAP@0,5, mAP@0,5:0,95, precisão, revocação, matriz de confusão, IoU e Dice; análise qualitativa de falsos positivos e negativos; métricas por tamanho de instância.
- **Vídeo:** gravação própria em vias urbanas (≥ 30 s) com inferência; rastreamento ByteTrack para contagem de buracos distintos (bônus).

## Riscos

Generalização para vias brasileiras (dataset estrangeiro, asfalto e sinalização diferentes); buracos pequenos e distantes; sombras, poças e remendos como falsos positivos; trocas de identidade no rastreamento em curvas e freadas.

# ADR-S05 — Protocolo de avaliação: `val` para todas as decisões, `test` tocado uma vez; métricas do barema; limiar escolhido por varredura P×R; análise de erros obrigatória

**Status:** Aceito
**Base:** enunciado (mAP@0.5, mAP@0.5:0.95, IoU, precisão/recall, matriz de confusão, análise qualitativa de erros; teste "nunca visto"); ADR-010, ADR-011, ADR-012

## Decisão

1. **Três splits, papéis estritos.** `train` ajusta; `val` escolhe modelo (`best.pt`), `conf`, `iou` do NMS e resolução; `test` é avaliado **uma única vez** na Fase 4, com os parâmetros já congelados, e o resultado entra no relatório sem retoques.
2. **Métricas reportadas (teste)**:
   - Detecção: Box **mAP@0,5**, **mAP@0,5:0,95**, **P**, **R** (no `conf` escolhido), **matriz de confusão** (`pothole` × `background`).
   - Segmentação: Mask mAP@0,5, mAP@0,5:0,95, P, R; **IoU** e **Dice** médios por imagem (máscara predita unida × máscara verdadeira unida).
   - Também em `val`, na mesma tabela, para mostrar a diferença val→test (sinal de sobreajuste ou de sorte).
3. **Ponto de operação:** varrer `conf` ∈ {0,05, 0,10, …, 0,60} em `val`, plotar P×R e escolher pelo **custo do erro no cenário**: para manutenção viária, um buraco não detectado (FN) custa mais que um alarme falso revisado por um operador (FP) → favorecer revocação, com precisão ≥ 0,7 como piso. Registrar o `conf` e o `iou` (NMS, padrão 0,7) no relatório.
4. **Análise de erros:** ≥ 4 falsos positivos e ≥ 4 falsos negativos do `test`, com imagem, caixa/máscara, score e uma hipótese por caso (sombra, poça, remendo, tampa de bueiro, rachadura, buraco pequeno/distante, corte na borda, anotação ausente no dataset).
5. **Por fatia:** ao menos uma partição — tamanho da caixa (pequena < 32² px, média, grande) sempre; luz/piso se o dataset marcar. A média esconde onde falha (ADR-019).
6. **Baseline de comparação:** `yolo11n` (30 épocas) como linha inferior e, se houver tempo, `yolo11s @ 800` — tabela com média de uma execução cada (sem CV: custo de GPU; declarar a limitação).
7. **Vídeo:** não há gabarito; reportar contagem de buracos únicos com ByteTrack vs. detecções por quadro, e observações qualitativas (ADR-S06).


## Decisão do ponto de operação (19/09, 01:50)

Curva P×R real do `det_s` em `val` (TP/FP/FN por casamento IoU ≥ 0,5, uma inferência em conf 0,001): conf 0,10 → P 0,46/R 0,62; 0,20 → 0,57/0,52; **0,25 → 0,62/0,49 (F1 0,55, máximo)**; 0,35 → 0,71/0,41; 0,50 → 0,84/0,31.

Opções apresentadas ao grupo: (A) regra original deste ADR, maior R com P ≥ 0,70 → conf 0,35, perde 59% dos buracos; (B) máximo F1 → conf 0,25; (C) piso P ≥ 0,55 → conf 0,20. **Adotada: B — conf = 0,25**, por ser a regra mais defensável e por o custo de um buraco não detectado pesar mais que o de um alarme revisado por operador; A foi descartada porque o piso de 0,70, fixado antes de ver a curva, sacrificava revocação demais neste modelo. Registro: `runs/logs/decisao_limiar.txt`. O mesmo `conf` vale para teste, vídeo e pitch; `IoU NMS = 0,7`.

## Alternativas rejeitadas

- **Validação cruzada k-fold:** ideal (ADR-011), mas 5 treinos × 2 modelos não cabem; declarar.
- **`conf` padrão (0,25) sem varredura:** ponto de operação arbitrário — perde o argumento de análise crítica (20%).
- **Reportar só mAP50:** o enunciado exige os dois.

## Consequências

- Se o `test` vier pior que o `val`, o relatório diz isso e explica; não se re-treina para "melhorar o teste".
- O `conf` escolhido é o mesmo usado no vídeo e no pitch.

## Relacionados

ADR-S03, ADR-S04, SPEC-S04.

# ADR-S03 — Detector: fine-tuning de YOLO11 (Ultralytics), variante `s`, 640 px, 50 épocas com parada antecipada

**Status:** Aceito
**Base:** exigência do enunciado ("YOLO (Ultralytics) ou Faster R-CNN (torchvision)"); ADR-005, ADR-007, ADR-008

## Contexto

O enunciado oferece YOLO (um estágio) ou Faster R-CNN (dois estágios). O cenário — vídeo de veículo em movimento, demonstração em tempo real, futura implantação embarcada — é o caso de manual para um estágio (ADR-008). O Ultralytics ainda entrega, no mesmo pacote, segmentação, validação com mAP50/mAP50-95/matriz de confusão, inferência em vídeo e rastreamento (ByteTrack) — cada um desses é um item do barema ou do bônus.

## Decisão

| Item | Valor | Por quê |
|---|---|---|
| Modelo | `yolo11s.pt` (pré-treinado COCO) | equilíbrio precisão × tempo na T4; `n` como baseline noturno, `m` só se sobrar tempo |
| Resolução | `imgsz=640` | padrão; buracos pequenos ao longe → testar 800 como experimento extra |
| Épocas | `epochs=50`, `patience=10` | parada antecipada por `val`; 1 355 imagens ≈ 1 min/época na T4 |
| Batch | `batch=16` (T4 16 GB suporta) | |
| Otimizador / LR | `optimizer="auto"` (AdamW/SGD conforme dataset), `lr0` padrão | não reabrir; documentar o que o Ultralytics escolheu (`args.yaml`) |
| Seed | `seed=0`, `deterministic=True` | reprodutibilidade |
| Augmentation | padrão (mosaic, fliplr, hsv, scale, translate) | documentar; não desativar |
| Congelamento | nenhum (`freeze=None`) | dataset > 1 000 imagens suporta fine-tuning completo; registrar como escolha |
| Métrica de seleção | `best.pt` = melhor fitness (0,1·mAP50 + 0,9·mAP50-95) em `val` | padrão Ultralytics |

## Alternativas rejeitadas

- **Faster R-CNN (torchvision):** mais preciso em objetos pequenos, mas 5–15 FPS, sem segmentação/tracking integrados e pipeline de treino mais longo de montar. Seria a escolha se o requisito fosse imagem estática de alta resolução (ADR-008).
- **YOLO11m/l desde o início:** 2–3× o tempo de treino sem garantia de ganho num dataset de 1–2 k imagens.
- **Vocabulário aberto (OWLv2/Grounding DINO):** não atende "fine-tuning de um detector moderno"; serve apenas como referência zero-shot comparativa se houver tempo (ADR-013).

## Consequências

- Treino e validação em uma linha cada; o tempo do grupo vai para dados, avaliação e escrita — onde estão 60% do barema.
- Limiar de confiança e IoU do NMS ficam explícitos e são escolhidos por varredura (ADR-S05, ADR-012).
- Exportação para ONNX é um comando, útil como "próximos passos" no relatório (ADR-017).

## Relacionados

ADR-S04, ADR-S05, SPEC-S02.

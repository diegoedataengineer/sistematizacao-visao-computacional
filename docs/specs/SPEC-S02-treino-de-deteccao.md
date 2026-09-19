# SPEC-S02 — Fase 2: treino de detecção (YOLO11)

**Entrega desta fase:** `runs/detect/det_s/weights/best.pt` no Drive · `results.csv` e curvas · tabela de hiperparâmetros · métricas em `val`
**ADRs:** S03, S05, S07

## 1. Baseline noturno (quinta)

```python
from ultralytics import YOLO
m = YOLO("yolo11n.pt")
m.train(data=f"{BASE}/dataset/data.yaml", epochs=30, imgsz=640, batch=16, seed=0,
        deterministic=True, project=f"{BASE}/runs/detect", name="baseline_n", exist_ok=True)
```

Objetivo: validar o pipeline (dados, labels, Drive). Se `mAP50` em `val` passar de ~0,5, está sadio.

## 2. Treino principal (sexta manhã)

```python
m = YOLO("yolo11s.pt")
r = m.train(data=f"{BASE}/dataset/data.yaml", epochs=50, imgsz=640, batch=16,
            patience=10, seed=0, deterministic=True, optimizer="auto",
            project=f"{BASE}/runs/detect", name="det_s", exist_ok=True, plots=True)
```

Se a sessão cair: `YOLO(f"{BASE}/runs/detect/det_s/weights/last.pt").train(resume=True)`.

Tempo esperado na T4: ~1 min/época para 1–1,5 k imagens → 40–60 min.

## 3. Validação em `val` e registro

```python
best = YOLO(f"{BASE}/runs/detect/det_s/weights/best.pt")
v = best.val(data=f"{BASE}/dataset/data.yaml", split="val", imgsz=640, plots=True,
             project=f"{BASE}/runs/detect", name="det_s_val", exist_ok=True)
print(f"mAP50={v.box.map50:.3f}  mAP50-95={v.box.map:.3f}  P={v.box.mp:.3f}  R={v.box.mr:.3f}")
```

Artefatos gerados automaticamente em `det_s/`: `results.png` (curvas de perda/mAP por época), `confusion_matrix.png`, `PR_curve.png`, `F1_curve.png`, `val_batch*_pred.jpg`, `args.yaml` (todos os hiperparâmetros efetivos — copie para a tabela).

## 4. Tabela de hiperparâmetros (obrigatória no relatório)

| Parâmetro | Valor | Origem |
|---|---|---|
| Modelo base | yolo11s.pt (COCO) | escolha |
| Épocas / parada antecipada | 50 / patience 10 (parou na época __) | `results.csv` |
| Tamanho de imagem | 640 | escolha |
| Batch | 16 | escolha |
| Otimizador, lr0, momentum, weight_decay | (ler de `args.yaml`) | auto |
| Augmentation | mosaic 1,0; fliplr 0,5; hsv_h/s/v 0,015/0,7/0,4; scale 0,5; translate 0,1 (ler de `args.yaml`) | padrão |
| Seed / determinístico | 0 / True | escolha |
| Camadas congeladas | nenhuma | escolha |
| Hardware / tempo | Colab T4 / __ min | medido |

## 5. Experimento extra (se sobrar tempo, sexta 17:00)

Uma **única** variação para a tabela comparativa: `imgsz=800` (buracos pequenos) **ou** `yolo11m`. Mesmo `seed`, mesmas épocas. Comparar em `val`; escolher o modelo final **antes** de abrir o `test`.

## 6. O que vai para o relatório (seção 3.1)

- A tabela acima; as curvas `results.png` (comentar: a perda de `val` estabilizou? houve sobreajuste — `train` cai, `val` sobe?).
- Uma frase sobre o que a parada antecipada fez.
- A comparação baseline `n` × `s` (× extra) em `val`, com a decisão do modelo final.

## Critério de pronto

- [ ] `best.pt` no Drive e link no README
- [ ] `args.yaml` copiado para a tabela
- [ ] mAP50/mAP50-95/P/R em `val` anotados
- [ ] Modelo final escolhido e **congelado** (tag `v0.1` no repo)

# SPEC-S04 — Fase 4: escolha do limiar, avaliação única no teste, IoU/Dice, matriz de confusão, análise de erros e fatias

**Entrega desta fase:** curva P×R em `val` + `conf` escolhido · tabela final (val e test, det e seg) · `confusion_matrix.png` · `figs/erros_fp.png`, `figs/erros_fn.png` com comentários · tabela por fatia
**ADRs:** S05

## 1. Varredura de limiar em `val` (antes de abrir o teste)

```python
import numpy as np, matplotlib.pyplot as plt
confs = np.round(np.arange(0.05, 0.65, 0.05), 2); P, R = [], []
for c in confs:
    v = det.val(data=f"{BASE}/dataset/data.yaml", split="val", conf=c, iou=0.7, verbose=False, plots=False)
    P.append(v.box.mp); R.append(v.box.mr)
fig, ax = plt.subplots(figsize=(5.5, 4.5)); ax.plot(R, P, "o-")
for c, r, p in zip(confs, R, P): ax.annotate(f"{c}", (r, p), fontsize=8, xytext=(4, 3), textcoords="offset points")
ax.set_xlabel("revocação"); ax.set_ylabel("precisão"); ax.grid(alpha=.3); ax.set_title("val: P×R por limiar de confiança")
plt.savefig(f"{BASE}/figs/pr_limiar.png", dpi=150)
```

Escolher `CONF` pelo critério do ADR-S05 (favorecer revocação com precisão ≥ 0,7). Fixar também `IOU_NMS = 0.7` (padrão) — ou 0,5 se buracos vizinhos forem suprimidos no painel. **Congelar** `CONF` e `IOU_NMS`; são os mesmos no teste, no vídeo e no pitch.

## 2. Avaliação única no `test`

```python
CONF, IOU_NMS = 0.20, 0.7                            # valores congelados
res = {}
for nome, modelo in (("det", det), ("seg", seg)):
    for split in ("val", "test"):
        v = modelo.val(data=f"{BASE}/dataset/data.yaml", split=split, conf=CONF, iou=IOU_NMS, imgsz=640,
                       plots=(split == "test"), project=f"{BASE}/runs/eval", name=f"{nome}_{split}", exist_ok=True)
        res[(nome, split)] = dict(box_map50=v.box.map50, box_map=v.box.map, box_p=v.box.mp, box_r=v.box.mr,
                                  **({"mask_map50": v.seg.map50, "mask_map": v.seg.map, "mask_p": v.seg.mp, "mask_r": v.seg.mr} if nome == "seg" else {}))
import pandas as pd; tab = pd.DataFrame(res).T.round(3); print(tab); tab.to_csv(f"{BASE}/figs/tabela_final.csv")
```

`runs/eval/det_test/confusion_matrix.png` é a matriz exigida (classe `pothole` × `background`: FP = fundo previsto como buraco; FN = buraco previsto como fundo).

## 3. IoU e Dice por imagem (segmentação, teste)

Compara a **união** das máscaras preditas com a união das máscaras anotadas, por imagem:

```python
import glob, cv2, numpy as np
def mascara_gt(lab, H, W):
    m = np.zeros((H, W), np.uint8)
    for l in open(lab):
        p = l.split()
        if len(p) > 5:
            xy = (np.array(p[1:], float).reshape(-1, 2) * [W, H]).astype(np.int32)
            cv2.fillPoly(m, [xy], 1)
    return m
ious, dices = [], []
for p in sorted(glob.glob(f"{BASE}/dataset/test/images/*")):
    lab = p.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
    img = cv2.imread(p); H, W = img.shape[:2]
    gt = mascara_gt(lab, H, W) if os.path.exists(lab) else np.zeros((H, W), np.uint8)
    r = seg(p, conf=CONF, iou=IOU_NMS, verbose=False)[0]
    pr = np.zeros((H, W), np.uint8)
    if r.masks is not None:
        for m in r.masks.data.cpu().numpy():
            pr |= cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
    inter, uni = (gt & pr).sum(), (gt | pr).sum()
    if uni == 0: continue                              # imagem sem buraco e sem predição
    ious.append(inter / uni); dices.append(2 * inter / (gt.sum() + pr.sum()))
print(f"IoU médio={np.mean(ious):.3f}  Dice médio={np.mean(dices):.3f}  (n={len(ious)})")
```

## 4. Análise de erros (≥ 4 FP e ≥ 4 FN, comentados)

```python
from ultralytics.utils.metrics import box_iou
import torch
fps, fns = [], []
for p in sorted(glob.glob(f"{BASE}/dataset/test/images/*")):
    lab = p.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
    img = cv2.imread(p); H, W = img.shape[:2]
    gts = []
    if os.path.exists(lab):
        for l in open(lab):
            q = l.split()
            if len(q) > 5:
                xy = np.array(q[1:], float).reshape(-1, 2) * [W, H]
                gts.append([xy[:, 0].min(), xy[:, 1].min(), xy[:, 0].max(), xy[:, 1].max()])
    gts = torch.tensor(gts, dtype=torch.float32) if gts else torch.zeros((0, 4))
    r = det(p, conf=CONF, iou=IOU_NMS, verbose=False)[0]
    pb = r.boxes.xyxy.cpu(); sc = r.boxes.conf.cpu()
    iou = box_iou(pb, gts) if len(pb) and len(gts) else torch.zeros((len(pb), len(gts)))
    for i in range(len(pb)):
        if len(gts) == 0 or iou[i].max() < 0.5: fps.append((p, pb[i].tolist(), float(sc[i])))
    for j in range(len(gts)):
        if len(pb) == 0 or iou[:, j].max() < 0.5: fns.append((p, gts[j].tolist()))
print(len(fps), "FP |", len(fns), "FN")

def painel(casos, cor, titulo, arq, k=8):
    fig, ax = plt.subplots(2, 4, figsize=(16, 7)); ax = ax.ravel()
    for a, caso in zip(ax, casos[:k]):
        img = cv2.imread(caso[0]); x1, y1, x2, y2 = map(int, caso[1])
        cv2.rectangle(img, (x1, y1), (x2, y2), cor, 3)
        a.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); a.axis("off")
        a.set_title(f"{os.path.basename(caso[0])[:18]}" + (f"  conf={caso[2]:.2f}" if len(caso) > 2 else ""), fontsize=8)
    fig.suptitle(titulo); plt.tight_layout(); plt.savefig(arq, dpi=130); plt.show()
painel(sorted(fps, key=lambda c: -c[2]), (0, 0, 255), "Falsos positivos (mais confiantes)", f"{BASE}/figs/erros_fp.png")
painel(fns, (255, 0, 0), "Falsos negativos", f"{BASE}/figs/erros_fn.png")
```

Para cada caso no relatório, uma linha: **imagem · o que o modelo fez · hipótese**. Hipóteses típicas: sombra de árvore/poste (FP), poça/mancha de óleo (FP), remendo de asfalto escuro (FP), tampa de bueiro (FP), rachadura larga (FP ou FN por ambiguidade da anotação), buraco pequeno/distante (FN), buraco cortado pela borda (FN), buraco parcialmente coberto por água (FN), **anotação faltante no dataset** (FP "correto" — registrar como ruído de rótulo).

## 5. Fatias

Por tamanho da caixa anotada (teste): pequena (< 32² px), média (32²–96²), grande (> 96²) — revocação em cada faixa a partir das listas `fns`/GT. Se o dataset tiver metadados de luz/piso, adicionar. Uma tabela de 3 linhas basta; comentar onde o modelo falha mais.

## 6. Tabela final (relatório, seção 4)

| Modelo | Split | mAP@0,5 | mAP@0,5:0,95 | P | R | IoU médio | Dice médio |
|---|---|---|---|---|---|---|---|
| YOLO11n (baseline) | val | | | | | — | — |
| YOLO11s (final) | val | | | | | — | — |
| YOLO11s (final) | **test** | | | | | — | — |
| YOLO11s-seg (box) | test | | | | | — | — |
| YOLO11s-seg (mask) | test | | | | | | |

`conf = __`, `IoU NMS = __`, `imgsz = 640`. Comentar val→test; comentar o que a matriz de confusão mostra; ligar à análise de erros.

## Critério de pronto

- [ ] `pr_limiar.png` e `CONF` justificado
- [ ] `tabela_final.csv`, `confusion_matrix.png` do teste
- [ ] IoU/Dice do teste
- [ ] `erros_fp.png`, `erros_fn.png` + comentários escritos
- [ ] Tabela por fatia

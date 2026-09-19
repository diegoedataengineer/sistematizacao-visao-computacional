# SPEC-S03 — Fase 3: treino de segmentação (YOLO11-seg) e comparação caixas × máscaras

**Entrega desta fase:** `runs/segment/seg_s/weights/best.pt` · métricas Mask em `val` · `figs/caixas_vs_mascaras.png` · razão área máscara/caixa
**ADRs:** S04, S05

## 1. Treino

```python
from ultralytics import YOLO
s = YOLO("yolo11s-seg.pt")
s.train(data=f"{BASE}/dataset/data.yaml", epochs=50, imgsz=640, batch=16, patience=10,
        seed=0, deterministic=True, project=f"{BASE}/runs/segment", name="seg_s", exist_ok=True, plots=True)
```

Mesmo `data.yaml` (os `labels/*.txt` já são polígonos). Tempo esperado: 60–80 min na T4.

## 2. Validação em `val`

```python
seg = YOLO(f"{BASE}/runs/segment/seg_s/weights/best.pt")
v = seg.val(data=f"{BASE}/dataset/data.yaml", split="val", imgsz=640, plots=True,
            project=f"{BASE}/runs/segment", name="seg_s_val", exist_ok=True)
print(f"Box  mAP50={v.box.map50:.3f} mAP50-95={v.box.map:.3f}")
print(f"Mask mAP50={v.seg.map50:.3f} mAP50-95={v.seg.map:.3f} P={v.seg.mp:.3f} R={v.seg.mr:.3f}")
```

## 3. Painel caixas × máscaras (8 imagens de `val`, mesmo `conf`)

```python
import glob, cv2, numpy as np, matplotlib.pyplot as plt
det = YOLO(f"{BASE}/runs/detect/det_s/weights/best.pt")
imgs = sorted(glob.glob(f"{BASE}/dataset/valid/images/*"))[:: max(1, len(glob.glob(f"{BASE}/dataset/valid/images/*")) // 8)][:8]
CONF = 0.25                                   # substituir pelo conf escolhido na SPEC-S04
fig, ax = plt.subplots(2, 8, figsize=(24, 6))
for j, p in enumerate(imgs):
    rd = det(p, conf=CONF, verbose=False)[0]; rs = seg(p, conf=CONF, verbose=False)[0]
    ax[0, j].imshow(cv2.cvtColor(rd.plot(), cv2.COLOR_BGR2RGB)); ax[0, j].set_title("detecção", fontsize=9)
    ax[1, j].imshow(cv2.cvtColor(rs.plot(), cv2.COLOR_BGR2RGB)); ax[1, j].set_title("segmentação", fontsize=9)
    ax[0, j].axis("off"); ax[1, j].axis("off")
plt.tight_layout(); plt.savefig(f"{BASE}/figs/caixas_vs_mascaras.png", dpi=130); plt.show()
```

## 4. O que a máscara revela que a caixa não mostra — quantificar

Para cada instância predita pelo modelo `-seg` em `val`: razão `área_máscara / área_caixa`.

```python
razoes = []
for p in glob.glob(f"{BASE}/dataset/valid/images/*"):
    r = seg(p, conf=CONF, verbose=False)[0]
    if r.masks is None: continue
    H, W = r.orig_shape
    for m, b in zip(r.masks.data.cpu().numpy(), r.boxes.xyxy.cpu().numpy()):
        m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        a_box = max(1, (b[2] - b[0]) * (b[3] - b[1]))
        razoes.append(m.sum() / a_box)
razoes = np.array(razoes)
print(f"n={len(razoes)}  razão média={razoes.mean():.2f}  mediana={np.median(razoes):.2f}  "
      f"P10={np.percentile(razoes, 10):.2f}  P90={np.percentile(razoes, 90):.2f}")
plt.hist(razoes, bins=30); plt.xlabel("área da máscara / área da caixa"); plt.savefig(f"{BASE}/figs/razao_area.png", dpi=150)
```

Interpretação para o relatório: uma razão mediana de ~0,5–0,7 significa que a caixa **superestima a área do buraco em 30–50%** — para priorizar manutenção por extensão do dano, a máscara é a medida certa. Mostrar um buraco alongado/diagonal onde a diferença é maior, e um caso em que a máscara vaza para sombra/poça (limitação).

## 5. Ajuste de anotações (só se necessário)

Se o painel mostrar polígonos sistematicamente grosseiros em `val`, **não** re-anotar no prazo: registrar como limitação com um exemplo. Se houver labels vazias por erro de export, corrigir o filtro da SPEC-S01 e re-treinar.

## 6. O que vai para o relatório (seção 3.2)

- Tabela Box × Mask (mAP50, mAP50-95, P, R) em `val`;
- `figs/caixas_vs_mascaras.png` com 2–3 frases por padrão observado;
- `figs/razao_area.png` + números do §4 e a conclusão de valor.

## Critério de pronto

- [ ] `best.pt` seg no Drive
- [ ] Métricas Mask em `val`
- [ ] Painel e razão de área salvos em `figs/`

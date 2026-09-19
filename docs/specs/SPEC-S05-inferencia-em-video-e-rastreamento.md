# SPEC-S05 — Fase 4: vídeo real com inferência (≥ 30 s) e rastreamento ByteTrack (bônus)

**Entrega desta fase:** `video/cenario.mp4` (bruto, não versionado) · `video/cenario_seg.mp4` (anotado) · `video/cenario_track.mp4` (IDs) · contagem de IDs únicos · latência ms/quadro
**ADRs:** S06

## 1. Captura

- Celular no para-brisa (suporte) ou na mão, **horizontal**, 1080p 30 FPS, 60–120 s, luz do dia, velocidade baixa (< 30 km/h) em trecho com buracos visíveis. Evitar contraluz forte.
- Alternativa: dashcam pública com licença compatível — anotar fonte e licença; recortar 30–60 s.
- Subir para `{BASE}/video/cenario.mp4`. Se > 200 MB, reduzir: `ffmpeg -i in.mp4 -vf scale=1280:-2 -r 30 -crf 23 cenario.mp4`.

## 2. Privacidade (se houver pessoas/placas)

```python
coco = YOLO("yolo11n.pt")                                  # COCO: person=0, car=2 (placas via car não é preciso; borrar carros próximos se necessário)
def borrar(quadro):
    r = coco(quadro, classes=[0], conf=0.3, verbose=False)[0]
    for x1, y1, x2, y2 in r.boxes.xyxy.cpu().numpy().astype(int):
        quadro[y1:y2, x1:x2] = cv2.GaussianBlur(quadro[y1:y2, x1:x2], (51, 51), 0)
    return quadro
```

Aplicar antes de gerar os vídeos publicados. Alternativa: editar no celular.

## 3. Inferência com máscaras (vídeo principal)

```python
seg = YOLO(f"{BASE}/runs/segment/seg_s/weights/best.pt")
seg.predict(source=f"{BASE}/video/cenario.mp4", conf=CONF, iou=IOU_NMS, imgsz=640,
            save=True, project=f"{BASE}/video", name="seg", exist_ok=True, vid_stride=1)
# saída: {BASE}/video/seg/cenario.avi → converter: ffmpeg -i cenario.avi -c:v libx264 -crf 23 cenario_seg.mp4
```

## 4. Bônus: ByteTrack

```python
import cv2, time, collections
seg = YOLO(f"{BASE}/runs/segment/seg_s/weights/best.pt")
ids, n_det, tempos = set(), 0, []
cap = cv2.VideoCapture(f"{BASE}/video/cenario.mp4"); W, H = int(cap.get(3)), int(cap.get(4)); fps = cap.get(5)
out = cv2.VideoWriter(f"{BASE}/video/cenario_track.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
while True:
    ok, q = cap.read()
    if not ok: break
    t0 = time.perf_counter()
    r = seg.track(q, persist=True, tracker="bytetrack.yaml", conf=CONF, iou=IOU_NMS, imgsz=640, verbose=False)[0]
    tempos.append((time.perf_counter() - t0) * 1000)
    if r.boxes.id is not None:
        ids.update(r.boxes.id.int().tolist()); n_det += len(r.boxes)
    out.write(r.plot())                                   # caixas + máscaras + IDs
cap.release(); out.release()
print(f"buracos únicos (IDs)={len(ids)}  detecções por quadro somadas={n_det}  "
      f"latência média={np.mean(tempos):.1f} ms/quadro (T4)  → {1000/np.mean(tempos):.1f} FPS de inferência")
```

`persist=True` mantém o rastreador entre quadros. Buracos são estáticos no mundo; o movimento é da câmera — o Kalman de velocidade constante funciona bem em velocidade estável e perde IDs em freadas/curvas (registrar).

## 5. O que reportar (relatório, seção 4.4 e 6)

- Link do vídeo anotado (YouTube não listado ou Drive) — duração ≥ 30 s.
- Tabela: detecções por quadro (soma) × IDs únicos; trocas de ID observadas (contar manualmente 2–3 casos).
- Latência ms/quadro na T4 (e na CPU local, se medida) e FPS — ligar ao orçamento de 33 ms (ADR-016): cabe? Se não, "detectar a cada N + rastrear" como próximo passo.
- Observação de generalização: vias brasileiras vs. dataset estrangeiro — o que funcionou, o que falhou (asfalto claro, sombras, faixas).

## Critério de pronto

- [ ] `cenario_seg.mp4` e `cenario_track.mp4` gerados e assistidos do início ao fim
- [ ] Contagem de IDs, latência e FPS anotados
- [ ] Pessoas/placas borradas, se aplicável
- [ ] Vídeo anotado publicado e link testado em aba anônima

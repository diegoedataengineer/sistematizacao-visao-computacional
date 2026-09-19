"""Inferência do segmentador no vídeo do cenário e rastreamento ByteTrack para contar buracos únicos.

Gera video/seg/ (predição quadro a quadro, salvo pelo Ultralytics), video/cenario_track.mp4 (IDs do
rastreador), figs/video_contagem.png, figs/video_quadros.png e figs/video_resumo.json.

Uso:
    python scripts/video.py --video video/cenario.mp4 --pesos runs/segment/seg_s/weights/best.pt --conf 0.25 --iou-nms 0.7
"""
import argparse
import json
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO


def iou_xyxy(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="video/cenario.mp4")
    ap.add_argument("--pesos", default="runs/segment/seg_s/weights/best.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou-nms", type=float, default=0.7)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--saida", default="figs")
    ap.add_argument("--sem-predict", action="store_true", help="pula o vídeo de predição sem rastreamento")
    args = ap.parse_args()
    saida = Path(args.saida); saida.mkdir(parents=True, exist_ok=True)
    seg = YOLO(args.pesos)

    if not args.sem_predict:
        seg.predict(source=args.video, conf=args.conf, iou=args.iou_nms, imgsz=args.imgsz, save=True,
                    project=str(Path("video").resolve()), name="seg", exist_ok=True, verbose=False)

    cap = cv2.VideoCapture(args.video)
    W, H, fps = int(cap.get(3)), int(cap.get(4)), cap.get(5)
    bruto = "video/cenario_track_raw.mp4"
    out = cv2.VideoWriter(bruto, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    por_quadro, tempos, trilhas = [], [], defaultdict(list)   # trilhas[id] = [(quadro, caixa)]
    quadros_guardados = {}
    q = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        r = seg.track(frame, persist=True, tracker="bytetrack.yaml", conf=args.conf, iou=args.iou_nms,
                      imgsz=args.imgsz, verbose=False)[0]
        tempos.append((time.perf_counter() - t0) * 1000)
        n = len(r.boxes)
        if r.boxes.id is not None:
            for i, cx in zip(r.boxes.id.int().tolist(), r.boxes.xyxy.tolist()):
                trilhas[i].append((q, cx))
        por_quadro.append(n)
        anot = r.plot()
        out.write(anot)
        if q in (int(fps * 10), int(fps * 30), int(fps * 50), int(fps * 70)):
            quadros_guardados[q] = anot
        q += 1
    cap.release(); out.release()
    # recodifica em H.264 para reprodução em navegador; mantém o bruto se o ffmpeg faltar
    final = "video/cenario_track.mp4"
    try:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", bruto, "-c:v", "libx264", "-preset", "fast",
                        "-crf", "23", "-pix_fmt", "yuv420p", final], check=True)
        Path(bruto).unlink()
    except Exception as e:  # noqa: BLE001
        print("ffmpeg indisponível, mantendo mp4v:", e); Path(bruto).rename(final)

    # métricas das trilhas
    vidas = {i: (t[0][0], t[-1][0], len(t)) for i, t in trilhas.items()}
    duracoes = np.array([v[2] for v in vidas.values()])
    curtas = int((duracoes < 5).sum())
    # candidatos a troca de ID: trilha nova nasce até 15 quadros depois do fim de outra, no mesmo lugar (IoU ≥ 0,3)
    trocas = []
    for i, (ini, fim, _) in vidas.items():
        for j, (ini2, _, _) in vidas.items():
            if j != i and 0 < ini2 - fim <= 15 and iou_xyxy(trilhas[i][-1][1], trilhas[j][0][1]) >= 0.3:
                trocas.append((i, j, fim, ini2))
    ids_estaveis = int((duracoes >= 5).sum())
    resumo = dict(video=args.video, largura=W, altura=H, fps=round(fps, 2), quadros=q, duracao_s=round(q / fps, 1),
                  conf=args.conf, iou_nms=args.iou_nms, deteccoes_somadas=int(sum(por_quadro)),
                  quadros_com_deteccao=int(sum(n > 0 for n in por_quadro)), ids_unicos=len(vidas),
                  ids_estaveis_5q=ids_estaveis, trilhas_curtas_lt5q=curtas, trocas_id_candidatas=len(trocas),
                  trocas=[dict(de=a, para=b, quadro_fim=c, quadro_ini=d, t_s=round(d / fps, 1)) for a, b, c, d in trocas],
                  vida_mediana_quadros=float(np.median(duracoes)) if len(duracoes) else 0.0,
                  vida_max_quadros=int(duracoes.max()) if len(duracoes) else 0,
                  ms_por_quadro=round(float(np.mean(tempos)), 1), fps_inferencia=round(1000 / float(np.mean(tempos)), 1))
    (saida / "video_resumo.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in resumo.items() if k != "trocas"}, indent=2, ensure_ascii=False))

    # figura: detecções por quadro × IDs únicos acumulados
    t = np.arange(q) / fps
    acum = np.zeros(q, int)
    for ini, _, _ in vidas.values():
        acum[ini:] += 1
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(t, por_quadro, color="#999", lw=0.8, label="detecções no quadro")
    ax.set_xlabel("tempo (s)"); ax.set_ylabel("detecções por quadro", color="#666")
    ax2 = ax.twinx(); ax2.plot(t, acum, color="#c00", lw=1.8, label="buracos únicos (IDs) acumulados")
    ax2.set_ylabel("IDs acumulados", color="#c00")
    for a, b, c, d in trocas:
        ax2.axvline(d / fps, color="#c00", ls=":", lw=0.8, alpha=0.6)
    ax.set_title(f"soma das detecções = {resumo['deteccoes_somadas']} · IDs únicos = {len(vidas)} · trocas de ID candidatas (pontilhado) = {len(trocas)}", fontsize=10)
    fig.tight_layout(); fig.savefig(saida / "video_contagem.png", dpi=150); plt.close(fig)

    # figura: quadros anotados
    if quadros_guardados:
        fig, axs = plt.subplots(1, len(quadros_guardados), figsize=(4.2 * len(quadros_guardados), 2.6))
        for a_, (qi, im) in zip(np.atleast_1d(axs), sorted(quadros_guardados.items())):
            a_.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); a_.set_title(f"t = {qi / fps:.0f} s", fontsize=9); a_.axis("off")
        fig.tight_layout(); fig.savefig(saida / "video_quadros.png", dpi=130); plt.close(fig)
    print(f"vídeos: video/seg/ e {final} · figuras em {saida}/")


if __name__ == "__main__":
    main()

"""Treino de detecção ou segmentação com Ultralytics, protocolo fixo do projeto.

Uso:
    python scripts/treinar.py --tarefa det --modelo yolo11n.pt --epocas 30 --nome baseline_n
    python scripts/treinar.py --tarefa det --modelo yolo11s.pt --epocas 50 --nome det_s
    python scripts/treinar.py --tarefa seg --modelo yolo11s-seg.pt --epocas 50 --nome seg_s
    python scripts/treinar.py --retomar runs/detect/det_s/weights/last.pt

Saídas: runs/<detect|segment>/<nome>/ (pesos, results.csv, curvas) e runs/logs/<nome>.log (log completo).
"""
import argparse
import logging
from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils import LOGGER

RAIZ = Path(__file__).resolve().parent.parent


def registrar_log(nome: str) -> Path:
    """Anexa um FileHandler ao logger do Ultralytics: tudo que ele imprime vai também para runs/logs/<nome>.log.
    As barras de progresso (tqdm) não passam pelo logger; para capturá-las use `| tee` no shell."""
    caminho = RAIZ / "runs" / "logs" / f"{nome}.log"
    caminho.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(caminho, mode="a", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))
    LOGGER.addHandler(fh)
    return caminho


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tarefa", choices=["det", "seg"], default="det")
    ap.add_argument("--modelo", default="yolo11s.pt")
    ap.add_argument("--dados", default=str(RAIZ / "dataset" / "data.yaml"))
    ap.add_argument("--epocas", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--paciencia", type=int, default=10)
    ap.add_argument("--nome", default=None)
    ap.add_argument("--retomar", default=None, help="caminho de last.pt para retomar")
    args = ap.parse_args()

    nome = args.nome or (Path(args.retomar).parents[1].name if args.retomar else f"{args.tarefa}_{Path(args.modelo).stem}")
    log = registrar_log(nome)
    LOGGER.info(f"log em {log}")

    if args.retomar:
        YOLO(args.retomar).train(resume=True)
        return

    projeto = RAIZ / "runs" / ("segment" if args.tarefa == "seg" else "detect")   # absoluto: evita runs/detect/runs/detect
    m = YOLO(args.modelo)
    res = m.train(
        data=args.dados, epochs=args.epocas, imgsz=args.imgsz, batch=args.batch,
        patience=args.paciencia, seed=0, deterministic=True, optimizer="auto",
        project=str(projeto), name=nome, exist_ok=True, plots=True,
    )
    save_dir = Path(res.save_dir)
    best = save_dir / "weights" / "best.pt"
    v = YOLO(str(best)).val(data=args.dados, split="val", imgsz=args.imgsz, plots=False,
                            project=str(projeto), name=f"{nome}_val", exist_ok=True)
    LOGGER.info(f"[{nome}] val: box mAP50={v.box.map50:.3f} mAP50-95={v.box.map:.3f} P={v.box.mp:.3f} R={v.box.mr:.3f}")
    if args.tarefa == "seg":
        LOGGER.info(f"[{nome}] val: mask mAP50={v.seg.map50:.3f} mAP50-95={v.seg.map:.3f} P={v.seg.mp:.3f} R={v.seg.mr:.3f}")
    LOGGER.info(f"pesos: {best}")


if __name__ == "__main__":
    main()

"""Fase 4 — avaliação do projeto (SPEC-S04/S03), com protocolo de teste único (ADR-S05).

Etapas:
  1. varredura de conf em VAL (det) → figs/pr_limiar.png; escolha do ponto de operação
     (maior revocação com precisão ≥ --piso-precisao; se impossível, maior F1);
  2. avaliação em VAL e TEST de cada modelo listado → figs/tabela_final.csv (+ resultados.json);
     matriz de confusão do teste do detector principal → figs/confusion_matrix_test.png;
  3. IoU/Dice por imagem no TEST (seg) contra as labels atuais e, se existir, contra labels_original;
  4. análise de erros do detector principal no TEST → figs/erros_fp.png, figs/erros_fn.png, figs/erros.csv;
  5. fatias por tamanho da instância (TEST) → figs/fatias.csv;
  6. razão área máscara/caixa (VAL, seg) → figs/razao_area.png;
  7. painel caixas × máscaras (VAL) → figs/caixas_vs_mascaras.png;
  8. cópia das curvas de treino → figs/curvas_<nome>.png.

Uso:
    python scripts/avaliar.py --det runs/detect/det_s --seg runs/segment/seg_s \
        --extras runs/detect/baseline_n runs/detect/det_s800 runs/detect/det_m
    python scripts/avaliar.py --det runs/detect/baseline_n --sem-test      # ensaio, não abre o test
"""
import argparse
import csv
import glob
import json
import os
import shutil
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO
from ultralytics.utils.metrics import box_iou

RAIZ = Path(__file__).resolve().parent.parent
DEVICE = None
DADOS = RAIZ / "dataset" / "data.yaml"
FIGS = RAIZ / "figs"
EXT = (".jpg", ".jpeg", ".png")


# ----------------------------------------------------------------------------- utilidades
def imagens(split: str) -> list[Path]:
    return sorted(p for p in (RAIZ / "dataset" / split / "images").iterdir() if p.suffix.lower() in EXT)


def label_de(img: Path, pasta: str = "labels") -> Path:
    return img.parent.parent / pasta / (img.name[: -len(img.suffix)] + ".txt")


def poligonos_gt(lab: Path, W: int, H: int) -> list[np.ndarray]:
    if not lab.exists():
        return []
    out = []
    for l in lab.read_text().splitlines():
        p = l.split()
        if len(p) >= 7:
            out.append(np.array(p[1:], float).reshape(-1, 2) * [W, H])
    return out


def caixas_gt(lab: Path, W: int, H: int) -> torch.Tensor:
    polys = poligonos_gt(lab, W, H)
    if not polys:
        return torch.zeros((0, 4))
    return torch.tensor([[p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()] for p in polys], dtype=torch.float32)


def mascara_uniao(polys: list[np.ndarray], H: int, W: int) -> np.ndarray:
    m = np.zeros((H, W), np.uint8)
    for p in polys:
        cv2.fillPoly(m, [p.astype(np.int32)], 1)
    return m


def nome_run(run: Path) -> str:
    return run.name


def carregar(run: Path) -> YOLO:
    return YOLO(str(run / "weights" / "best.pt"))


def imgsz_de(run: Path) -> int:
    try:
        import yaml
        return int(yaml.safe_load((run / "args.yaml").read_text()).get("imgsz", 640))
    except Exception:
        return 640


# ----------------------------------------------------------------------------- 1. varredura de limiar
def _casar(pb: torch.Tensor, sc: torch.Tensor, gts: torch.Tensor, lim: float) -> tuple[int, int, int]:
    """TP/FP/FN em uma imagem para as predições com score >= lim (casamento guloso por IoU >= 0,5)."""
    keep = sc >= lim; pb, sc = pb[keep], sc[keep]
    if len(pb) == 0:
        return 0, 0, int(len(gts))
    if len(gts) == 0:
        return 0, int(len(pb)), 0
    ordem = torch.argsort(sc, descending=True); iou = box_iou(pb[ordem], gts); usado = torch.zeros(len(gts), dtype=torch.bool); tp = 0
    for i in range(len(pb)):
        cand = iou[i].clone(); cand[usado] = 0
        j = int(torch.argmax(cand))
        if float(cand[j]) >= 0.5:
            usado[j] = True; tp += 1
    return tp, int(len(pb)) - tp, int(len(gts)) - tp


def predicoes_val(det: YOLO, imgsz: int, iou_nms: float, split: str = "valid") -> list[tuple]:
    """Uma única passada em conf=0,001; devolve [(caixas, scores, gts)] por imagem."""
    out = []
    for img_p in imagens(split):
        img = cv2.imread(str(img_p)); H, W = img.shape[:2]
        r = det(str(img_p), conf=0.001, iou=iou_nms, imgsz=imgsz, verbose=False, device=DEVICE)[0]
        out.append((r.boxes.xyxy.cpu(), r.boxes.conf.cpu(), caixas_gt(label_de(img_p), W, H)))
    return out


def varredura(det: YOLO, imgsz: int, piso: float, iou_nms: float = 0.7, regra: str = "piso") -> tuple[float, dict]:
    """P/R/F1 reais por limiar de confiança em VAL (TP/FP/FN por casamento IoU>=0,5), sem depender do val() do Ultralytics."""
    preds = predicoes_val(det, imgsz, iou_nms)
    confs = np.round(np.arange(0.05, 0.95, 0.05), 2); P, R, F1 = [], [], []
    for c in confs:
        tp = fp = fn = 0
        for pb, sc, gts in preds:
            a, b, d = _casar(pb, sc, gts, float(c)); tp += a; fp += b; fn += d
        p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
        P.append(p); R.append(r); F1.append(2 * p * r / (p + r) if p + r else 0.0)
    ok = [i for i, p in enumerate(P) if p >= piso] if regra == "piso" else []
    i = max(ok, key=lambda i: R[i]) if ok else int(np.argmax(F1))
    escolhido = float(confs[i])
    fig, ax = plt.subplots(figsize=(5.8, 4.8)); ax.plot(R, P, "o-", color="#333", ms=4)
    for c, r, p in zip(confs, R, P):
        if round(c * 100) % 10 == 0: ax.annotate(f"{c:.1f}", (r, p), fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.plot(R[i], P[i], "o", ms=13, mfc="none", mec="r", mew=2, label=f"escolhido: conf={escolhido:.2f}")
    if regra == "piso": ax.axhline(piso, color="r", ls=":", lw=1, label=f"piso de precisão {piso}")
    ax.set_xlabel("revocação (val)"); ax.set_ylabel("precisão (val)"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    ax.set_title("Detector: precisão × revocação por limiar de confiança (IoU ≥ 0,5)")
    plt.tight_layout(); plt.savefig(FIGS / "pr_limiar.png", dpi=150); plt.close()
    tab = {"conf": confs.tolist(), "P": P, "R": R, "F1": F1, "escolhido": escolhido, "regra": f"max R com P>={piso}" if ok else "max F1"}
    with open(FIGS / "pr_limiar.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["conf", "P", "R", "F1"]); w.writerows(zip(confs, P, R, F1))
    print(f"[1] limiar escolhido conf={escolhido:.2f} ({tab['regra']}): P={P[i]:.3f} R={R[i]:.3f} F1={F1[i]:.3f}")
    return escolhido, tab


# ----------------------------------------------------------------------------- 2. tabela final
def avaliar_run(run: Path, tarefa: str, split: str, conf: float, iou_nms: float, plots: bool) -> dict:
    """mAP no conf padrão (0,001 — protocolo COCO/Ultralytics); P/R e matriz de confusão no ponto de operação."""
    m = carregar(run); imgsz = imgsz_de(run)
    vm = m.val(data=str(DADOS), split=split, iou=iou_nms, imgsz=imgsz, plots=False, verbose=False, device=DEVICE,
               project=str(RAIZ / "runs" / "eval"), name=f"{nome_run(run)}_{split}_map", exist_ok=True)
    vo = m.val(data=str(DADOS), split=split, conf=conf, iou=iou_nms, imgsz=imgsz, plots=plots, verbose=False, device=DEVICE,
               project=str(RAIZ / "runs" / "eval"), name=f"{nome_run(run)}_{split}", exist_ok=True)
    preds = predicoes_val(m, imgsz, iou_nms, "valid" if split == "val" else split)
    tp = fp = fn = 0
    for pb, sc, gts in preds:
        a, b, c_ = _casar(pb, sc, gts, conf); tp += a; fp += b; fn += c_
    P_op = tp / (tp + fp) if tp + fp else 0.0; R_op = tp / (tp + fn) if tp + fn else 0.0
    d = dict(modelo=nome_run(run), tarefa=tarefa, split=split, imgsz=imgsz,
             box_mAP50=round(float(vm.box.map50), 4), box_mAP5095=round(float(vm.box.map), 4),
             box_P=round(P_op, 4), box_R=round(R_op, 4), TP=tp, FP=fp, FN=fn)
    if tarefa == "seg":
        d.update(mask_mAP50=round(float(vm.seg.map50), 4), mask_mAP5095=round(float(vm.seg.map), 4),
                 mask_P=round(float(vo.seg.mp), 4), mask_R=round(float(vo.seg.mr), 4))
    return d


# ----------------------------------------------------------------------------- 3. IoU / Dice
def iou_dice(seg: YOLO, imgsz: int, conf: float, iou_nms: float, split: str, pasta_labels: str) -> dict:
    ious, dices = [], []
    for img_p in imagens(split):
        img = cv2.imread(str(img_p)); H, W = img.shape[:2]
        gt = mascara_uniao(poligonos_gt(label_de(img_p, pasta_labels), W, H), H, W)
        r = seg(str(img_p), conf=conf, iou=iou_nms, imgsz=imgsz, verbose=False, device=DEVICE)[0]
        pr = np.zeros((H, W), np.uint8)
        if r.masks is not None:
            for m in r.masks.data.cpu().numpy():
                pr |= cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        inter, uni = int((gt & pr).sum()), int((gt | pr).sum())
        if uni == 0:
            continue
        ious.append(inter / uni); dices.append(2 * inter / (int(gt.sum()) + int(pr.sum())))
    return dict(n=len(ious), iou_medio=round(float(np.mean(ious)), 4), dice_medio=round(float(np.mean(dices)), 4),
                iou_mediano=round(float(np.median(ious)), 4))


# ----------------------------------------------------------------------------- 4/5. erros e fatias
def erros_e_fatias(det: YOLO, imgsz: int, conf: float, iou_nms: float, split: str) -> dict:
    fps, fns, tps = [], [], []
    tam = lambda b: "pequena" if (b[2] - b[0]) * (b[3] - b[1]) < 32 ** 2 else ("média" if (b[2] - b[0]) * (b[3] - b[1]) < 96 ** 2 else "grande")
    fat = {k: [0, 0] for k in ("pequena", "média", "grande")}          # [detectadas, total]
    for img_p in imagens(split):
        img = cv2.imread(str(img_p)); H, W = img.shape[:2]
        gts = caixas_gt(label_de(img_p), W, H)
        r = det(str(img_p), conf=conf, iou=iou_nms, imgsz=imgsz, verbose=False, device=DEVICE)[0]
        pb, sc = r.boxes.xyxy.cpu(), r.boxes.conf.cpu()
        ordem = torch.argsort(sc, descending=True); pb, sc = pb[ordem], sc[ordem]
        iou = box_iou(pb, gts) if len(pb) and len(gts) else torch.zeros((len(pb), len(gts)))
        usado = torch.zeros(len(gts), dtype=torch.bool)
        for i in range(len(pb)):                                   # guloso por score, um-para-um (igual à varredura)
            if len(gts):
                cand = iou[i].clone(); cand[usado] = 0; j = int(torch.argmax(cand))
                if float(cand[j]) >= 0.5:
                    usado[j] = True; tps.append((str(img_p), gts[j].tolist())); continue
            fps.append((str(img_p), pb[i].tolist(), float(sc[i])))
        for j in range(len(gts)):
            b = gts[j].tolist(); k = tam(b); fat[k][1] += 1
            if bool(usado[j]):
                fat[k][0] += 1
            else:
                fns.append((str(img_p), b))
    with open(FIGS / "fatias.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["tamanho", "instancias", "detectadas", "revocacao"])
        for k, (d, t) in fat.items():
            w.writerow([k, t, d, round(d / t, 4) if t else ""])
    with open(FIGS / "erros.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["tipo", "imagem", "x1", "y1", "x2", "y2", "conf"])
        for p, b, s in fps: w.writerow(["FP", os.path.basename(p), *[round(v) for v in b], round(s, 3)])
        for p, b in fns: w.writerow(["FN", os.path.basename(p), *[round(v) for v in b], ""])

    def painel(casos, cor, titulo, arq, k=8):
        fig, ax = plt.subplots(2, 4, figsize=(16, 7.2)); ax = ax.ravel()
        for a in ax: a.axis("off")
        for a, caso in zip(ax, casos[:k]):
            img = cv2.imread(caso[0]); x1, y1, x2, y2 = map(int, caso[1])
            cv2.rectangle(img, (x1, y1), (x2, y2), cor, 3)
            a.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            a.set_title(os.path.basename(caso[0])[:20] + (f"  conf={caso[2]:.2f}" if len(caso) > 2 else ""), fontsize=8)
        fig.suptitle(titulo); plt.tight_layout(); plt.savefig(arq, dpi=130); plt.close()
    painel(sorted(fps, key=lambda c: -c[2]), (0, 0, 255), "Falsos positivos no teste (mais confiantes primeiro) — caixa prevista sem buraco anotado (IoU < 0,5)", FIGS / "erros_fp.png")
    painel(fns, (255, 0, 0), "Falsos negativos no teste — buracos anotados sem detecção (IoU < 0,5)", FIGS / "erros_fn.png")
    res = dict(FP=len(fps), FN=len(fns), TP=len(tps), fatias={k: dict(instancias=t, detectadas=d, revocacao=round(d / t, 4) if t else None) for k, (d, t) in fat.items()})
    print(f"[4/5] {split}: TP={len(tps)} FP={len(fps)} FN={len(fns)} · fatias: " + ", ".join(f"{k} {d}/{t}" for k, (d, t) in fat.items()))
    return res


# ----------------------------------------------------------------------------- 6/7. máscaras × caixas (VAL)
def razao_area_e_painel(det: YOLO, seg: YOLO, imgsz: int, conf: float, iou_nms: float) -> dict:
    razoes = []; imgs = imagens("valid")
    for img_p in imgs:
        r = seg(str(img_p), conf=conf, iou=iou_nms, imgsz=imgsz, verbose=False, device=DEVICE)[0]
        if r.masks is None:
            continue
        H, W = r.orig_shape
        for m, b in zip(r.masks.data.cpu().numpy(), r.boxes.xyxy.cpu().numpy()):
            m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
            razoes.append(float(m.sum()) / max(1.0, float((b[2] - b[0]) * (b[3] - b[1]))))
    razoes = np.array(razoes)
    fig, ax = plt.subplots(figsize=(6, 4)); ax.hist(razoes, bins=30, color="#444")
    ax.axvline(np.median(razoes), color="r", ls="--", label=f"mediana {np.median(razoes):.2f}")
    ax.set_xlabel("área da máscara / área da caixa (instâncias previstas em val)"); ax.set_ylabel("instâncias"); ax.legend()
    plt.tight_layout(); plt.savefig(FIGS / "razao_area.png", dpi=150); plt.close()

    sel = imgs[:: max(1, len(imgs) // 8)][:8]
    fig, ax = plt.subplots(2, 8, figsize=(24, 6.4))
    for j, p in enumerate(sel):
        rd = det(str(p), conf=conf, iou=iou_nms, imgsz=imgsz, verbose=False, device=DEVICE)[0]
        rs = seg(str(p), conf=conf, iou=iou_nms, imgsz=imgsz, verbose=False, device=DEVICE)[0]
        ax[0, j].imshow(cv2.cvtColor(rd.plot(labels=False), cv2.COLOR_BGR2RGB)); ax[0, j].set_title("detecção (caixas)", fontsize=9)
        ax[1, j].imshow(cv2.cvtColor(rs.plot(labels=False), cv2.COLOR_BGR2RGB)); ax[1, j].set_title("segmentação (máscaras)", fontsize=9)
        ax[0, j].axis("off"); ax[1, j].axis("off")
    plt.tight_layout(); plt.savefig(FIGS / "caixas_vs_mascaras.png", dpi=120); plt.close()
    res = dict(n=int(len(razoes)), media=round(float(razoes.mean()), 4), mediana=round(float(np.median(razoes)), 4),
               p10=round(float(np.percentile(razoes, 10)), 4), p90=round(float(np.percentile(razoes, 90)), 4))
    print(f"[6/7] razão área máscara/caixa (val): n={res['n']} mediana={res['mediana']} P10={res['p10']} P90={res['p90']}")
    return res


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", required=True, help="run do detector principal (ex.: runs/detect/det_s)")
    ap.add_argument("--seg", default=None, help="run do segmentador principal (ex.: runs/segment/seg_s)")
    ap.add_argument("--extras", nargs="*", default=[], help="runs de detecção extras para a tabela")
    ap.add_argument("--piso-precisao", type=float, default=0.7)
    ap.add_argument("--regra", choices=["piso", "f1"], default="piso", help="piso: maior R com P>=piso; f1: máximo F1")
    ap.add_argument("--iou-nms", type=float, default=0.7)
    ap.add_argument("--sem-test", action="store_true", help="ensaio: não abre o conjunto de teste")
    ap.add_argument("--device", default=None, help="cuda | cpu (padrão: automático)")
    args = ap.parse_args()
    if args.device:
        import ultralytics.utils as _u; _u.DEFAULT_CFG.device = args.device  # noqa
        global DEVICE; DEVICE = args.device
    FIGS.mkdir(exist_ok=True)
    det_run = RAIZ / args.det; seg_run = RAIZ / args.seg if args.seg else None
    det = carregar(det_run); imgsz = imgsz_de(det_run)
    resultados = {"det": nome_run(det_run), "seg": nome_run(seg_run) if seg_run else None, "iou_nms": args.iou_nms}

    conf, resultados["varredura"] = varredura(det, imgsz, args.piso_precisao, args.iou_nms, args.regra)
    resultados["conf"] = conf
    splits = ["val"] if args.sem_test else ["val", "test"]

    linhas = []
    for run in [det_run] + [RAIZ / e for e in args.extras if (RAIZ / e / "weights" / "best.pt").exists()]:
        for s in splits:
            linhas.append(avaliar_run(run, "det", s, conf, args.iou_nms, plots=(s == "test" and run == det_run)))
    if seg_run:
        for s in splits:
            linhas.append(avaliar_run(seg_run, "seg", s, conf, args.iou_nms, plots=False))
    campos = ["modelo", "tarefa", "split", "imgsz", "box_mAP50", "box_mAP5095", "box_P", "box_R", "TP", "FP", "FN", "mask_mAP50", "mask_mAP5095", "mask_P", "mask_R"]
    with open(FIGS / "tabela_final.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=campos); w.writeheader(); [w.writerow({k: l.get(k, "") for k in campos}) for l in linhas]
    resultados["tabela"] = linhas
    print("[2] tabela_final.csv:"); [print("    ", {k: v for k, v in l.items() if v != ""}) for l in linhas]

    if not args.sem_test:
        cm = RAIZ / "runs" / "eval" / f"{nome_run(det_run)}_test" / "confusion_matrix.png"
        if cm.exists():
            shutil.copy(cm, FIGS / "confusion_matrix_test.png")
        if seg_run:
            seg = carregar(seg_run)
            resultados["iou_dice_test"] = iou_dice(seg, imgsz_de(seg_run), conf, args.iou_nms, "test", "labels")
            print(f"[3] test IoU/Dice vs labels atuais: {resultados['iou_dice_test']}")
            if (RAIZ / "dataset" / "test" / "labels_original").exists():
                resultados["iou_dice_test_labels_original"] = iou_dice(seg, imgsz_de(seg_run), conf, args.iou_nms, "test", "labels_original")
                print(f"[3] test IoU/Dice vs labels_original (retângulos): {resultados['iou_dice_test_labels_original']}")
        resultados["erros_test"] = erros_e_fatias(det, imgsz, conf, args.iou_nms, "test")
    if seg_run:
        resultados["razao_area_val"] = razao_area_e_painel(det, carregar(seg_run), imgsz, conf, args.iou_nms)

    for run in [det_run, seg_run] + [RAIZ / e for e in args.extras]:
        if run and (run / "results.png").exists():
            shutil.copy(run / "results.png", FIGS / f"curvas_{nome_run(run)}.png")
    (FIGS / "resultados.json").write_text(json.dumps(resultados, indent=2, ensure_ascii=False))
    print(f"[ok] figs/resultados.json escrito; conf={conf:.2f} iou_nms={args.iou_nms}")


if __name__ == "__main__":
    main()

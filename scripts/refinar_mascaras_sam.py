"""Refina as anotações só-caixa do dataset gerando máscaras com SAM (caixa como prompt).

Duas operações:

  qc       controle de qualidade — para instâncias que TÊM polígono humano, deriva a caixa do polígono,
           pede a máscara ao SAM com essa caixa e mede IoU contra o polígono humano. Gera figs/qc_sam.png
           e imprime mediana/média. Não altera nada.
  refinar  para cada instância anotada só por caixa (retângulo alinhado de 4 pontos, resultado de
           filtrar_classes.py), gera a máscara com SAM, recorta pela caixa, pega o maior componente,
           simplifica o contorno e substitui a linha na label. Originais preservados em <split>/labels_original/.

Uso:
    python scripts/refinar_mascaras_sam.py qc dataset --amostra 300
    python scripts/refinar_mascaras_sam.py refinar dataset --splits train valid test
"""
import argparse
import random
import shutil
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXT = (".jpg", ".jpeg", ".png")
MODELO = "facebook/sam-vit-base"


# ----------------------------------------------------------------------------- utilidades de label
def imagem_de(label: Path) -> Path | None:
    base = str(label).replace("/labels/", "/images/")[: -len(label.suffix)]
    for e in EXT:
        p = Path(base + e)
        if p.exists():
            return p
    return None


def ler_label(label: Path) -> list[tuple[int, np.ndarray]]:
    itens = []
    for l in label.read_text().splitlines():
        p = l.split()
        if len(p) >= 7:
            itens.append((int(p[0]), np.array(p[1:], float).reshape(-1, 2)))
    return itens


def eh_retangulo(poly_n: np.ndarray, tol: float = 1e-6) -> bool:
    """4 pontos formando retângulo alinhado aos eixos (assinatura das caixas convertidas)."""
    if len(poly_n) != 4:
        return False
    xs, ys = poly_n[:, 0], poly_n[:, 1]
    return (abs(xs[0] - xs[3]) < tol and abs(xs[1] - xs[2]) < tol
            and abs(ys[0] - ys[1]) < tol and abs(ys[2] - ys[3]) < tol)


def caixa_de(poly_px: np.ndarray) -> list[float]:
    return [float(poly_px[:, 0].min()), float(poly_px[:, 1].min()), float(poly_px[:, 0].max()), float(poly_px[:, 1].max())]


def mascara_de_poligono(poly_px: np.ndarray, H: int, W: int) -> np.ndarray:
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [poly_px.astype(np.int32)], 1)
    return m


def poligono_de_mascara(m: np.ndarray, caixa: list[float], H: int, W: int) -> np.ndarray | None:
    """Maior componente dentro da caixa → contorno simplificado → polígono normalizado (None se inválido)."""
    x1, y1, x2, y2 = (int(round(v)) for v in caixa)
    clip = np.zeros_like(m); clip[max(0, y1):min(H, y2 + 1), max(0, x1):min(W, x2 + 1)] = 1
    m = (m & clip).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n < 2:
        return None
    maior = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    m = (lab == maior).astype(np.uint8)
    area_caixa = max(1.0, (x2 - x1) * (y2 - y1))
    if m.sum() < 0.05 * area_caixa:                       # SAM devolveu quase nada: manter a caixa
        return None
    cont, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cont, key=cv2.contourArea)
    eps = 0.004 * cv2.arcLength(c, True)
    c = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
    if len(c) < 3:
        return None
    return np.clip(c / [W, H], 0, 1)


# ----------------------------------------------------------------------------- SAM
class Sam:
    def __init__(self):
        from transformers import SamModel, SamProcessor
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.proc = SamProcessor.from_pretrained(MODELO)
        self.model = SamModel.from_pretrained(MODELO).to(self.dev).eval()
        print(f"SAM {MODELO} em {self.dev}")

    @torch.no_grad()
    def mascaras(self, img_rgb: np.ndarray, caixas: list[list[float]]) -> np.ndarray:
        """(N, H, W) bool — uma máscara por caixa; encoder roda uma vez por imagem."""
        ent = self.proc(img_rgb, input_boxes=[caixas], return_tensors="pt").to(self.dev)
        s = self.model(**ent, multimask_output=False)
        m = self.proc.image_processor.post_process_masks(
            s.pred_masks.cpu(), ent["original_sizes"].cpu(), ent["reshaped_input_sizes"].cpu())[0]
        return m.squeeze(1).numpy().astype(bool)


# ----------------------------------------------------------------------------- operações
def qc(raiz: Path, amostra: int, saida: Path, splits: list[str]) -> float:
    random.seed(0)
    casos = []                                              # (label, idx da instância com polígono humano)
    for split in splits:
        for lab in sorted((raiz / split / "labels").glob("*.txt")):
            for i, (_, poly) in enumerate(ler_label(lab)):
                if not eh_retangulo(poly):
                    casos.append((lab, i))
    random.shuffle(casos); casos = casos[:amostra]
    por_label: dict[Path, list[int]] = {}
    for lab, i in casos:
        por_label.setdefault(lab, []).append(i)

    sam = Sam(); ious, exemplos = [], []
    for k, (lab, idxs) in enumerate(por_label.items(), 1):
        img_p = imagem_de(lab); img = cv2.imread(str(img_p)); H, W = img.shape[:2]
        itens = ler_label(lab)
        polys_px = [itens[i][1] * [W, H] for i in idxs]
        caixas = [caixa_de(p) for p in polys_px]
        ms = sam.mascaras(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caixas)
        for poly_px, caixa, m in zip(polys_px, caixas, ms):
            gt = mascara_de_poligono(poly_px, H, W)
            pn = poligono_de_mascara(m.astype(np.uint8), caixa, H, W)
            pr = mascara_de_poligono(pn * [W, H], H, W) if pn is not None else mascara_de_poligono(
                np.array([[caixa[0], caixa[1]], [caixa[2], caixa[1]], [caixa[2], caixa[3]], [caixa[0], caixa[3]]]), H, W)
            inter, uni = (gt & pr).sum(), (gt | pr).sum()
            iou = inter / uni if uni else 0.0
            ious.append(iou)
            if len(exemplos) < 8 and random.random() < 0.15:
                exemplos.append((img.copy(), poly_px, pn, iou, W, H))
        if k % 25 == 0:
            print(f"  {k}/{len(por_label)} imagens · IoU mediano parcial = {np.median(ious):.3f}", flush=True)

    ious = np.array(ious)
    print(f"\nQC SAM × polígono humano — n={len(ious)}: mediana={np.median(ious):.3f}  média={ious.mean():.3f}  "
          f"P10={np.percentile(ious, 10):.3f}  P90={np.percentile(ious, 90):.3f}  ≥0,7: {(ious >= 0.7).mean() * 100:.0f}%")

    saida.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(16, 8)); gs = fig.add_gridspec(2, 5)
    ax = fig.add_subplot(gs[:, 0]); ax.hist(ious, bins=25, color="#444"); ax.axvline(0.7, color="r", ls="--", label="0,7")
    ax.set_xlabel("IoU (SAM × humano)"); ax.set_title(f"n={len(ious)} · mediana {np.median(ious):.2f}"); ax.legend()
    for j, (img, poly_px, pn, iou, W, H) in enumerate(exemplos[:8]):
        a = fig.add_subplot(gs[j // 4, 1 + j % 4])
        vis = img.copy()
        cv2.polylines(vis, [poly_px.astype(np.int32)], True, (0, 0, 255), 2)                 # humano: vermelho
        if pn is not None:
            cv2.polylines(vis, [(pn * [W, H]).astype(np.int32)], True, (0, 255, 0), 2)     # SAM: verde
        x1, y1, x2, y2 = (int(v) for v in caixa_de(poly_px)); pad = int(0.5 * max(x2 - x1, y2 - y1)) + 10
        vis = vis[max(0, y1 - pad):min(H, y2 + pad), max(0, x1 - pad):min(W, x2 + pad)]
        a.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)); a.set_title(f"IoU {iou:.2f}", fontsize=9); a.axis("off")
    fig.suptitle("Controle de qualidade: polígono humano (vermelho) × máscara do SAM a partir da caixa (verde)")
    plt.tight_layout(); plt.savefig(saida / "qc_sam.png", dpi=130)
    print(f"figura: {saida / 'qc_sam.png'}")
    return float(np.median(ious))


def refinar(raiz: Path, splits: list[str], saida: Path) -> None:
    sam = Sam(); tot_caixas = tot_refinadas = 0; exemplos = []
    for split in splits:
        ldir, bdir = raiz / split / "labels", raiz / split / "labels_original"
        if not bdir.exists():
            shutil.copytree(ldir, bdir)
            print(f"[{split}] backup em {bdir}")
        labels = sorted(ldir.glob("*.txt"))
        for k, lab in enumerate(labels, 1):
            itens = ler_label(lab)
            idxs = [i for i, (_, p) in enumerate(itens) if eh_retangulo(p)]
            if not idxs:
                continue
            img_p = imagem_de(lab); img = cv2.imread(str(img_p)); H, W = img.shape[:2]
            caixas = [caixa_de(itens[i][1] * [W, H]) for i in idxs]
            ms = sam.mascaras(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caixas)
            novos = {}
            for i, caixa, m in zip(idxs, caixas, ms):
                tot_caixas += 1
                pn = poligono_de_mascara(m.astype(np.uint8), caixa, H, W)
                if pn is not None:
                    novos[i] = pn; tot_refinadas += 1
                    if len(exemplos) < 8 and random.random() < 0.02:
                        exemplos.append((img.copy(), caixa, pn, W, H))
            linhas = []
            for i, (cls, poly) in enumerate(itens):
                p = novos.get(i, poly)
                linhas.append(f"{cls} " + " ".join(f"{v:.6f}" for v in p.ravel()))
            lab.write_text("\n".join(linhas) + "\n")
            if k % 100 == 0:
                print(f"  [{split}] {k}/{len(labels)} labels · refinadas {tot_refinadas}/{tot_caixas}", flush=True)
    for cache in raiz.glob("*/labels.cache"):
        cache.unlink()
    print(f"\ncaixas encontradas: {tot_caixas} · refinadas com SAM: {tot_refinadas} "
          f"({100 * tot_refinadas / max(1, tot_caixas):.1f}%) · mantidas como retângulo: {tot_caixas - tot_refinadas}")

    saida.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(2, 4, figsize=(16, 7)); ax = ax.ravel()
    for a, (img, caixa, pn, W, H) in zip(ax, exemplos):
        x1, y1, x2, y2 = (int(v) for v in caixa)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.polylines(img, [(pn * [W, H]).astype(np.int32)], True, (0, 255, 0), 2)
        pad = int(0.5 * max(x2 - x1, y2 - y1)) + 10
        a.imshow(cv2.cvtColor(img[max(0, y1 - pad):min(H, y2 + pad), max(0, x1 - pad):min(W, x2 + pad)], cv2.COLOR_BGR2RGB)); a.axis("off")
    fig.suptitle("Refinamento: caixa original (vermelho) → máscara do SAM (verde)")
    plt.tight_layout(); plt.savefig(saida / "refino_sam_exemplos.png", dpi=130)
    print(f"figura: {saida / 'refino_sam_exemplos.png'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("operacao", choices=["qc", "refinar"])
    ap.add_argument("raiz")
    ap.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    ap.add_argument("--amostra", type=int, default=300, help="qc: nº de instâncias com polígono a avaliar")
    ap.add_argument("--saida", default="figs")
    args = ap.parse_args()
    raiz, saida = Path(args.raiz).resolve(), Path(args.saida)
    if args.operacao == "qc":
        med = qc(raiz, args.amostra, saida, args.splits)
        print("APROVADO (mediana ≥ 0,7)" if med >= 0.7 else "REPROVADO (mediana < 0,7) — manter rota A")
        sys.exit(0 if med >= 0.7 else 2)
    refinar(raiz, args.splits, saida)


if __name__ == "__main__":
    main()

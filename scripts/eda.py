"""Análise exploratória de um dataset YOLO-seg: contagens, resoluções, luz, tamanho das instâncias.

Gera figs/eda.png, figs/amostras.png e figs/eda_tabela.md.

Uso:
    python scripts/eda.py dataset --saida figs
"""
import argparse
import random
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXT = (".jpg", ".jpeg", ".png")


def imagem_de(label: Path) -> Path | None:
    # nomes do Roboflow têm pontos ("x_jpg.rf.<hash>.txt"): concatenar, não usar with_suffix
    base = str(label).replace("/labels/", "/images/")[: -len(label.suffix)]
    for e in EXT:
        p = Path(base + e)
        if p.exists():
            return p
    return None


def area_poligono(p: list[str], w: int, h: int) -> float:
    xy = np.array(p[1:], float).reshape(-1, 2) * [w, h]
    return 0.5 * abs(np.dot(xy[:, 0], np.roll(xy[:, 1], 1)) - np.dot(xy[:, 1], np.roll(xy[:, 0], 1)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("raiz")
    ap.add_argument("--saida", default="figs")
    args = ap.parse_args()
    raiz, saida = Path(args.raiz), Path(args.saida)
    saida.mkdir(parents=True, exist_ok=True)

    rows, amostras = [], []
    for split in ("train", "valid", "test"):
        for lab in sorted((raiz / split / "labels").glob("*.txt")):
            img_p = imagem_de(lab)
            if img_p is None:
                continue
            img = cv2.imread(str(img_p))
            h, w = img.shape[:2]
            polys = [l.split() for l in lab.read_text().splitlines() if l.strip()]
            areas = [area_poligono(p, w, h) for p in polys if len(p) > 5]
            lum = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean()
            rows.append(dict(split=split, w=w, h=h, n_inst=len(polys), lum=lum,
                             area_med=float(np.median(areas)) if areas else 0.0,
                             n_peq=sum(a < 32 * 32 for a in areas)))
            if split == "train" and polys:
                amostras.append((img_p, polys))
    df = pd.DataFrame(rows)

    resumo = df.groupby("split").agg(imagens=("n_inst", "size"), instancias=("n_inst", "sum"),
                                     inst_por_img=("n_inst", "mean"), sem_buraco=("n_inst", lambda s: int((s == 0).sum())),
                                     lum_media=("lum", "mean"), inst_pequenas=("n_peq", "sum")).round(2)
    resol = df.groupby(["w", "h"]).size().sort_values(ascending=False).head(5)
    md = ["| split | imagens | instâncias | inst/img | imgs sem buraco | luminância média | inst < 32² px |",
          "|---|---|---|---|---|---|---|"]
    for s, r in resumo.iterrows():
        md.append(f"| {s} | {int(r.imagens)} | {int(r.instancias)} | {r.inst_por_img} | {int(r.sem_buraco)} | {r.lum_media} | {int(r.inst_pequenas)} |")
    md += ["", "Resoluções mais frequentes (w×h → imagens):", ""] + [f"- {w}×{h}: {n}" for (w, h), n in resol.items()]
    (saida / "eda_tabela.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    df.n_inst.hist(bins=range(0, int(df.n_inst.max()) + 2), ax=ax[0], color="#444")
    ax[0].set_title("instâncias por imagem"); ax[0].set_xlabel("buracos anotados")
    df.lum.hist(bins=30, ax=ax[1], color="#444"); ax[1].set_title("luminância média da imagem"); ax[1].set_xlabel("0 = escuro, 255 = claro")
    a = df.area_med[df.area_med > 0]
    np.log10(a).hist(bins=30, ax=ax[2], color="#444"); ax[2].axvline(np.log10(32 * 32), color="r", ls="--", label="32² px")
    ax[2].set_title("área mediana do buraco por imagem"); ax[2].set_xlabel("log10(px²)"); ax[2].legend()
    plt.tight_layout(); plt.savefig(saida / "eda.png", dpi=150)

    random.seed(0)
    fig, ax = plt.subplots(2, 4, figsize=(16, 7)); ax = ax.ravel()
    for a_, (img_p, polys) in zip(ax, random.sample(amostras, min(8, len(amostras)))):
        img = cv2.imread(str(img_p)); h, w = img.shape[:2]
        for p in polys:
            if len(p) > 5:
                xy = (np.array(p[1:], float).reshape(-1, 2) * [w, h]).astype(np.int32)
                cv2.polylines(img, [xy], True, (0, 0, 255), 3)
        a_.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); a_.set_title(img_p.name[:24], fontsize=8); a_.axis("off")
    plt.suptitle("amostras de treino com polígonos anotados"); plt.tight_layout(); plt.savefig(saida / "amostras.png", dpi=130)
    print(f"figuras em {saida}/")


if __name__ == "__main__":
    main()

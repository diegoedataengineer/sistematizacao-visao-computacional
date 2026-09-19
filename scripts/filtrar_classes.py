"""Normaliza as labels YOLO do dataset: classe única e formato único (polígono).

1. Mantém apenas as linhas cujo índice de classe está em --manter e remapeia todas para 0.
2. Converte linhas em formato caixa (cls cx cy w h) para polígono retangular de 4 pontos —
   o Ultralytics rejeita arquivos que misturam caixas e polígonos ("labels mix").
3. Reescreve o data.yaml com nc: 1 e caminho absoluto; remove caches de labels antigos.

Uso:
    python scripts/filtrar_classes.py dataset --manter 0
"""
import argparse
from pathlib import Path

import yaml


def caixa_para_poligono(cx: float, cy: float, w: float, h: float) -> list[str]:
    x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    clip = lambda v: f"{min(max(v, 0.0), 1.0):.6f}"
    return [clip(x1), clip(y1), clip(x2), clip(y1), clip(x2), clip(y2), clip(x1), clip(y2)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("raiz")
    ap.add_argument("--manter", type=int, action="append", required=True, help="índices de classe a manter (repetível)")
    args = ap.parse_args()

    raiz = Path(args.raiz).resolve()
    manter = set(args.manter)
    antes = depois = caixas = poligonos = arquivos_mistos = 0
    for lab in raiz.glob("*/labels/*.txt"):
        linhas = [l for l in lab.read_text().splitlines() if l.strip()]
        antes += len(linhas)
        novas, tem_caixa, tem_poly = [], False, False
        for l in linhas:
            p = l.split()
            if int(p[0]) not in manter:
                continue
            if len(p) == 5:                                   # caixa → polígono retangular
                tem_caixa = True; caixas += 1
                novas.append("0 " + " ".join(caixa_para_poligono(*map(float, p[1:]))))
            else:
                tem_poly = True; poligonos += 1
                novas.append("0 " + " ".join(p[1:]))
        if tem_caixa and tem_poly:
            arquivos_mistos += 1
        depois += len(novas)
        lab.write_text("\n".join(novas) + ("\n" if novas else ""))

    for cache in raiz.glob("*/labels.cache"):
        cache.unlink()

    cfg = {"path": str(raiz), "train": "train/images", "val": "valid/images", "test": "test/images",
           "nc": 1, "names": ["pothole"]}
    (raiz / "data.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    print(f"instâncias: {antes} → {depois}  (classes mantidas: {sorted(manter)} → 0)")
    print(f"formato: {poligonos} polígonos originais + {caixas} caixas convertidas em retângulos "
          f"({100 * caixas / max(1, depois):.1f}% das instâncias); arquivos que misturavam formatos: {arquivos_mistos}")
    print(f"data.yaml reescrito: {raiz / 'data.yaml'}; caches removidos")


if __name__ == "__main__":
    main()

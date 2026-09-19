"""Baixa um dataset do Roboflow Universe no formato YOLO (segmentação).

Uso:
    export ROBOFLOW_API_KEY=...
    python scripts/baixar_dataset.py --workspace <ws> --project <proj> --version <n> [--destino dataset]

A chave nunca é gravada no repositório: vem só da variável de ambiente.
"""
import argparse
import os
import sys
from pathlib import Path

import yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--project", required=True)
    ap.add_argument("--version", type=int, required=True)
    ap.add_argument("--destino", default="dataset")
    ap.add_argument("--formato", default="yolov11", help="yolov11 | yolov8 (ambos exportam polígonos)")
    args = ap.parse_args()

    chave = os.environ.get("ROBOFLOW_API_KEY")
    if not chave:
        sys.exit("defina ROBOFLOW_API_KEY no ambiente (chave gratuita em app.roboflow.com → Settings → API)")

    from roboflow import Roboflow

    ds = (
        Roboflow(api_key=chave)
        .workspace(args.workspace)
        .project(args.project)
        .version(args.version)
        .download(args.formato, location=args.destino, overwrite=True)
    )
    raiz = Path(ds.location)
    cfg = yaml.safe_load((raiz / "data.yaml").read_text())
    print(f"baixado em {raiz}")
    print(f"classes originais ({cfg.get('nc')}): {cfg.get('names')}")
    for split in ("train", "valid", "test"):
        imgs = list((raiz / split / "images").glob("*")) if (raiz / split).exists() else []
        print(f"  {split:5s}: {len(imgs)} imagens")


if __name__ == "__main__":
    main()

"""Gera notebooks/sistematizacao.ipynb — notebook autossuficiente para o Colab (T4 ou A100).

O notebook é um invólucro fino sobre os scripts do repositório: clona, instala, baixa o dataset,
filtra/refina labels, treina, avalia e roda o vídeo. Assim o que roda no Colab é o mesmo código do repo.

Uso: python scripts/gerar_notebook.py
"""
import json
from pathlib import Path

REPO = "https://github.com/diegoedataengineer/sistematizacao-visao-computacional"
SAIDA = Path(__file__).resolve().parent.parent / "notebooks" / "sistematizacao.ipynb"


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n")}


def code(src: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src.strip("\n")}


cells = [
md(f"""
# Detecção e segmentação de buracos em vias urbanas com YOLO11

**Sistematização — Visão Computacional e Reconhecimento de Padrões (CEUB) · Prof. Romes Heriberto**

Integrantes: Diego Nunes de Morais <!-- adicionar os demais -->

Repositório: {REPO}

Este notebook executa, na ordem, o pipeline completo do projeto usando os scripts do repositório:
dados → EDA → refino de máscaras (SAM) → treino (detecção e segmentação) → avaliação (protocolo de teste único) → vídeo.

**Antes de executar:** `Ambiente de execução → Alterar tipo de ambiente → GPU` (A100 se disponível; T4 funciona).
Tempos aproximados no A100: refino SAM ~3 min · YOLO11s det ~15 min · YOLO11s-seg ~20 min · avaliação ~5 min.
"""),
code("""
#@title 0. Configuração
TREINAR   = True    #@param {type:"boolean"}   — False: baixa os pesos já treinados (Release do GitHub) em vez de treinar
REFINAR   = True    #@param {type:"boolean"}   — refinar máscaras das instâncias só-caixa com SAM (rota B, ADR-S04)
EXTRAS    = False   #@param {type:"boolean"}   — treinar também yolo11s@800 e yolo11m (tabela comparativa)
USAR_DRIVE = False  #@param {type:"boolean"}   — montar o Drive e trabalhar em MyDrive/visao-computacional/sistematizacao

import os, subprocess, sys
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True).stdout)
if USAR_DRIVE:
    from google.colab import drive; drive.mount('/content/drive')
    BASE = '/content/drive/MyDrive/visao-computacional/sistematizacao'; os.makedirs(BASE, exist_ok=True)
else:
    BASE = '/content'
os.chdir(BASE); print("pasta de trabalho:", BASE)
"""),
code(f"""
#@title 1. Clonar o repositório e instalar dependências (≈2 min)
import os, subprocess
if not os.path.isdir("sistematizacao-visao-computacional"):
    r = subprocess.run(["git", "clone", "-q", "{REPO}.git"], capture_output=True, text=True)
    if r.returncode != 0:                       # repositório privado: pedir token
        from getpass import getpass
        tok = getpass("Token do GitHub (repo privado): ")
        subprocess.run(["git", "clone", "-q", f"https://{{tok}}@github.com/diegoedataengineer/sistematizacao-visao-computacional.git"], check=True)
os.chdir("sistematizacao-visao-computacional")
!git pull -q
!pip install -q -r requirements.txt transformers accelerate
import ultralytics, torch; print("ultralytics", ultralytics.__version__, "| torch", torch.__version__, "| cuda", torch.cuda.is_available())
"""),
code("""
#@title 2. Dataset: download (Roboflow) e classe única `pothole`
import os
from getpass import getpass
if not os.path.isdir("dataset/train/images"):
    os.environ["ROBOFLOW_API_KEY"] = getpass("Chave da API do Roboflow (app.roboflow.com → Settings → API): ")
    !python scripts/baixar_dataset.py --workspace pothole-vsmtu --project potholes-and-roads-instance-segmentation --version 5
    !python scripts/filtrar_classes.py dataset --manter 0
else:
    print("dataset já presente")
!for s in train valid test; do printf "%-6s %s imagens\\n" $s "$(ls dataset/$s/images | wc -l)"; done
"""),
code("""
#@title 3. Análise exploratória
!python scripts/eda.py dataset --saida figs
from IPython.display import Image, display, Markdown
display(Markdown(open("figs/eda_tabela.md").read()))
display(Image("figs/eda.png")); display(Image("figs/amostras.png", width=1000))
"""),
code("""
#@title 4. Refino das máscaras com SAM (rota B) — QC e refinamento
import os
if REFINAR and not os.path.isdir("dataset/train/labels_original"):
    !python scripts/refinar_mascaras_sam.py qc dataset --amostra 300 --saida figs
    !python scripts/refinar_mascaras_sam.py refinar dataset --splits train valid test --saida figs
    from IPython.display import Image, display
    display(Image("figs/qc_sam.png", width=1100)); display(Image("figs/refino_sam_exemplos.png", width=1000))
else:
    print("refino desativado ou já aplicado (labels_original existe)")
"""),
code("""
#@title 5. Treino — detecção (YOLO11s) e segmentação (YOLO11s-seg)
import os, subprocess
if TREINAR:
    !python scripts/treinar.py --tarefa det --modelo yolo11n.pt     --epocas 30 --nome baseline_n
    !python scripts/treinar.py --tarefa det --modelo yolo11s.pt     --epocas 50 --nome det_s
    !python scripts/treinar.py --tarefa seg --modelo yolo11s-seg.pt --epocas 50 --nome seg_s
    if EXTRAS:
        !python scripts/treinar.py --tarefa det --modelo yolo11s.pt --epocas 50 --imgsz 800 --nome det_s800
        !python scripts/treinar.py --tarefa det --modelo yolo11m.pt --epocas 50 --nome det_m
else:
    # pesos publicados na Release do GitHub (ver README) — coloca em runs/<tarefa>/<nome>/weights/best.pt
    for nome, tarefa in (("det_s", "detect"), ("seg_s", "segment"), ("baseline_n", "detect")):
        d = f"runs/{tarefa}/{nome}/weights"; os.makedirs(d, exist_ok=True)
        subprocess.run(["gh", "release", "download", "--pattern", f"{nome}_best.pt", "-O", f"{d}/best.pt", "--clobber"], check=False)
    print("pesos baixados (se a Release existir); caso contrário ative TREINAR")
!grep -h "val:" runs/logs/*.log
"""),
code("""
#@title 6. Avaliação — varredura de limiar em val, avaliação ÚNICA no teste, erros, fatias, máscaras × caixas
!python scripts/avaliar.py --det runs/detect/det_s --seg runs/segment/seg_s --extras runs/detect/baseline_n runs/detect/det_s800 runs/detect/det_m
import pandas as pd, json
from IPython.display import Image, display
display(pd.read_csv("figs/tabela_final.csv"))
print(json.dumps({k: v for k, v in json.load(open("figs/resultados.json")).items() if k in ("conf", "iou_dice_test", "iou_dice_test_labels_original", "erros_test", "razao_area_val")}, indent=2, ensure_ascii=False))
for f in ("pr_limiar.png", "confusion_matrix_test.png", "erros_fp.png", "erros_fn.png", "caixas_vs_mascaras.png", "razao_area.png"):
    display(Image(f"figs/{f}", width=1000))
"""),
code("""
#@title 7. Vídeo real (≥ 30 s): inferência com máscaras e rastreamento ByteTrack (bônus)
import json, os, cv2, time, numpy as np
from ultralytics import YOLO
from google.colab import files
os.makedirs("video", exist_ok=True)
CONF = json.load(open("figs/resultados.json"))["conf"]; IOU_NMS = 0.7
VIDEO = "video/cenario.mp4"
if not os.path.exists(VIDEO):
    up = files.upload(); os.rename(list(up)[0], VIDEO)
seg = YOLO("runs/segment/seg_s/weights/best.pt")
seg.predict(source=VIDEO, conf=CONF, iou=IOU_NMS, imgsz=640, save=True, project="video", name="seg", exist_ok=True)

ids, n_det, tempos = set(), 0, []
cap = cv2.VideoCapture(VIDEO); W, H, fps = int(cap.get(3)), int(cap.get(4)), cap.get(5)
out = cv2.VideoWriter("video/cenario_track.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
while True:
    ok, q = cap.read()
    if not ok: break
    t0 = time.perf_counter()
    r = seg.track(q, persist=True, tracker="bytetrack.yaml", conf=CONF, iou=IOU_NMS, imgsz=640, verbose=False)[0]
    tempos.append((time.perf_counter() - t0) * 1000)
    if r.boxes.id is not None:
        ids.update(r.boxes.id.int().tolist()); n_det += len(r.boxes)
    out.write(r.plot())
cap.release(); out.release()
print(f"buracos únicos (IDs) = {len(ids)} · detecções somadas por quadro = {n_det} · latência média = {np.mean(tempos):.1f} ms/quadro ({1000/np.mean(tempos):.1f} FPS de inferência)")
!ls -la video/ video/seg/ 2>/dev/null
"""),
md("""
## Protocolo e observações

- **Teste único:** o conjunto `test` é avaliado uma única vez pela célula 6, com `conf` e `iou` escolhidos em `val`. Não re-treine olhando os números do teste (ADR-S05).
- **Máscaras de referência:** 55% das instâncias do dataset original estavam anotadas só por caixa; a célula 4 as refina com SAM (QC: IoU mediano 0,703 contra polígonos humanos, vs 0,684 do retângulo). `labels_original/` preserva as anotações originais e a avaliação reporta IoU/Dice contra as duas.
- **Relatório:** os números de `figs/tabela_final.csv` e `figs/resultados.json` são os que entram em `docs/relatorio.md`.
- **Entrega:** `Arquivo → Fazer download → .ipynb` com as saídas visíveis, após `Ambiente de execução → Reiniciar e executar tudo`.
"""),
]

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}, "colab": {"provenance": [], "gpuType": "A100"},
                                   "accelerator": "GPU"}, "nbformat": 4, "nbformat_minor": 5}
SAIDA.parent.mkdir(exist_ok=True)
SAIDA.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print(f"notebook gerado: {SAIDA} ({len(cells)} células)")

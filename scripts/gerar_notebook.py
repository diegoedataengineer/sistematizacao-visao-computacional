"""Gera notebooks/sistematizacao.ipynb — notebook executável que reproduz os resultados do relatório.

Roda no Colab (clona o repositório) ou localmente (na raiz do repositório). Cada célula mostra o resultado
da etapa: inspeção das labels, EDA, controle de qualidade do SAM, treinos, avaliação com tabela e figuras,
vídeo. Com TREINAR = False usa os pesos já treinados (runs/ ou Release v1.0) e reproduz exatamente os
números do relatório; com TREINAR = True refaz os treinos.

Uso: python scripts/gerar_notebook.py
Executar com saídas: jupyter nbconvert --to notebook --execute --inplace notebooks/sistematizacao.ipynb
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

**Sistematização — Visão Computacional e Reconhecimento de Padrões (CEUB) · Prof. Dr. Romes Heriberto Pires de Araújo**

Autor: Diego Nunes de Morais — trabalho individual · Repositório: {REPO}

Este notebook reproduz, na ordem, o pipeline do projeto e os resultados do relatório (`reports/relatorio.pdf`):

1. dataset (download, classe única, correção das labels mistas) e análise exploratória;
2. refino das máscaras com SAM, com controle de qualidade contra os polígonos humanos;
3. treino — YOLO11n (baseline), YOLO11s (detecção) e YOLO11s-seg (segmentação);
4. avaliação — limiar escolhido em validação, **teste avaliado uma única vez**, matriz de confusão, IoU/Dice, análise de erros, máscaras × caixas;
5. inferência em vídeo com rastreamento (ByteTrack).

Com `TREINAR = False` (padrão) as células de treino usam os pesos já treinados (`runs/` ou Release `v1.0`) e a avaliação reproduz exatamente os números do relatório. No Colab: `Ambiente de execução → Alterar tipo → GPU`.
"""),
code("""
#@title 0. Configuração e ambiente
TREINAR = False        #@param {type:"boolean"}  — True: refaz os treinos (~1 h num A100, ~5 h numa GTX 1060)
EXTRAS  = False        #@param {type:"boolean"}  — treinar também yolo11s@800 e yolo11m (só com TREINAR)

import os, sys, subprocess, json, csv
from pathlib import Path
IN_COLAB = "google.colab" in sys.modules or os.path.exists("/content")
if IN_COLAB and not Path("scripts/treinar.py").exists():
    os.chdir("/content")
    if not Path("sistematizacao-visao-computacional").exists():
        r = subprocess.run(["git", "clone", "-q", "%s.git" % "REPO_URL"], capture_output=True, text=True)
        if r.returncode != 0:                                   # repositório privado: token
            from getpass import getpass
            tok = getpass("Token do GitHub (repo privado): ")
            subprocess.run(["git", "clone", "-q", f"https://{tok}@github.com/diegoedataengineer/sistematizacao-visao-computacional.git"], check=True)
    os.chdir("sistematizacao-visao-computacional")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt", "transformers", "accelerate"], check=True)
else:                                                           # local: sobe até a raiz do repositório
    while not Path("scripts/treinar.py").exists() and Path.cwd() != Path.cwd().parent:
        os.chdir("..")
RAIZ = Path.cwd(); print("raiz do projeto:", RAIZ)
import torch, ultralytics
print("ultralytics", ultralytics.__version__, "| torch", torch.__version__, "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "nenhuma")
"""),
md("""
## 1. Dataset

*Potholes and Roads Instance Segmentation* (workspace `pothole-vsmtu`, Roboflow Universe, v5, CC BY 4.0). Mantemos só a classe `pothole` e convertemos as instâncias anotadas por caixa em polígonos retangulares — o Ultralytics descarta arquivos que misturam os dois formatos (ver relatório, seção 3.1).
"""),
code("""
#@title 1.1 Download (se necessário), classe única e contagem por split
if not Path("dataset/train/images").exists():
    from getpass import getpass
    os.environ["ROBOFLOW_API_KEY"] = getpass("Chave da API do Roboflow (app.roboflow.com → Settings → API): ")
    subprocess.run([sys.executable, "scripts/baixar_dataset.py", "--workspace", "pothole-vsmtu", "--project", "potholes-and-roads-instance-segmentation", "--version", "5"], check=True)
    subprocess.run([sys.executable, "scripts/filtrar_classes.py", "dataset", "--manter", "0"], check=True)
for split in ("train", "valid", "test"):
    imgs = len(list(Path(f"dataset/{split}/images").glob("*")))
    inst = sum(1 for lab in Path(f"dataset/{split}/labels").glob("*.txt") for l in lab.read_text().splitlines() if l.strip())
    print(f"{split:5s}: {imgs:5d} imagens · {inst:5d} instâncias pothole")
"""),
code("""
#@title 1.2 Inspeção das anotações: polígonos × caixas (nas labels originais, preservadas em labels_original/)
import numpy as np
def formato(linha):
    p = linha.split(); return "caixa" if len(p) == 5 else ("retângulo" if len(p) == 9 and float(p[1]) == float(p[7]) and float(p[3]) == float(p[5]) else "polígono")
origem = "labels_original" if Path("dataset/train/labels_original").exists() else "labels"
cont = {"polígono": 0, "retângulo": 0, "caixa": 0}; mistos = 0
for split in ("train", "valid", "test"):
    for lab in Path(f"dataset/{split}/{origem}").glob("*.txt"):
        tipos = [formato(l) for l in lab.read_text().splitlines() if l.strip()]
        for t in tipos: cont[t] += 1
        if len(set(tipos)) > 1: mistos += 1
tot = sum(cont.values())
print(f"fonte: {origem}/ · instâncias: {tot}")
for k, v in cont.items():
    if v: print(f"  {k:10s} {v:5d} ({100*v/tot:.0f}%)")
print(f"arquivos que misturavam formatos: {mistos}")
"""),
code("""
#@title 1.3 Análise exploratória
%run scripts/eda.py dataset --saida figs
from IPython.display import Image, display, Markdown
display(Image("figs/eda.png")); display(Image("figs/amostras.png", width=1000))
"""),
md("""
## 2. Refino das máscaras com SAM

55% das instâncias estavam anotadas só por caixa. Para cada uma, o SAM (ViT-B) gera a máscara com a caixa como prompt. Antes de substituir as anotações, medimos em 300 instâncias **com** polígono humano se a máscara do SAM concorda mais com o humano do que o retângulo.
"""),
code("""
#@title 2.1 Controle de qualidade e refinamento (pulados se já aplicados)
if not Path("dataset/train/labels_original").exists():
    %run scripts/refinar_mascaras_sam.py qc dataset --amostra 300 --saida figs
    %run scripts/refinar_mascaras_sam.py refinar dataset --splits train valid test --saida figs
log = Path("runs/logs/qc_sam.log")
if log.exists():
    print([l for l in log.read_text().splitlines() if l.startswith("QC SAM")][-1])
ret = Path("runs/logs/qc_retangulo.txt")
if ret.exists(): print("RETÂNGULO × polígono humano —", ret.read_text().strip())
display(Image("figs/qc_sam.png", width=1100))
"""),
md("""
## 3. Treino

Protocolo comum: pesos pré-treinados no COCO, 640 px, batch 16, 50 épocas com *patience* 10, seed 0, augmentation padrão do Ultralytics. Baseline YOLO11n por 30 épocas.
"""),
code("""
#@title 3.1 Treinos (ou resumo dos treinos existentes)
def resumo(nome, tarefa):
    d = Path("runs") / ("segment" if tarefa == "seg" else "detect") / nome
    if not (d / "results.csv").exists():
        return print(f"{nome}: sem resultados")
    rows = list(csv.DictReader(open(d / "results.csv")))
    m50 = [float(r["metrics/mAP50(B)"]) for r in rows]; m = [float(r["metrics/mAP50-95(B)"]) for r in rows]
    best = int(np.argmax(m)) + 1; t = float(rows[-1]["time"])
    extra = f" · mask mAP50 final {float(rows[-1]['metrics/mAP50(M)']):.3f}" if tarefa == "seg" else ""
    print(f"{nome:11s} {len(rows):2d} épocas · {t/60:4.0f} min · val mAP50 final {m50[-1]:.3f} · melhor mAP50-95 {m[best-1]:.3f} (época {best}){extra}")

treinos = [("baseline_n", "det", "yolo11n.pt", 30, 640), ("det_s", "det", "yolo11s.pt", 50, 640), ("seg_s", "seg", "yolo11s-seg.pt", 50, 640)]
if EXTRAS: treinos += [("det_s800", "det", "yolo11s.pt", 50, 800), ("det_m", "det", "yolo11m.pt", 50, 640)]
for nome, tarefa, modelo, ep, sz in treinos:
    pesos = Path("runs") / ("segment" if tarefa == "seg" else "detect") / nome / "weights" / "best.pt"
    if TREINAR or not pesos.exists():
        if not TREINAR:                                          # tenta os pesos publicados antes de treinar
            pesos.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["gh", "release", "download", "v1.0", "--pattern", f"{nome}_best.pt", "-O", str(pesos), "--clobber"], capture_output=True)
        if TREINAR or not pesos.exists():
            %run scripts/treinar.py --tarefa {tarefa} --modelo {modelo} --epocas {ep} --imgsz {sz} --nome {nome}
    resumo(nome, tarefa)
display(Image("figs/curvas_det_s.png" if Path("figs/curvas_det_s.png").exists() else "runs/detect/det_s/results.png", width=1000))
"""),
md("""
## 4. Avaliação

`val` escolhe o limiar (máximo F1 na curva precisão × revocação, casamento um-para-um com IoU ≥ 0,5); `test` é avaliado uma única vez. mAP no protocolo padrão (conf 0,001); precisão e revocação no ponto de operação.
"""),
code("""
#@title 4.1 Varredura de limiar em validação, avaliação no teste, erros, fatias e máscaras × caixas
extras = [p for p in ("runs/detect/baseline_n", "runs/detect/det_s800", "runs/detect/det_m") if Path(p, "weights/best.pt").exists()]
%run scripts/avaliar.py --det runs/detect/det_s --seg runs/segment/seg_s --extras {" ".join(extras)} --regra f1
"""),
code("""
#@title 4.2 Tabela final (val e teste)
import pandas as pd
tab = pd.read_csv("figs/tabela_final.csv")
R = json.load(open("figs/resultados.json"))
print(f"ponto de operação: conf = {R['conf']:.2f} ({R['varredura']['regra']}) · IoU NMS = {R['iou_nms']}")
display(tab.set_index(["modelo", "split"]).round(3))
"""),
code("""
#@title 4.3 Curva precisão × revocação por limiar e matriz de confusão do teste
display(Image("figs/pr_limiar.png", width=520)); display(Image("figs/confusion_matrix_test.png", width=520))
e = R["erros_test"]; print(f"teste (conf {R['conf']:.2f}): TP={e['TP']} FP={e['FP']} FN={e['FN']}")
print(pd.read_csv("figs/fatias.csv").to_string(index=False))
"""),
code("""
#@title 4.4 Análise de erros: falsos positivos e falsos negativos no teste
display(Image("figs/erros_fp.png", width=1100)); display(Image("figs/erros_fn.png", width=1100))
"""),
code("""
#@title 4.5 Segmentação: máscaras × caixas, razão de área, IoU/Dice
display(Image("figs/caixas_vs_mascaras.png", width=1200)); display(Image("figs/razao_area.png", width=520))
ra = R["razao_area_val"]; print(f"razão área máscara/caixa (val, n={ra['n']}): mediana {ra['mediana']:.3f} · P10 {ra['p10']:.3f} · P90 {ra['p90']:.3f}")
print("IoU/Dice no teste vs referências refinadas :", R["iou_dice_test"])
if "iou_dice_test_labels_original" in R: print("IoU/Dice no teste vs anotações originais   :", R["iou_dice_test_labels_original"])
"""),
md("""
## 5. Vídeo

Inferência do segmentador (caixas + máscaras) no vídeo real do cenário, com o mesmo ponto de operação, e rastreamento ByteTrack para contar buracos únicos. Coloque o vídeo em `video/cenario.mp4` (no Colab a célula pede o upload).
"""),
code("""
#@title 5.1 Inferência e rastreamento (ByteTrack)
import cv2, time
from ultralytics import YOLO
os.makedirs("video", exist_ok=True); VIDEO = "video/cenario.mp4"
if not Path(VIDEO).exists() and IN_COLAB:
    from google.colab import files
    up = files.upload(); os.rename(list(up)[0], VIDEO)
if Path(VIDEO).exists():
    CONF, IOU_NMS = R["conf"], R["iou_nms"]
    seg = YOLO("runs/segment/seg_s/weights/best.pt")
    seg.predict(source=VIDEO, conf=CONF, iou=IOU_NMS, imgsz=640, save=True, project="video", name="seg", exist_ok=True, verbose=False)
    ids, n_det, tempos = set(), 0, []
    cap = cv2.VideoCapture(VIDEO); W, H, fps = int(cap.get(3)), int(cap.get(4)), cap.get(5)
    out = cv2.VideoWriter("video/cenario_track.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    while True:
        ok, q = cap.read()
        if not ok: break
        t0 = time.perf_counter()
        r = seg.track(q, persist=True, tracker="bytetrack.yaml", conf=CONF, iou=IOU_NMS, imgsz=640, verbose=False)[0]
        tempos.append((time.perf_counter() - t0) * 1000)
        if r.boxes.id is not None: ids.update(r.boxes.id.int().tolist()); n_det += len(r.boxes)
        out.write(r.plot())
    cap.release(); out.release()
    print(f"{W}x{H} @ {fps:.0f} FPS · buracos únicos (IDs) = {len(ids)} · detecções somadas por quadro = {n_det} · {np.mean(tempos):.1f} ms/quadro ({1000/np.mean(tempos):.1f} FPS)")
else:
    print("vídeo ausente: grave um vídeo (≥ 30 s) do cenário e salve em video/cenario.mp4")
"""),
md("""
---
**Protocolo.** O conjunto de teste é avaliado uma única vez pela célula 4.1, com `conf` e `IoU NMS` escolhidos em validação; nenhum parâmetro é ajustado a partir dele. Todos os números exibidos acima são os do relatório (`reports/relatorio.pdf`).
"""),
]

nb = {"cells": [{**c, "source": c["source"].replace("REPO_URL", REPO)} for c in cells],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}, "colab": {"provenance": [], "gpuType": "A100"}, "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}
SAIDA.parent.mkdir(exist_ok=True)
SAIDA.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print(f"notebook gerado: {SAIDA} ({len(cells)} células)")

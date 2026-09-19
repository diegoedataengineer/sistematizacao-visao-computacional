# SPEC-S01 — Fase 1: dataset, EDA e proposta de 1 página

**Entrega desta fase:** `data.yaml` + splits no Drive · 3 figuras de EDA · tabela de contagens · `docs/proposta.md`
**ADRs:** S01, S02, S07

## 1. Critérios de seleção do dataset

Abrir os candidatos do ADR-S02 no Roboflow Universe e aprovar o primeiro que atender a todos:

- [ ] Tipo **Instance Segmentation** (polígonos) — a página mostra máscaras nas amostras
- [ ] ≥ 600 imagens no total
- [ ] Classe `pothole` (ou equivalente) com ≥ 80% das instâncias; classes auxiliares descartáveis
- [ ] Licença CC BY 4.0 (ou mais permissiva)
- [ ] Amostras com variação (ângulo de veículo/pedestre, luz, piso molhado)
- [ ] Anotações visualmente plausíveis em 10 amostras aleatórias (polígono segue a borda do buraco)

Registrar: nome, autor, URL, versão, licença, data do download.

## 2. Download (Colab)

Exportar no site como **YOLOv11** (formato "YOLOv8/YOLOv11 segmentation") → "Show download code":

```python
from google.colab import drive; drive.mount('/content/drive')
BASE = '/content/drive/MyDrive/visao-computacional/sistematizacao'
!mkdir -p {BASE}/dataset {BASE}/runs {BASE}/figs {BASE}/video
!pip -q install ultralytics roboflow supervision

from roboflow import Roboflow
rf = Roboflow(api_key="SUA_CHAVE")                     # não versionar a chave
ds = rf.workspace("WORKSPACE").project("PROJETO").version(N).download("yolov11", location=f"{BASE}/dataset")
```

Alternativa sem chave: baixar o zip pelo site e descompactar em `{BASE}/dataset`.

## 3. `data.yaml` — classe única e caminhos absolutos

```yaml
path: /content/drive/MyDrive/visao-computacional/sistematizacao/dataset
train: train/images
val: valid/images
test: test/images
nc: 1
names: ['pothole']
```

Se o dataset tiver classes extras (ex.: `manhole`, `crack`), filtrar as labels:

```python
import glob, os
MANTER = {0}                       # índice da classe pothole no data.yaml original
for f in glob.glob(f"{BASE}/dataset/*/labels/*.txt"):
    linhas = [l for l in open(f) if int(l.split()[0]) in MANTER]
    with open(f, "w") as out: out.writelines(("0" + l[l.index(" "):]) for l in linhas)   # remapeia para 0
```

Se só houver `train/valid` (sem `test`), criar o `test` a partir do `train` com seed:

```python
import random, shutil
random.seed(0)
imgs = sorted(glob.glob(f"{BASE}/dataset/train/images/*"))
random.shuffle(imgs); n_test = int(0.15 * len(imgs))
for p in imgs[:n_test]:
    for sub, ext in (("images", None), ("labels", ".txt")):
        src = p if sub == "images" else p.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
        dst = src.replace("/train/", "/test/"); os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(src): shutil.move(src, dst)
```

O `test` **não é aberto** até a Fase 4.

## 4. EDA — célula pronta

```python
import glob, cv2, numpy as np, pandas as pd, matplotlib.pyplot as plt
rows = []
for split in ("train", "valid", "test"):
    for lab in glob.glob(f"{BASE}/dataset/{split}/labels/*.txt"):
        img = lab.replace("/labels/", "/images/").rsplit(".", 1)[0]
        cand = [c for c in glob.glob(img + ".*")]
        h, w = cv2.imread(cand[0]).shape[:2] if cand else (None, None)
        polys = [l.split() for l in open(lab) if l.strip()]
        areas = []
        for p in polys:                                   # área do polígono normalizado (shoelace)
            xy = np.array(p[1:], float).reshape(-1, 2) * [w, h]
            areas.append(0.5 * abs(np.dot(xy[:, 0], np.roll(xy[:, 1], 1)) - np.dot(xy[:, 1], np.roll(xy[:, 0], 1))))
        lum = cv2.imread(cand[0], cv2.IMREAD_GRAYSCALE).mean() if cand else None
        rows.append(dict(split=split, w=w, h=h, n_inst=len(polys), lum=lum,
                         area_med=np.median(areas) if areas else 0))
df = pd.DataFrame(rows)
print(df.groupby("split").agg(imagens=("n_inst", "size"), instancias=("n_inst", "sum"),
                              inst_por_img=("n_inst", "mean"), lum_media=("lum", "mean")))
print(df.groupby(["w", "h"]).size().sort_values(ascending=False).head())

fig, ax = plt.subplots(1, 3, figsize=(13, 3.5))
df.n_inst.hist(bins=range(0, df.n_inst.max() + 2), ax=ax[0]); ax[0].set_title("instâncias por imagem")
df.lum.hist(bins=30, ax=ax[1]); ax[1].set_title("luminância média (escuro → claro)")
np.log10(df.area_med[df.area_med > 0]).hist(bins=30, ax=ax[2]); ax[2].set_title("log10 área mediana do buraco (px²)")
plt.tight_layout(); plt.savefig(f"{BASE}/figs/eda.png", dpi=150); plt.show()
```

Também salvar uma **grade de 8 amostras anotadas** (`supervision` ou `results.plot()` do Ultralytics após o treino) como `figs/amostras.png`.

O que comentar no relatório (seção 2): contagens por split; instâncias por imagem (muitas com 0? isso vira `background` na matriz de confusão); resolução dominante e implicação do `imgsz=640`; distribuição de luz (há noite? há molhado?); tamanho dos buracos (fração de instâncias pequenas < 32² px — antecipa FN); desbalanceamento (classe única → o desbalanceamento é objeto × fundo).

## 5. Proposta de 1 página — template (`docs/proposta.md`)

```
# Proposta — Detecção e segmentação de buracos em vias urbanas

Grupo: <nomes>            Cenário: Cidades Inteligentes         Data: 18/09/2026

## Problema
Priorizar manutenção viária a partir de imagens capturadas por veículos em circulação.
O sistema deve localizar buracos (detecção) e delimitar sua extensão (segmentação),
permitindo estimar área relativa e contagem por trecho.

## Classes-alvo
`pothole` (classe única). Não tratamos rachaduras nem tampas de bueiro; aparecem na
análise de erros como confusores.

## Dados
<Nome do dataset>, <autor>, Roboflow Universe, v<N>, CC BY 4.0, <URL>.
<T> imagens / <I> instâncias — treino <..> / validação <..> / teste <..> (seed 0).
Anotação em polígono (usada para caixas e máscaras).

## Ferramenta de anotação
Dataset já anotado (Roboflow). Correções pontuais, se houver, no Roboflow Annotate.

## Métodos
Detecção: YOLO11s (Ultralytics), fine-tuning a partir de pesos COCO.
Segmentação: YOLO11s-seg, mesmo protocolo. Avaliação: mAP@0,5, mAP@0,5:0,95, P/R,
matriz de confusão, IoU/Dice; análise de erros no teste. Vídeo próprio ≥ 30 s;
rastreamento ByteTrack (bônus).

## Riscos
Generalização para vias brasileiras (dataset estrangeiro); buracos pequenos/distantes;
sombras e poças como falsos positivos.
```

## Critério de pronto

- [ ] `data.yaml` com `nc: 1` e três splits com contagens registradas
- [ ] `figs/eda.png` e `figs/amostras.png`
- [ ] `docs/proposta.md` preenchido
- [ ] Treino baseline `yolo11n` disparado (SPEC-S02 §1)

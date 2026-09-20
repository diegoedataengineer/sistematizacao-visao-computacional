# Detecção e segmentação de buracos em vias urbanas com YOLO11

Sistematização da disciplina **Visão Computacional e Reconhecimento de Padrões** (pós-graduação CEUB, Prof. Dr. Romes Heriberto Pires de Araújo) — cenário *Cidades Inteligentes*.

**Autor:** Diego Nunes de Morais — trabalho individual

## Problema

Priorizar a manutenção viária a partir de imagens capturadas por veículos em circulação: o sistema localiza buracos (detecção) e delimita sua extensão (segmentação de instâncias), permitindo estimar a área relativa de cada dano e contar buracos por trecho em vídeo.

## Resultados

| Modelo | Split | mAP@0,5 | mAP@0,5:0,95 | P | R | Mask mAP@0,5 | Mask mAP@0,5:0,95 |
|---|---|---|---|---|---|---|---|
| YOLO11s | val | 0,498 | 0,298 | 0,619 | 0,493 | — | — |
| YOLO11s | **teste** | 0,487 | 0,287 | 0,645 | 0,518 | — | — |
| YOLO11n (baseline, 30 ép.) | val | 0,437 | 0,234 | 0,640 | 0,431 | — | — |
| YOLO11n (baseline, 30 ép.) | **teste** | 0,421 | 0,201 | 0,612 | 0,450 | — | — |
| YOLO11s @ 800 px | val | 0,508 | 0,302 | 0,647 | 0,501 | — | — |
| YOLO11s @ 800 px | **teste** | 0,517 | 0,306 | 0,655 | 0,485 | — | — |
| YOLO11m | val | 0,491 | 0,288 | 0,639 | 0,455 | — | — |
| YOLO11m | **teste** | 0,502 | 0,284 | 0,674 | 0,487 | — | — |
| YOLO11s-seg | val | 0,508 | 0,297 | 0,649 | 0,514 | 0,306 | 0,132 |
| YOLO11s-seg | **teste** | 0,493 | 0,292 | 0,663 | 0,496 | 0,319 | 0,156 |

`conf = 0,25` · `IoU NMS = 0,7` · avaliação única no teste. Relatório completo em [reports/relatorio.md](reports/relatorio.md) e em [PDF](reports/relatorio.pdf).

## Vídeo

Vídeo de 90,7 s de uma rodovia da Paraíba ([YouTube](https://www.youtube.com/watch?v=I0HsZ2rsW8M), 1280×720 a 30 FPS), usado como teste de generalização de domínio. `scripts/video.py` roda o segmentador no ponto de operação do teste e o ByteTrack para contar buracos únicos:

| Detecções somadas por quadro | Buracos únicos (IDs) | Trilhas ≥ 5 quadros | Trocas de ID candidatas | Latência (GTX 1060) |
|---|---|---|---|---|
| 3 617 | 227 | 107 | 22 | 18–21 ms/quadro (≈ 50 FPS) |

Vídeo com rastreamento (`cenario_track.mp4`): https://drive.google.com/file/d/1MBH4MUW7n8PJhkpvSn6E6l2-FT04-bHW/view?usp=sharing · Vídeo com máscaras (`cenario_seg.mp4`): https://drive.google.com/file/d/1bt9ZMDrYmxOyFZ0Gp3_BlIqrtE5oky2r/view?usp=sharing · Vídeo-pitch: <!-- link -->

## Dataset

*Potholes and Roads Instance Segmentation* (workspace `pothole-vsmtu`), Roboflow Universe, v5, CC BY 4.0 — https://universe.roboflow.com/pothole-vsmtu/potholes-and-roads-instance-segmentation/dataset/5 (download em 18/09/2026). Duas classes originais (`pothole`, `road`); usamos apenas `pothole`. Anotações em polígono (55% originalmente só em caixa; máscaras refinadas com SAM — ver relatório, seção 3.3).

| split | imagens | instâncias `pothole` |
|---|---|---|
| treino | 1 076 | 5 097 |
| validação | 138 | 627 |
| teste | 141 | 540 |

Detalhes da análise exploratória em `figs/eda.png` e na seção 3 do relatório. Imagens **não** são versionadas; ver [Reprodução](#reprodução).

## Estrutura

```
.
├── data.yaml              configuração do dataset (classe única: pothole)
├── requirements.txt
├── scripts/
│   ├── baixar_dataset.py  download via Roboflow (chave por variável de ambiente)
│   ├── filtrar_classes.py remapeia/filtra labels para classe única
│   ├── eda.py             figuras e tabela da análise exploratória
│   ├── refinar_mascaras_sam.py  controle de qualidade e refino das máscaras com SAM
│   ├── treinar.py         treino de detecção ou segmentação (Ultralytics)
│   ├── pipeline.sh        sequência completa de treinos (idempotente, retoma checkpoints)
│   ├── avaliar.py         limiar em validação, teste único, matriz, IoU/Dice, erros, fatias
│   ├── video.py           inferência em vídeo + ByteTrack, contagem por ID e figuras
│   └── gerar_notebook.py  gera notebooks/sistematizacao.ipynb
├── notebooks/             notebook executável (local ou Colab), com saídas
├── reports/               relatório (Markdown, HTML e PDF) e figuras numeradas
├── tools/build_report.py  Markdown → HTML → PDF com apêndice de código gerado do repositório
├── figs/                  figuras geradas (EDA, curvas, painéis, erros)
├── dataset/               (ignorado) imagens e labels
├── runs/                  (ignorado) saídas de treino/avaliação
└── video/                 (ignorado) vídeos; links no README
```

## Reprodução

Passo a passo no Linux, na ordem em que os scripts se encadeiam. Cada etapa grava suas saídas em `figs/` ou `runs/`, e as seguintes leem de lá. Seed 0 em todos os treinos; versões em `requirements.txt`.

### 0. Ambiente

```bash
git clone https://github.com/diegoedataengineer/sistematizacao-visao-computacional.git
cd sistematizacao-visao-computacional
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126   # GPU com CUDA 12.x
pip install -r requirements.txt transformers accelerate                            # transformers: SAM
```

### 1. Dados e análise exploratória

```bash
export ROBOFLOW_API_KEY=...            # chave própria (gratuita) em app.roboflow.com
python scripts/baixar_dataset.py --workspace pothole-vsmtu --project potholes-and-roads-instance-segmentation --version 5
python scripts/filtrar_classes.py dataset --manter 0        # mantém só pothole; caixas viram polígonos de 4 pontos
python scripts/eda.py dataset --saida figs                  # figs/eda.png, figs/amostras.png, figs/eda_tabela.md
```

### 2. Refino das máscaras com SAM

```bash
python scripts/refinar_mascaras_sam.py qc dataset --amostra 300               # SAM × 300 polígonos humanos
python scripts/refinar_mascaras_sam.py refinar dataset --splits train valid test   # caixas → máscaras; originais em labels_original/
```

### 3. Treino

Cerca de 6 h no total numa GTX 1060 (6 GB); bem menos numa A100 do Colab.

```bash
python scripts/treinar.py --tarefa det --modelo yolo11n.pt     --epocas 50 --nome baseline_n
python scripts/treinar.py --tarefa det --modelo yolo11s.pt     --epocas 50 --nome det_s
python scripts/treinar.py --tarefa seg --modelo yolo11s-seg.pt --epocas 50 --nome seg_s
python scripts/treinar.py --tarefa det --modelo yolo11s.pt     --epocas 50 --nome det_s800 --imgsz 800   # extra
python scripts/treinar.py --tarefa det --modelo yolo11m.pt     --epocas 50 --nome det_m                  # extra
```

Alternativa: tudo de uma vez, idempotente, retomando do último checkpoint se interrompido.

```bash
setsid nohup bash scripts/pipeline.sh > runs/logs/pipeline.out 2>&1 < /dev/null &
tail -f runs/logs/pipeline.log
```

Atalho sem treinar: pesos da [Release v1.0](https://github.com/diegoedataengineer/sistematizacao-visao-computacional/releases/tag/v1.0) (só as três configurações principais; nesse caso retire `det_s800` e `det_m` do `--extras` no passo 4).

```bash
gh release download v1.0 -p "*.pt"
mkdir -p runs/detect/det_s/weights runs/segment/seg_s/weights runs/detect/baseline_n/weights
mv det_s_best.pt      runs/detect/det_s/weights/best.pt
mv seg_s_best.pt      runs/segment/seg_s/weights/best.pt
mv baseline_n_best.pt runs/detect/baseline_n/weights/best.pt
```

### 4. Avaliação

Escolhe o limiar em validação (máximo F1), avalia o teste uma única vez e grava tabela, matriz de confusão, IoU/Dice, painéis de erros e caixas × máscaras em `figs/`.

```bash
python scripts/avaliar.py --det runs/detect/det_s --seg runs/segment/seg_s \
    --extras runs/detect/baseline_n runs/detect/det_s800 runs/detect/det_m --regra f1 --iou-nms 0.7
```

Sem GPU: acrescente `--device cpu`.

### 5. Vídeo

O arquivo `video/cenario.mp4` não é versionado: é o [vídeo do YouTube](https://www.youtube.com/watch?v=I0HsZ2rsW8M) citado acima, ou qualquer vídeo próprio.

```bash
python scripts/video.py --video video/cenario.mp4 --conf 0.25 --iou-nms 0.7
# saídas: video/seg/ (máscaras), video/cenario_track.mp4 (IDs), figs/video_contagem.png, figs/video_quadros.png, figs/video_resumo.json
```

### 6. Relatório e notebook

```bash
pip install -r requirements-report.txt
python tools/build_report.py                     # reports/relatorio.html e .pdf (10 páginas); precisa do Google Chrome
python tools/build_report.py --com-apendice      # versão com o apêndice de código

python scripts/gerar_notebook.py
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/sistematizacao.ipynb
```

No Colab: abra `notebooks/sistematizacao.ipynb`, ative a GPU e execute em ordem; a primeira célula clona o repositório e instala as dependências. Com `TREINAR = False` (padrão) o notebook usa os pesos da Release e reproduz os números do relatório.

## Referências

- Dataset: *Potholes and Roads Instance Segmentation*, workspace pothole-vsmtu, Roboflow Universe, v5, CC BY 4.0 — https://universe.roboflow.com/pothole-vsmtu/potholes-and-roads-instance-segmentation/dataset/5
- Jocher, G. et al. *Ultralytics YOLO11*. https://github.com/ultralytics/ultralytics
- Redmon, J. et al. *You Only Look Once* (2016). Zhang, Y. et al. *ByteTrack* (2022). Kalman, R. (1960).
- Apostilas da disciplina (Prof. Romes Heriberto): Visão Computacional Vol. I e II; Vídeo com Visão Computacional; Reconhecimento de Padrões.

## Licença

Código sob MIT. O dataset segue a licença do autor (CC BY 4.0).

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

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

export ROBOFLOW_API_KEY=...            # chave própria (gratuita) em app.roboflow.com
python scripts/baixar_dataset.py --workspace pothole-vsmtu --project potholes-and-roads-instance-segmentation --version 5
python scripts/filtrar_classes.py dataset --manter 0     # mantém só pothole (descarta road)
python scripts/eda.py dataset --saida figs

python scripts/refinar_mascaras_sam.py qc dataset          # SAM × polígonos humanos (300 instâncias)
python scripts/refinar_mascaras_sam.py refinar dataset     # caixas → máscaras SAM (originais em labels_original/)

python scripts/treinar.py --tarefa det --modelo yolo11s.pt --epocas 50
python scripts/treinar.py --tarefa seg --modelo yolo11s-seg.pt --epocas 50
python scripts/avaliar.py --det runs/detect/det_s --seg runs/segment/seg_s --extras runs/detect/det_s800 runs/detect/det_m --regra f1
python scripts/video.py --video video/cenario.mp4 --conf 0.25 --iou-nms 0.7
```

No Colab: abra `notebooks/sistematizacao.ipynb`, ative GPU T4 e execute em ordem. Pesos treinados: [Release v1.0](https://github.com/diegoedataengineer/sistematizacao-visao-computacional/releases/tag/v1.0) — `gh release download v1.0 -p "*.pt"` (det_s_best.pt, seg_s_best.pt, baseline_n_best.pt).

Versões: ver `requirements.txt`; seed 0 em todos os treinos.

## Referências

- Dataset: *Potholes and Roads Instance Segmentation*, workspace pothole-vsmtu, Roboflow Universe, v5, CC BY 4.0 — https://universe.roboflow.com/pothole-vsmtu/potholes-and-roads-instance-segmentation/dataset/5
- Jocher, G. et al. *Ultralytics YOLO11*. https://github.com/ultralytics/ultralytics
- Redmon, J. et al. *You Only Look Once* (2016). Zhang, Y. et al. *ByteTrack* (2022). Kalman, R. (1960).
- Apostilas da disciplina (Prof. Romes Heriberto): Visão Computacional Vol. I e II; Vídeo com Visão Computacional; Reconhecimento de Padrões.

## Licença

Código sob MIT. O dataset segue a licença do autor (CC BY 4.0).

# Sistematização — Plano de execução, ADRs e Specs do projeto

Projeto final da disciplina VCRP: **sistema de detecção e segmentação de buracos em vias urbanas** (cenário Cidades Inteligentes), com demonstração em vídeo.

| | |
|---|---|
| Cenário | Cidades Inteligentes → **buracos em vias** ([ADR-S01](adr/ADR-S01-cenario-buracos-em-vias.md)) |
| Detector | **YOLO11 (Ultralytics)**, fine-tuning ([ADR-S03](adr/ADR-S03-detector-yolo11-ultralytics.md)) |
| Segmentação | **YOLO11-seg** (instâncias) sobre as mesmas anotações em polígono ([ADR-S04](adr/ADR-S04-segmentacao-yolo11-seg.md)) |
| Dataset | Público, Roboflow Universe, polígonos, CC BY 4.0 ([ADR-S02](adr/ADR-S02-dataset-publico-roboflow-poligonos.md)) |
| Bônus | Rastreamento **ByteTrack** no vídeo ([ADR-S06](adr/ADR-S06-video-e-bonus-bytetrack.md)) |
| Prazo | **Domingo, 20/09/2026, 23:55** — Moodle, tentativa única, sem atraso |

## Comece por aqui

1. [PLANO-DE-EXECUCAO.md](PLANO-DE-EXECUCAO.md) — cronograma hora a hora até domingo, papéis, riscos, definição de pronto.
2. ADRs — as decisões do projeto e por que (8 registros curtos).
3. Specs — o que produzir em cada etapa, com código pronto para o Colab e o esqueleto do relatório.

## Estrutura

```
docs/sistematizacao/
├── README.md
├── PLANO-DE-EXECUCAO.md
├── adr/
│   ├── ADR-S01  cenário: buracos em vias
│   ├── ADR-S02  dataset público com polígonos; splits fixos
│   ├── ADR-S03  detector: YOLO11 (Ultralytics)
│   ├── ADR-S04  segmentação: YOLO11-seg (instâncias)
│   ├── ADR-S05  protocolo de avaliação e métricas
│   ├── ADR-S06  vídeo e bônus (ByteTrack)
│   ├── ADR-S07  ambiente: Colab + Drive + GitHub
│   └── ADR-S08  autoria, voz do grupo e declaração de uso de IA
└── specs/
    ├── SPEC-S01  dataset, EDA e proposta (Fase 1)
    ├── SPEC-S02  treino de detecção (Fase 2)
    ├── SPEC-S03  treino de segmentação (Fase 3)
    ├── SPEC-S04  avaliação e análise de erros (Fase 4)
    ├── SPEC-S05  inferência em vídeo e rastreamento (Fase 4)
    └── SPEC-S06  entregáveis: relatório, repositório, notebook, pitch (Fase 5)
```

## Mapeamento barema → onde é produzido

| Critério (peso) | Spec | Evidência no relatório |
|---|---|---|
| 1. Problema e dataset (10%) | S01 | seção 2 + figuras da EDA + tabela de splits |
| 2. Detecção (25%) | S02 | seção 3.1 + curvas de treino + tabela de hiperparâmetros |
| 3. Segmentação (20%) | S03 | seção 3.2 + painel caixas × máscaras |
| 4. Avaliação e análise crítica (20%) | S04 | seção 4 + matriz de confusão + painel FP/FN comentado |
| 5. Vídeo (10%) | S05 | link do vídeo anotado (≥ 30 s) |
| 6. Relatório e repositório (10%) | S06 | README de reprodução, fontes citadas |
| 7. Vídeo-pitch (5%) | S06 | roteiro de 5–8 min com todos os integrantes |

A base teórica de cada decisão está na documentação da disciplina (pasta `docs/` do projeto de estudo); os ADRs daqui apenas aplicam aquela doutrina a este projeto.

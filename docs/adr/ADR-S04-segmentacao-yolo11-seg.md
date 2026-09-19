# ADR-S04 — Segmentação de instâncias com YOLO11-seg sobre as mesmas anotações em polígono

**Status:** Aceito
**Base:** enunciado ("instâncias — YOLO-seg ou Mask R-CNN — ou semântica — DeepLab/U-Net — no mesmo domínio"); ADR-009

## Contexto

O barema pede segmentação no mesmo domínio (20%) e a **comparação caixas × máscaras** ("o que a segmentação revela que a detecção não mostra?"). O dataset já tem polígonos; o Ultralytics treina um modelo `-seg` com o mesmo `data.yaml`. Buracos são **instâncias** (contam-se, medem-se um a um), não regiões amorfas — segmentação de instâncias é a semântica correta do problema.

## Decisão

- Modelo `yolo11s-seg.pt`, mesmos `imgsz`, `epochs`, `batch`, `seed` e augmentation do detector (ADR-S03) — a comparação fica justa.
- Métricas: **Mask mAP50 / mAP50-95, P, R** (Ultralytics reporta Box e Mask separadamente) + **IoU e Dice médios por imagem** calculados no teste (SPEC-S04).
- **Argumento de valor da máscara**: área do buraco em pixels (e fração da imagem) por instância; a caixa superestima a área em buracos alongados/diagonais — quantificar `area_mask / area_box` no teste (razão média e distribuição).
- Painel obrigatório: 8 imagens de `val` com caixa (det) e máscara (seg) lado a lado, mesmas imagens, mesmo `conf`.

## Alternativas rejeitadas

- **Mask R-CNN (torchvision):** dois estágios, precisa de conversão COCO e de um loop de treino próprio; sem ganho justificável no prazo.
- **U-Net / DeepLab (semântica):** exigiria converter polígonos em máscaras por classe e implementar treino; perderia a noção de instância (contagem de buracos).
- **SAM a partir das caixas do YOLO (sem treino):** válido como comparação extra (ADR-013), mas o barema pede modelo treinado/adaptado no domínio e "qualidade das máscaras" — o SAM promptado por caixa larga tende a vazar para o asfalto.

## Decisão: refinar as máscaras retangulares com SAM (rota B, escolhida em 18/09 à noite)

A inspeção (ADR-S02) mostrou que 55% das instâncias de referência são retângulos. Duas rotas:

| Rota | O que fazer | Custo | Efeito |
|---|---|---|---|
| **A — aceitar** | Treinar YOLO11-seg sobre as labels como estão; declarar que 55% das máscaras de referência são caixas | zero | Mask mAP mede em parte "quão bem o modelo desenha retângulos"; a comparação caixas × máscaras só é honesta nas 45% em polígono (filtrar o painel e a razão de área para essas) |
| **B — refinar com SAM** | Para cada instância anotada só por caixa, gerar a máscara com SAM (`facebook/sam-vit-base`, caixa como prompt), substituir o polígono na label; revisar uma amostra de 30 casos e medir IoU entre a máscara do SAM e o polígono humano nas instâncias que **têm** polígono (controle de qualidade) | ~15–25 min de GPU na 1060 (1 270 imagens) + 30 min de revisão + script | Máscaras de referência com contorno real; história de projeto forte (pipeline agêntico da Apostila Vol. II, Cap. 4); risco: SAM vazar para sombra/água em caixas largas — medido pelo controle |

Recomendação: **B, se o controle de qualidade der IoU mediano ≥ 0,7** contra os polígonos humanos; senão A. Em qualquer caso, o relatório informa a proporção e o método.

### Resultado do controle de qualidade (18/09, 23:15) — rota B confirmada

Amostra de 300 instâncias com polígono humano (seed 0), mesma amostra para os dois lados:

| Referência comparada ao polígono humano | IoU mediano | IoU médio | P10 | P90 | ≥ 0,7 |
|---|---|---|---|---|---|
| **Retângulo** (a anotação atual das instâncias só-caixa) | 0,684 | 0,669 | 0,522 | 0,790 | 43% |
| **Máscara do SAM** a partir da caixa | **0,703** | **0,685** | 0,497 | **0,851** | **51%** |

Leitura: os polígonos humanos desta base são pouco precisos (um simples retângulo já concorda 0,68 com eles), por isso o SAM não "dispara" na mediana; mas supera o retângulo em todas as estatísticas, com ganho maior na cauda alta (P90 0,85 vs 0,79). Decisão: **refinar** (rota B), preservando `labels_original/` para que a avaliação reporte IoU/Dice contra as duas referências (`avaliar.py`). No relatório: declarar que 55% das máscaras de referência foram geradas por SAM a partir de caixas humanas, com este controle de qualidade como evidência, e que a métrica de máscara deve ser lida à luz da imprecisão dos polígonos humanos originais.

## Consequências

- Dois modelos, um dataset, um notebook: baixo risco de inconsistência.
- Máscaras herdam a grosseria dos polígonos anotados — registrar como limitação e mostrar um exemplo.
- No vídeo, a versão com máscaras (`yolo11s-seg` + `track`) é visualmente mais convincente para o pitch.

## Relacionados

ADR-S02, ADR-S03, SPEC-S03, SPEC-S04.

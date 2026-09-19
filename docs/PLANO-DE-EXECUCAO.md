# Plano de execução — Sistematização VCRP

**Hoje:** sexta, 18/09/2026 (noite) · **Prazo:** domingo, 20/09/2026, 23:55 · **Janela útil:** ~50 h

## Princípio

Com dois dias, o projeto é uma **linha de montagem com uma única passagem por cada fase**. A regra é: primeiro fechar um pipeline mínimo ponta a ponta (dataset → YOLO → YOLO-seg → métricas → vídeo) até **sábado ao meio-dia**, e só depois refinar. Nenhuma fase começa "do zero" no domingo.

## Escolhas fixas (não reabrir)

| Decisão | Escolha | ADR |
|---|---|---|
| Sub-cenário | Buracos em vias (pothole) | S01 |
| Dataset | Roboflow Universe, instance segmentation, CC BY 4.0, ≥ 600 imagens, classe única `pothole` | S02 |
| Detector | `yolo11s.pt` (fallback `yolo11n.pt`), 640 px, 50 épocas | S03 |
| Segmentação | `yolo11s-seg.pt` sobre o mesmo dataset | S04 |
| Vídeo | Gravação própria com celular (≥ 60 s) **ou** dashcam pública citada | S06 |
| Bônus | `model.track(tracker="bytetrack.yaml")` | S06 |
| Ambiente | Colab T4 + Drive (`visao-computacional/sistematizacao/`) + GitHub pessoal | S07 |

## Cronograma

### Sexta 18/09 — noite (2–3 h) · Fase 1

| Hora | Tarefa | Saída | Quem |
|---|---|---|---|
| 21:45 | Confirmar grupo e integrantes; criar repo `vcrp-buracos-vias` (github-pessoal) | repo vazio com README inicial | todos |
| 22:00 | Escolher o dataset (critérios em SPEC-S01), baixar em formato **YOLOv11 segmentation** via Roboflow | `data.yaml` + `train/valid/test` | 1 |
| 22:30 | Rodar a EDA (célula pronta em SPEC-S01): contagens, resoluções, luz, instâncias por imagem | 3 figuras + tabela | 1 |
| 23:00 | Escrever a **proposta de 1 página** (template em SPEC-S01) | `docs/proposta.md` | 2 |
| 23:15 | Disparar o **primeiro treino de detecção** (`yolo11n`, 30 épocas) para dormir com o pipeline validado | `runs/detect/baseline_n/` no Drive | 1 |

### Sábado 19/09 · Fases 2 e 3

| Hora | Tarefa | Saída |
|---|---|---|
| 08:00 | Conferir o baseline noturno; se mAP50 > 0,5 o pipeline está sadio | `results.csv`, curvas |
| 08:30 | Treino principal de detecção: `yolo11s`, 50 épocas, `patience=10` (~40–60 min na T4) | `best.pt` det |
| 09:00 | Em paralelo: gravar/obter o **vídeo** (≥ 60 s, ruas com buracos, luz do dia) | `video/cenario.mp4` |
| 10:00 | Treino de segmentação: `yolo11s-seg`, 50 épocas (~60–80 min) | `best.pt` seg |
| 11:30 | Validação em `val` dos dois modelos; registrar tabela de hiperparâmetros | tabela + `val` metrics |
| 14:00 | Painel **caixas × máscaras** (8 imagens de val, mesmo índice) — SPEC-S03 | figura |
| 15:00 | Varredura de limiar (`conf` 0,05–0,6) e escolha do ponto de operação — SPEC-S04 | curva P×R + `conf` escolhido |
| 16:00 | Inferência no vídeo com `predict` e com `track` (ByteTrack) | 2 vídeos anotados |
| 17:00 | Se sobrar tempo: segundo treino com `imgsz=800` ou `yolo11m` para comparar | linha extra na tabela |
| 19:00 | **Congelar modelos.** Commit de tudo (pesos no Drive, link no README) | tag `v0.1` |
| 20:00 | Escrever seções 1–3 do relatório (problema, dataset, metodologia) | rascunho |

### Sábado 19/09 (tarde/noite) · Fase 4

| Hora | Tarefa | Saída |
|---|---|---|
| 09:00 | **Avaliação única no `test`** (nunca tocado antes) — det e seg: mAP50, mAP50-95, P, R, IoU/Dice das máscaras, matriz de confusão | tabela final |
| 10:00 | **Análise de erros**: 4 FP + 4 FN do teste, comentados um a um (sombra, poça, remendo, tampa de bueiro, buraco pequeno, borda da imagem) | painel + texto |
| 11:30 | Métricas por fatia (dia/noite, seco/molhado se o dataset tiver; senão por tamanho de caixa) | tabela |
| 14:00 | Seções 4–6 do relatório (resultados, análise, limitações e próximos passos) | rascunho completo |
| 16:00 | Notebook Colab final: executar **do início ao fim** com saídas visíveis; `Restart & run all` | `.ipynb` limpo |
| 18:00 | README do repositório com reprodução passo a passo; `requirements.txt`; link do dataset e do vídeo | repo final |
| 20:00 | Roteiro do pitch (SPEC-S06) distribuído entre os integrantes; ensaio 1× | roteiro |

### Domingo 20/09 · Fase 5 — entrega até 23:55

| Hora | Tarefa |
|---|---|
| 09:00 | Gravar o vídeo-pitch (5–8 min, todos aparecem; sistema rodando + vídeo anotado) |
| 11:00 | Editar e subir (YouTube não listado ou Drive); testar o link em aba anônima |
| 12:00 | Revisão final do relatório: leitura em voz alta, números batem com o notebook, fontes citadas, declaração de IA presente (ADR-S08). Exportar PDF |
| 14:00 | Checklist de entrega (abaixo). Submeter no Moodle — **todos os integrantes** submetem |
| 16:00 | Margem para imprevistos. **Não deixar para depois das 20:00.** |

## Papéis (adapte ao tamanho do grupo)

| Papel | Responsável por |
|---|---|
| **Dados e treino** | download, EDA, treinos, pesos no Drive |
| **Avaliação e vídeo** | limiar, métricas no teste, análise de erros, vídeo anotado, tracking |
| **Relatório e repo** | texto, figuras, README, notebook final, PDF |
| **Pitch** | roteiro, gravação, edição, upload |

Em grupo de 1–2 pessoas: siga o cronograma na ordem; os papéis viram fases sequenciais.

## Riscos e planos B

| Risco | Sinal | Plano B |
|---|---|---|
| Colab derruba a sessão no meio do treino | perda de `runs/` | pesos e `last.pt` sempre no Drive (`project=` apontando para o Drive); retomar com `resume=True` |
| Dataset tem classes confusas (manhole, crack) | mAP baixo por classe | filtrar para classe única `pothole` no `data.yaml` (SPEC-S01) |
| mAP50 do teste < 0,4 | modelo fraco | aumentar `imgsz` para 800, `yolo11m`, 80 épocas; relatar honestamente — a análise crítica vale 20% |
| Não conseguir gravar vídeo próprio | sem cenário real | dashcam pública (Kaggle/YouTube CC) com fonte citada; recortar 30–60 s |
| Sem GPU no Colab (cota) | `cuda` indisponível | conta alternativa, ou treinar `yolo11n` em CPU local (GTX 1060 6 GB: instalar torch CUDA e rodar local — funciona para `n`/`s` em 640) |
| Falta de tempo no domingo | relatório incompleto | relatório de 6 páginas com tudo obrigatório vale mais que 10 com lacunas |

## Definição de pronto (checklist de entrega)

- [ ] Dataset citado (nome, autor, URL, licença), ≥ 300 imagens, splits treino/val/teste com seed fixa e contagens no relatório
- [ ] Detecção: `yolo11s` fine-tuned; hiperparâmetros documentados; curvas de treino no notebook
- [ ] Segmentação: `yolo11s-seg`; painel caixas × máscaras com comentário
- [ ] Teste: mAP@0,5, mAP@0,5:0,95, P, R, matriz de confusão; IoU/Dice das máscaras
- [ ] Análise de erros com ≥ 4 FP e ≥ 4 FN comentados
- [ ] Vídeo ≥ 30 s com inferência (link) + versão com ByteTrack (bônus)
- [ ] Relatório 6–10 páginas em PDF (estrutura em SPEC-S06), com declaração de uso de IA e citação de fontes
- [ ] Repositório GitHub com README de reprodução, `requirements.txt`, notebook, link do dataset e dos pesos
- [ ] Notebook Colab executado do início ao fim, saídas visíveis
- [ ] Vídeo-pitch 5–8 min, todos os integrantes, link testado
- [ ] Todos os integrantes submeteram no Moodle antes de 23:55

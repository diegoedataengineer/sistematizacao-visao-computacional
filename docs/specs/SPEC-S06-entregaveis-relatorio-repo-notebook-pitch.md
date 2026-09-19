# SPEC-S06 — Fase 5: entregáveis — relatório técnico, repositório, notebook Colab e vídeo-pitch

**ADRs:** S07, S08

## 1. Relatório técnico (PDF ou MD, 6–10 páginas)

Estrutura fixa, uma seção por critério do barema. Escrever em primeira pessoa do plural, com número ou figura em cada afirmação (ADR-S08).

```
Título: Detecção e segmentação de buracos em vias urbanas com YOLO11
Integrantes · Disciplina VCRP · Prof. Romes Heriberto · CEUB · set/2026

1. Problema e cenário (½ página)
   Manutenção viária; entrada (vídeo de veículo) → saída (buracos localizados, área relativa,
   contagem por trecho). Por que detecção E segmentação. Escopo: classe única.

2. Dataset e análise exploratória (1–1½ página)                         [critério 1]
   Fonte, licença, versão, data. Tabela de splits (imagens/instâncias). Fig. EDA
   (instâncias por imagem, luminância, tamanho). Grade de amostras. Observações que
   antecipam erros (buracos pequenos, imagens sem buraco, luz).

3. Metodologia (1½–2 páginas)                                            [critérios 2, 3]
   3.1 Detecção — YOLO11s, fine-tuning; tabela de hiperparâmetros; curvas de treino;
       baseline n × s (× extra). Decisão do modelo final em val.
   3.2 Segmentação — YOLO11s-seg; tabela Box × Mask em val; painel caixas × máscaras;
       razão área máscara/caixa e o que ela significa para o problema.
   3.3 Protocolo de avaliação — papéis dos splits; escolha de conf pela curva P×R
       (figura) e critério de custo; IoU do NMS.
   Declaração de uso de IA (uma frase — ADR-S08).

4. Resultados (1½–2 páginas)                                             [critérios 4, 5]
   4.1 Tabela final (val e test; det e seg; mAP@0,5, mAP@0,5:0,95, P, R, IoU, Dice).
   4.2 Matriz de confusão do teste + leitura.
   4.3 Análise de erros: painéis FP e FN, uma linha por caso com hipótese; fatias por
       tamanho.
   4.4 Vídeo: link, duração, contagem por quadro × IDs únicos (ByteTrack), latência.

5. Limitações (½ página)
   Rótulos herdados (polígonos grosseiros, anotações faltantes); dataset estrangeiro;
   sem validação cruzada (uma execução por configuração); sombras/poças; buracos
   pequenos; trocas de ID em curvas.

6. Próximos passos (¼ página)
   Anotação de vias locais; fatias por condição; exportação ONNX/TensorRT e medição
   de latência; detectar a cada N quadros; demo Gradio.

Referências
   Dataset; Ultralytics (versão); Redmon et al. 2016 (YOLO); Zhang et al. 2022
   (ByteTrack); Kalman 1960; apostilas da disciplina (Vol. I Caps. 6, 8, 9; Vídeo
   Caps. 4, 5, 7; RP Cap. 9).
```

Exportar: `pandoc relatorio.md -o relatorio.pdf` ou Google Docs → PDF. Conferir que **todo número do relatório existe no notebook**.

## 2. Repositório GitHub (`vcrp-buracos-vias`)

```
vcrp-buracos-vias/
├── README.md               ← problema, integrantes, resultados-chave (tabela), como reproduzir, links
├── requirements.txt        ← ultralytics==<versão>, roboflow, supervision, opencv-python, pandas, matplotlib
├── data.yaml               ← caminhos relativos (ajustar path) e nc: 1
├── notebooks/
│   └── sistematizacao.ipynb  ← executado, saídas visíveis
├── docs/
│   ├── proposta.md
│   └── relatorio.md / relatorio.pdf
├── figs/                   ← eda, amostras, curvas, caixas_vs_mascaras, razao_area, pr_limiar, confusion_matrix, erros_fp, erros_fn
├── video/
│   └── README.md           ← links dos vídeos anotados (não versionar mp4 > 50 MB)
├── scripts/                ← opcional: eda.py, avaliar.py, video.py (extraídos do notebook)
├── .gitignore              ← runs/, dataset/, *.pt, *.mp4, *.avi, chaves
└── LICENSE                 ← MIT para o código; dataset segue CC BY 4.0 (citar)
```

README — seção "Reprodução" (deve funcionar numa conta Colab nova):

```
1. Abra notebooks/sistematizacao.ipynb no Colab; ative GPU T4.
2. Célula 1: monta o Drive e instala requirements.
3. Célula 2: baixa o dataset (Roboflow, chave própria) ou descompacta o zip de <link>.
4. Células 3–5: treino det/seg (≈2 h) — ou pule e baixe os pesos de <link Release/Drive>.
5. Células 6–8: avaliação no teste, análise de erros, vídeo.
Versões: ultralytics X.Y.Z, torch A.B, CUDA; seed 0.
```

Git (nesta máquina): remoto pelo alias pessoal e autoria por repositório —

```bash
git init && git remote add origin git@github-pessoal:diegoedataengineer/vcrp-buracos-vias.git
git config user.name "Diego Nunes de Morais"; git config user.email "diego.dataengineer1987@gmail.com"
git log -1 --format='%an <%ae>'   # conferir após o primeiro commit
```

## 3. Notebook Colab final

Ordem de células = ordem das specs S01→S05. Regras:
- Uma célula de **configuração** no topo (`BASE`, `CONF`, `IOU_NMS`, seeds, versões impressas).
- Treinos com `if TREINAR:` para permitir pular e carregar pesos.
- `Runtime → Restart and run all` no sábado à tarde; salvar com saídas; sem células de rascunho; sem chave de API em texto (usar `getpass`).
- Título e cabeçalho em Markdown com integrantes e link do repo.

## 4. Vídeo-pitch (5–8 min, todos os integrantes)

| Tempo | Bloco | Quem | Tela |
|---|---|---|---|
| 0:00–0:45 | Problema e proposta de valor (manutenção viária; detectar + medir) | A | slide 1 / proposta |
| 0:45–1:45 | Dataset e EDA — o que vimos e o que antecipava erros | B | figs EDA |
| 1:45–3:15 | Metodologia — YOLO11s e YOLO11s-seg; hiperparâmetros; por que um estágio; escolha do `conf` na curva P×R | C | tabela + curva |
| 3:15–4:45 | Resultados — tabela final, matriz de confusão, 2 FP e 2 FN explicados, caixas × máscaras e razão de área | D | figs |
| 4:45–6:15 | **Demo ao vivo/gravada**: vídeo anotado com máscaras e IDs; contagem de buracos únicos; latência | A/B | vídeo |
| 6:15–7:00 | Limitações e próximos passos; fechamento | todos | slide final |

Regras: cada pessoa fala pelo menos 1 min; câmera ligada; sem ler texto; números de cabeça (os 4 principais: mAP50 teste, mAP50-95, IoU médio, IDs únicos no vídeo). Gravar em Meet/Zoom ou OBS; exportar 1080p; YouTube **não listado** ou Drive com link de leitura; testar em aba anônima.

## 5. Submissão no Moodle

- Item 1: `relatorio.pdf`. Item 2: link do repositório. Item 3: `sistematizacao.ipynb` (arquivo + link Colab). Item 4: link do pitch.
- **Todos os integrantes** submetem; README nomeia todos.
- Enviar até as 20:00 de domingo; prazo formal 23:55.

## Critério de pronto

- [ ] PDF com as 6 seções, figuras numeradas, referências e declaração de IA
- [ ] Repositório público com README de reprodução e `requirements.txt`
- [ ] Notebook executado ponta a ponta com saídas
- [ ] Pitch 5–8 min, todos aparecem, link testado
- [ ] Submissão feita por todos

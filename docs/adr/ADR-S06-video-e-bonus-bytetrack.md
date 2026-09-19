# ADR-S06 — Vídeo real do cenário com inferência ≥ 30 s; bônus com ByteTrack integrado ao Ultralytics

**Status:** Aceito
**Base:** enunciado (vídeo real ≥ 30 s; bônus rastreamento ByteTrack/DeepSORT ou demo Gradio/HF Spaces); ADR-015, ADR-016

## Contexto

O vídeo vale 10% e é a peça central do pitch. O bônus (+0,5) tem duas opções; o rastreamento é **uma linha** no Ultralytics (`model.track(tracker="bytetrack.yaml")`), enquanto uma demo Gradio publicada exige conta, deploy e testes.

## Decisão

1. **Fonte do vídeo (ordem de preferência):**
   a. **Gravação própria** com celular fixado no para-brisa (ou pedestre), 60–120 s, luz do dia, ruas com buracos conhecidos, 1080p 30 FPS, horizontal. Vantagens: licença própria, domínio brasileiro (teste de generalização), material para o pitch.
   b. Dashcam pública (Kaggle/YouTube com licença CC) com fonte citada, recortada para 30–60 s.
2. **Privacidade:** borrar placas e rostos se aparecerem (`cv2.GaussianBlur` sobre detecções de um YOLO COCO para `person`, ou edição simples). Não publicar o vídeo bruto no repositório; só o anotado.
3. **Inferência:** `predict` com `yolo11s-seg` (máscaras + caixas), `conf` do ADR-S05, `imgsz=640`, salvar MP4 anotado. Reduzir para 720p se a decodificação atrasar.
4. **Bônus: ByteTrack** via `model.track(source, tracker="bytetrack.yaml", persist=True)`; reportar o número de **IDs únicos** (buracos distintos) vs. total de detecções por quadro — o argumento de "não contar o mesmo buraco duas vezes" (SPEC-09). Descrever trocas de ID observadas como limitação.
5. **Latência:** medir ms/quadro na T4 e, se houver, na CPU local; incluir tabela curta (ADR-016) em "próximos passos".
6. Demo Gradio: **não** no prazo; citar como trabalho futuro.

## Alternativas rejeitadas

- **DeepSORT:** exige modelo de ReID e instalação extra; buracos não se cruzam (são estáticos no mundo; o movimento é da câmera), então aparência acrescenta pouco. ByteTrack basta.
- **Demo HF Spaces:** custo de tempo alto para o mesmo +0,5.

## Consequências

- O vídeo anotado com máscaras + IDs é a tomada principal do pitch.
- Contagem de IDs únicos é uma métrica de produto (buracos por km) que o relatório pode citar como aplicação.

## Relacionados

ADR-S04, ADR-S05, SPEC-S05, SPEC-S06.

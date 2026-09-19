# Vídeos

Os arquivos .mp4 não são versionados (ficam no Drive).

- `cenario.mp4` — bruto: trecho de 90,7 s de *Estrada esburacada na Paraíba* (YouTube, ID I0HsZ2rsW8M), 1280×720, 29,97 FPS
- `cenario_seg.mp4` — inferência do segmentador (caixas + máscaras), `conf = 0,25`, `IoU NMS = 0,7`: <!-- link -->
- `cenario_track.mp4` — rastreamento ByteTrack com IDs: <!-- link -->

Gerados por `python scripts/video.py --video video/cenario.mp4`; métricas em `figs/video_resumo.json`.

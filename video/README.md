# Vídeos

Os arquivos .mp4 não são versionados (ficam no Drive).

- `cenario.mp4` — bruto: *Estrada esburacada na Paraíba* (YouTube, https://www.youtube.com/watch?v=I0HsZ2rsW8M), 90,7 s, 1280×720, 29,97 FPS
- `cenario_seg.mp4` — inferência do segmentador (caixas + máscaras), `conf = 0,25`, `IoU NMS = 0,7`: https://drive.google.com/file/d/1bt9ZMDrYmxOyFZ0Gp3_BlIqrtE5oky2r/view?usp=sharing
- `cenario_track.mp4` — rastreamento ByteTrack com IDs: https://drive.google.com/file/d/1MBH4MUW7n8PJhkpvSn6E6l2-FT04-bHW/view?usp=sharing

Gerados por `python scripts/video.py --video video/cenario.mp4`; métricas em `figs/video_resumo.json`.

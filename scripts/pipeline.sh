#!/usr/bin/env bash
# Pipeline de treino (idempotente): baseline → QC SAM → refinar labels → det_s → seg_s → extras (det_s800, det_m).
# Cada etapa deixa um marcador; relançar o script continua de onde parou.
# Treinos retomam do last.pt se interrompidos; em "out of memory" o batch cai para 8 (depois 4).
#
#   setsid nohup bash scripts/pipeline.sh > runs/logs/pipeline.out 2>&1 < /dev/null &
#   cat runs/logs/pipeline_status.txt     # etapa atual
#   tail -f runs/logs/pipeline.log
set -uo pipefail
cd "$(dirname "$0")/.."
P=$HOME/.venvs/vcrp/bin/python
LOGS=runs/logs; mkdir -p "$LOGS"
LOG=$LOGS/pipeline.log
STATUS=$LOGS/pipeline_status.txt

log()    { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
status() { echo "$(date '+%F %T') $*" > "$STATUS"; log "STATUS: $*"; }
idade()  { [ -f "$1" ] && echo $(( $(date +%s) - $(stat -c %Y "$1") )) || echo 999999; }
concluido() { grep -q "\[$1\] val:" "$LOGS/$1.log" 2>/dev/null; }

# treina <nome> <tarefa> <modelo> <epocas> [imgsz]; retoma se houver last.pt; até 3 tentativas
treinar() {
  local nome=$1 tarefa=$2 modelo=$3 epocas=$4 imgsz=${5:-640} batch=16 tent=0
  local dir; [ "$tarefa" = seg ] && dir=runs/segment/$nome || dir=runs/detect/$nome
  while ! concluido "$nome" && [ $tent -lt 3 ]; do
    tent=$((tent+1))
    if [ -f "$dir/weights/last.pt" ]; then
      status "$nome: retomando de last.pt (tentativa $tent)"
      $P scripts/treinar.py --retomar "$dir/weights/last.pt" 2>&1 | tr '\r' '\n' >> "$LOGS/$nome.progress.log"
    else
      status "$nome: treino novo, batch=$batch imgsz=$imgsz (tentativa $tent)"
      $P scripts/treinar.py --tarefa "$tarefa" --modelo "$modelo" --epocas "$epocas" --batch $batch --imgsz $imgsz --nome "$nome" \
        2>&1 | tr '\r' '\n' >> "$LOGS/$nome.progress.log"
    fi
    if ! concluido "$nome"; then
      if tail -n 300 "$LOGS/$nome.progress.log" | grep -qi "out of memory"; then
        batch=$(( batch / 2 )); [ $batch -lt 4 ] && batch=4
        log "$nome: OOM detectado — reiniciando do zero com batch=$batch"; rm -rf "$dir"
      else
        log "$nome: terminou sem marcador de conclusão (tentativa $tent); ver $LOGS/$nome.progress.log"
      fi
      if [ -f "$dir/weights/best.pt" ] && grep -q "epochs completed" "$LOGS/$nome.progress.log"; then
        log "$nome: pesos finais existem; validando separadamente"
        $P - "$nome" "$tarefa" "$dir/weights/best.pt" "$imgsz" <<'PY' 2>&1 | tee -a "$LOGS/$nome.log"
import sys; from ultralytics import YOLO
nome, tarefa, best, imgsz = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
v = YOLO(best).val(data="dataset/data.yaml", split="val", imgsz=imgsz, plots=False, project="runs/eval", name=f"{nome}_val", exist_ok=True)
print(f"[{nome}] val: box mAP50={v.box.map50:.3f} mAP50-95={v.box.map:.3f} P={v.box.mp:.3f} R={v.box.mr:.3f}")
if tarefa == "seg": print(f"[{nome}] val: mask mAP50={v.seg.map50:.3f} mAP50-95={v.seg.map:.3f} P={v.seg.mp:.3f} R={v.seg.mr:.3f}")
PY
      fi
    fi
  done
  concluido "$nome" && log "$nome: CONCLUÍDO — $(grep "\[$nome\] val:" "$LOGS/$nome.log" | tail -2 | tr '\n' ' ')" \
                     || log "$nome: FALHOU após $tent tentativas"
}

log "===== pipeline iniciado (pid $$) ====="

# 0) baseline: pode estar rodando fora deste script; espera progresso; se parar sem concluir, retoma
status "baseline_n: aguardando conclusão"
while ! concluido baseline_n; do
  if [ "$(idade "$LOGS/baseline_n.progress.log")" -gt 600 ]; then
    log "baseline_n: sem progresso há 10 min — assumindo controle"
    treinar baseline_n det yolo11n.pt 30; break
  fi
  sleep 60
done
concluido baseline_n && log "baseline_n: $(grep '\[baseline_n\] val:' "$LOGS/baseline_n.log" | tail -1)"

# 1) QC do SAM (só uma vez)
if [ ! -f "$LOGS/qc_resultado.txt" ]; then
  status "qc_sam: medindo IoU SAM × polígonos humanos (300 instâncias)"
  $P scripts/refinar_mascaras_sam.py qc dataset --amostra 300 --saida figs 2>&1 | tee "$LOGS/qc_sam.log"
  rc=${PIPESTATUS[0]}
  if [ $rc -eq 0 ]; then echo APROVADO > "$LOGS/qc_resultado.txt"; elif [ $rc -eq 2 ]; then echo REPROVADO > "$LOGS/qc_resultado.txt"; else echo ERRO > "$LOGS/qc_resultado.txt"; fi
fi
log "qc_sam: $(cat "$LOGS/qc_resultado.txt") — $(grep 'QC SAM' "$LOGS/qc_sam.log" 2>/dev/null | tail -1)"

# 2) refinar labels (só se aprovado e ainda não feito)
if [ "$(cat "$LOGS/qc_resultado.txt")" = APROVADO ] && [ ! -d dataset/train/labels_original ]; then
  status "refinar_sam: gerando máscaras para as instâncias só-caixa (train/valid/test)"
  $P scripts/refinar_mascaras_sam.py refinar dataset --splits train valid test --saida figs 2>&1 | tee "$LOGS/refinar_sam.log"
  grep -q "caixas encontradas" "$LOGS/refinar_sam.log" && log "refinar_sam: $(grep 'caixas encontradas' "$LOGS/refinar_sam.log")" \
                                                        || log "refinar_sam: ERRO — labels originais preservadas em labels_original"
fi

# 3) detecção principal · 4) segmentação principal
treinar det_s det yolo11s.pt 50
treinar seg_s seg yolo11s-seg.pt 50

# 5) extras para a tabela comparativa
treinar det_s800 det yolo11s.pt 50 800
treinar det_m    det yolo11m.pt 50

status "TREINOS CONCLUÍDOS: $(date '+%F %T')"
log "===== pipeline terminado ====="

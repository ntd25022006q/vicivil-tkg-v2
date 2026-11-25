#!/bin/bash
# daily_pipeline: chuỗi realtime hoàn chỉnh — chạy thay cho watch --once đơn lẻ
cd /home/devlab1/vicivil_tkg || exit 1
BEFORE=$(cat delta/delta_*.jsonl 2>/dev/null | wc -l)
python3 -m vtg.watch --once
python3 -m vtg.integrate          # delta -> integrated/ (gate)
python3 -m vtg.temporalize        # refresh entity intervals
python3 -m vtg.local_update       # calibrate embedding cho entity mới (GEDUM/DINGAL-O)
python3 -m vtg.topology_audit     # dual-support audit
python3 -m vtg.queue_export       # refresh human_queue.csv
AFTER=$(cat delta/delta_*.jsonl 2>/dev/null | wc -l)
echo "[$(date '+%F %T')] pipeline done: delta records before=$BEFORE after=$AFTER"

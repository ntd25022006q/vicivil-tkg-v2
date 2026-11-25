#!/bin/bash
# daily_pipeline: chuỗi realtime hoàn chỉnh — chạy thay cho watch --once đơn lẻ
cd /home/devlab1/vicivil_tkg || exit 1
BEFORE=$(wc -l < delta/delta_*.jsonl 2>/dev/null | tail -1)
python3 -m vtg.watch --once
python3 -m vtg.integrate          # delta -> integrated/ (gate)
python3 -m vtg.temporalize        # refresh entity intervals
python3 -m vtg.local_update       # calibrate embedding cho entity mới (GEDUM/DINGAL-O)
python3 -m vtg.topology_audit     # dual-support audit
python3 -m vtg.queue_export       # refresh human_queue.csv
AFTER=$(wc -l < delta/delta_*.jsonl 2>/dev/null | tail -1)
echo "[$(date '+%F %T')] pipeline done: delta lines before=$BEFORE after=$AFTER"

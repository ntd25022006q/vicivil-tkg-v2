#!/bin/bash
# finalize: dọn dẹp + xử lý nốt 100% phần còn lại → v0.8.0
set -x
cd /home/devlab1/vicivil_tkg || exit 1

# 1. dọn đồ tự tạo: logs về logs/, bỏ venv hỏng + pycache + lock rác
mkdir -p logs
mv -f build_*.log p_*.log p_review*.log p4*.log p_tar.log nohup.out .watch.lock logs/ 2>/dev/null
rm -rf .venv vtg/__pycache__ __pycache__
rm -f ";" "print(\"hello world\").py" 2>/dev/null

# 2. thẩm nốt 539 pending tier3
python3 -m vtg.pending_adjudicate

# 3. agent review round 2 trên P1/P2 mới lộ sau rebuild
python3 -m vtg.agent_review

# 4. rebuild áp toàn bộ veto/approve
python3 -m vtg.run_all --llm --version 0.8.0

# 5. downstream đầy đủ
python3 -m vtg.temporalize
python3 -m vtg.integrate
python3 -m vtg.topology_audit
python3 -m vtg.local_update
python3 -m vtg.queue_export
python3 -m vtg.triage --queue-only

# 6. CHANGES + SHA256 v0.8.0
python3 - <<'EOF'
import json, hashlib, os
G = "/home/devlab1/vicivil_tkg/graph"
c = {
    "version": "0.8.0",
    "date": "2026-11-26",
    "changes": [
        "repo cleanup: logs/ archived, broken .venv + pycache + ad-hoc files removed",
        "pending tier3 band fully adjudicated (round 2)",
        "agent review round 2 applied to residual P1/P2 (convergence pass)",
        "full rebuild + downstream; SHACL/drift re-verified",
    ],
}
json.dump(c, open(os.path.join(G, "CHANGES.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
sums = []
for fn in sorted(os.listdir(G)):
    p = os.path.join(G, fn)
    if os.path.isfile(p) and fn not in ("SHA256SUMS", "CHANGES.json"):
        sums.append(f"{hashlib.sha256(open(p,'rb').read()).hexdigest()}  {fn}")
open(os.path.join(G, "SHA256SUMS"), "w", encoding="utf-8").write("\n".join(sums) + "\n")
print("v0.8.0 |", len(sums), "files hashed")
EOF
echo "FINALIZE_DONE"

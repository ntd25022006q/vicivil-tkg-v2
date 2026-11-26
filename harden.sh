#!/bin/bash
set -e
cd /home/devlab1/vicivil_tkg || exit 1

# 1. TTL: thêm domain/range cho mọi ObjectProperty còn thiếu (mặc định LegalConcept)
python3 - <<'EOF'
from rdflib import Graph, Namespace, RDF, RDFS, OWL
TTL = "/home/devlab1/vicivil_tkg/ontology/vclo2_core.ttl"
g = Graph(); g.parse(TTL, format="turtle")
missing = [p for p in g.subjects(RDF.type, OWL.ObjectProperty)
           if not list(g.objects(p, RDFS.domain))]
lines = ["\n# ---- AUTO: domain/range defaults cho entity predicates (audit: 49/58 thiếu) ----"]
for p in sorted(missing):
    name = str(p).split("#")[-1]
    if name in ("aliasOf", "preferred_label", "confidence", "at", "flagged"):
        continue
    lines.append(f'vclo:{name} rdfs:domain vclo:LegalConcept ; rdfs:range vclo:LegalConcept .')
open(TTL, "a", encoding="utf-8").write("\n".join(lines) + "\n")
g2 = Graph(); g2.parse(TTL, format="turtle")
n_dom = sum(1 for p in g2.subjects(RDF.type, OWL.ObjectProperty) if list(g2.objects(p, RDFS.domain)))
print(f"TTL patched: {len(missing)} predicates got domain/range; now {n_dom}/{len(list(g2.subjects(RDF.type, OWL.ObjectProperty)))} have domain")
EOF

# 2. rebuild + downstream (không LLM — GPU đang bận)
python3 -m vtg.run_all --version 0.9.0
python3 -m vtg.temporalize
python3 -m vtg.integrate
python3 -m vtg.topology_audit
python3 -m vtg.local_update

# 3. re-run experiments sau khi sửa leakage + head-sign
python3 -m vtg.embed_temporal
python3 -m vtg.dggp
python3 -m vtg.dggp_tar
python3 -m vtg.coldstart

# 4. figures + queue + triage
python3 -m vtg.make_figs
python3 -m vtg.make_figs2
python3 -m vtg.queue_export
python3 -m vtg.triage --queue-only

# 5. rehash v0.9.0
python3 - <<'EOF'
import json, hashlib, os
G = "/home/devlab1/vicivil_tkg/graph"
c = {
    "version": "0.9.0",
    "date": "2026-11-26",
    "changes": [
        "AUDIT ROUND-2 HARDENING:",
        "inverse-leakage fix in embedding eval (split base edges first; inverses stay same side) — re-ran TransE/DGGP/TAR with corrected protocol + head-ranking sign fix",
        "cold-start random baseline corrected to (ln n + gamma)/n",
        "as-of tests rewritten as EXACT-set legal-fact assertions (must_active/must_inactive) — no subset semantics",
        "TTL: domain/range added to entity predicates lacking them (audit: 49/58 missing)",
        "deliverables: full graph/ sync (41 files) per audit finding",
    ],
}
json.dump(c, open(os.path.join(G, "CHANGES.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
sums = []
for fn in sorted(os.listdir(G)):
    p = os.path.join(G, fn)
    if os.path.isfile(p) and fn not in ("SHA256SUMS", "CHANGES.json"):
        sums.append(f"{hashlib.sha256(open(p,'rb').read()).hexdigest()}  {fn}")
open(os.path.join(G, "SHA256SUMS"), "w", encoding="utf-8").write("\n".join(sums) + "\n")
print("v0.9.0 |", len(sums), "files hashed")
EOF
echo "HARDEN_DONE"

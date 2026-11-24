"""run_all: orchestrator for the 6-step build. Usage:
    python3 -m vtg.run_all [--ws ~/law_workspace] [--out ~/vicivil_tkg/graph] [--skip-llm]
Read-only on --ws; all outputs go to --out.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vtg.canonicalize import run as canon_run
from vtg.graft_temporal import run as graft_run
from vtg.build_graph import (build_edges_norm, build_nodes_legal, export_ttl,
                             release, retype_relationships, validate)
from vtg.case_wire import run as case_run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ws", default=os.path.expanduser("~/law_workspace"))
    ap.add_argument("--out", default=os.path.expanduser("~/vicivil_tkg/graph"))
    ap.add_argument("--version", default="0.1.0")
    ap.add_argument("--max-judgments", type=int, default=None)
    ap.add_argument("--llm", action="store_true",
                    help="enable tier-3 LLM adjudication (ollama, small model)")
    args = ap.parse_args()
    ws, out = os.path.expanduser(args.ws), os.path.expanduser(args.out)
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    report = {"started": time.strftime("%Y-%m-%dT%H:%M:%S"), "ws": ws, "out": out}

    print("=" * 70)
    print("ViCivil-TKG v2 build — JSONL source of truth, TTL on publish")
    print("=" * 70)

    tg, nodes_legal = build_nodes_legal(ws, out, log=print)
    edges_norm = build_edges_norm(ws, out, tg, log=print)
    rstats, _ = retype_relationships(ws, out, log=print)
    gstats = graft_run(ws, out, log=print)
    cstats = canon_run(ws, out, log=print, use_llm=args.llm, llm_model="qwen3.5:4b")
    sstats = case_run(ws, out, log=print, max_judgments=args.max_judgments)
    vstats = validate(out, nodes_legal, log=print)
    ttl = export_ttl(out, ws, log=print)
    release(out, args.version, log=print)

    report.update({"retype": rstats, "graft": gstats, "canonicalize": cstats,
                   "case_wire": sstats, "validate": vstats, "ttl": ttl,
                   "seconds": round(time.time() - t0, 1)})
    with open(os.path.join(out, "build_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print("=" * 70)
    print("BUILD REPORT")
    print(json.dumps(report, ensure_ascii=False, indent=1)[:2000])
    print("=" * 70)


if __name__ == "__main__":
    main()

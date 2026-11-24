# vicivil_tkg — ViCivil-TKG v2 build package

Ontology-based temporal KG for Vietnamese civil law: grafts (A) VCLO 2.0 schema,
(B) article-level temporality, (C) entity canonicalization onto the EXISTING
law_workspace assets — **read-only** wrt `~/law_workspace`.

## Run
```bash
cd ~/vicivil_tkg
python3 -m vtg.run_all                    # full 6-step build
python3 -m vtg.run_all --max-judgments 200  # smoke test
```
Outputs → `~/vicivil_tkg/graph/` (JSONL source of truth + TTL export + SHA256SUMS + CHANGES.json + build_report.json).

## Registry contract
- `nodes_legal.jsonl` — documents (so_hieu/ten_full/validity/status)
- `nodes_entity.jsonl` — canonical entities (class VCLO + aliases + norm_key + confidence + law_scope)
- `edges_norm.jsonl` — AMENDS/REPLACES/CONSOLIDATES/GUIDES/CITES/SUCCEEDS with provenance
- `edges_entity.jsonl` — typed relations (≤40 predicates, from 6,093 free-form keywords)
- `edges_case.jsonl` — APPLIES(verdict → Law|DieuN) with as_of=verdict_date
- `merge_log.jsonl` — every canonicalization merge (rollback = union-find replay)
- `article_validity.json` + `article_amend_facts.jsonl` — T1 graft
- `literals.jsonl` — demoted date/duration/money pseudo-entities

## AC (acceptance criteria v0.1)
1. 0 entities untyped; date/duration/money demoted to literals
2. ≤40 distinct predicates; SHACL-style validator 0 violations in kept set
3. Resolver doc-level tests (3 as_of windows) pass
4. ≥1 APPLIES edge for judgments with resolvable citations
5. SHA256SUMS + CHANGES.json present (audit discipline v1.0.1)

## Next (not in v0.1)
- tier-3 LLM adjudication via ollama (small model) — hook ready in canonicalize.py
- amendment text ingestion for historical article text
- realtime loop (crawl queue → hash diff → delta) plugging into this build

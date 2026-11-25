"""onto_check: prove the ontology is LOAD-BEARING.
1. Self-test TTL parse (classes/predicates from vclo2_core.ttl via rdflib).
2. Drift check: every predicate used in edges_norm/edges_entity must be DECLARED in the TTL.
3. Real pyshacl validation of the exported graph TTL against vclo2_shapes.ttl.
Output: graph/shacl_report.json
"""
import json
import os

from .ontology import self_test, declared_predicates
from .build_graph import jl

G = os.path.expanduser("~/vicivil_tkg/graph")
ONTO = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "ontology", "vclo2_shapes.ttl")


def main():
    self_test()
    # ---- drift check (normalize: registry stores short 'vclo:x'; TTL stores full URIs)
    PREFIX = "http://synapselex.org/ontology/vclo#"
    declared_short = {p.replace(PREFIX, "vclo:") for p in declared_predicates()}
    drift = {}
    for name in ("edges_norm.jsonl", "edges_entity.jsonl", "edges_case.jsonl"):
        used = {e["relation"] for e in jl(os.path.join(G, name))}
        undeclared = sorted(used - declared_short)
        drift[name] = {"used": len(used), "undeclared": undeclared}
    print("[DRIFT]", json.dumps(drift, ensure_ascii=False))

    # ---- real SHACL validation (pyshacl)
    from pyshacl import validate
    from rdflib import Graph
    data = Graph()
    data.parse(os.path.join(G, "vcivil_tkg_graph.ttl"), format="turtle")
    shapes = Graph()
    shapes.parse(ONTO, format="turtle")
    conforms, results_graph, results_text = validate(
        data_graph=data, shacl_graph=shapes, inference="none", advanced=True)
    n_viol = results_text.count("Constraint Violation")
    print(f"[SHACL] conforms={conforms} violations={n_viol}")
    preview = "\n".join(l for l in results_text.splitlines()
                        if "Constraint Violation" in l or "Focus Node" in l)[:1000]
    print(preview)
    report = {"conforms": bool(conforms), "violations": n_viol, "drift": drift,
              "pyshacl": "0.40.2", "shapes": ONTO}
    json.dump(report, open(os.path.join(G, "shacl_report.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()

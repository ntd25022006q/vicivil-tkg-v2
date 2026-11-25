"""ontology: SINGLE SOURCE OF TRUTH — load vclo2_core.ttl via rdflib.
Every module must derive its predicate/class sets from here, never hardcode.
(Bài học từ kiểm toán SynapseLex: ontology không được load = ontology trang trí.)
"""
import os
from functools import lru_cache

from rdflib import Graph, Namespace, RDF, RDFS, OWL

VCLO = Namespace("http://synapselex.org/ontology/vclo#")
TTL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "ontology", "vclo2_core.ttl")

_NORM_DOMAINS = {
    "http://synapselex.org/ontology/vclo#NormativeDocument",
    "http://synapselex.org/ontology/vclo#ConsolidatedVersion",
    "http://synapselex.org/ontology/vclo#SubLawDecree",
    "http://synapselex.org/ontology/vclo#DocumentVersion",
    "http://synapselex.org/ontology/vclo#Article",
    "http://synapselex.org/ontology/vclo#Clause",
    "http://synapselex.org/ontology/vclo#Point",
    "http://synapselex.org/ontology/vclo#CourtVerdict",
}


@lru_cache(maxsize=1)
def _graph():
    g = Graph()
    g.parse(TTL, format="turtle")
    return g


@lru_cache(maxsize=1)
def _predicates():
    """All vclo: owl:ObjectProperty URIs declared in the TTL."""
    return {str(p) for p in _graph().subjects(RDF.type, OWL.ObjectProperty)
            if str(p).startswith(str(VCLO))}


@lru_cache(maxsize=1)
def _domains():
    d = {}
    for p in _graph().subjects(RDF.type, OWL.ObjectProperty):
        for dom in _graph().objects(p, RDFS.domain):
            d[str(p)] = str(dom)
    return d


@lru_cache(maxsize=1)
def classes():
    return {str(c) for c in _graph().subjects(RDF.type, OWL.Class)
            if str(c).startswith(str(VCLO))}


def class_exists(uri: str) -> bool:
    return uri in classes()


def norm_predicates() -> set:
    return {p for p in _predicates() if _domains().get(p) in _NORM_DOMAINS}


def entity_predicates() -> set:
    return _predicates() - norm_predicates() - {str(VCLO.aliasOf)}


def declared_predicates() -> set:
    return set(_predicates())


def extraction_predicates() -> list:
    """Predicates offered to the LLM extractor (entity-level, no junk catcher)."""
    return sorted(p.replace(str(VCLO), "") for p in entity_predicates()
                  if p != str(VCLO.relatedTo))


def self_test():
    g = _graph()
    print(f"[ONTOLOGY] parsed {TTL}")
    print(f"  triples={len(g)} classes={len(classes())} predicates={len(_predicates())}")
    print(f"  norm_predicates={len(norm_predicates())} entity_predicates={len(entity_predicates())}")
    print(f"  extraction list ({len(extraction_predicates())}):", ", ".join(extraction_predicates()[:10]), "...")


if __name__ == "__main__":
    self_test()

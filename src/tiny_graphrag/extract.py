from dataclasses import dataclass
from functools import cache
from typing import List, Tuple

import glirel  # noqa: F401 Import time side effect
import spacy

from tiny_graphrag.config import DEVICE


@dataclass
class ExtractionResult:
    """Represents the result of entity and relation extraction from text.

    Contains lists of extracted entities and their relationships.
    """

    entities: List[Tuple[str, str]]  # (text, label)
    relations: List[Tuple[str, str, str]]  # (head_text, label, tail_text)


@cache
def nlp_model(threshold: float, entity_types: tuple[str], device: str = DEVICE):
    """Instantiate a spacy model with GLiNER and GLiREL components."""
    custom_spacy_config = {
        "gliner_model": "urchade/gliner_mediumv2.1",
        "chunk_size": 250,
        "labels": entity_types,
        "style": "ent",
        "threshold": threshold,
        "map_location": device,
    }
    if device != "cpu":
        spacy.require_gpu()  # type: ignore

    nlp = spacy.blank("en")
    nlp.add_pipe("gliner_spacy", config=custom_spacy_config)
    nlp.add_pipe("glirel", after="gliner_spacy")
    return nlp


def extract_rels(
    text: str,
    entity_types: List[str],
    relation_types: List[str],
    threshold: float = 0.75,
) -> ExtractionResult:
    """Extract entities and relations from text using GLiNER and GLiREL."""
    nlp = nlp_model(threshold, tuple(entity_types))
    docs = list(nlp.pipe([(text, {"glirel_labels": relation_types})], as_tuples=True))
    relations = docs[0][0]._.relations

    sorted_data_desc = sorted(relations, key=lambda x: x["score"], reverse=True)

    # Extract entities
    ents = [(ent.text, ent.label_) for ent in docs[0][0].ents]

    # Extract relations
    rels = [
        (" ".join(item["head_text"]), item["label"], " ".join(item["tail_text"]))
        for item in sorted_data_desc
        if item["score"] >= threshold
    ]

    return ExtractionResult(entities=ents, relations=rels)

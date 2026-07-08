from __future__ import annotations

from backend.rag import ingest, retrieval
from backend.shared.schemas import ScoredProcedure


def test_search_returns_scored_procedures(sample_settings, fake_embeddings):
    ingest.build_index(sample_settings)

    results = retrieval.search("select a rectangle", top_k=3, settings=sample_settings)

    assert len(results) == 3
    assert all(isinstance(r, ScoredProcedure) for r in results)
    assert all(isinstance(r.distance, float) for r in results)
    names = {r.procedure.name for r in results}
    assert names.issubset(
        {
            "gimp-image-select-rectangle",
            "gimp-context-set-foreground",
            "gimp-image-flatten",
            "gimp-layer-new",
            "gimp-image-resize",
        }
    )


def test_search_uses_settings_default_top_k(sample_settings, fake_embeddings):
    ingest.build_index(sample_settings)
    sample_settings.rag_top_k = 2

    results = retrieval.search("resize the canvas", settings=sample_settings)

    assert len(results) == 2


def test_search_embeds_the_query(sample_settings, fake_embeddings):
    ingest.build_index(sample_settings)

    retrieval.search("flatten the image", settings=sample_settings)

    assert fake_embeddings.embed_query_calls == ["flatten the image"]


def test_row_to_scored_procedure_parses_json_fields():
    row = {
        "name": "gimp-layer-new",
        "proc_type": "PLUGIN",
        "blurb": "Create a new layer.",
        "help": "",
        "menu_label": "",
        "authors": "",
        "copyright": "",
        "date": "",
        "args": '[{"name": "image", "type": "GimpImage", "nick": "", "description": ""}]',
        "return_values": '[{"name": "layer", "type": "GimpLayer", "nick": "", "description": ""}]',
        "deprecated": False,
        "embedding_text": "gimp layer new",
        "vector": [0.1, 0.2],
        "_distance": 0.123,
    }

    scored = retrieval._row_to_scored_procedure(row)

    assert scored.distance == 0.123
    assert scored.procedure.name == "gimp-layer-new"
    assert scored.procedure.args[0].name == "image"
    assert scored.procedure.return_values[0].name == "layer"

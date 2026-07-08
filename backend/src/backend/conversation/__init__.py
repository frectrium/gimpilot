"""LangGraph conversation agent (milestone 4, not yet implemented).

This is where the retrieve -> agent graph will live: a "retrieve" node
querying `backend.rag.search` for candidate PDB procedures, and an "agent"
node calling Gemini with those candidates bound as per-turn dynamic tools,
checkpointed with LangGraph's `MemorySaver` keyed by `thread_id`. See the
root README's "Backend architecture" section for the full design.
"""

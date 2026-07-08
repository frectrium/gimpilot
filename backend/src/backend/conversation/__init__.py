"""The LangGraph conversation agent: retrieve candidate PDB procedures via
RAG, then ask Gemini to call one or respond directly. One retrieve+agent
pass per `/converse` HTTP call — see `graph.py` for why.
"""

from backend.conversation.graph import build_graph

__all__ = ["build_graph"]
